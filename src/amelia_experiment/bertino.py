"""Fine-tuning e inferenza con BERTino (DistilBERT italiano).

Usa ``transformers.Trainer`` per il fine-tuning su classificazione binaria
(prem vs conc) e fornisce funzioni per predizione batch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from amelia_experiment.artifacts import save_dict_json, save_metrics_csv
from amelia_experiment.config import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_SEED,
    ID2LABEL,
    LABEL2ID,
    LABEL_COL,
    MODEL_ID,
    MODELS_DIR,
    NUM_LABELS,
    RESULTS_METRICS_DIR,
    TEXT_COL,
)
from amelia_experiment.metrics import classification_dict, compute_metrics_for_trainer
from amelia_experiment.preprocess import encode_label

logger = logging.getLogger(__name__)

# ── Directory di default per i checkpoint ────────────────────────────────────
BERTINO_OUTPUT_DIR: Path = MODELS_DIR / "bertino_finetuned"


# ── Seed riproducibile ──────────────────────────────────────────────────────


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Imposta seed per random, numpy e torch."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Tokenizzazione ──────────────────────────────────────────────────────────


def tokenize_dataset(ds, tokenizer, max_length: int = DEFAULT_MAX_LENGTH):
    """Tokenizza un dataset HF aggiungendo colonne ``input_ids``, ``attention_mask``, ``labels``.

    Args:
        ds: dataset HuggingFace con colonne Text e Component.
        tokenizer: tokenizer HF.
        max_length: lunghezza massima in token.

    Returns:
        Dataset tokenizzato, pronto per il Trainer.
    """

    def _tokenize_fn(batch):
        encoded = tokenizer(
            batch[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = [encode_label(lab) for lab in batch[LABEL_COL]]
        return encoded

    cols_to_remove = [
        c for c in ds.column_names if c not in ("input_ids", "attention_mask", "labels")
    ]
    tokenized = ds.map(_tokenize_fn, batched=True, remove_columns=cols_to_remove)
    tokenized.set_format("torch")
    return tokenized


# ── Fine-tuning ──────────────────────────────────────────────────────────────


def finetune_bertino(
    train_ds,
    val_ds,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_steps: int = -1,
    output_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Fine-tuning di BERTino per classificazione prem/conc.

    Args:
        train_ds: dataset di training (HF Dataset).
        val_ds: dataset di validazione.
        max_length: lunghezza massima tokenizzazione.
        num_train_epochs: numero di epoche.
        per_device_train_batch_size: batch size training.
        per_device_eval_batch_size: batch size eval.
        learning_rate: learning rate.
        weight_decay: weight decay.
        warmup_ratio: percentuale di warmup.
        max_train_samples: limite campioni training (None = tutti).
        max_eval_samples: limite campioni eval (None = tutti).
        max_steps: se > 0, prevale su num_train_epochs.
        output_dir: directory output del Trainer.
        seed: seed riproducibilità.

    Returns:
        Dizionario con ``model``, ``tokenizer``, ``val_metrics``, ``trainer``.
    """
    set_seed(seed)
    if output_dir is None:
        output_dir = BERTINO_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Caricamento tokenizer e modello %s …", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Sottocampionamento opzionale
    if max_train_samples is not None and max_train_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=seed).select(range(max_train_samples))
        logger.info("Training sottocampionato a %d esempi.", len(train_ds))
    if max_eval_samples is not None and max_eval_samples < len(val_ds):
        val_ds = val_ds.shuffle(seed=seed).select(range(max_eval_samples))
        logger.info("Validation sottocampionata a %d esempi.", len(val_ds))

    # Tokenizzazione
    logger.info("Tokenizzazione (max_length=%d) …", max_length)
    train_tok = tokenize_dataset(train_ds, tokenizer, max_length)
    val_tok = tokenize_dataset(val_ds, tokenizer, max_length)

    # Training arguments
    import math

    # Calcola warmup_steps da warmup_ratio (warmup_ratio deprecato in transformers >= 5.2)
    total_steps = (
        math.ceil(len(train_tok) / per_device_train_batch_size) * num_train_epochs
        if max_steps <= 0
        else max_steps
    )
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_for_trainer,
    )

    logger.info("Inizio fine-tuning BERTino …")
    trainer.train()

    # Valutazione finale su validation
    eval_result = trainer.evaluate()
    logger.info("Risultati validation: %s", eval_result)

    # Salvataggio modello + tokenizer
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Modello e tokenizer salvati in %s", final_dir)

    # Metriche strutturate
    preds_output = trainer.predict(val_tok)
    predictions = preds_output.predictions
    # predictions può essere una tupla (logits, hidden_states, …)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1)
    val_metrics = classification_dict(preds_output.label_ids, preds)

    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_dict_json(val_metrics, RESULTS_METRICS_DIR / "bertino_validation.json")
    save_metrics_csv(val_metrics, RESULTS_METRICS_DIR / "bertino_validation.csv")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "val_metrics": val_metrics,
    }


# ── Inferenza ────────────────────────────────────────────────────────────────


def load_bertino(model_dir: Path | None = None):
    """Carica modello BERTino fine-tuned e tokenizer.

    Args:
        model_dir: directory del checkpoint (default: models/bertino_finetuned/final).

    Returns:
        Tupla (model, tokenizer).
    """
    if model_dir is None:
        model_dir = BERTINO_OUTPUT_DIR / "final"
    model_dir = Path(model_dir)
    logger.info("Caricamento BERTino da %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return model, tokenizer


def predict_texts(
    texts: list[str],
    model=None,
    tokenizer=None,
    model_dir: Path | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[dict]:
    """Predice label e probabilità per una lista di testi.

    Args:
        texts: lista di stringhe.
        model: modello HF (opzionale, se già caricato).
        tokenizer: tokenizer HF (opzionale).
        model_dir: directory del checkpoint (usata se model/tokenizer non forniti).
        max_length: lunghezza massima token.

    Returns:
        Lista di dizionari con ``label``, ``label_id``, ``confidence``.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_bertino(model_dir)

    device = next(model.parameters()).device
    results = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, pred_id].item())

        results.append(
            {
                "label": ID2LABEL[pred_id],
                "label_id": pred_id,
                "confidence": confidence,
            }
        )

    return results
