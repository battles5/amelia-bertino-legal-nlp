"""Wrapper per fastText (opzionale).

Se il modulo ``fasttext`` non è installato, le funzioni sollevano
un errore chiaro senza far crashare l'intero progetto.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from amelia_experiment.artifacts import save_dict_json, save_metrics_csv
from amelia_experiment.config import (
    LABEL_COL,
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    TEXT_COL,
)
from amelia_experiment.metrics import classification_dict
from amelia_experiment.preprocess import encode_label, normalize_text

logger = logging.getLogger(__name__)

# ── Verifica disponibilità fasttext ──────────────────────────────────────────

_FASTTEXT_AVAILABLE = False
try:
    import fasttext  # noqa: F401

    _FASTTEXT_AVAILABLE = True
except ImportError:
    pass


def _check_fasttext() -> None:
    """Verifica che il modulo fasttext sia disponibile.

    Raises:
        ImportError: con messaggio chiaro su come installare.
    """
    if not _FASTTEXT_AVAILABLE:
        raise ImportError(
            "Il modulo 'fasttext' non è installato. "
            "Installalo con: pip install fasttext  (oppure pip install fasttext-wheel su Windows). "
            "La baseline fastText è opzionale — il resto del progetto funziona senza."
        )


# ── Conversione dataset → formato fastText ──────────────────────────────────


def dataset_to_fasttext_file(ds, output_path: Path) -> Path:
    """Converte un dataset HF in un file di testo nel formato fastText.

    Ogni riga ha il formato: ``__label__prem testo normalizzato``

    Args:
        ds: dataset HF con colonne Text e Component.
        output_path: percorso del file di output.

    Returns:
        Percorso del file creato.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for text, label in zip(ds[TEXT_COL], ds[LABEL_COL], strict=True):
            clean = normalize_text(text).replace("\n", " ")
            f.write(f"__label__{label} {clean}\n")

    logger.info("File fastText scritto: %s (%d righe)", output_path, len(ds))
    return output_path


# ── Training ─────────────────────────────────────────────────────────────────


def train_fasttext(
    train_ds,
    val_ds,
    *,
    lr: float = 0.5,
    epoch: int = 25,
    word_ngrams: int = 2,
    dim: int = 100,
    seed: int = 42,
) -> dict:
    """Addestra un classificatore fastText e salva modello + metriche.

    Args:
        train_ds: dataset di training.
        val_ds: dataset di validazione.
        lr: learning rate.
        epoch: numero di epoche.
        word_ngrams: n-grammi di parole.
        dim: dimensione degli embedding.
        seed: seed per riproducibilità.

    Returns:
        Dizionario con chiavi ``model``, ``val_metrics``, ``val_preds``.
    """
    _check_fasttext()
    import fasttext as ft

    # Scrivi file temporanei nel formato fastText
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = dataset_to_fasttext_file(train_ds, Path(tmpdir) / "train.txt")

        logger.info("Addestramento fastText (epoch=%d, lr=%.2f) …", epoch, lr)
        model = ft.train_supervised(
            input=str(train_path),
            lr=lr,
            epoch=epoch,
            wordNgrams=word_ngrams,
            dim=dim,
            seed=seed,
            verbose=0,
        )

    # Predizioni su validation
    val_texts = [normalize_text(t) for t in val_ds[TEXT_COL]]
    val_labels = [encode_label(lab) for lab in val_ds[LABEL_COL]]

    val_preds = []
    for text in val_texts:
        pred_labels, _ = model.predict(text.replace("\n", " "))
        # pred_labels è tipo ['__label__prem']
        pred_str = pred_labels[0].replace("__label__", "")
        val_preds.append(encode_label(pred_str))

    val_metrics = classification_dict(val_labels, val_preds)
    logger.info("Macro-F1 validation (fastText): %.4f", val_metrics["macro_f1"])

    # Salvataggio modello
    model_path = MODELS_DIR / "fasttext.bin"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info("Modello fastText salvato in %s", model_path)

    # Salvataggio metriche
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_dict_json(val_metrics, RESULTS_METRICS_DIR / "fasttext_validation.json")
    save_metrics_csv(val_metrics, RESULTS_METRICS_DIR / "fasttext_validation.csv")

    return {
        "model": model,
        "val_metrics": val_metrics,
        "val_preds": val_preds,
    }


def load_fasttext_model(path: Path | None = None):
    """Carica un modello fastText salvato.

    Args:
        path: percorso del file .bin.

    Returns:
        Modello fastText.
    """
    _check_fasttext()
    import fasttext as ft

    if path is None:
        path = MODELS_DIR / "fasttext.bin"
    logger.info("Caricamento modello fastText da %s", path)
    return ft.load_model(str(path))
