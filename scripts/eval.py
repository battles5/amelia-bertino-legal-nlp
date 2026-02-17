#!/usr/bin/env python
"""CLI per valutazione modelli su validation/test e generazione report.

Carica i modelli salvati (baseline e/o BERTino) e produce:
- metriche JSON/CSV in results/metrics/
- confusion matrix in results/plots/
- tabella comparativa in results/tables/

Uso:
    python eval.py
    python eval.py --models baseline bertino
    python eval.py --splits validation test
    python eval.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amelia_experiment.artifacts import (
    generate_results_table_latex,
    save_dict_json,
    save_metrics_csv,
)
from amelia_experiment.config import (
    DEFAULT_SEED,
    LABEL_COL,
    MODELS_DIR,
    RESULTS_METRICS_DIR,
    RESULTS_TABLES_DIR,
    TEXT_COL,
)
from amelia_experiment.dataset import load_amelia
from amelia_experiment.logging_utils import setup_logging
from amelia_experiment.metrics import classification_dict
from amelia_experiment.plotting import plot_confusion_matrix
from amelia_experiment.preprocess import encode_label, normalize_text

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Valutazione modelli su AMELIA e generazione report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "bertino", "fasttext"],
        default=["baseline", "bertino"],
        help="Modelli da valutare.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["validation", "test"],
        default=["validation", "test"],
        help="Split su cui valutare.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed riproducibilità.")
    return parser.parse_args()


def evaluate_baseline(ds, split_name: str) -> dict:
    """Valuta la baseline TF-IDF + LR su uno split."""
    from amelia_experiment.baseline import load_baseline_pipeline

    pipeline = load_baseline_pipeline()
    texts = [normalize_text(t) for t in ds[TEXT_COL]]
    labels = [encode_label(lab) for lab in ds[LABEL_COL]]
    preds = pipeline.predict(texts).tolist()

    metrics = classification_dict(labels, preds)
    logger.info("Baseline — %s — Macro-F1: %.4f", split_name, metrics["macro_f1"])

    # Salva metriche
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_dict_json(metrics, RESULTS_METRICS_DIR / f"baseline_{split_name}.json")
    save_metrics_csv(metrics, RESULTS_METRICS_DIR / f"baseline_{split_name}.csv")

    # Plot confusion matrix
    plot_confusion_matrix(metrics["confusion_matrix"], "baseline", split_name)

    return metrics


def evaluate_bertino(ds, split_name: str) -> dict:
    """Valuta BERTino fine-tuned su uno split."""
    from amelia_experiment.bertino import load_bertino, predict_texts

    model, tokenizer = load_bertino()
    texts = list(ds[TEXT_COL])
    labels = [encode_label(lab) for lab in ds[LABEL_COL]]

    predictions = predict_texts(texts, model=model, tokenizer=tokenizer)
    preds = [p["label_id"] for p in predictions]

    metrics = classification_dict(labels, preds)
    logger.info("BERTino — %s — Macro-F1: %.4f", split_name, metrics["macro_f1"])

    # Salva metriche
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_dict_json(metrics, RESULTS_METRICS_DIR / f"bertino_{split_name}.json")
    save_metrics_csv(metrics, RESULTS_METRICS_DIR / f"bertino_{split_name}.csv")

    # Plot confusion matrix
    plot_confusion_matrix(metrics["confusion_matrix"], "bertino", split_name)

    return metrics


def evaluate_fasttext(ds, split_name: str) -> dict:
    """Valuta il modello fastText su uno split."""
    from amelia_experiment.fasttext_model import load_fasttext_model

    model = load_fasttext_model()
    texts = [normalize_text(t) for t in ds[TEXT_COL]]
    labels = [encode_label(lab) for lab in ds[LABEL_COL]]

    preds = []
    for text in texts:
        pred_labels, _ = model.predict(text.replace("\n", " "))
        pred_str = pred_labels[0].replace("__label__", "")
        preds.append(encode_label(pred_str))

    metrics = classification_dict(labels, preds)
    logger.info("fastText — %s — Macro-F1: %.4f", split_name, metrics["macro_f1"])

    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_dict_json(metrics, RESULTS_METRICS_DIR / f"fasttext_{split_name}.json")
    save_metrics_csv(metrics, RESULTS_METRICS_DIR / f"fasttext_{split_name}.csv")

    plot_confusion_matrix(metrics["confusion_matrix"], "fasttext", split_name)

    return metrics


EVALUATORS = {
    "baseline": evaluate_baseline,
    "bertino": evaluate_bertino,
    "fasttext": evaluate_fasttext,
}


def main() -> None:
    """Entry point principale."""
    setup_logging()
    args = parse_args()

    logger.info("=== Valutazione modelli: %s su split: %s ===", args.models, args.splits)

    # Raccoglie metriche per tabella comparativa
    all_metrics: dict[str, dict[str, dict]] = {}

    for model_name in args.models:
        all_metrics[model_name] = {}
        evaluator = EVALUATORS[model_name]

        for split_name in args.splits:
            # Controlla che il modello esista
            if model_name == "baseline" and not (MODELS_DIR / "baseline_tfidf_lr.joblib").exists():
                logger.warning("Modello baseline non trovato, skip.")
                continue
            if (
                model_name == "bertino"
                and not (MODELS_DIR / "bertino_finetuned" / "final").exists()
            ):
                logger.warning("Modello BERTino non trovato, skip.")
                continue
            if model_name == "fasttext" and not (MODELS_DIR / "fasttext.bin").exists():
                logger.warning("Modello fastText non trovato, skip.")
                continue

            ds = load_amelia(split=split_name, seed=args.seed)
            metrics = evaluator(ds, split_name)
            all_metrics[model_name][split_name] = metrics

    # Genera tabella comparativa
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = RESULTS_TABLES_DIR / "results_table.tex"
    generate_results_table_latex(all_metrics, table_path)

    logger.info("=== Valutazione completata ===")


if __name__ == "__main__":
    main()
