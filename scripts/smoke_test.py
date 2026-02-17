#!/usr/bin/env python
"""Smoke test leggero per CI e verifica rapida del progetto.

Esegue:
1. Caricamento dataset AMELIA (subset ridotto)
2. Addestramento baseline TF-IDF + LR
3. Verifica che le metriche e i file di output vengano generati

Pensato per durare pochi minuti, adatto alla CI.

Uso:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Aggiungi src/ al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amelia_experiment.config import MODELS_DIR, RESULTS_METRICS_DIR
from amelia_experiment.logging_utils import setup_logging

logger = logging.getLogger(__name__)

TRAIN_LIMIT = 200
VAL_LIMIT = 50


def main() -> None:
    """Esegue lo smoke test."""
    setup_logging()
    logger.info("=== Smoke Test ===")

    # 1. Caricamento dataset (subset ridotto)
    logger.info("1) Caricamento dataset AMELIA (train=%d, val=%d) …", TRAIN_LIMIT, VAL_LIMIT)
    from amelia_experiment.dataset import load_amelia

    train_ds = load_amelia(split="train", limit=TRAIN_LIMIT, seed=42)
    val_ds = load_amelia(split="validation", limit=VAL_LIMIT, seed=42)

    assert len(train_ds) == TRAIN_LIMIT, f"Attesi {TRAIN_LIMIT} train, trovati {len(train_ds)}"
    assert len(val_ds) == VAL_LIMIT, f"Attesi {VAL_LIMIT} val, trovati {len(val_ds)}"
    logger.info("   Dataset caricato correttamente.")

    # 2. Addestramento baseline
    logger.info("2) Addestramento baseline TF-IDF + LR …")
    from amelia_experiment.baseline import train_tfidf_lr

    result = train_tfidf_lr(train_ds, val_ds, seed=42)
    macro_f1 = result["val_metrics"]["macro_f1"]
    logger.info("   Macro-F1 validation: %.4f", macro_f1)

    # 3. Verifica file di output
    logger.info("3) Verifica artefatti …")
    model_path = MODELS_DIR / "baseline_tfidf_lr.joblib"
    assert model_path.exists(), f"File modello non trovato: {model_path}"

    metrics_json = RESULTS_METRICS_DIR / "baseline_validation.json"
    assert metrics_json.exists(), f"File metriche JSON non trovato: {metrics_json}"

    metrics_csv = RESULTS_METRICS_DIR / "baseline_validation.csv"
    assert metrics_csv.exists(), f"File metriche CSV non trovato: {metrics_csv}"

    logger.info("   Tutti gli artefatti presenti.")

    # 4. Verifica metriche ragionevoli
    assert 0.0 < macro_f1 <= 1.0, f"Macro-F1 fuori range: {macro_f1}"
    logger.info("   Metriche ragionevoli (macro_f1=%.4f).", macro_f1)

    logger.info("=== Smoke Test SUPERATO ===")


if __name__ == "__main__":
    main()
