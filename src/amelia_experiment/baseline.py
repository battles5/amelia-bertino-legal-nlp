"""Baseline TF-IDF + Logistic Regression per classificazione prem/conc.

Addestra una pipeline scikit-learn (TfidfVectorizer → LogisticRegression),
salva il modello con joblib e le metriche in results/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

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


def train_tfidf_lr(
    train_ds,
    val_ds,
    *,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = 50_000,
    C: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Addestra la pipeline TF-IDF + Logistic Regression.

    Args:
        train_ds: dataset di training (colonne Text, Component).
        val_ds: dataset di validazione.
        ngram_range: range n-grammi per TfidfVectorizer.
        min_df: frequenza minima di documento per termine.
        max_features: numero massimo di feature.
        C: parametro di regolarizzazione per LogisticRegression.
        seed: seed per riproducibilità.

    Returns:
        Dizionario con chiavi ``pipeline``, ``val_metrics``, ``val_preds``.
    """
    logger.info("Preparazione dati per baseline TF-IDF + LR …")

    # Estrai testi e label
    train_texts = [normalize_text(t) for t in train_ds[TEXT_COL]]
    train_labels = [encode_label(lab) for lab in train_ds[LABEL_COL]]
    val_texts = [normalize_text(t) for t in val_ds[TEXT_COL]]
    val_labels = [encode_label(lab) for lab in val_ds[LABEL_COL]]

    # Pipeline scikit-learn
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    logger.info("Addestramento pipeline (train=%d) …", len(train_texts))
    pipeline.fit(train_texts, train_labels)

    # Predizioni su validation
    val_preds = pipeline.predict(val_texts).tolist()
    val_metrics = classification_dict(val_labels, val_preds)
    logger.info("Macro-F1 validation: %.4f", val_metrics["macro_f1"])

    # Salvataggio modello
    model_path = MODELS_DIR / "baseline_tfidf_lr.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Pipeline salvata in %s", model_path)

    # Salvataggio metriche
    RESULTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_dict_json(val_metrics, RESULTS_METRICS_DIR / "baseline_validation.json")
    save_metrics_csv(val_metrics, RESULTS_METRICS_DIR / "baseline_validation.csv")

    return {
        "pipeline": pipeline,
        "val_metrics": val_metrics,
        "val_preds": val_preds,
    }


def load_baseline_pipeline(path: Path | None = None) -> Pipeline:
    """Carica la pipeline baseline salvata con joblib.

    Args:
        path: percorso del file .joblib (default: models/baseline_tfidf_lr.joblib).

    Returns:
        Pipeline scikit-learn.
    """
    if path is None:
        path = MODELS_DIR / "baseline_tfidf_lr.joblib"
    logger.info("Caricamento pipeline da %s", path)
    return joblib.load(path)
