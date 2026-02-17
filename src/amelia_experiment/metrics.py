"""Calcolo metriche di classificazione.

Fornisce macro-F1, accuracy, classification report strutturato
e confusion matrix come array numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from amelia_experiment.config import ID2LABEL


def macro_f1(y_true, y_pred) -> float:
    """Calcola il Macro-F1 score.

    Args:
        y_true: etichette vere (int o array-like).
        y_pred: etichette predette.

    Returns:
        Macro-F1 come float.
    """
    return float(f1_score(y_true, y_pred, average="macro"))


def classification_dict(y_true, y_pred) -> dict[str, Any]:
    """Calcola un report strutturato con macro-F1, accuracy e metriche per classe.

    Args:
        y_true: etichette vere.
        y_pred: etichette predette.

    Returns:
        Dizionario con chiavi:
        - ``macro_f1``, ``accuracy``
        - ``per_class``: lista di dict con precision/recall/f1 per ogni classe
        - ``confusion_matrix``: matrice di confusione come lista di liste
    """
    mf1 = macro_f1(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))

    labels = sorted(ID2LABEL.keys())
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class = []
    for i, label_id in enumerate(labels):
        per_class.append(
            {
                "label_id": label_id,
                "label": ID2LABEL[label_id],
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(sup[i]),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "macro_f1": mf1,
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def compute_metrics_for_trainer(eval_pred) -> dict[str, float]:
    """Funzione ``compute_metrics`` compatibile con ``transformers.Trainer``.

    Args:
        eval_pred: ``EvalPrediction`` fornita dal Trainer (predictions, label_ids).

    Returns:
        Dizionario con ``macro_f1`` e ``accuracy``.
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # predictions può essere una tupla (logits, hidden_states, …)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=-1)
    return {
        "macro_f1": macro_f1(labels, preds),
        "accuracy": float(accuracy_score(labels, preds)),
    }
