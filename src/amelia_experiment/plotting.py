"""Generazione di grafici per l'esperimento (confusion matrix).

Produce file PNG/PDF in ``results/plots/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from amelia_experiment.config import ID2LABEL, RESULTS_PLOTS_DIR

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    model_name: str,
    split: str,
    output_dir: Path | None = None,
    fmt: str = "d",
) -> Path:
    """Genera e salva un grafico della matrice di confusione.

    Args:
        cm: matrice di confusione (lista di liste o array numpy).
        model_name: nome del modello (per il titolo e il nome file).
        split: nome dello split (``validation`` o ``test``).
        output_dir: directory di output (default: results/plots/).
        fmt: formato numerico per le celle.

    Returns:
        Percorso del file PNG salvato.
    """
    if output_dir is None:
        output_dir = RESULTS_PLOTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(cm)
    labels = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm_array.shape[1]),
        yticks=np.arange(cm_array.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion Matrix - {model_name} ({split})",
    )

    # Annota le celle
    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j,
                i,
                format(cm_array[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_array[i, j] > thresh else "black",
            )

    fig.tight_layout()

    filename = f"cm_{model_name}_{split}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix salvata: %s", filepath)
    return filepath
