"""Caricamento e validazione del dataset AMELIA da Hugging Face.

Il dataset è identificato come ``nlp-unibo/AMELIA`` e deve avere
licenza ``cc-by-4.0``. Contiene tre split: train, validation, test.
"""

from __future__ import annotations

import logging
import os

from datasets import load_dataset, load_dataset_builder

from amelia_experiment.config import (
    DATASET_ID,
    EXPECTED_LICENSE,
    LABEL_COL,
    TEXT_COL,
)

logger = logging.getLogger(__name__)


# ── Controllo licenza ────────────────────────────────────────────────────────


def assert_license(expected: str = EXPECTED_LICENSE) -> None:
    """Verifica che la licenza dichiarata dal dataset sia quella attesa.

    Se il campo ``license`` non è disponibile nei metadati del builder,
    emette un warning anziché bloccare l'esecuzione (la licenza è comunque
    indicata nella scheda del dataset su Hugging Face).

    Raises:
        ValueError: se la licenza è presente ma non corrisponde.
    """
    builder = load_dataset_builder(DATASET_ID)
    actual = getattr(builder.info, "license", None) or ""
    actual_clean = actual.strip().lower()
    expected_clean = expected.strip().lower()

    if not actual_clean:
        logger.warning(
            "Campo 'license' vuoto nei metadati del builder. "
            "Verifica manualmente la licenza su https://huggingface.co/datasets/%s",
            DATASET_ID,
        )
        return

    if expected_clean not in actual_clean:
        raise ValueError(
            f"Licenza attesa '{expected}', trovata '{actual}'. "
            "Verifica i termini d'uso del dataset prima di procedere."
        )
    logger.info("Licenza dataset verificata: %s", actual)


# ── Caricamento dati ─────────────────────────────────────────────────────────


def load_amelia(
    split: str = "train",
    cache_dir: str | None = None,
    limit: int | None = None,
    seed: int = 42,
):
    """Carica uno split del dataset AMELIA con caching e validazione.

    Args:
        split: nome dello split (``train``, ``validation``, ``test``).
        cache_dir: directory di cache opzionale (default: HF_HOME o ~/.cache/huggingface).
        limit: numero massimo di esempi da restituire (utile per debug/smoke test).
        seed: seed per lo shuffle quando si usa ``limit``.

    Returns:
        ``datasets.Dataset`` con le colonne ``Text`` e ``Component``.

    Raises:
        ValueError: se colonne attese mancanti o licenza non valida.
    """
    # Risolvi cache dir (supporta variabile d'ambiente HF_HOME)
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", None)

    logger.info("Caricamento AMELIA split='%s' (limit=%s) …", split, limit)

    # Controllo licenza
    assert_license()

    ds = load_dataset(DATASET_ID, split=split, cache_dir=cache_dir)

    # Validazione colonne
    _validate_columns(ds)

    # Sottocampionamento opzionale
    if limit is not None and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
        logger.info("Sottocampionato a %d esempi.", limit)

    logger.info("Dataset caricato: %d esempi.", len(ds))
    return ds


# ── Helper interni ───────────────────────────────────────────────────────────


def _validate_columns(ds) -> None:
    """Verifica che le colonne attese siano presenti nel dataset.

    Raises:
        ValueError: se manca ``Text`` o ``Component``.
    """
    cols = set(ds.column_names)
    required = {TEXT_COL, LABEL_COL}
    missing = required - cols
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {missing}. Colonne trovate: {cols}")
