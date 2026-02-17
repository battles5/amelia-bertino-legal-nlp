"""Preprocessing del testo e mapping delle etichette.

Funzioni per normalizzazione, mascheramento (per la demo) e
conversione label stringa ↔ intero.
"""

from __future__ import annotations

import re

from amelia_experiment.config import DEMO_MAX_CHARS, ID2LABEL, LABEL2ID

# ── Normalizzazione testo ────────────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Normalizza il testo: stripping e collasso spazi multipli.

    Args:
        text: testo grezzo.

    Returns:
        Testo normalizzato (senza spazi iniziali/finali, spazi multipli → singolo).
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ── Mascheramento testo (per demo / slide) ───────────────────────────────────


def mask_text(text: str, max_chars: int = DEMO_MAX_CHARS) -> str:
    """Maschera cifre e tronca il testo, adatto per presentazioni.

    - Sostituisce sequenze di cifre con ``###``.
    - Tronca a ``max_chars`` caratteri aggiungendo ``…`` se necessario.

    Args:
        text: testo originale (già normalizzato o grezzo).
        max_chars: lunghezza massima del testo in output.

    Returns:
        Testo mascherato e troncato.
    """
    # Maschera sequenze di cifre
    masked = re.sub(r"\d+", "###", text)
    # Tronca
    if len(masked) > max_chars:
        masked = masked[: max_chars - 1] + "…"
    return masked


# ── Encoding / decoding etichette ────────────────────────────────────────────


def encode_label(label: str) -> int:
    """Converte una etichetta stringa nel suo id intero.

    Args:
        label: ``'prem'`` o ``'conc'``.

    Returns:
        Intero corrispondente (0 per prem, 1 per conc).

    Raises:
        KeyError: se l'etichetta non è nel mapping.
    """
    return LABEL2ID[label]


def decode_label(label_id: int) -> str:
    """Converte un id intero nella corrispondente etichetta stringa.

    Args:
        label_id: 0 o 1.

    Returns:
        Stringa ``'prem'`` o ``'conc'``.

    Raises:
        KeyError: se l'id non è nel mapping.
    """
    return ID2LABEL[label_id]
