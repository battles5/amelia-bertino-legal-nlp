"""Utility per la configurazione del logging di progetto."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configura il logging di base per tutto il progetto.

    Formato: ``[LIVELLO] nome_modulo — messaggio``
    Output su stderr per non interferire con stdout degli script CLI.
    """
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )
