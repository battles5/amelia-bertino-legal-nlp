#!/usr/bin/env python
"""CLI per addestramento della baseline (TF-IDF + LR oppure fastText).

Uso:
    python train_baseline.py --algo tfidf_lr
    python train_baseline.py --algo fasttext
    python train_baseline.py --help
"""

from __future__ import annotations

import argparse
import logging

from amelia_experiment.config import DEFAULT_SEED
from amelia_experiment.dataset import load_amelia
from amelia_experiment.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Addestramento baseline per classificazione prem/conc su AMELIA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        choices=["tfidf_lr"],
        default="tfidf_lr",
        help="Algoritmo da usare: tfidf_lr.",
    )
    parser.add_argument(
        "--ngram_range",
        type=int,
        nargs=2,
        default=[1, 2],
        help="Range n-grammi per TF-IDF (es. 1 2).",
    )
    parser.add_argument("--min_df", type=int, default=2, help="Frequenza minima di documento.")
    parser.add_argument(
        "--max_features", type=int, default=50000, help="Numero max feature TF-IDF."
    )
    parser.add_argument(
        "--C", type=float, default=1.0, help="Regolarizzazione Logistic Regression."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed riproducibilità.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite campioni per split (utile per debug veloce).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point principale."""
    setup_logging()
    args = parse_args()

    logger.info("=== Training baseline: %s ===", args.algo)

    # Caricamento dataset
    train_ds = load_amelia(split="train", limit=args.limit, seed=args.seed)
    val_ds = load_amelia(split="validation", limit=args.limit, seed=args.seed)

    if args.algo == "tfidf_lr":
        from amelia_experiment.baseline import train_tfidf_lr

        result = train_tfidf_lr(
            train_ds,
            val_ds,
            ngram_range=tuple(args.ngram_range),
            min_df=args.min_df,
            max_features=args.max_features,
            C=args.C,
            seed=args.seed,
        )

    logger.info("Macro-F1 validation: %.4f", result["val_metrics"]["macro_f1"])
    logger.info("=== Training completato ===")


if __name__ == "__main__":
    main()
