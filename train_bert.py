#!/usr/bin/env python
"""CLI per fine-tuning di BERTino su classificazione prem/conc.

Uso:
    python train_bert.py
    python train_bert.py --num_train_epochs 2 --max_train_samples 400
    python train_bert.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from amelia_experiment.config import DEFAULT_MAX_LENGTH, DEFAULT_SEED
from amelia_experiment.dataset import load_amelia
from amelia_experiment.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning di BERTino (DistilBERT italiano) su AMELIA — prem vs conc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Numero di epoche di training."
    )
    parser.add_argument(
        "--max_length", type=int, default=DEFAULT_MAX_LENGTH, help="Lunghezza massima token."
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=16, help="Batch size training."
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=32, help="Batch size eval."
    )
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limite campioni training (None = tutti). Utile su CPU.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limite campioni eval (None = tutti).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Max step di training (se > 0, prevale su epoche).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory output checkpoint (default: models/bertino_finetuned).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed riproducibilità.")
    return parser.parse_args()


def main() -> None:
    """Entry point principale."""
    setup_logging()
    args = parse_args()

    logger.info("=== Fine-tuning BERTino ===")

    # Caricamento dataset
    train_ds = load_amelia(split="train", seed=args.seed)
    val_ds = load_amelia(split="validation", seed=args.seed)

    from amelia_experiment.bertino import finetune_bertino

    output_dir = Path(args.output_dir) if args.output_dir else None

    result = finetune_bertino(
        train_ds,
        val_ds,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_steps=args.max_steps,
        output_dir=output_dir,
        seed=args.seed,
    )

    logger.info("Macro-F1 validation: %.4f", result["val_metrics"]["macro_f1"])
    logger.info("=== Fine-tuning completato ===")


if __name__ == "__main__":
    main()
