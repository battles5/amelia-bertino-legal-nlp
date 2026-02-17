#!/usr/bin/env python
"""CLI demo: mostra 5 esempi mascherati dal test set con predizioni.

Carica un modello salvato (baseline o BERTino) e stampa:
- Testo mascherato e troncato (max 240 char)
- Etichetta gold
- Etichetta predetta
- Confidenza (se disponibile)

Uso:
    python demo.py --model baseline
    python demo.py --model bertino
    python demo.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from amelia_experiment.config import (
    DEFAULT_SEED,
    LABEL_COL,
    TEXT_COL,
)
from amelia_experiment.dataset import load_amelia
from amelia_experiment.logging_utils import setup_logging
from amelia_experiment.preprocess import decode_label, mask_text, normalize_text

logger = logging.getLogger(__name__)

NUM_EXAMPLES = 5


def parse_args() -> argparse.Namespace:
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Demo: 5 esempi mascherati dal test set con predizioni.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "bertino"],
        default="baseline",
        help="Modello da usare per le predizioni.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Seed per selezionare esempi."
    )
    parser.add_argument("--n", type=int, default=NUM_EXAMPLES, help="Numero di esempi da mostrare.")
    return parser.parse_args()


def predict_baseline(texts: list[str]) -> list[dict]:
    """Predice con la baseline TF-IDF + LR."""
    from amelia_experiment.baseline import load_baseline_pipeline

    pipeline = load_baseline_pipeline()
    normalized = [normalize_text(t) for t in texts]
    preds = pipeline.predict(normalized)
    probas = pipeline.predict_proba(normalized)

    results = []
    for pred_id, proba in zip(preds, probas, strict=True):
        results.append(
            {
                "label": decode_label(int(pred_id)),
                "label_id": int(pred_id),
                "confidence": float(max(proba)),
            }
        )
    return results


def predict_bertino(texts: list[str]) -> list[dict]:
    """Predice con BERTino fine-tuned."""
    from amelia_experiment.bertino import predict_texts

    return predict_texts(texts)


def main() -> None:
    """Entry point principale."""
    setup_logging()
    args = parse_args()

    logger.info("=== Demo: %d esempi con modello '%s' ===", args.n, args.model)

    # Carica test set e seleziona esempi con seed fisso
    test_ds = load_amelia(split="test", seed=args.seed)
    test_ds_shuffled = test_ds.shuffle(seed=args.seed)
    selected = test_ds_shuffled.select(range(min(args.n, len(test_ds_shuffled))))

    texts = list(selected[TEXT_COL])
    golds = list(selected[LABEL_COL])

    # Predizioni
    predictions = predict_baseline(texts) if args.model == "baseline" else predict_bertino(texts)

    # Stampa risultati
    print("\n" + "=" * 80)
    print(f"  DEMO — {args.model.upper()} — {args.n} esempi dal test set")
    print("=" * 80)

    for i, (text, gold, pred_info) in enumerate(zip(texts, golds, predictions, strict=True), 1):
        masked = mask_text(text)
        pred_label = pred_info["label"]
        confidence = pred_info.get("confidence", None)

        # Indicatore corretto/errato
        marker = "✓" if gold == pred_label else "✗"

        print(f"\n--- Esempio {i} [{marker}] ---")
        print(f"  Testo:      {masked}")
        print(f"  Gold:       {gold}")
        print(f"  Predetto:   {pred_label}", end="")
        if confidence is not None:
            print(f"  (conf: {confidence:.3f})", end="")
        print()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
