"""Gestione artefatti: salvataggio metriche, tabelle, snippet esempi.

Funzioni utility per esportare risultati in JSON, CSV e tabelle formattate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── JSON ─────────────────────────────────────────────────────────────────────


def save_dict_json(data: dict[str, Any], path: Path) -> None:
    """Salva un dizionario in formato JSON.

    Args:
        data: dizionario da salvare.
        path: percorso del file di output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("JSON salvato: %s", path)


# ── CSV ──────────────────────────────────────────────────────────────────────


def save_metrics_csv(metrics: dict[str, Any], path: Path) -> None:
    """Salva le metriche globali + per-classe in un file CSV.

    Args:
        metrics: dizionario (output di ``classification_dict``).
        path: percorso del file di output.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [{"metric": "macro_f1", "value": metrics["macro_f1"]}]
    rows.append({"metric": "accuracy", "value": metrics["accuracy"]})
    for pc in metrics.get("per_class", []):
        rows.append({"metric": f"{pc['label']}_precision", "value": pc["precision"]})
        rows.append({"metric": f"{pc['label']}_recall", "value": pc["recall"]})
        rows.append({"metric": f"{pc['label']}_f1", "value": pc["f1"]})

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("CSV salvato: %s", path)


# ── Tabella risultati comparativi ─────────────────────────────────────────────


def generate_results_table_latex(
    metrics_dict: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Genera una tabella con i risultati di tutti i modelli.

    Args:
        metrics_dict: dizionario ``{nome_modello: {split: metrics_dict}}``.
            Esempio: ``{"baseline": {"validation": {...}, "test": {...}}, "bertino": {...}}``.
        output_path: percorso del file .tex di output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Risultati: Macro-F1 su validation e test}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Modello} & \textbf{Validation} & \textbf{Test} \\",
        r"\midrule",
    ]

    for model_name, splits in metrics_dict.items():
        val_f1 = splits.get("validation", {}).get("macro_f1", "—")
        test_f1 = splits.get("test", {}).get("macro_f1", "—")
        val_str = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)
        test_str = f"{test_f1:.4f}" if isinstance(test_f1, float) else str(test_f1)
        lines.append(f"{model_name} & {val_str} & {test_str} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("Tabella risultati salvata: %s", output_path)
