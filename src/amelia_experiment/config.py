"""Configurazione centralizzata per l'esperimento AMELIA + BERTino.

Contiene costanti, path di default e mapping delle etichette.
"""

from pathlib import Path

# ── Identificativi dataset e modello ──────────────────────────────────────────
DATASET_ID: str = "nlp-unibo/AMELIA"
MODEL_ID: str = "indigo-ai/BERTino"
EXPECTED_LICENSE: str = "cc-by-4.0"

# ── Mapping etichette ────────────────────────────────────────────────────────
LABEL2ID: dict[str, int] = {"prem": 0, "conc": 1}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS: int = len(LABEL2ID)

# ── Nomi colonne dataset ─────────────────────────────────────────────────────
TEXT_COL: str = "Text"
LABEL_COL: str = "Component"

# ── Directory di progetto (relative alla root del repo) ──────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_METRICS_DIR: Path = RESULTS_DIR / "metrics"
RESULTS_PLOTS_DIR: Path = RESULTS_DIR / "plots"
RESULTS_TABLES_DIR: Path = RESULTS_DIR / "tables"

# ── Parametri di default ─────────────────────────────────────────────────────
DEFAULT_SEED: int = 42
DEFAULT_MAX_LENGTH: int = 256  # lunghezza massima token per BERTino
DEMO_MAX_CHARS: int = 240  # troncamento testo nella demo
