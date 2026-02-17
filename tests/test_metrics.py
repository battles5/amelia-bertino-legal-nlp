"""Test unitari per il modulo metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amelia_experiment.metrics import classification_dict, macro_f1

# ── macro_f1 ─────────────────────────────────────────────────────────────────


class TestMacroF1:
    """Test per la funzione macro_f1."""

    def test_perfect_predictions(self):
        """Predizioni perfette → macro_f1 = 1.0."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        assert macro_f1(y_true, y_pred) == 1.0

    def test_worst_predictions(self):
        """Tutto sbagliato → macro_f1 = 0.0."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        assert macro_f1(y_true, y_pred) == 0.0

    def test_partial(self):
        """Predizioni parziali → valore tra 0 e 1."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 0, 1]
        result = macro_f1(y_true, y_pred)
        assert 0.0 < result < 1.0

    def test_all_same_class(self):
        """Tutte le predizioni sono una sola classe."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 0, 0]
        result = macro_f1(y_true, y_pred)
        # F1 per classe 1 = 0, per classe 0 > 0 → media > 0 ma < 1
        assert 0.0 < result < 1.0


# ── classification_dict ──────────────────────────────────────────────────────


class TestClassificationDict:
    """Test per la funzione classification_dict."""

    def test_structure(self):
        """Verifica che il dizionario abbia le chiavi attese."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        result = classification_dict(y_true, y_pred)

        assert "macro_f1" in result
        assert "accuracy" in result
        assert "per_class" in result
        assert "confusion_matrix" in result
        assert len(result["per_class"]) == 2

    def test_accuracy_correct(self):
        """Verifica accuracy su un caso noto."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 0]  # 3 corretti su 4
        result = classification_dict(y_true, y_pred)
        assert result["accuracy"] == 0.75

    def test_confusion_matrix_shape(self):
        """Matrice di confusione deve essere 2×2."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        result = classification_dict(y_true, y_pred)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_per_class_labels(self):
        """Verifica che per_class contenga le etichette corrette."""
        y_true = [0, 1]
        y_pred = [0, 1]
        result = classification_dict(y_true, y_pred)
        labels = {pc["label"] for pc in result["per_class"]}
        assert labels == {"prem", "conc"}
