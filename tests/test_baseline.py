"""Test unitari per il modulo baseline."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from amelia_experiment.baseline import train_tfidf_lr
from amelia_experiment.config import LABEL_COL, TEXT_COL


def _make_fake_dataset(n: int = 100, seed: int = 42):
    """Crea un dataset finto con interfaccia __getitem__ simile a HF Dataset."""
    rng = np.random.RandomState(seed)

    texts = []
    labels = []
    for i in range(n):
        if rng.random() < 0.8:
            texts.append(f"premessa numero {i} con argomento giuridico dettagliato")
            labels.append("prem")
        else:
            texts.append(f"conclusione numero {i} la corte decide pertanto")
            labels.append("conc")

    class FakeDataset:
        """Oggetto che simula un datasets.Dataset con __getitem__ e column_names."""

        def __init__(self, data: dict):
            self._data = data

        def __len__(self):
            return len(self._data[TEXT_COL])

        def __getitem__(self, key):
            return self._data[key]

    return FakeDataset({TEXT_COL: texts, LABEL_COL: labels})


class TestTrainTfidfLr:
    """Test per train_tfidf_lr con dati sintetici."""

    def test_returns_pipeline(self, tmp_path, monkeypatch):
        """Il risultato deve contenere una Pipeline scikit-learn."""
        monkeypatch.setattr("amelia_experiment.baseline.MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr("amelia_experiment.baseline.RESULTS_METRICS_DIR", tmp_path / "metrics")

        train_ds = _make_fake_dataset(80, seed=1)
        val_ds = _make_fake_dataset(20, seed=2)

        result = train_tfidf_lr(train_ds, val_ds)
        assert isinstance(result["pipeline"], Pipeline)

    def test_returns_val_metrics(self, tmp_path, monkeypatch):
        """Il risultato deve contenere metriche di validazione."""
        monkeypatch.setattr("amelia_experiment.baseline.MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr("amelia_experiment.baseline.RESULTS_METRICS_DIR", tmp_path / "metrics")

        train_ds = _make_fake_dataset(80, seed=1)
        val_ds = _make_fake_dataset(20, seed=2)

        result = train_tfidf_lr(train_ds, val_ds)
        metrics = result["val_metrics"]

        assert "macro_f1" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_saves_model_file(self, tmp_path, monkeypatch):
        """Il modello .joblib deve essere salvato su disco."""
        models_dir = tmp_path / "models"
        monkeypatch.setattr("amelia_experiment.baseline.MODELS_DIR", models_dir)
        monkeypatch.setattr("amelia_experiment.baseline.RESULTS_METRICS_DIR", tmp_path / "metrics")

        train_ds = _make_fake_dataset(80, seed=1)
        val_ds = _make_fake_dataset(20, seed=2)

        train_tfidf_lr(train_ds, val_ds)
        assert (models_dir / "baseline_tfidf_lr.joblib").exists()

    def test_saves_metrics_files(self, tmp_path, monkeypatch):
        """JSON e CSV delle metriche devono essere salvati."""
        metrics_dir = tmp_path / "metrics"
        monkeypatch.setattr("amelia_experiment.baseline.MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr("amelia_experiment.baseline.RESULTS_METRICS_DIR", metrics_dir)

        train_ds = _make_fake_dataset(80, seed=1)
        val_ds = _make_fake_dataset(20, seed=2)

        train_tfidf_lr(train_ds, val_ds)
        assert (metrics_dir / "baseline_validation.json").exists()
        assert (metrics_dir / "baseline_validation.csv").exists()

    def test_predictions_length(self, tmp_path, monkeypatch):
        """Le predizioni devono avere la stessa lunghezza del validation set."""
        monkeypatch.setattr("amelia_experiment.baseline.MODELS_DIR", tmp_path / "models")
        monkeypatch.setattr("amelia_experiment.baseline.RESULTS_METRICS_DIR", tmp_path / "metrics")

        val_ds = _make_fake_dataset(25, seed=3)
        train_ds = _make_fake_dataset(80, seed=1)

        result = train_tfidf_lr(train_ds, val_ds)
        assert len(result["val_preds"]) == len(val_ds)
