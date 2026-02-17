"""Test unitari per il modulo dataset."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from amelia_experiment.dataset import _validate_columns, assert_license

# ── assert_license ───────────────────────────────────────────────────────────


class TestAssertLicense:
    """Test per la funzione assert_license."""

    def test_valid_license(self):
        """Licenza corretta → nessun errore."""
        mock_builder = MagicMock()
        mock_builder.info.license = "cc-by-4.0"
        with patch("amelia_experiment.dataset.load_dataset_builder", return_value=mock_builder):
            assert_license("cc-by-4.0")  # non deve sollevare eccezioni

    def test_invalid_license(self):
        """Licenza sbagliata → ValueError."""
        mock_builder = MagicMock()
        mock_builder.info.license = "mit"
        with (
            patch("amelia_experiment.dataset.load_dataset_builder", return_value=mock_builder),
            pytest.raises(ValueError, match="Licenza attesa"),
        ):
            assert_license("cc-by-4.0")

    def test_empty_license_warns(self):
        """Licenza vuota → warning, nessun errore."""
        mock_builder = MagicMock()
        mock_builder.info.license = ""
        with patch("amelia_experiment.dataset.load_dataset_builder", return_value=mock_builder):
            assert_license("cc-by-4.0")  # non deve sollevare eccezioni

    def test_none_license_warns(self):
        """Licenza None → warning, nessun errore."""
        mock_builder = MagicMock()
        mock_builder.info.license = None
        with patch("amelia_experiment.dataset.load_dataset_builder", return_value=mock_builder):
            assert_license("cc-by-4.0")


# ── _validate_columns ────────────────────────────────────────────────────────


class TestValidateColumns:
    """Test per la funzione _validate_columns."""

    def test_valid_columns(self):
        """Colonne corrette → nessun errore."""
        ds = MagicMock()
        ds.column_names = ["Text", "Component", "extra"]
        _validate_columns(ds)  # non deve sollevare eccezioni

    def test_missing_text_column(self):
        """Manca 'Text' → ValueError."""
        ds = MagicMock()
        ds.column_names = ["Component"]
        with pytest.raises(ValueError, match="Colonne mancanti"):
            _validate_columns(ds)

    def test_missing_component_column(self):
        """Manca 'Component' → ValueError."""
        ds = MagicMock()
        ds.column_names = ["Text"]
        with pytest.raises(ValueError, match="Colonne mancanti"):
            _validate_columns(ds)

    def test_missing_both(self):
        """Mancano entrambe → ValueError."""
        ds = MagicMock()
        ds.column_names = ["other"]
        with pytest.raises(ValueError, match="Colonne mancanti"):
            _validate_columns(ds)
