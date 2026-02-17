"""Test unitari per il modulo preprocess."""

from __future__ import annotations

from amelia_experiment.preprocess import (
    decode_label,
    encode_label,
    mask_text,
    normalize_text,
)

# ── normalize_text ───────────────────────────────────────────────────────────


class TestNormalizeText:
    """Test per la funzione normalize_text."""

    def test_stripping(self):
        assert normalize_text("  ciao  ") == "ciao"

    def test_whitespace_collapse(self):
        assert normalize_text("ciao   mondo") == "ciao mondo"

    def test_tabs_and_newlines(self):
        assert normalize_text("ciao\t\nmondo") == "ciao mondo"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_already_clean(self):
        assert normalize_text("tutto ok") == "tutto ok"

    def test_mixed_whitespace(self):
        assert normalize_text("  a  b\tc\n\n d  ") == "a b c d"


# ── mask_text ────────────────────────────────────────────────────────────────


class TestMaskText:
    """Test per la funzione mask_text."""

    def test_digit_masking(self):
        assert mask_text("codice 12345 articolo") == "codice ### articolo"

    def test_multiple_digit_sequences(self):
        result = mask_text("art. 42 comma 3 del 2023")
        assert "42" not in result
        assert "###" in result

    def test_truncation(self):
        long_text = "a" * 300
        result = mask_text(long_text, max_chars=240)
        assert len(result) <= 240
        assert result.endswith("…")

    def test_no_truncation_if_short(self):
        short_text = "testo breve"
        result = mask_text(short_text)
        assert result == short_text

    def test_empty_string(self):
        assert mask_text("") == ""

    def test_combined(self):
        """Cifre mascherate E troncamento."""
        text = "art. 123 " * 50  # lungo
        result = mask_text(text, max_chars=100)
        assert "123" not in result
        assert len(result) <= 100


# ── encode_label / decode_label ──────────────────────────────────────────────


class TestLabelEncoding:
    """Test per encode_label e decode_label."""

    def test_encode_prem(self):
        assert encode_label("prem") == 0

    def test_encode_conc(self):
        assert encode_label("conc") == 1

    def test_decode_0(self):
        assert decode_label(0) == "prem"

    def test_decode_1(self):
        assert decode_label(1) == "conc"

    def test_roundtrip(self):
        for label in ("prem", "conc"):
            assert decode_label(encode_label(label)) == label

    def test_encode_invalid(self):
        import pytest

        with pytest.raises(KeyError):
            encode_label("invalido")
