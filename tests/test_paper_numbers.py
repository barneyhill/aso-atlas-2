"""Smoke tests for paper_numbers.json model metrics."""

import json
from pathlib import Path

import pytest

NUMBERS_PATH = Path("typst/data/paper_numbers.json")


@pytest.fixture
def numbers():
    assert NUMBERS_PATH.exists(), f"{NUMBERS_PATH} not found — run `just export` first"
    return json.loads(NUMBERS_PATH.read_text())


def test_model_key_exists(numbers):
    assert "model" in numbers, "Missing 'model' key — run `just evaluate && just export`"


def test_model_required_fields(numbers):
    model = numbers["model"]
    required = [
        "n_params",
        "median_spearman",
        "iv_spearman",
        "alt_spearman",
        "ast_spearman",
        "fob_spearman",
        "top_k_fraction",
        "inhibition_enrichment_factor",
        "ALT_enrichment_factor",
        "FOB_enrichment_factor",
    ]
    for field in required:
        assert field in model, f"Missing model.{field}"


def test_spearman_values_in_range(numbers):
    model = numbers["model"]
    for key in ("median_spearman", "iv_spearman", "alt_spearman", "ast_spearman", "fob_spearman"):
        val = model[key]
        assert -1 <= val <= 1, f"model.{key} = {val} not in [-1, 1]"


def test_enrichment_factors_above_one(numbers):
    model = numbers["model"]
    for stage in ("inhibition", "ALT", "FOB"):
        key = f"{stage}_enrichment_factor"
        val = model[key]
        assert val > 1.0, f"model.{key} = {val} should be > 1.0"


def test_n_params_positive(numbers):
    assert numbers["model"]["n_params"] > 0


def test_base_rates_are_probabilities(numbers):
    model = numbers["model"]
    for stage in ("inhibition", "ALT", "FOB"):
        for suffix in ("base_rate", "top_k_pass_rate"):
            key = f"{stage}_{suffix}"
            val = model[key]
            assert 0 <= val <= 1, f"model.{key} = {val} not in [0, 1]"


# ── Ablation tests ──────────────────────────────────────────────────

def test_ablation_key_exists(numbers):
    assert "ablation" in numbers, "Missing 'ablation' key — run `just ablation && just export`"


def test_ablation_has_three_conditions(numbers):
    ablation = numbers.get("ablation", {})
    for condition in ("full", "no_warmup", "vivo_only"):
        assert condition in ablation, f"Missing ablation.{condition}"


def test_ablation_spearman_values_in_range(numbers):
    ablation = numbers.get("ablation", {})
    for condition, metrics in ablation.items():
        for key in ("alt_spearman", "ast_spearman", "fob_spearman", "median_vivo_spearman"):
            val = metrics[key]
            assert -1 <= val <= 1, f"ablation.{condition}.{key} = {val} not in [-1, 1]"


def test_ablation_median_spearman_positive(numbers):
    """All ablation conditions should achieve positive median vivo Spearman."""
    ablation = numbers.get("ablation", {})
    for condition in ("full", "no_warmup", "vivo_only"):
        val = ablation.get(condition, {}).get("median_vivo_spearman", -999)
        assert val > 0, f"ablation.{condition}.median_vivo_spearman = {val} should be > 0"
