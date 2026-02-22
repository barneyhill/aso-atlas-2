"""Smoke tests for paper_numbers.json."""

import json
from pathlib import Path

import pytest

NUMBERS_PATH = Path("typst/data/paper_numbers.json")


@pytest.fixture
def numbers():
    assert NUMBERS_PATH.exists(), f"{NUMBERS_PATH} not found — run `just export` first"
    return json.loads(NUMBERS_PATH.read_text())


# ── Dataset keys ──────────────────────────────────────────────────

def test_dataset_keys_exist(numbers):
    for key in ("in_vitro", "dose_response", "hepatic", "neuro", "genes"):
        assert key in numbers, f"Missing '{key}' key"


def test_dataset_counts_positive(numbers):
    assert numbers["in_vitro"]["n_measurements"] > 0
    assert numbers["in_vitro"]["n_asos"] > 0
    assert numbers["dose_response"]["n_measurements"] > 0
    assert numbers["hepatic"]["n_records"] > 0
    assert numbers["neuro"]["n_records"] > 0
    assert numbers["genes"]["n_unique"] > 0


# ── Hagerdorn hepatotoxicity ─────────────────────────────────────

def test_hagerdorn_hepatotox_key_exists(numbers):
    assert "hagerdorn_hepatotox" in numbers, \
        "Missing 'hagerdorn_hepatotox' key — run `just hagerdorn && just export`"


def test_hagerdorn_hepatotox_fields(numbers):
    h = numbers["hagerdorn_hepatotox"]
    for field in ("oob_accuracy", "accuracy", "sensitivity", "specificity", "auc", "n"):
        assert field in h, f"Missing hagerdorn_hepatotox.{field}"


def test_hagerdorn_hepatotox_accuracy_in_range(numbers):
    acc = numbers["hagerdorn_hepatotox"]["accuracy"]
    assert 0 <= acc <= 1, f"accuracy = {acc} not in [0, 1]"


def test_hagerdorn_hepatotox_auc_in_range(numbers):
    auc = numbers["hagerdorn_hepatotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── Hagerdorn neurotoxicity ──────────────────────────────────────

def test_hagerdorn_neurotox_key_exists(numbers):
    assert "hagerdorn_neurotox" in numbers, \
        "Missing 'hagerdorn_neurotox' key — run `just hagerdorn && just export`"


def test_hagerdorn_neurotox_fields(numbers):
    h = numbers["hagerdorn_neurotox"]
    for field in ("accuracy", "sensitivity", "specificity", "auc", "n", "n_groups"):
        assert field in h, f"Missing hagerdorn_neurotox.{field}"


def test_hagerdorn_neurotox_accuracy_in_range(numbers):
    acc = numbers["hagerdorn_neurotox"]["accuracy"]
    assert 0 <= acc <= 1, f"accuracy = {acc} not in [0, 1]"


def test_hagerdorn_neurotox_auc_in_range(numbers):
    auc = numbers["hagerdorn_neurotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── Pipeline ─────────────────────────────────────────────────────

def test_pipeline_key_exists(numbers):
    assert "pipeline" in numbers, \
        "Missing 'pipeline' key — run `just analysis && just export`"


def test_pipeline_baseline_costs(numbers):
    p = numbers["pipeline"]
    assert p["baseline_n_initial"] > 0
    assert p["baseline_total_cost"] > 0


def test_pipeline_hagerdorn_savings(numbers):
    p = numbers["pipeline"]
    if "hagerdorn_total_cost" in p:
        assert p["hagerdorn_total_cost"] < p["baseline_total_cost"], \
            "Hagerdorn pipeline should cost less than baseline"
        assert 0 < p["hagerdorn_savings_pct"] < 100


def test_pipeline_oligoai_savings(numbers):
    p = numbers["pipeline"]
    if "oligoai_total_cost" in p:
        assert p["oligoai_total_cost"] < p["baseline_total_cost"], \
            "OligoAI pipeline should cost less than baseline"
        assert 0 < p["oligoai_savings_pct"] < 100


def test_pipeline_combined_savings(numbers):
    p = numbers["pipeline"]
    if "combined_total_cost" in p:
        assert p["combined_total_cost"] < p["baseline_total_cost"], \
            "Combined pipeline should cost less than baseline"
        assert p["combined_total_cost"] <= p["hagerdorn_total_cost"], \
            "Combined should be at least as good as Hagerdorn alone"
        assert 0 < p["combined_savings_pct"] < 100
