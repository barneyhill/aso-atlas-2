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


# ── Hagedorn hepatotoxicity ─────────────────────────────────────

def test_hagerdorn_hepatotox_key_exists(numbers):
    assert "hagerdorn_hepatotox" in numbers, \
        "Missing 'hagerdorn_hepatotox' key — run `just hagerdorn && just export`"


def test_hagerdorn_hepatotox_fields(numbers):
    h = numbers["hagerdorn_hepatotox"]
    for field in ("accuracy", "sensitivity", "specificity", "auc", "n", "n_groups"):
        assert field in h, f"Missing hagerdorn_hepatotox.{field}"


def test_hagerdorn_hepatotox_accuracy_in_range(numbers):
    acc = numbers["hagerdorn_hepatotox"]["accuracy"]
    assert 0 <= acc <= 1, f"accuracy = {acc} not in [0, 1]"


def test_hagerdorn_hepatotox_auc_in_range(numbers):
    auc = numbers["hagerdorn_hepatotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── Hagedorn neurotoxicity ──────────────────────────────────────

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
            "Hagedorn pipeline should cost less than baseline"
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
            "Combined should be at least as good as Hagedorn alone"
        assert 0 < p["combined_savings_pct"] < 100


# ── Cross-species concordance ────────────────────────────────────

def test_cross_species_hepatotox_key_exists(numbers):
    assert "cross_species_hepatotox" in numbers, \
        "Missing 'cross_species_hepatotox' key — run `just hagerdorn && just export`"


def test_cross_species_hepatotox_biomarkers(numbers):
    cs = numbers["cross_species_hepatotox"]
    for bm in ("ALT", "AST", "TBIL"):
        assert bm in cs, f"Missing cross_species_hepatotox.{bm}"
        for field in ("n_shared", "spearman_rho", "spearman_p", "concordance_rate", "concordance_n"):
            assert field in cs[bm], f"Missing cross_species_hepatotox.{bm}.{field}"


def test_cross_species_hepatotox_values(numbers):
    for bm in ("ALT", "AST", "TBIL"):
        v = numbers["cross_species_hepatotox"][bm]
        assert v["n_shared"] > 50, f"{bm} n_shared too low: {v['n_shared']}"
        assert -1 <= v["spearman_rho"] <= 1, f"{bm} rho out of range: {v['spearman_rho']}"
        assert v["spearman_p"] < 0.05, f"{bm} p-value not significant: {v['spearman_p']}"
        assert 0 <= v["concordance_rate"] <= 1, f"{bm} concordance out of range"


def test_cross_species_neurotox_key_exists(numbers):
    assert "cross_species_neurotox" in numbers, \
        "Missing 'cross_species_neurotox' key — run `just hagerdorn && just export`"


# ── Rat hepatotoxicity ────────────────────────────────────────────

def test_rat_hepatotox_key_exists(numbers):
    assert "rat_hepatotox" in numbers, \
        "Missing 'rat_hepatotox' key — run `just hagerdorn && just export`"


def test_rat_hepatotox_fields(numbers):
    h = numbers["rat_hepatotox"]
    for field in ("accuracy", "sensitivity", "specificity", "auc", "n", "n_groups"):
        assert field in h, f"Missing rat_hepatotox.{field}"


def test_rat_hepatotox_accuracy_in_range(numbers):
    acc = numbers["rat_hepatotox"]["accuracy"]
    assert 0 <= acc <= 1, f"accuracy = {acc} not in [0, 1]"


def test_rat_hepatotox_auc_in_range(numbers):
    auc = numbers["rat_hepatotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── Rat neurotoxicity ────────────────────────────────────────────

def test_rat_neurotox_key_exists(numbers):
    assert "rat_neurotox" in numbers, \
        "Missing 'rat_neurotox' key — run `just hagerdorn && just export`"


def test_rat_neurotox_fields(numbers):
    h = numbers["rat_neurotox"]
    for field in ("accuracy", "sensitivity", "specificity", "auc", "n", "n_groups"):
        assert field in h, f"Missing rat_neurotox.{field}"


def test_rat_neurotox_accuracy_in_range(numbers):
    acc = numbers["rat_neurotox"]["accuracy"]
    assert 0 <= acc <= 1, f"accuracy = {acc} not in [0, 1]"


def test_rat_neurotox_auc_in_range(numbers):
    auc = numbers["rat_neurotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── Cross-species concordance ────────────────────────────────────

def test_cross_species_neurotox_fob(numbers):
    cs = numbers["cross_species_neurotox"]
    assert "FOB" in cs, "Missing cross_species_neurotox.FOB"
    v = cs["FOB"]
    for field in ("n_shared", "spearman_rho", "spearman_p", "concordance_rate", "concordance_n"):
        assert field in v, f"Missing cross_species_neurotox.FOB.{field}"
    assert v["n_shared"] > 100, f"FOB n_shared too low: {v['n_shared']}"
    assert -1 <= v["spearman_rho"] <= 1
    assert v["spearman_p"] < 0.05
    assert 0 <= v["concordance_rate"] <= 1


# ── OligoAI-tox hepatotoxicity ──────────────────────────────────

def test_oligoai_tox_hepatotox_key_exists(numbers):
    assert "oligoai_tox_hepatotox" in numbers, \
        "Missing 'oligoai_tox_hepatotox' key — run `just hagerdorn && just export`"


def test_oligoai_tox_hepatotox_fields(numbers):
    h = numbers["oligoai_tox_hepatotox"]
    for field in ("accuracy", "sensitivity", "specificity", "auc", "n"):
        assert field in h, f"Missing oligoai_tox_hepatotox.{field}"


def test_oligoai_tox_hepatotox_auc_in_range(numbers):
    auc = numbers["oligoai_tox_hepatotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── OligoAI-tox neurotoxicity ───────────────────────────────────

def test_oligoai_tox_neurotox_key_exists(numbers):
    assert "oligoai_tox_neurotox" in numbers, \
        "Missing 'oligoai_tox_neurotox' key — run `just hagerdorn && just export`"


def test_oligoai_tox_neurotox_fields(numbers):
    h = numbers["oligoai_tox_neurotox"]
    for field in ("accuracy", "sensitivity", "specificity", "auc", "n"):
        assert field in h, f"Missing oligoai_tox_neurotox.{field}"


def test_oligoai_tox_neurotox_auc_in_range(numbers):
    auc = numbers["oligoai_tox_neurotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── OligoAI-tox pipeline ────────────────────────────────────────

def test_pipeline_oligoai_tox_savings(numbers):
    p = numbers["pipeline"]
    if "oligoai_tox_total_cost" in p:
        assert p["oligoai_tox_total_cost"] < p["baseline_total_cost"], \
            "OligoAI-tox pipeline should cost less than baseline"
        assert 0 < p["oligoai_tox_savings_pct"] < 100


def test_pipeline_combined_cnn_savings(numbers):
    p = numbers["pipeline"]
    if "combined_cnn_total_cost" in p:
        assert p["combined_cnn_total_cost"] < p["baseline_total_cost"], \
            "Combined CNN pipeline should cost less than baseline"
        assert 0 < p["combined_cnn_savings_pct"] < 100
