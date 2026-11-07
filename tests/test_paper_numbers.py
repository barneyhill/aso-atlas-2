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


# ── Hagedorn et al. 2022 baseline (neurotox only) ──────────────

def test_hagedorn_linear_neurotox_key_exists(numbers):
    assert "hagedorn_linear_neurotox" in numbers, \
        "Missing 'hagedorn_linear_neurotox' key — run `just hagerdorn && just export`"


def test_hagedorn_linear_neurotox_fields(numbers):
    h = numbers["hagedorn_linear_neurotox"]
    for field in ("accuracy", "sensitivity", "specificity", "auc", "n", "n_groups"):
        assert field in h, f"Missing hagedorn_linear_neurotox.{field}"


def test_hagedorn_linear_neurotox_auc_in_range(numbers):
    auc = numbers["hagedorn_linear_neurotox"]["auc"]
    assert 0.4 <= auc <= 1, f"AUC = {auc} not in [0.4, 1]"


# ── Pipeline ─────────────────────────────────────────────────────

def test_pipeline_key_exists(numbers):
    assert "pipeline" in numbers, \
        "Missing 'pipeline' key — run `just analysis && just export`"


def test_pipeline_baseline_costs(numbers):
    p = numbers["pipeline"]
    assert p["baseline_n_initial"] > 0
    assert p["baseline_total_cost"] > 0


def test_pipeline_oligoai_tox_savings(numbers):
    p = numbers["pipeline"]
    if "oligoai_tox_total_cost" in p:
        assert p["oligoai_tox_total_cost"] < p["baseline_total_cost"], \
            "OligoAI-tox pipeline should cost less than baseline"
        assert 0 < p["oligoai_tox_savings_pct"] < 100


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
        assert p["combined_total_cost"] <= p["oligoai_tox_total_cost"], \
            "Combined should be at least as good as OligoAI-tox alone"
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


# ── ef_table.json consistency ───────────────────────────────────

EF_TABLE_PATH = Path("data/results/ef_table.json")


@pytest.fixture
def ef_rows():
    assert EF_TABLE_PATH.exists(), f"{EF_TABLE_PATH} not found — run `just analysis` first"
    return json.loads(EF_TABLE_PATH.read_text())


def test_ef_table_has_one_tox_only_row(ef_rows):
    tox_rows = [r for r in ef_rows if r.get("group") == "tox_only"]
    assert len(tox_rows) == 1, f"Expected exactly 1 tox_only row, got {len(tox_rows)}"
    assert tox_rows[0].get("source_model") == "XGBoost", \
        f"OligoAI-tox source_model should be XGBoost, got {tox_rows[0].get('source_model')}"


def test_ef_table_costs_match_marginal_sensitivity_numbers(numbers, ef_rows):
    p = numbers["marginal_pipeline"]
    tox_row = next((r for r in ef_rows if r.get("oligoai_tox")), None)
    comb_row = next((r for r in ef_rows if r.get("group") == "combined"), None)
    if tox_row and "oligoai_tox_total_cost" in p:
        assert p["oligoai_tox_total_cost"] == pytest.approx(tox_row["total_cost"]), \
            "paper_numbers oligoai_tox_total_cost != ef_table tox_only row"
    if comb_row and "combined_total_cost" in p:
        assert p["combined_total_cost"] == pytest.approx(comb_row["total_cost"]), \
            "paper_numbers combined_total_cost != ef_table combined row"
