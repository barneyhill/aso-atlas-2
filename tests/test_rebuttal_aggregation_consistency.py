import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"


def _load(name):
    return json.loads((RESULTS / name).read_text())


def test_threshold_and_primary_use_identical_pooled_oof_precisions():
    ef_rows = _load("ef_table.json")
    threshold = _load("threshold_sweep.json")
    combined = next(row for row in ef_rows if row.get("group") == "combined")

    canonical_points = {
        block["stage_idx"]: next(
            point for point in block["points"] if point["is_canonical"]
        )
        for block in threshold["tornado"]
        if block["stage_idx"] < 6
    }
    for stage in range(6):
        assert combined["prec_by_stage"][str(stage)] == pytest.approx(
            canonical_points[stage]["model_selected_pass_rate"], abs=5e-5
        )


def test_threshold_reference_matches_primary_marginal_projection():
    threshold = _load("threshold_sweep.json")
    uncertainty = _load("ac_uncertainty.json")
    assert threshold["canonical_expected_savings_pct"] == uncertainty["canonical"][
        "marginal_expected_savings_pct"
    ]


def test_model_choice_xgboost_matches_corrected_primary_projection():
    uncertainty = _load("ac_uncertainty.json")
    xgboost = next(
        row for row in uncertainty["model_choice_sensitivity"]
        if row["model"] == "XGBoost"
    )
    assert xgboost["corrected_expected_savings_pct"] == uncertainty["canonical"][
        "corrected_expected_savings_pct"
    ]


def test_corrected_stage_counts_conserve_expected_yield_and_cost():
    """A headline change must carry its underlying funnel and costs with it."""
    uncertainty = _load("ac_uncertainty.json")
    pipeline = _load("pipeline_results.json")
    scenarios = uncertainty["corrected_pipeline"]
    for row in scenarios.values():
        counts = row["asos_at_stage"]
        assert counts[-1] == 1
        for i, rate in enumerate(row["proportions"]):
            assert counts[i] * rate == pytest.approx(counts[i + 1])
            assert row["costs_per_stage"][i] == pytest.approx(
                counts[i] * pipeline["stages"][i]["cost_per_aso"]
            )
        assert row["total_cost"] == pytest.approx(sum(row["costs_per_stage"]))
        assert row["animals"] == pytest.approx(4 * sum(counts[2:-1]))
        assert row["savings_pct"] == pytest.approx(
            100 * (1 - row["total_cost"] / scenarios["baseline"]["total_cost"])
        )
    assert round(scenarios["combined"]["savings_pct"], 1) == uncertainty[
        "canonical"
    ]["corrected_expected_savings_pct"]


def test_exported_headline_costs_and_animals_use_corrected_scenarios():
    uncertainty = _load("ac_uncertainty.json")
    paper = json.loads((ROOT / "typst/data/paper_numbers.json").read_text())
    corrected = uncertainty["corrected_pipeline"]
    assert paper["pipeline"]["combined_savings_pct"] == uncertainty["canonical"][
        "corrected_expected_savings_pct"
    ]
    assert paper["pipeline"]["baseline_total_cost"] == corrected["baseline"]["total_cost"]
    assert paper["pipeline"]["combined_total_cost"] == corrected["combined"]["total_cost"]
    assert paper["pipeline"]["baseline_animals"] == round(corrected["baseline"]["animals"], 1)
    assert paper["marginal_pipeline"]["combined_savings_pct"] == uncertainty[
        "canonical"
    ]["marginal_ceiling_savings_pct"]


def test_plotted_histograms_include_every_sensitivity_draw():
    uncertainty = _load("ac_uncertainty.json")
    for histogram in uncertainty["sensitivity_histograms"].values():
        assert sum(histogram["counts"]) == uncertainty["method"]["baseline_rate_draws"]
        assert len(histogram["edges"]) == len(histogram["counts"]) + 1
