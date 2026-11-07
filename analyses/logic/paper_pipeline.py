"""Presentation of the existing corrected expected-count analysis."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_corrected_analysis():
    report = json.loads((ROOT / "data/results/ac_uncertainty.json").read_text())
    if "corrected_pipeline" not in report:
        raise ValueError("Run just stats to export corrected stage counts before plotting.")
    return report


def paper_pipeline(report, stages):
    scenarios = report["corrected_pipeline"]
    baseline = scenarios["baseline"]
    result = {
        "recursion": "conditional expected counts",
        "baseline_n_initial": round(baseline["n_initial"], 1),
        "baseline_total_cost": baseline["total_cost"],
        "baseline_animals": round(baseline["animals"], 1),
        "stages": [
            {**stage, "p_k": baseline["proportions"][i],
             "r_k": stage["cost_per_aso"], "N_k": baseline["asos_at_stage"][i]}
            for i, stage in enumerate(stages)
        ],
    }
    for key, prefix in [("in_vitro", "oligoai"), ("in_vivo", "oligoai_tox"),
                        ("combined", "combined")]:
        row = scenarios[key]
        result.update({
            prefix + "_n_initial": round(row["n_initial"], 1),
            prefix + "_total_cost": row["total_cost"],
            prefix + "_savings_pct": round(row["savings_pct"], 1),
        })
    result["oligoai_tox_source_model"] = "XGBoost"
    result["oligoai_tox_animals"] = round(scenarios["in_vivo"]["animals"], 1)
    result["animal_reduction_pct"] = round(
        100 * (1 - scenarios["in_vivo"]["animals"] / baseline["animals"]), 1,
    )
    result["in_vitro_cost_pct"] = round(
        100 * sum(baseline["costs_per_stage"][:2]) / baseline["total_cost"], 1,
    )
    return result
