"""
Supplementary figure: enrichment and pipeline savings vs top-X% selection.

Top-X% means selecting the X% compounds with the lowest predicted biomarker
value for each toxicity endpoint, using CatBoost (OligoAI-tox) regression
predictions from the OligoGym benchmark.

Panel A: Enrichment factor curves vs top-X%.
Panel B: Pipeline savings (%) vs top-X% (OligoAI-tox only, Combined with OligoAI).
"""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analyses.logic.enrichment import enrichment_at_top_k, stage_for_dataset
from analyses.logic.ef_table import OLIGOAI_TOX_MODEL

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
BENCH_PATH = _root / "data/results/oligogym_benchmark.json"
PIPE_PATH = _root / "data/results/pipeline_results.json"
EF_TABLE_PATH = _root / "data/results/ef_table.json"
OUT_DIR = _root / "typst/plots/fig_enrichment"

DATASETS = ["mouse_hepatic", "mouse_neuro", "rat_hepatic", "rat_neuro"]
STAGE_MAP = {"mouse_hepatic": 2, "mouse_neuro": 3, "rat_hepatic": 4, "rat_neuro": 5}
TASK_LABELS = {
    "mouse_hepatic": "Mouse ALT",
    "mouse_neuro": "Mouse bFOB",
    "rat_hepatic": "Rat ALT",
    "rat_neuro": "Rat mFOB",
}
TASK_COLORS = {
    "mouse_hepatic": "#D4A574",
    "mouse_neuro": "#E8B88A",
    "rat_hepatic": "#7BAA97",
    "rat_neuro": "#94C4A7",
}


def _load_catboost_predictions(bench: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract pooled y_true/y_pred for each tox dataset from the benchmark."""
    out = {}
    for r in bench.get("all_results", []):
        if r["model"] != OLIGOAI_TOX_MODEL or r["dataset"] not in DATASETS:
            continue
        yt, yp = [], []
        for fm in r.get("fold_metrics", []):
            if "y_true" in fm and "y_pred" in fm:
                yt.extend(fm["y_true"])
                yp.extend(fm["y_pred"])
        if yt:
            out[r["dataset"]] = (np.asarray(yt, dtype=float), np.asarray(yp, dtype=float))
    return out


def _back_calculate_total_cost(proportions, costs, toxicity_ef, oligoai_ef=None):
    eff = list(proportions)
    for ds, stage_idx in STAGE_MAP.items():
        if ds in toxicity_ef:
            eff[stage_idx] = min(eff[stage_idx] * toxicity_ef[ds], 1.0)
    if oligoai_ef is not None:
        eff[0] = min(eff[0] * oligoai_ef, 1.0)

    asos_at_stage = [0] * (len(eff) + 1)
    asos_at_stage[-1] = 1
    for i in range(len(eff) - 1, -1, -1):
        asos_at_stage[i] = math.ceil(asos_at_stage[i + 1] / eff[i]) if eff[i] > 0 else float("inf")
    total_cost = sum(asos_at_stage[i] * costs[i] for i in range(len(eff)))
    return total_cost


def main():
    if not (BENCH_PATH.exists() and PIPE_PATH.exists()):
        raise FileNotFoundError("Missing result files. Run `just analysis` and `just oligogym` first.")

    bench = json.loads(BENCH_PATH.read_text())
    preds = _load_catboost_predictions(bench)

    pipe = json.loads(PIPE_PATH.read_text())
    base_cost = pipe["baseline"]["total_cost"]
    proportions = pipe["proportions"]
    stage_costs = [s["cost_per_aso"] for s in pipe["stages"]]

    # OligoAI EF from ef_table (OligoAI row, efficacy stage)
    oligoai_ef = 1.0
    if EF_TABLE_PATH.exists():
        ef_rows = json.loads(EF_TABLE_PATH.read_text())
        for row in ef_rows:
            if row.get("name") == "OligoAI":
                ef_val = row["ef_by_stage"].get("0")
                if ef_val is not None:
                    oligoai_ef = ef_val
                break

    top_x = list(range(100, 0, -1))
    fractions = [x / 100 for x in top_x]

    ef_curves = {ds: [] for ds in DATASETS}
    tox_savings = []
    comb_savings = []

    for frac in fractions:
        task_efs = {}
        for ds in DATASETS:
            if ds not in preds:
                ef_curves[ds].append(1.0)
                task_efs[ds] = 1.0
                continue
            y_true, y_pred = preds[ds]
            stage = stage_for_dataset(ds)
            result = enrichment_at_top_k(y_true, y_pred, stage, k=frac)
            ef = result["enrichment_factor"]
            ef_curves[ds].append(ef if np.isfinite(ef) else 1.0)
            task_efs[ds] = ef if np.isfinite(ef) else 1.0

        tox_cost = _back_calculate_total_cost(proportions, stage_costs, task_efs, oligoai_ef=None)
        comb_cost = _back_calculate_total_cost(proportions, stage_costs, task_efs, oligoai_ef=oligoai_ef)
        tox_savings.append((1 - tox_cost / base_cost) * 100)
        comb_savings.append((1 - comb_cost / base_cost) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Panel A: EF curves
    for ds in DATASETS:
        ax1.plot(top_x, ef_curves[ds], linewidth=2, color=TASK_COLORS[ds], label=TASK_LABELS[ds])
    ax1.axvline(25, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax1.set_xlabel("Top X% selected (lowest predicted toxicity)")
    ax1.set_ylabel("Enrichment factor (EF)")
    ax1.set_title("A. Enrichment vs selection budget")
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, frameon=False, loc="upper right")

    # Panel B: pipeline savings curves
    ax2.plot(top_x, tox_savings, linewidth=2.2, color="#2A9D8F", label="OligoAI-tox only")
    ax2.plot(top_x, comb_savings, linewidth=2.2, color="#264653", label="Combined (OligoAI + OligoAI-tox)")
    ax2.axvline(25, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax2.set_xlabel("Top X% selected (lowest predicted toxicity)")
    ax2.set_ylabel("Pipeline cost savings (%)")
    ax2.set_title("B. Pipeline savings vs selection budget")
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    ax1.set_xlim(100, 0)
    ax2.set_xlim(100, 0)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "fig_enrichment.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
