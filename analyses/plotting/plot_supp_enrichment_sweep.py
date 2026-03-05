"""
Supplementary figure: enrichment and pipeline savings vs top-X% selection.

Top-X% means selecting the X% compounds with lowest predicted P(high toxicity)
for each toxicity endpoint (ALT, FOB, rat_ALT, rat_FOB).

Panel A: Enrichment factor curves vs top-X%.
Panel B: Pipeline savings (%) vs top-X% (OligoAI-tox only, Combined with OligoAI).
"""

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
HEP_PATH = _root / "data/results/hepatotox.json"
NEURO_PATH = _root / "data/results/neurotox.json"
PIPE_PATH = _root / "data/results/pipeline_results.json"
OUT_DIR = _root / "typst/plots/supp_enrichment"

TASKS = ["ALT", "FOB", "rat_ALT", "rat_FOB"]
STAGE_MAP = {"ALT": 2, "FOB": 3, "rat_ALT": 4, "rat_FOB": 5}
TASK_LABELS = {
    "ALT": "Mouse ALT",
    "FOB": "Mouse bFOB",
    "rat_ALT": "Rat ALT",
    "rat_FOB": "Rat mFOB",
}
TASK_COLORS = {
    "ALT": "#D4A574",
    "FOB": "#E8B88A",
    "rat_ALT": "#7BAA97",
    "rat_FOB": "#94C4A7",
}


def _load_prediction_tables():
    with open(HEP_PATH) as f:
        hep = json.load(f)
    with open(NEURO_PATH) as f:
        neuro = json.load(f)

    preds = {}
    preds.update(hep["predictions"])
    preds.update(neuro["predictions"])
    preds.update(hep.get("rat_predictions", {}))
    preds.update(neuro.get("rat_predictions", {}))
    return preds


def _enrichment_at_top_fraction(predictions, labels, fraction):
    preds = np.asarray(predictions, dtype=float)
    y = np.asarray(labels, dtype=int)  # 1=high tox, 0=low tox/pass
    n = len(y)
    k = max(1, int(np.floor(fraction * n)))
    selected_idx = np.argsort(preds, kind="mergesort")[:k]

    base_pass = float((y == 0).mean())
    selected_pass = float((y[selected_idx] == 0).mean())
    ef = selected_pass / base_pass if base_pass > 0 else np.nan
    return ef, selected_pass, base_pass, k


def _back_calculate_total_cost(proportions, costs, toxicity_ef, oligoai_ef=None):
    eff = list(proportions)
    for task, stage_idx in STAGE_MAP.items():
        eff[stage_idx] = min(eff[stage_idx] * toxicity_ef[task], 1.0)
    if oligoai_ef is not None:
        eff[0] = min(eff[0] * oligoai_ef, 1.0)

    asos_at_stage = [0] * (len(eff) + 1)
    asos_at_stage[-1] = 1
    for i in range(len(eff) - 1, -1, -1):
        asos_at_stage[i] = math.ceil(asos_at_stage[i + 1] / eff[i]) if eff[i] > 0 else float("inf")
    total_cost = sum(asos_at_stage[i] * costs[i] for i in range(len(eff)))
    return total_cost


def main():
    if not (HEP_PATH.exists() and NEURO_PATH.exists() and PIPE_PATH.exists()):
        raise FileNotFoundError("Missing result files. Run `just analysis` first.")

    preds = _load_prediction_tables()
    with open(PIPE_PATH) as f:
        pipe = json.load(f)

    base_cost = pipe["baseline"]["total_cost"]
    proportions = pipe["proportions"]
    stage_costs = [s["cost_per_aso"] for s in pipe["stages"]]
    oligoai_ef = pipe.get("oligoai", {}).get("enriched_stages", {}).get("0", {}).get("enrichment_factor", 3.14)

    top_x = list(range(50, 0, -1))  # 50 -> 1
    fractions = [x / 100 for x in top_x]

    ef_curves = {t: [] for t in TASKS}
    selected_curves = {t: [] for t in TASKS}
    tox_savings = []
    comb_savings = []

    for frac in fractions:
        task_efs = {}
        for task in TASKS:
            d = preds[task]
            ef, sel_pass, _, k = _enrichment_at_top_fraction(d["predictions"], d["labels"], frac)
            ef_curves[task].append(ef)
            selected_curves[task].append(k / len(d["labels"]))
            task_efs[task] = ef

        tox_cost = _back_calculate_total_cost(proportions, stage_costs, task_efs, oligoai_ef=None)
        comb_cost = _back_calculate_total_cost(proportions, stage_costs, task_efs, oligoai_ef=oligoai_ef)
        tox_savings.append((1 - tox_cost / base_cost) * 100)
        comb_savings.append((1 - comb_cost / base_cost) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    # Panel A: EF curves
    for task in TASKS:
        ax1.plot(top_x, ef_curves[task], linewidth=2, color=TASK_COLORS[task], label=TASK_LABELS[task])
    ax1.axvline(25, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax1.text(25.8, ax1.get_ylim()[1] * 0.96, "chosen = top 25%", fontsize=8, color="#444444", va="top")
    ax1.set_xlabel("Top X% selected (lowest predicted toxicity)")
    ax1.set_ylabel("Enrichment factor (EF)")
    ax1.set_title("A. Enrichment vs selection budget")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, frameon=False, loc="upper right")

    # Panel B: pipeline savings curves
    ax2.plot(top_x, tox_savings, linewidth=2.2, color="#2A9D8F", label="OligoAI-tox only")
    ax2.plot(top_x, comb_savings, linewidth=2.2, color="#264653", label="Combined (OligoAI + OligoAI-tox)")
    ax2.axvline(25, color="#444444", linestyle="--", linewidth=1, alpha=0.8)
    ax2.set_xlabel("Top X% selected (lowest predicted toxicity)")
    ax2.set_ylabel("Pipeline cost savings (%)")
    ax2.set_title("B. Pipeline savings vs selection budget")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    # Match x-axis direction to requested top50 -> top1 sweep.
    ax1.set_xlim(50, 1)
    ax2.set_xlim(50, 1)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "supp_enrichment_sweep.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
