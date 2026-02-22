"""
Figure 5: Cost savings from computational pre-screening.

Panel A — Stacked bar chart of pipeline costs: baseline vs screened scenarios.
Panel B — Summary table (total cost, savings %, enrichment factors).

Scenarios:
  - Baseline: no computational pre-screening
  - Hagerdorn: hepatotox + neurotox classifiers (ALT 1.03x, FOB 1.57x)
  - OligoAI: in vitro efficacy model (inhibition 3.14x, Gehrmann et al. 2025)
  - Combined: all three enrichment stages

Reads: data/results/pipeline_results.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
RESULTS_PATH = _root / "data/results/pipeline_results.json"
OUT_DIR = _root / "typst/plots/fig5"


def draw_cost_comparison(scenarios, stages, ax):
    """Panel A: Stacked bar chart — costs by stage for each scenario."""
    categories = [s["short_name"].replace("\n", " ") for s in reversed(stages)]
    colors = [s["color"] for s in reversed(stages)]

    n_scenarios = len(scenarios)
    x_positions = list(range(n_scenarios))
    bar_width = 0.55

    bottoms = [0.0] * n_scenarios
    for i, (cat, color) in enumerate(zip(categories, colors)):
        for j, (label, scenario) in enumerate(scenarios):
            val = list(reversed(scenario["costs_per_stage"]))[i] / 1e6
            legend_label = cat if j == 0 else None
            ax.bar(x_positions[j], val, bar_width, bottom=bottoms[j],
                   color=color, label=legend_label, edgecolor="white", linewidth=0.5)
            bottoms[j] += val

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in scenarios], fontsize=10)

    totals = [sum(s["costs_per_stage"]) / 1e6 for _, s in scenarios]
    for j, total in enumerate(totals):
        ax.text(x_positions[j], total + max(totals) * 0.02, f"${total:.2f}M",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
              title="Stage", title_fontsize=9)
    ax.set_ylim(0, max(totals) * 1.18)
    ax.set_ylabel("Total Cost ($M)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_summary_table(baseline, scenarios, ax):
    """Panel B: Summary table of cost savings."""
    ax.axis("off")

    # Header
    y = 0.92
    ax.text(0.05, y, "Scenario", fontsize=10, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.45, y, "Cost", fontsize=10, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.65, y, "Savings", fontsize=10, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.82, y, "ASOs", fontsize=10, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.04
    ax.plot([0.03, 0.97], [y, y], color="black", linewidth=0.5,
            transform=ax.transAxes, clip_on=False)

    for label, scenario in scenarios:
        y -= 0.1
        cost = scenario["total_cost"] / 1e6
        savings = (1 - scenario["total_cost"] / baseline["total_cost"]) * 100
        n_init = scenario["n_initial"]

        ax.text(0.05, y, label, fontsize=10, transform=ax.transAxes, va="top")
        ax.text(0.45, y, f"${cost:.2f}M", fontsize=10, transform=ax.transAxes, va="top")
        if label == "Baseline":
            ax.text(0.65, y, "---", fontsize=10, color="#999999",
                    transform=ax.transAxes, va="top")
        else:
            ax.text(0.65, y, f"{savings:.0f}%", fontsize=10, fontweight="bold",
                    color="#2ca02c", transform=ax.transAxes, va="top")
        ax.text(0.82, y, f"{n_init:,}", fontsize=10, transform=ax.transAxes, va="top")

    # Enrichment factors annotation
    y -= 0.18
    ax.text(0.05, y, "Enrichment factors:", fontsize=9, fontweight="bold",
            transform=ax.transAxes, va="top", color="#555555")
    y -= 0.09
    ef_lines = [
        "OligoAI: Inhibition >80% = 3.14x",
        "Hagerdorn: Mouse ALT <100 = 1.03x",
        "Hagerdorn: Mouse FOB <=1 = 1.57x",
    ]
    for line in ef_lines:
        ax.text(0.07, y, line, fontsize=9, transform=ax.transAxes, va="top",
                color="#666666")
        y -= 0.08


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found. Run `just analysis` first.")

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    baseline = data["baseline"]
    hagerdorn = data.get("hagerdorn")
    oligoai = data.get("oligoai")
    combined = data.get("combined")
    stages = data["stages"]

    # Build scenario list
    scenarios = [("Baseline", baseline)]
    if hagerdorn:
        scenarios.append(("Hagerdorn", hagerdorn))
    if oligoai:
        scenarios.append(("OligoAI", oligoai))
    if combined:
        scenarios.append(("Combined", combined))

    if len(scenarios) < 2:
        raise RuntimeError("No enriched pipeline scenarios found.")

    fig, (ax_cost, ax_summary) = plt.subplots(
        1, 2, figsize=(15, 6), dpi=300,
        gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.35},
    )

    ax_cost.text(-0.08, 1.08, "A", transform=ax_cost.transAxes,
                 fontsize=14, fontweight="bold")
    ax_summary.text(-0.08, 1.08, "B", transform=ax_summary.transAxes,
                    fontsize=14, fontweight="bold")

    draw_cost_comparison(scenarios, stages, ax_cost)
    draw_summary_table(baseline, scenarios, ax_summary)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig5.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
