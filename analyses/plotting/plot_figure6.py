"""
Figure 6: Cost savings from computational pre-screening.

Panel A — Stacked bar chart of pipeline costs: baseline vs screened scenarios.
Panel B — Summary table (total cost, savings %, enrichment factors).

Scenarios:
  - Baseline: no computational pre-screening
  - Hagedorn: hepatotox + neurotox classifiers (ALT 1.03x, FOB 1.57x)
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
OUT_DIR = _root / "typst/plots/fig6"


def draw_cost_comparison(scenarios, stages, ax):
    """Panel A: Stacked bar chart — costs by stage for each scenario.

    Labels are placed to the right of the last bar, connected by lines
    to the midpoint of each segment (bottom-to-top order matches label order).
    """
    # Reversed so the stack goes bottom→top matching the pipeline order
    categories = [s["short_name"].replace("\n", " ") for s in reversed(stages)]
    colors = [s["color"] for s in reversed(stages)]

    n_scenarios = len(scenarios)
    x_positions = list(range(n_scenarios))
    bar_width = 0.55

    # Track segment midpoints on the last (rightmost) bar for annotations
    last_idx = n_scenarios - 1
    last_midpoints = []  # (y_mid, category_name, color) bottom-to-top

    bottoms = [0.0] * n_scenarios
    for i, (cat, color) in enumerate(zip(categories, colors)):
        for j, (label, scenario) in enumerate(scenarios):
            val = list(reversed(scenario["costs_per_stage"]))[i] / 1e6
            ax.bar(x_positions[j], val, bar_width, bottom=bottoms[j],
                   color=color, edgecolor="white", linewidth=0.5)
            if j == last_idx:
                mid_y = bottoms[j] + val / 2
                last_midpoints.append((mid_y, cat, color))
            bottoms[j] += val

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in scenarios], fontsize=10)

    totals = [sum(s["costs_per_stage"]) / 1e6 for _, s in scenarios]
    for j, total in enumerate(totals):
        ax.text(x_positions[j], total + max(totals) * 0.02, f"${total:.2f}M",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(totals) * 1.28)
    ax.set_ylabel("Total Cost ($M)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Bracket between first and last bar showing cost decrease ──
    if n_scenarios >= 2:
        first_total = totals[0]
        last_total = totals[-1]
        savings_pct = (1 - last_total / first_total) * 100
        bracket_y = max(totals) * 1.12
        tick_h = max(totals) * 0.02
        x0, x1 = x_positions[0], x_positions[-1]
        ax.plot([x0, x0, x1, x1],
                [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
                color="black", lw=1.2, clip_on=False)
        ax.text((x0 + x1) / 2, bracket_y + max(totals) * 0.015,
                f"{savings_pct:.0f}% cost decrease",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Inline labels with connecting lines ──
    bar_right = x_positions[last_idx] + bar_width / 2
    elbow_x = bar_right + 0.15   # horizontal kink point
    label_x = bar_right + 0.30   # where text starts

    n_labels = len(last_midpoints)
    y_max = max(totals) * 0.78  # ~70% of full plot height
    spacing = y_max / (n_labels + 1)
    even_ys = [spacing * (i + 1) for i in range(n_labels)]

    # Size of colour swatch in data coords
    swatch_size = spacing * 0.35

    for (seg_y, cat, color), label_y in zip(last_midpoints, even_ys):
        # Straight line from segment midpoint to label
        ax.plot(
            [bar_right, label_x - 0.04],
            [seg_y, label_y],
            color="black", lw=0.8, clip_on=False,
        )
        # Colour swatch (square marker in points — aspect-ratio independent)
        ax.plot(label_x, label_y, marker='s', markersize=8,
                color=color, clip_on=False, zorder=5)
        ax.annotate(cat, xy=(label_x, label_y),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=10, va="center", ha="left", color="#333333",
                    clip_on=False)


ANIMALS_PER_ASO = 4
IN_VIVO_START = 2  # stages 0-1 are in vitro, 2+ are in vivo


def draw_animal_comparison(scenarios, stages, ax):
    """Stacked bar chart — animals per in vivo stage for each scenario."""
    # Only in vivo stages (reversed for bottom-to-top stacking)
    vivo_stages = list(reversed(stages[IN_VIVO_START:]))
    categories = [s["short_name"].replace("\n", " ") for s in vivo_stages]
    colors = [s["color"] for s in vivo_stages]

    n_scenarios = len(scenarios)
    x_positions = list(range(n_scenarios))
    bar_width = 0.55

    last_idx = n_scenarios - 1
    last_midpoints = []

    bottoms = [0.0] * n_scenarios
    for i, (cat, color) in enumerate(zip(categories, colors)):
        for j, (label, scenario) in enumerate(scenarios):
            # asos_at_stage has n_stages+1 entries; in vivo stages start at index IN_VIVO_START
            vivo_asos = list(reversed(scenario["asos_at_stage"][IN_VIVO_START:-1]))
            animals = vivo_asos[i] * ANIMALS_PER_ASO
            ax.bar(x_positions[j], animals, bar_width, bottom=bottoms[j],
                   color=color, edgecolor="white", linewidth=0.5)
            if j == last_idx:
                mid_y = bottoms[j] + animals / 2
                last_midpoints.append((mid_y, cat, color))
            bottoms[j] += animals

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in scenarios], fontsize=10)

    totals = [sum(s["asos_at_stage"][IN_VIVO_START:-1]) * ANIMALS_PER_ASO for _, s in scenarios]
    for j, total in enumerate(totals):
        ax.text(x_positions[j], total + max(totals) * 0.02, f"{total:,}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(totals) * 1.28)
    ax.set_ylabel("Animals", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bracket between first and last bar
    if n_scenarios >= 2:
        first_total = totals[0]
        last_total = totals[-1]
        reduction_pct = (1 - last_total / first_total) * 100
        bracket_y = max(totals) * 1.12
        tick_h = max(totals) * 0.02
        x0, x1 = x_positions[0], x_positions[-1]
        ax.plot([x0, x0, x1, x1],
                [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
                color="black", lw=1.2, clip_on=False)
        ax.text((x0 + x1) / 2, bracket_y + max(totals) * 0.015,
                f"{reduction_pct:.0f}% fewer animals",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Inline labels
    bar_right = x_positions[last_idx] + bar_width / 2
    label_x = bar_right + 0.30

    n_labels = len(last_midpoints)
    y_max = max(totals) * 0.78
    spacing = y_max / (n_labels + 1)
    even_ys = [spacing * (i + 1) for i in range(n_labels)]

    for (seg_y, cat, color), label_y in zip(last_midpoints, even_ys):
        ax.plot([bar_right, label_x - 0.04], [seg_y, label_y],
                color="black", lw=0.8, clip_on=False)
        ax.plot(label_x, label_y, marker='s', markersize=8,
                color=color, clip_on=False, zorder=5)
        ax.annotate(cat, xy=(label_x, label_y),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=10, va="center", ha="left", color="#333333",
                    clip_on=False)


def draw_summary_table(baseline, scenarios, ax, enriched_stages=None):
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

    # Enrichment factors annotation — read dynamically from enriched_stages
    y -= 0.18
    ax.text(0.05, y, "Enrichment factors:", fontsize=9, fontweight="bold",
            transform=ax.transAxes, va="top", color="#555555")
    y -= 0.09
    ef_lines = [
        "OligoAI: Inhibition >80% = 3.14x",
    ]
    if enriched_stages:
        stage_labels = {
            "2": "Mouse ALT",
            "3": "Mouse bFOB",
            "4": "Rat ALT",
            "5": "Rat mFOB",
        }
        for stage_idx, info in sorted(enriched_stages.items()):
            label = stage_labels.get(stage_idx, f"Stage {stage_idx}")
            ef = info["enrichment_factor"]
            ef_lines.append(f"Hagedorn: {label} = {ef:.2f}x")
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
    if oligoai:
        scenarios.append(("OligoAI", oligoai))
    if hagerdorn:
        scenarios.append(("Hagedorn", hagerdorn))
    if combined:
        scenarios.append(("OligoAI+Hagedorn", combined))

    if len(scenarios) < 2:
        raise RuntimeError("No enriched pipeline scenarios found.")

    fig, ax_cost = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

    draw_cost_comparison(scenarios, stages, ax_cost)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig6.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # ── Variant: Baseline + OligoAI only ──
    if oligoai:
        scenarios_oa = [("Baseline", baseline), ("OligoAI", oligoai)]
        fig_oa, ax_oa = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
        draw_cost_comparison(scenarios_oa, stages, ax_oa)
        oa_path = OUT_DIR / "fig6-just-oligoAI.svg"
        fig_oa.savefig(oa_path, format="svg", bbox_inches="tight")
        plt.close(fig_oa)
        print(f"Saved {oa_path}")


    # ── Variant: Animals (Baseline vs Hagedorn only) ──
    scenarios_an = [("Baseline", baseline)]
    if hagerdorn:
        scenarios_an.append(("Hagedorn", hagerdorn))
    fig_an, ax_an = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    draw_animal_comparison(scenarios_an, stages, ax_an)
    an_path = OUT_DIR / "fig6-animals.svg"
    fig_an.savefig(an_path, format="svg", bbox_inches="tight")
    plt.close(fig_an)
    print(f"Saved {an_path}")


if __name__ == "__main__":
    main()
