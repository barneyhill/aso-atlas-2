"""
Figure 3: Demonstration — enrichment factors and cost savings.

Panel A — Enrichment factor per endpoint (horizontal bars).
           Collapsed to 5 logical endpoints (efficacy, potency,
           hepatotox, neurotox, monkey) rather than 7 per-species stages.
Panel B — Relative cost by scenario (stacked bars, same style as Fig 2C).

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
OUT_DIR = _root / "typst/plots/fig3_demo"

# ── Style constants (match Fig 2) ──
TEXT_SIZE = 16
PANEL_LABEL_SIZE = 22

# Logical endpoints: each maps to one or more pipeline stage indices.
# Enrichment factor is averaged across mapped stages when a cross-species
# model enriches both (e.g. a single hepatotox model enriches mouse + rat ALT).
ENDPOINTS = [
    {
        "name": "In vitro efficacy",
        "stage_indices": [0],
        "color": "#4878A8",
        "no_model_text": None,  # has OligoAI
    },
    {
        "name": "In vitro potency",
        "stage_indices": [1],
        "color": "#6A9BC3",
        "no_model_text": "no baseline",
    },
    {
        "name": "Hepatotoxicity (ALT)",
        "stage_indices": [2, 4],  # mouse ALT + rat ALT
        "color": "#D4A574",
        "no_model_text": None,
    },
    {
        "name": "Neurotoxicity (FOB)",
        "stage_indices": [3, 5],  # mouse FOB + rat FOB
        "color": "#E8B88A",
        "no_model_text": None,
    },
    {
        "name": "Monkey hepatotoxicity",
        "stage_indices": [6],
        "color": "#9B8AA6",
        "no_model_text": "insufficient data",
    },
]


# ---------------------------------------------------------------------------
# Panel A: Enrichment factor per endpoint
# ---------------------------------------------------------------------------

def draw_enrichment_bars(ax, combined_scenario):
    """Horizontal bar chart of enrichment factors per logical endpoint."""
    enriched = combined_scenario.get("enriched_stages", {})

    n = len(ENDPOINTS)
    y_pos = np.arange(n)[::-1]  # top-to-bottom matches pipeline order

    for i, ep in enumerate(ENDPOINTS):
        y = y_pos[i]

        # Collect enrichment factors for all stages this endpoint maps to
        efs_for_ep = []
        for si in ep["stage_indices"]:
            key = str(si)
            if key in enriched:
                efs_for_ep.append(enriched[key]["enrichment_factor"])

        if efs_for_ep:
            ef = np.mean(efs_for_ep)
            ax.barh(y, ef, height=0.6, color=ep["color"], edgecolor="white",
                    linewidth=0.5, zorder=3)
            ax.text(ef + 0.05, y, f"{ef:.2f}x",
                    va="center", ha="left", fontsize=TEXT_SIZE - 1, fontweight="bold")
        else:
            ax.barh(y, 0, height=0.6)
            text = ep["no_model_text"] or "no baseline"
            ax.text(0.15, y, text,
                    va="center", ha="left", fontsize=TEXT_SIZE - 2,
                    color="#888888", style="italic")

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([ep["name"] for ep in ENDPOINTS], fontsize=TEXT_SIZE - 1)
    ax.set_xlabel("Enrichment factor", fontsize=TEXT_SIZE)

    max_ef = max(
        (np.mean([enriched.get(str(si), {}).get("enrichment_factor", 0)
                  for si in ep["stage_indices"]]) for ep in ENDPOINTS),
        default=2,
    )
    ax.set_xlim(0, max_ef * 1.25 if max_ef > 0 else 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="x", labelsize=TEXT_SIZE - 1)


# ---------------------------------------------------------------------------
# Panel B: Relative cost reduction (stacked bars)
# ---------------------------------------------------------------------------

def draw_relative_cost_bars(ax, scenarios, stages, bar_width=0.45):
    """Stacked bar chart showing cost as % of baseline, same style as Fig 2C."""
    baseline_total = scenarios[0][1]["total_cost"]

    categories = [s["name"].replace("\n", " ") for s in reversed(stages)]
    colors = [s["color"] for s in reversed(stages)]

    n = len(scenarios)
    x_pos = list(range(n))
    last = n - 1
    midpoints = []

    bottoms = [0.0] * n
    for i, (cat, color) in enumerate(zip(categories, colors)):
        for j, (label, scenario) in enumerate(scenarios):
            val = list(reversed(scenario["costs_per_stage"]))[i] / baseline_total * 100
            ax.bar(x_pos[j], val, bar_width, bottom=bottoms[j],
                   color=color, edgecolor="white", linewidth=0.5)
            if j == last:
                midpoints.append((bottoms[j] + val / 2, cat, color))
            bottoms[j] += val

    ax.set_xticks(x_pos)
    ax.set_xticklabels([label for label, _ in scenarios], fontsize=TEXT_SIZE)

    # Total % labels on top
    totals = [sum(s["costs_per_stage"]) / baseline_total * 100 for _, s in scenarios]
    for j, total in enumerate(totals):
        reduction = 100 - total
        if reduction > 0:
            ax.text(x_pos[j], total + 1.5, f"-{reduction:.0f}%",
                    ha="center", va="bottom", fontsize=TEXT_SIZE, fontweight="bold",
                    color="#2a7f2a")
        else:
            ax.text(x_pos[j], total + 1.5, "baseline",
                    ha="center", va="bottom", fontsize=TEXT_SIZE, color="#666666")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Cost (% of baseline)", fontsize=TEXT_SIZE)
    ax.tick_params(axis="both", labelsize=TEXT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(False)

    # Inline legend (same style as Fig 2C)
    bar_right = x_pos[last] + bar_width / 2
    label_x = bar_right + 0.30
    x_left = x_pos[0] - 0.5
    grid_right = bar_right + 0.08
    ax.set_xlim(x_left, label_x + 1.0)
    ax.spines["bottom"].set_bounds(x_left, grid_right)

    for yt in ax.get_yticks():
        if ax.get_ylim()[0] <= yt <= ax.get_ylim()[1]:
            ax.hlines(yt, x_left, grid_right, colors="#666666",
                      linestyles="--", linewidth=0.7, alpha=0.35, zorder=0)

    n_labels = len(midpoints)
    y_min, y_max = ax.get_ylim()
    label_ys = np.linspace(y_min + 0.05 * (y_max - y_min), y_max * 0.85, n_labels)

    for (seg_y, cat, color), ly in zip(midpoints, label_ys):
        ax.plot([bar_right, label_x - 0.04], [seg_y, ly],
                color="black", lw=0.8, clip_on=False)
        ax.plot(label_x, ly, marker="s", markersize=8,
                color=color, clip_on=False, zorder=5)
        ax.annotate(cat, xy=(label_x, ly), xytext=(10, 0),
                    textcoords="offset points", fontsize=TEXT_SIZE,
                    va="center", ha="left", color="#333333", clip_on=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    stages = data["stages"]
    baseline = data["baseline"]

    # Build scenarios list
    scenarios = [("Baseline", baseline)]
    for key, label in [("oligoai", "OligoAI"), ("oligoai_tox", "OligoAI-tox")]:
        if data.get(key):
            scenarios.append((label, data[key]))
    if data.get("combined"):
        scenarios.append(("OligoAI +\nOligoAI-tox", data["combined"]))

    # Use combined scenario for enrichment factors (has all stages)
    combined = data.get("combined") or data.get("oligoai_tox") or {}

    # ── Figure layout: A left, B right ──
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(18, 7), dpi=300,
        gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.35},
    )

    draw_enrichment_bars(ax_a, combined)
    draw_relative_cost_bars(ax_b, scenarios, stages)

    # Panel labels
    for ax, letter in [(ax_a, "A"), (ax_b, "B")]:
        ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
                fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="bottom", ha="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig3_demo"
    fig.savefig(out.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
