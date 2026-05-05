"""
Figure 3: Demonstration — enrichment factors and cost savings.

Panel A — Enrichment factor per endpoint (horizontal bars).
           Collapsed to 5 logical endpoints (efficacy, potency,
           hepatotox, neurotox, monkey) rather than 7 per-species stages.
Panel B — Relative cost by scenario (stacked bars, same style as Fig 2C).

Also generates fig_animal_reduction (appendix): stacked bar comparing
animal usage across pipeline scenarios.

Reads: data/results/ef_table.json, data/results/pipeline_results.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
EF_TABLE_PATH = _root / "data/results/ef_table.json"
PIPELINE_PATH = _root / "data/results/pipeline_results.json"
OUT_DIR = _root / "typst/plots/fig_cost"
ANIMAL_OUT_DIR = _root / "typst/plots/fig_animal_reduction"

ANIMALS_PER_ASO = 4
IN_VIVO_STAGES = [2, 3, 4, 5, 6]

# ── Style constants (match Fig 2) ──
TEXT_SIZE = 16
PANEL_LABEL_SIZE = 22

# Logical endpoints: each maps to one or more pipeline stage indices.
ENDPOINTS = [
    {
        "name": "In vitro efficacy",
        "stage_indices": [0],
        "color": "#4878A8",
        "no_model_text": None,
    },
    {
        "name": "In vitro potency",
        "stage_indices": [1],
        "color": "#6A9BC3",
        "no_model_text": "no baseline",
    },
    {
        "name": "Hepatotoxicity (ALT)",
        "stage_indices": [2, 4],
        "color": "#D4A574",
        "no_model_text": None,
    },
    {
        "name": "Neurotoxicity (FOB)",
        "stage_indices": [3, 5],
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


def _find_row(rows, **criteria):
    for r in rows:
        if all(r.get(k) == v for k, v in criteria.items()):
            return r
    return None


# ---------------------------------------------------------------------------
# Panel A: Enrichment factor per endpoint
# ---------------------------------------------------------------------------

def draw_enrichment_bars(ax, combined_row):
    """Horizontal bar chart of EFs per logical endpoint from the combined row."""
    ef_by_stage = combined_row["ef_by_stage"]
    ef_std_by_stage = combined_row.get("ef_std_by_stage", {})

    n = len(ENDPOINTS)
    y_pos = np.arange(n)[::-1]

    for i, ep in enumerate(ENDPOINTS):
        y = y_pos[i]
        efs_for_ep = []
        stds_for_ep = []
        for si in ep["stage_indices"]:
            val = ef_by_stage.get(str(si))
            if val is not None:
                efs_for_ep.append(val)
                std = ef_std_by_stage.get(str(si))
                stds_for_ep.append(std if std is not None else 0.0)

        if efs_for_ep:
            ef = np.mean(efs_for_ep)
            ef_std = np.mean(stds_for_ep) if stds_for_ep else None
            ax.barh(y, ef, height=0.6, color=ep["color"], edgecolor="white",
                    linewidth=0.5, zorder=3)
            if ef_std and ef_std > 0:
                ax.errorbar(ef, y, xerr=ef_std, fmt='none',
                            ecolor='black', capsize=3, capthick=1, elinewidth=1, zorder=10)
                ax.text(ef + ef_std + 0.08, y, f"{ef:.2f}×",
                        va="center", ha="left", fontsize=TEXT_SIZE - 1, fontweight="bold")
            else:
                ax.text(ef + 0.05, y, f"{ef:.2f}×",
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

    max_val = 0
    for ep in ENDPOINTS:
        vals = [ef_by_stage.get(str(si), 0) or 0 for si in ep["stage_indices"]]
        stds = [ef_std_by_stage.get(str(si), 0) or 0 for si in ep["stage_indices"]]
        if vals:
            max_val = max(max_val, np.mean(vals) + np.mean(stds))
    ax.set_xlim(0, max_val * 1.25 if max_val > 0 else 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="x", labelsize=TEXT_SIZE - 1)


# ---------------------------------------------------------------------------
# Panel B: Relative cost reduction (stacked bars)
# ---------------------------------------------------------------------------

def draw_relative_cost_bars(ax, scenarios, stages, bar_width=0.45):
    """Stacked bar chart showing cost as % of baseline."""
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

    bar_right = x_pos[last] + bar_width / 2
    label_x = bar_right + 0.55
    x_left = x_pos[0] - 0.5
    grid_right = bar_right + 0.08
    ax.set_xlim(x_left, label_x + 1.2)
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
# Animal reduction figure (appendix)
# ---------------------------------------------------------------------------

def draw_animal_reduction(stages, baseline_row, tox_row):
    """Stacked bar chart comparing animal usage: baseline vs OligoAI-tox."""
    scenarios = [
        ("Baseline", baseline_row),
        ("RF", tox_row),
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bar_width = 0.45
    x_pos = list(range(len(scenarios)))

    categories = []
    colors = []
    for si in reversed(IN_VIVO_STAGES):
        categories.append(stages[si]["name"].replace("\n", " "))
        colors.append(stages[si]["color"])

    bottoms = [0.0] * len(scenarios)
    last = len(scenarios) - 1
    midpoints = []

    for i, (cat, color, si) in enumerate(
        zip(categories, colors, reversed(IN_VIVO_STAGES))
    ):
        for j, (_, scenario) in enumerate(scenarios):
            val = scenario["asos_at_stage"][si] * ANIMALS_PER_ASO
            ax.bar(x_pos[j], val, bar_width, bottom=bottoms[j],
                   color=color, edgecolor="white", linewidth=0.5)
            if j == last:
                midpoints.append((bottoms[j] + val / 2, cat, color))
            bottoms[j] += val

    totals = []
    for _, scenario in scenarios:
        totals.append(sum(
            scenario["asos_at_stage"][si] * ANIMALS_PER_ASO
            for si in IN_VIVO_STAGES
        ))

    for j, total in enumerate(totals):
        ax.text(x_pos[j], total + max(totals) * 0.02, f"{total}",
                ha="center", va="bottom", fontsize=TEXT_SIZE, fontweight="bold")

    reduction = (1 - totals[1] / totals[0]) * 100
    mid_x = (x_pos[0] + x_pos[1]) / 2
    top_y = max(totals) * 1.15
    ax.annotate("", xy=(x_pos[1], top_y), xytext=(x_pos[0], top_y),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="black"))
    ax.text(mid_x, top_y + max(totals) * 0.02,
            f"{reduction:.0f}% fewer animals",
            ha="center", va="bottom", fontsize=TEXT_SIZE, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([label for label, _ in scenarios], fontsize=TEXT_SIZE)
    ax.set_ylabel("Animals", fontsize=TEXT_SIZE)
    ax.tick_params(axis="both", labelsize=TEXT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(False)
    ax.set_ylim(0, max(totals) * 1.35)

    bar_right = x_pos[last] + bar_width / 2
    label_x = bar_right + 0.55
    x_left = x_pos[0] - 0.5
    grid_right = bar_right + 0.08
    ax.set_xlim(x_left, label_x + 1.2)
    ax.spines["bottom"].set_bounds(x_left, grid_right)

    for yt in ax.get_yticks():
        if ax.get_ylim()[0] <= yt <= ax.get_ylim()[1]:
            ax.hlines(yt, x_left, grid_right, colors="#666666",
                      linestyles="--", linewidth=0.7, alpha=0.35, zorder=0)

    n_labels = len(midpoints)
    y_min, y_max = ax.get_ylim()
    label_ys = np.linspace(y_min + 0.05 * (y_max - y_min), y_max * 0.65, n_labels)

    for (seg_y, cat, color), ly in zip(midpoints, label_ys):
        ax.plot([bar_right, label_x - 0.04], [seg_y, ly],
                color="black", lw=0.8, clip_on=False)
        ax.plot(label_x, ly, marker="s", markersize=8,
                color=color, clip_on=False, zorder=5)
        ax.annotate(cat, xy=(label_x, ly), xytext=(10, 0),
                    textcoords="offset points", fontsize=TEXT_SIZE - 1,
                    va="center", ha="left", color="#333333", clip_on=False)

    ANIMAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = ANIMAL_OUT_DIR / "fig_animal_reduction"
    fig.savefig(out.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.svg')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ef_rows = json.loads(Path(EF_TABLE_PATH).read_text())
    pipeline = json.loads(Path(PIPELINE_PATH).read_text())
    stages = pipeline["stages"]

    baseline_row = _find_row(ef_rows, is_baseline=True)
    oligoai_row = _find_row(ef_rows, name="OligoAI")
    tox_row = _find_row(ef_rows, group="tox_only")
    combined_row = _find_row(ef_rows, group="combined")

    scenarios = [("Baseline", baseline_row)]
    if oligoai_row:
        scenarios.append(("OligoAI\n(in vitro)", oligoai_row))
    if tox_row:
        scenarios.append(("RF\n(in vivo)", tox_row))
    if combined_row:
        scenarios.append(("OligoAI +\nRF", combined_row))

    ef_source = combined_row or tox_row or {}

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(18, 7), dpi=300,
        gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.30},
    )

    draw_enrichment_bars(ax_a, ef_source)
    draw_relative_cost_bars(ax_b, scenarios, stages)

    for ax, letter in [(ax_a, "A"), (ax_b, "B")]:
        ax.text(-0.08, 1.05, letter, transform=ax.transAxes,
                fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="bottom", ha="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig_cost"
    fig.savefig(out.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.with_suffix('.svg')}")

    draw_animal_reduction(stages, baseline_row, tox_row)


if __name__ == "__main__":
    main()
