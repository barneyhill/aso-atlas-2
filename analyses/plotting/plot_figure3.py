"""
Figure 3: ION582 benchmarking — UBE3A-ATS vs other targets.

Panel A — IC50 distribution (gymnosis/free uptake) split by target_RNA.
Panel B — Mouse FOB score distribution (ICV, 700 ug, 3 h) split by target_RNA.
Panel C — Rat FOB score distribution (IT, 3000 ug, 3 h) split by target_RNA.

ION582 is marked as a vertical line with percentile label.
"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=RuntimeWarning)

_root = Path(__file__).resolve().parents[2]
DATA_DIR = _root / "data" / "oligostack" / "processed"
OUT_DIR = _root / "typst" / "plots" / "fig3"

ION582 = {"id": 1273062, "name": "ION582", "color": "#E64B35"}
TARGET = "UBE3A-ATS"


# ---------------------------------------------------------------------------
# IC50 fitting (mirrors analyses/logic/pipeline.py)
# ---------------------------------------------------------------------------

def _hill(dose, bottom, top, ic50, slope):
    return bottom + (top - bottom) / (1 + (ic50 / dose) ** slope)


def _fit_ic50(doses, responses):
    mask = ~(np.isnan(doses) | np.isnan(responses))
    doses, responses = np.array(doses)[mask], np.array(responses)[mask]
    if len(doses) < 4:
        return np.nan
    min_d, max_d = doses.min(), doses.max()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _hill, doses, responses,
                p0=[np.min(responses), np.max(responses), np.median(doses), 1.0],
                bounds=([-50, 20, min_d / 100, 0.1], [50, 120, max_d * 100, 10]),
                maxfev=5000,
            )
        ic50 = popt[2]
        pred = _hill(doses, *popt)
        ss_res = np.sum((responses - pred) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 > 0.5 and min_d / 10 <= ic50 <= max_d * 10:
            return ic50
        return np.nan
    except (RuntimeError, ValueError):
        return np.nan


def _fit_ic50s_for_subset(dr):
    """Fit IC50 per compound from a pre-filtered dataframe."""
    ic50s = {}
    for cid, g in dr.groupby("Compound ID"):
        vals = []
        for _, eg in g.groupby(["USPTO ID", "Table Number"]):
            v = _fit_ic50(eg["dosage_nm"].values, eg["Inhibition_pct"].values)
            if not np.isnan(v):
                vals.append(v)
        if vals:
            ic50s[cid] = np.median(vals)
    return ic50s


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _mark_drug(ax, x_val, dist_vals, name, color, y_top_frac=0.95):
    """Mark a drug on a histogram with a downward arrow and label above."""
    ylim = ax.get_ylim()
    arrow_top = ylim[1] * 0.35
    arrow_bot = 0
    ax.annotate(
        "", xy=(x_val, arrow_bot), xytext=(x_val, arrow_top),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=12),
        zorder=5,
    )
    ax.text(
        x_val, arrow_top + ylim[1] * 0.03, name,
        color=color, fontsize=8, fontweight="bold",
        va="bottom", ha="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )


def draw_ic50_panel(dr, ax, show_drug=True, show_target=True):
    """Panel A: gymnosis/free-uptake IC50 distribution split by target_RNA."""
    free = dr[
        (dr["transfection_method"] == "Gymnosis") | (dr["transfection_method"].isna())
    ]
    ic50s = _fit_ic50s_for_subset(free)

    # Build compound→target lookup
    cid_target = dr.drop_duplicates("Compound ID").set_index("Compound ID")["target_RNA"]

    if show_target:
        target_vals = np.array([v for k, v in ic50s.items() if cid_target.get(k) == TARGET])
        other_vals = np.array([v for k, v in ic50s.items() if cid_target.get(k) != TARGET])
    else:
        target_vals = np.array([])
        other_vals = np.array(list(ic50s.values()))
    all_vals = np.array(list(ic50s.values()))

    if len(all_vals) == 0:
        return
    log_all = np.log10(all_vals)
    bins = np.linspace(log_all.min(), log_all.max(), 31)

    if len(other_vals) > 0:
        ax.hist(np.log10(other_vals), bins=bins, color="#D0D0D0", edgecolor="grey",
                linewidth=0.4, alpha=0.7, zorder=2)
    if len(target_vals) > 0:
        ax.hist(np.log10(target_vals), bins=bins, color="#4878A8", edgecolor="black",
                linewidth=0.4, alpha=0.85, zorder=3)

    if show_drug:
        ion582_ic50 = ic50s.get(ION582["id"])
        if ion582_ic50 and not np.isnan(ion582_ic50):
            _mark_drug(ax, np.log10(ion582_ic50), log_all,
                       ION582["name"], ION582["color"])

    ax.set_xlabel("log\u2081\u2080(IC\u2085\u2080, nM)")
    ax.set_ylabel("Count")
    ax.set_title("In vitro potency — free uptake", fontsize=10)
    ax.grid(True, alpha=0.2)


def draw_neuro_panel(neuro, species, ax, show_drug=True, show_target=True):
    """FOB distribution split by target_RNA with ION582 marked."""
    sub = neuro[(neuro["species"] == species) & (neuro["FOB_val"].notna())].copy()
    sub["FOB_int"] = sub["FOB_val"].round().astype(int)

    if show_target:
        target_sub = sub[sub["target_RNA"] == TARGET]
        other_sub = sub[sub["target_RNA"] != TARGET]
    else:
        target_sub = sub.iloc[0:0]
        other_sub = sub

    bins = np.arange(-0.5, 8.5, 1)

    ax.hist(other_sub["FOB_int"].values, bins=bins, color="#D0D0D0", edgecolor="grey",
            linewidth=0.4, alpha=0.7, zorder=2)
    if len(target_sub) > 0:
        ax.hist(target_sub["FOB_int"].values, bins=bins, color="#4878A8", edgecolor="black",
                linewidth=0.4, alpha=0.85, zorder=3)

    if show_drug:
        rows = neuro[(neuro["Compound ID"] == ION582["id"]) & (neuro["species"] == species)]
        if len(rows) > 0:
            fob = round(rows["FOB_val"].iloc[0])
            _mark_drug(ax, fob, sub["FOB_int"].values, ION582["name"], ION582["color"])

    species_config = {
        "Mouse": ("bFOB", "ICV 700 \u00b5g, 3 h post-dose"),
        "Rat": ("mFOB", "IT 3,000 \u00b5g, 3 h post-dose"),
    }
    fob_label, detail = species_config[species]
    ax.set_xlabel(f"{fob_label} score")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{species} {fob_label} — {detail}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.2)


def main():
    dr = pd.read_parquet(DATA_DIR / "dose_response_processed.parquet")
    neuro = pd.read_parquet(DATA_DIR / "neurotoxicity_processed.parquet")
    neuro["FOB_val"] = neuro["FOB_score"].apply(
        lambda x: float(x[0]) if isinstance(x, np.ndarray) and len(x) > 0 else np.nan
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    draw_ic50_panel(dr, axes[0])
    draw_neuro_panel(neuro, "Mouse", axes[1])
    draw_neuro_panel(neuro, "Rat", axes[2])

    for i, letter in enumerate("ABC"):
        axes[i].text(-0.08, 1.08, letter, transform=axes[i].transAxes,
                     fontsize=14, fontweight="bold")

    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / "fig3.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(svg_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {svg_path}")

    # ── Variant: without ION582 ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.2))
    draw_ic50_panel(dr, axes2[0], show_drug=False)
    draw_neuro_panel(neuro, "Mouse", axes2[1], show_drug=False)
    draw_neuro_panel(neuro, "Rat", axes2[2], show_drug=False)
    for i, letter in enumerate("ABC"):
        axes2[i].text(-0.08, 1.08, letter, transform=axes2[i].transAxes,
                      fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = OUT_DIR / "fig3-without-zilg.svg"
    fig2.savefig(p2, format="svg", bbox_inches="tight")
    fig2.savefig(p2.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {p2}")

    # ── Variant: without ION582 and without UBE3A-ATS highlight ──
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.2))
    draw_ic50_panel(dr, axes3[0], show_drug=False, show_target=False)
    draw_neuro_panel(neuro, "Mouse", axes3[1], show_drug=False, show_target=False)
    draw_neuro_panel(neuro, "Rat", axes3[2], show_drug=False, show_target=False)
    for i, letter in enumerate("ABC"):
        axes3[i].text(-0.08, 1.08, letter, transform=axes3[i].transAxes,
                      fontsize=14, fontweight="bold")
    plt.tight_layout()
    p3 = OUT_DIR / "fig3-without-zilg-without-ube3a-ats.svg"
    fig3.savefig(p3, format="svg", bbox_inches="tight")
    fig3.savefig(p3.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {p3}")


if __name__ == "__main__":
    main()
