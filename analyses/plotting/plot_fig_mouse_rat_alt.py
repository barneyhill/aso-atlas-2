"""
Supplementary figure: Mouse vs Rat ALT concordance.

Reads: data/oligostack/processed/hepatictoxicity_processed.parquet
Writes: typst/plots/fig_mouse_rat_alt/mouse_vs_rat_alt.{svg,png}
        data/results/mouse_rat_alt.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
_data_dir = _root / "data/oligostack/processed"
OUT_DIR = _root / "typst/plots/fig_mouse_rat_alt"


def main():
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df = df[df["species"].isin(["mouse", "rat"])]

    # Filter to the most common dose per species to reduce heterogeneity
    common_dose_by_species = (
        df.groupby("species")["dosage_mg_per_kg"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    df = df[
        df.apply(
            lambda r: r["dosage_mg_per_kg"] == common_dose_by_species.get(r["species"]),
            axis=1,
        )
    ]
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna()]

    per_compound = (
        df.groupby(["Compound ID", "species"], as_index=False)["mean_ALT"]
        .mean()
    )
    wide = per_compound.pivot(
        index="Compound ID", columns="species", values="mean_ALT"
    ).dropna()

    x = wide["mouse"].values
    y = wide["rat"].values

    rho, pval = spearmanr(x, y)
    dose_mouse = common_dose_by_species.get("mouse")
    dose_rat = common_dose_by_species.get("rat")

    # Export stats to JSON
    results_dir = _root / "data/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "spearman_rho": round(float(rho), 3),
        "p": float(pval),
        "n": int(len(wide)),
        "mouse_dose_mg_per_kg": float(dose_mouse),
        "rat_dose_mg_per_kg": float(dose_rat),
    }
    stats_path = results_dir / "mouse_rat_alt.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(x, y, s=14, alpha=0.55, c="#4878A8", edgecolors="none")


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mouse mean ALT (IU/L)", fontsize=10)
    ax.set_ylabel("Rat mean ALT (IU/L)", fontsize=10)

    ticks = [10, 100, 1000, 10000]
    tick_labels = ["10", "100", "1,000", "10,000"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / "mouse_vs_rat_alt.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_svg}")


if __name__ == "__main__":
    main()
