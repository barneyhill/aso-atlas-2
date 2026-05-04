"""
Hepatotoxicity data analysis.

Cross-species concordance and biomarker correlations for hepatic
toxicity endpoints (ALT, AST, TBIL).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import calc_uln, mean_of_array
from analyses.logic.pipeline import MOUSE_ALT_ULN, RAT_ALT_ULN

_root = Path(__file__).resolve().parents[3]
_data_dir = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"

BIOMARKERS = ["ALT", "AST", "TBIL"]
THRESHOLDS = {
    "ALT": (1.5, 1.5),
    "AST": (5, 2),
    "TBIL": (5, 2),
}

# Literature ULN: mean of male + female 97.5th percentiles
# ALT: imported from pipeline.py (single source of truth)
# AST Mouse (C57BL/6J): Otto et al. 2016, JAALAS 55(4):375-386, PMID 27423143
LITERATURE_ULN = {
    "ALT": MOUSE_ALT_ULN,
    "AST": 111,   # mouse: (100 + 122) / 2
}
RAT_LITERATURE_ULN = {
    "ALT": RAT_ALT_ULN,
}


def load_and_filter():
    """Load hepatic data, filter to mouse + valid chemistry, compute means."""
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    print(f"Loaded hepatic data: {len(df):,} rows")

    df = df[
        (df["species"] == "mouse") & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"After filter (mouse, valid chemistry): {len(df):,} rows")

    for bm in BIOMARKERS:
        df[f"mean_{bm}"] = df[bm].apply(mean_of_array)
    return df.reset_index(drop=True)


def cross_species_concordance() -> dict:
    """Compare mouse vs rat biomarker values for shared compounds.

    For each biomarker (ALT, AST, TBIL):
      - Spearman rho on per-compound mean continuous values
      - Binary concordance using species-specific ULN thresholds
    """
    df_all = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df_all = df_all[df_all["HELM Annotation"].apply(Helm.valid_chemistry)].copy()

    for bm in BIOMARKERS:
        df_all[f"mean_{bm}"] = df_all[bm].apply(mean_of_array)

    mouse = df_all[df_all["species"] == "mouse"]
    rat = df_all[df_all["species"] == "rat"]

    shared_cids = set(mouse["Compound ID"].dropna()) & set(rat["Compound ID"].dropna())
    print(f"\nCross-species concordance: {len(shared_cids)} shared compounds")

    results = {}
    for bm in BIOMARKERS:
        col = f"mean_{bm}"

        mouse_means = mouse.groupby("Compound ID")[col].mean()
        rat_means = rat.groupby("Compound ID")[col].mean()
        common = mouse_means.index.intersection(rat_means.index)
        m_vals = mouse_means.loc[common].dropna()
        r_vals = rat_means.loc[common].dropna()
        both_valid = m_vals.index.intersection(r_vals.index)
        m_vals = m_vals.loc[both_valid]
        r_vals = r_vals.loc[both_valid]

        rho, pval = spearmanr(m_vals.values, r_vals.values)
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(len(m_vals) - 3)
        ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        print(f"  {bm}: n={len(m_vals)}, Spearman rho={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={pval:.2e}")

        high_mult, low_mult = THRESHOLDS[bm]
        uln = LITERATURE_ULN.get(bm)
        if uln is None:
            uln = calc_uln(mouse[col].dropna().values)

        m_label = pd.Series(index=m_vals.index, dtype=str)
        m_label[m_vals >= high_mult * uln] = "high"
        m_label[m_vals < low_mult * uln] = "low"

        r_label = pd.Series(index=r_vals.index, dtype=str)
        r_label[r_vals >= high_mult * uln] = "high"
        r_label[r_vals < low_mult * uln] = "low"

        labelled = m_label.isin(["high", "low"]) & r_label.isin(["high", "low"])
        m_lab = m_label[labelled]
        r_lab = r_label[labelled]

        hh = int(((m_lab == "high") & (r_lab == "high")).sum())
        hl = int(((m_lab == "high") & (r_lab == "low")).sum())
        lh = int(((m_lab == "low") & (r_lab == "high")).sum())
        ll = int(((m_lab == "low") & (r_lab == "low")).sum())
        n_lab = len(m_lab)
        concordance = (hh + ll) / n_lab if n_lab > 0 else 0.0
        print(f"    Binary: concordance={concordance:.1%} (n={n_lab}), "
              f"hh={hh} hl={hl} lh={lh} ll={ll}")

        results[bm] = {
            "n_shared": len(m_vals),
            "spearman_rho": round(float(rho), 3),
            "spearman_p": float(pval),
            "spearman_ci_lo": round(float(ci_lo), 3),
            "spearman_ci_hi": round(float(ci_hi), 3),
            "concordance_rate": round(concordance, 3),
            "concordance_n": n_lab,
            "crosstab": {"hh": hh, "hl": hl, "lh": lh, "ll": ll},
            "mouse_values": m_vals.tolist(),
            "rat_values": r_vals.tolist(),
        }

    return results


MOUSE_BIOMARKERS = ["ALB", "ALT", "AST", "BUN", "CREA", "TBIL"]


def mouse_biomarker_correlations() -> dict:
    """Compute Spearman correlation matrix between mouse biomarkers."""
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == "mouse") & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()

    for bm in MOUSE_BIOMARKERS:
        df[f"mean_{bm}"] = df[bm].apply(mean_of_array)

    cols = [f"mean_{bm}" for bm in MOUSE_BIOMARKERS]
    mat = df[cols].dropna(how="all")

    n = len(MOUSE_BIOMARKERS)
    rho_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)
    n_matrix = np.full((n, n), 0, dtype=int)

    for i in range(n):
        for j in range(n):
            valid = mat[[cols[i], cols[j]]].dropna()
            n_matrix[i, j] = len(valid)
            if len(valid) >= 10:
                rho, p = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
                rho_matrix[i, j] = round(float(rho), 3)
                p_matrix[i, j] = float(p)

    print(f"\nMouse biomarker correlations ({len(mat)} compounds):")
    for i, bm in enumerate(MOUSE_BIOMARKERS):
        vals = [f"{rho_matrix[i, j]:+.3f}" if not np.isnan(rho_matrix[i, j]) else "  N/A" for j in range(n)]
        print(f"  {bm:>5s}: {', '.join(vals)}")

    return {
        "biomarkers": MOUSE_BIOMARKERS,
        "rho": rho_matrix.tolist(),
        "p_values": p_matrix.tolist(),
        "n_pairs": n_matrix.tolist(),
    }


def main():
    print("=" * 60)
    print("Hepatotoxicity Analysis")
    print("=" * 60)

    cross_species = cross_species_concordance()
    biomarker_corr = mouse_biomarker_correlations()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "hepatotox.json"
    with open(out_path, "w") as f:
        json.dump({
            "cross_species": cross_species,
            "mouse_biomarker_correlations": biomarker_corr,
        }, f, indent=2)
    print(f"\nSaved {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
