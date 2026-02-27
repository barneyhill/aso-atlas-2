"""
Hagedorn 2013 hepatotoxicity model replication.

Random forest classifiers for ALT, AST, TBIL using dinucleotide
and other sequence features. Adapted from LNA to DNA/MOE/cET chemistry
with target-level train/test splits (GroupKFold).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import (
    MODELS,
    calc_uln,
    prepare_data,
    run_model_grouped,
    train_and_evaluate_grouped,
)
_root = Path(__file__).resolve().parents[3]
_data_dir = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"

BIOMARKERS = ["ALT", "AST", "TBIL"]
THRESHOLDS = {
    "ALT": (2, 2),   # 2× ULN: matches pipeline pass threshold
    "AST": (5, 2),   # Hagedorn 2013 original
    "TBIL": (5, 2),
}

# Literature ULN: mean of male + female 97.5th percentiles
# Mouse (C57BL/6J): Otto et al. 2016, JAALAS 55(4):375-386, PMID 27423143
# Rat (Sprague-Dawley): He et al. 2017, PLoS ONE 12(12):e0189837, PMID 29261747
LITERATURE_ULN = {
    "ALT": 75,    # mouse: (56 + 94) / 2
    "AST": 111,   # mouse: (100 + 122) / 2
}
RAT_LITERATURE_ULN = {
    "ALT": 39,    # rat: (47 + 30) / 2
}


def _mean_of_array(val):
    """Compute mean from array/scalar biomarker column."""
    if isinstance(val, (np.ndarray, list)):
        valid = [v for v in val if v is not None and not np.isnan(v)]
        return np.mean(valid) if valid else np.nan
    if val is not None and not np.isnan(val):
        return float(val)
    return np.nan


def load_and_filter():
    """Load hepatic data, filter to mouse + valid chemistry, compute means."""
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    print(f"Loaded hepatic data: {len(df):,} rows")

    df = df[
        (df["species"] == "mouse") & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"After filter (mouse, valid chemistry): {len(df):,} rows")

    for bm in BIOMARKERS:
        df[f"mean_{bm}"] = df[bm].apply(_mean_of_array)
    return df.reset_index(drop=True)


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Assign target-level groups for GroupKFold.

    Priority:
    1. target_RNA column (direct)
    2. Compound ID join to in_vitro_inhibition target_RNA
    3. USPTO ID as proxy (92% of patents have 1 target)
    """
    groups = df["target_RNA"].copy()

    # Fill gaps via in_vitro Compound ID -> target_RNA
    missing = groups.isna()
    if missing.any():
        invitro = pd.read_parquet(_data_dir / "in_vitro_inhibition_processed.parquet")
        cid_to_target = (
            invitro.dropna(subset=["target_RNA"])
            .groupby("Compound ID")["target_RNA"]
            .first()
        )
        filled = df.loc[missing, "Compound ID"].map(cid_to_target)
        groups.loc[missing] = filled
        n_filled = filled.notna().sum()
        print(f"  Filled {n_filled} groups via in_vitro Compound ID join")

    # Remaining: use USPTO ID as proxy
    still_missing = groups.isna()
    if still_missing.any():
        groups.loc[still_missing] = df.loc[still_missing, "USPTO ID"].apply(
            lambda x: f"patent_{x}" if pd.notna(x) else None
        )
        n_patent = still_missing.sum() - groups.isna().sum()
        print(f"  Filled {n_patent} groups via USPTO ID proxy")

    print(f"  Total grouped: {groups.notna().sum()}/{len(df)}, {groups.nunique()} unique groups")
    return groups


def build_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Build dosing covariate matrix."""
    cov = pd.DataFrame(index=df.index)
    for col in ["dosage_mg_per_kg", "num_doses", "dosing_period_days"]:
        cov[col] = df[col].fillna(df[col].median())

    cov["admin_subcut"] = (
        df["administration_method"].str.lower().str.contains("subcut", na=False).astype(int)
        if "administration_method" in df.columns
        else 0
    )
    return cov


def run_all_models(df, groups, cov_df):
    """Run all model x biomarker combinations with GroupKFold CV.

    Returns (results_df, predictions_dict) where predictions_dict contains
    the dinucleotide model × ALT predictions for pipeline enrichment.
    """
    results = []
    predictions_dict = {}

    for model_key, spec in MODELS.items():
        feature_df = prepare_data(df, model_key)

        for bm in BIOMARKERS:
            print(f"  {spec.name} x {bm}...", end=" ")

            high_mult, low_mult = THRESHOLDS[bm]

            gk_result = run_model_grouped(
                df, feature_df, cov_df, groups, bm, spec.name,
                high_mult=high_mult, low_mult=low_mult,
                uln=LITERATURE_ULN.get(bm),
            )

            row = {"model": spec.name, "biomarker": bm}
            if gk_result:
                row.update({
                    "N": gk_result["n"],
                    "N_high": gk_result["n_high"],
                    "N_low": gk_result["n_low"],
                    "GK_accuracy": gk_result["accuracy"],
                    "GK_sensitivity": gk_result["sensitivity"],
                    "GK_specificity": gk_result["specificity"],
                    "GK_AUC": gk_result["auc"],
                    "GK_pvalue": gk_result["p_value"],
                    "N_groups": gk_result["n_groups"],
                })
                print(f"GK acc={gk_result['accuracy']:.3f}")

                # Save dinucleotide × ALT predictions for pipeline enrichment
                if model_key == "dinucleotide" and bm == "ALT":
                    col = f"mean_{bm}"
                    uln_val = LITERATURE_ULN.get(bm)
                    y_full = pd.Series(index=df.index, dtype=str)
                    y_full[df[col] >= high_mult * uln_val] = "high"
                    y_full[df[col] < low_mult * uln_val] = "low"
                    mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
                    y = (y_full[mask] == "high")

                    predictions_dict["ALT"] = {
                        "predictions": gk_result["predictions"].tolist(),
                        "labels": y.astype(int).tolist(),
                        "n": int(gk_result["n"]),
                        "accuracy": float(gk_result["accuracy"]),
                        "auc": float(gk_result["auc"]),
                        "sensitivity": float(gk_result["sensitivity"]),
                        "specificity": float(gk_result["specificity"]),
                        "confusion": {k: int(v) for k, v in gk_result["confusion"].items()},
                    }
            else:
                print("GK=skip")

            results.append(row)

    return pd.DataFrame(results), predictions_dict


def cross_species_concordance() -> dict:
    """Compare mouse vs rat biomarker values for shared compounds.

    For each biomarker (ALT, AST, TBIL):
      - Spearman ρ on per-compound mean continuous values
      - Binary concordance using species-specific ULN thresholds
    """
    df_all = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df_all = df_all[df_all["HELM Annotation"].apply(Helm.valid_chemistry)].copy()

    for bm in BIOMARKERS:
        df_all[f"mean_{bm}"] = df_all[bm].apply(_mean_of_array)

    mouse = df_all[df_all["species"] == "mouse"]
    rat = df_all[df_all["species"] == "rat"]

    shared_cids = set(mouse["Compound ID"].dropna()) & set(rat["Compound ID"].dropna())
    print(f"\nCross-species concordance: {len(shared_cids)} shared compounds")

    results = {}
    for bm in BIOMARKERS:
        col = f"mean_{bm}"

        # Per-compound mean in each species
        mouse_means = mouse.groupby("Compound ID")[col].mean()
        rat_means = rat.groupby("Compound ID")[col].mean()
        common = mouse_means.index.intersection(rat_means.index)
        m_vals = mouse_means.loc[common].dropna()
        r_vals = rat_means.loc[common].dropna()
        both_valid = m_vals.index.intersection(r_vals.index)
        m_vals = m_vals.loc[both_valid]
        r_vals = r_vals.loc[both_valid]

        rho, pval = spearmanr(m_vals.values, r_vals.values)
        # Fisher z 95% CI
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(len(m_vals) - 3)
        ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        print(f"  {bm}: n={len(m_vals)}, Spearman ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={pval:.2e}")

        # Binary labels using literature ULN (same for both species)
        high_mult, low_mult = THRESHOLDS[bm]
        uln = LITERATURE_ULN.get(bm)
        if uln is None:
            # Fallback for TBIL (no literature value available)
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


def load_and_filter_rat():
    """Load hepatic data, filter to rat + valid chemistry, compute means."""
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[
        (df["species"] == "rat") & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"Rat hepatic data: {len(df):,} rows")

    for bm in BIOMARKERS:
        df[f"mean_{bm}"] = df[bm].apply(_mean_of_array)
    return df.reset_index(drop=True)


def run_rat_models(df, groups, cov_df):
    """Run all RF model variants on rat ALT data with GroupKFold.

    Returns (results_df, predictions_dict) — same structure as mouse models.
    """
    results = []
    predictions_dict = {}
    uln = RAT_LITERATURE_ULN["ALT"]
    high_mult, low_mult = THRESHOLDS["ALT"]

    for model_key, spec in MODELS.items():
        print(f"  {spec.name}...", end=" ")
        gk_result = run_model_grouped(
            df, prepare_data(df, model_key), cov_df, groups, "ALT", spec.name,
            high_mult=high_mult, low_mult=low_mult, uln=uln,
        )

        row = {"model": spec.name, "biomarker": "ALT"}
        if gk_result:
            row.update({
                "N": gk_result["n"],
                "N_high": gk_result["n_high"],
                "N_low": gk_result["n_low"],
                "GK_accuracy": gk_result["accuracy"],
                "GK_sensitivity": gk_result["sensitivity"],
                "GK_specificity": gk_result["specificity"],
                "GK_AUC": gk_result["auc"],
                "GK_pvalue": gk_result["p_value"],
                "N_groups": gk_result["n_groups"],
            })
            print(f"GK acc={gk_result['accuracy']:.3f}")

            if model_key == "dinucleotide":
                predictions_dict["rat_ALT"] = {
                    "predictions": gk_result["predictions"].tolist(),
                    "labels": (gk_result["predictions"] > 0).astype(int).tolist(),  # placeholder
                    "n": int(gk_result["n"]),
                    "accuracy": float(gk_result["accuracy"]),
                    "auc": float(gk_result["auc"]),
                    "sensitivity": float(gk_result["sensitivity"]),
                    "specificity": float(gk_result["specificity"]),
                    "confusion": {k: int(v) for k, v in gk_result["confusion"].items()},
                }
        else:
            print("GK=skip")

        results.append(row)

    return pd.DataFrame(results), predictions_dict


def _run_rat_dinuc_alt(df, groups, cov_df):
    """Run just the dinucleotide model on rat ALT to get proper labels+predictions.

    We need both predictions AND true labels for enrichment calculation,
    which run_model_grouped doesn't directly return.
    """
    uln = RAT_LITERATURE_ULN["ALT"]
    high_mult, low_mult = THRESHOLDS["ALT"]
    feature_df = prepare_data(df, "dinucleotide")

    col = "mean_ALT"
    y_full = pd.Series(index=df.index, dtype=str)
    y_full[df[col] >= high_mult * uln] = "high"
    y_full[df[col] < low_mult * uln] = "low"

    mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
    if mask.sum() < 50:
        return None

    X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
    y = (y_full[mask] == "high")
    g = groups[mask]

    result = train_and_evaluate_grouped(X, y, g)
    if result is None:
        return None

    return {
        "predictions": result["predictions"].tolist(),
        "labels": y.astype(int).tolist(),
        "n": int(result["n"]),
        "accuracy": float(result["accuracy"]),
        "auc": float(result["auc"]),
        "sensitivity": float(result["sensitivity"]),
        "specificity": float(result["specificity"]),
        "confusion": {k: int(v) for k, v in result["confusion"].items()},
    }


MOUSE_BIOMARKERS = ["ALB", "ALT", "AST", "BUN", "CREA", "TBIL"]


def mouse_biomarker_correlations() -> dict:
    """Compute Spearman correlation matrix between mouse biomarkers."""
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == "mouse") & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()

    for bm in MOUSE_BIOMARKERS:
        df[f"mean_{bm}"] = df[bm].apply(_mean_of_array)

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
    print("Hagedorn 2013 — Hepatotoxicity Model Replication")
    print("=" * 60)

    df = load_and_filter()
    groups = assign_groups(df)
    cov_df = build_covariates(df)

    print(f"\nRunning models...")
    results_df, predictions_dict = run_all_models(df, groups, cov_df)

    cross_species = cross_species_concordance()
    biomarker_corr = mouse_biomarker_correlations()

    # ── Rat models (independent CV on rat data) ──
    print("\n" + "=" * 60)
    print("Rat Hepatotoxicity Models (independent)")
    print("=" * 60)
    rat_df = load_and_filter_rat()
    rat_groups = assign_groups(rat_df)
    rat_cov = build_covariates(rat_df)
    rat_results_df, _ = run_rat_models(rat_df, rat_groups, rat_cov)

    # Get dinucleotide predictions with proper labels for enrichment
    rat_dinuc = _run_rat_dinuc_alt(rat_df, rat_groups, rat_cov)
    rat_predictions = {}
    if rat_dinuc:
        rat_predictions["rat_ALT"] = rat_dinuc

    # Save consolidated JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "hepatotox.json"
    with open(out_path, "w") as f:
        json.dump({
            "models": results_df.to_dict(orient="records"),
            "predictions": predictions_dict,
            "cross_species": cross_species,
            "mouse_biomarker_correlations": biomarker_corr,
            "rat_models": rat_results_df.to_dict(orient="records"),
            "rat_predictions": rat_predictions,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False, float_format="%.3f"))
    print("=" * 90)

    print("\nDone.")


if __name__ == "__main__":
    main()
