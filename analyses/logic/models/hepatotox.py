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
from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import confusion_matrix, roc_auc_score

from analyses.utils.helm import Helm
from analyses.utils.models import (
    MODELS,
    _optimal_threshold,
    calc_uln,
    mean_of_array,
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


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Assign patent-level groups for GroupKFold (patent_<USPTO ID>)."""
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    print(f"  Total grouped: {groups.notna().sum()}/{len(df)}, {groups.nunique()} unique patents")
    return groups


def build_dosing_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Build dosing covariate matrix (num_doses, dosing_period_days)."""
    cov = pd.DataFrame(index=df.index)
    for col in ["num_doses", "dosing_period_days"]:
        cov[col] = df[col].fillna(df[col].median())
    return cov


def _eval_on_dose_subset(df, y, result, dose_mg_per_kg=50):
    """Re-evaluate OOF predictions on a dosage subset.

    Returns a dict with subset metrics, or None if insufficient data.
    ``df`` must share the index of ``y`` (the labelled training subset).
    """
    dose_col = df.loc[y.index, "dosage_mg_per_kg"]
    subset = dose_col == dose_mg_per_kg

    preds_sub = result["predictions"][subset]
    labels_sub = y[subset].astype(int)

    n = len(labels_sub)
    n_high = int(labels_sub.sum())
    n_low = n - n_high
    print(f"  {dose_mg_per_kg} mg/kg subset: n={n} ({n_high} high, {n_low} low)")

    if n_high < 2 or n_low < 2:
        print(f"  {dose_mg_per_kg} mg/kg subset: insufficient class balance")
        return None

    auc = roc_auc_score(labels_sub, preds_sub)
    threshold = _optimal_threshold(labels_sub, preds_sub)
    pred_labels = (preds_sub > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_sub, pred_labels).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    _, pval = fisher_exact([[tp, fn], [fp, tn]])

    print(f"  {dose_mg_per_kg} mg/kg metrics: AUC={auc:.3f}, acc={acc:.3f}")

    return {
        "predictions": preds_sub.tolist(),
        "labels": labels_sub.tolist(),
        "n": n,
        "accuracy": float(acc),
        "auc": float(auc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "n_train_total": int(result["n"]),
    }


def run_species_models(df, groups, biomarkers, uln_dict, pred_key_prefix="",
                       cov_df=None):
    """Run all model variants × specified biomarkers.

    Evaluates dinucleotide × ALT predictions on 50 mg/kg subset.
    Returns (results_df, predictions_dict).
    """
    results = []
    predictions_dict = {}

    for model_key, spec in MODELS.items():
        feature_df = prepare_data(df, model_key)

        for bm in biomarkers:
            print(f"  {spec.name} x {bm}...", end=" ")

            high_mult, low_mult = THRESHOLDS[bm]

            gk_result = run_model_grouped(
                df, feature_df, groups, bm, spec.name,
                high_mult=high_mult, low_mult=low_mult,
                uln=uln_dict.get(bm),
                make_classifier=spec.make_classifier,
                cov_df=cov_df,
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
                if model_key in ("dinucleotide", "dinucleotide_decomposed") and bm == "ALT":
                    col = f"mean_{bm}"
                    uln_val = uln_dict.get(bm)
                    y_full = pd.Series(index=df.index, dtype=str)
                    y_full[df[col] >= high_mult * uln_val] = "high"
                    y_full[df[col] < low_mult * uln_val] = "low"
                    mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
                    y = (y_full[mask] == "high")

                    suffix = "" if model_key == "dinucleotide" else "_decomposed"
                    pred_key = f"{pred_key_prefix}ALT{suffix}"

                    # Evaluate on 50 mg/kg subset
                    subset_metrics = _eval_on_dose_subset(df, y, gk_result)
                    if subset_metrics:
                        predictions_dict[pred_key] = {**subset_metrics,
                            "feature_names": gk_result["feature_names"],
                            "fold_importances": gk_result["fold_importances"],
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
        df_all[f"mean_{bm}"] = df_all[bm].apply(mean_of_array)

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
        df[f"mean_{bm}"] = df[bm].apply(mean_of_array)
    return df.reset_index(drop=True)


def run_combined_models():
    """Run combined mouse+rat hepatotox model: sequence + species features only.

    Train on ALL data (all dosage regimes) for maximum power, then report
    metrics on OOF predictions filtered to the 50 mg/kg subset (controlled
    evaluation, no dosage confounding).

    Uses species-specific ULN thresholds for binary labels:
      - Mouse ALT ULN = 75 (Otto et al. 2016)
      - Rat ALT ULN = 39 (He et al. 2017)
    """
    print("\n" + "=" * 60)
    print("Combined (Mouse+Rat) Hepatotoxicity Model")
    print("=" * 60)

    df_all = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df_all = df_all[
        df_all["species"].isin(["mouse", "rat"])
        & df_all["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"Combined hepatic data: {len(df_all):,} rows")

    for bm in BIOMARKERS:
        df_all[f"mean_{bm}"] = df_all[bm].apply(mean_of_array)

    # Species-specific ULN for binary labels
    high_mult, low_mult = THRESHOLDS["ALT"]
    col = "mean_ALT"
    y_full = pd.Series(index=df_all.index, dtype=str)
    mouse_mask = df_all["species"] == "mouse"
    rat_mask = df_all["species"] == "rat"

    mouse_uln = LITERATURE_ULN["ALT"]  # 75
    rat_uln = RAT_LITERATURE_ULN["ALT"]  # 39

    y_full.loc[mouse_mask & (df_all[col] >= high_mult * mouse_uln)] = "high"
    y_full.loc[mouse_mask & (df_all[col] < low_mult * mouse_uln)] = "low"
    y_full.loc[rat_mask & (df_all[col] >= high_mult * rat_uln)] = "high"
    y_full.loc[rat_mask & (df_all[col] < low_mult * rat_uln)] = "low"

    n_mouse = mouse_mask.sum()
    n_rat = rat_mask.sum()
    print(f"  Mouse: {n_mouse}, Rat: {n_rat}")
    print(f"  Labels: {(y_full == 'high').sum()} high, {(y_full == 'low').sum()} low")

    df_all = df_all.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)
    rat_mask = (df_all["species"] == "rat").astype(int)

    groups = assign_groups(df_all)

    # Features: dinucleotide (128) + species_rat (1) + dosing covariates (2) = 131.
    model_key = "dinucleotide"
    spec = MODELS[model_key]
    feature_df = prepare_data(df_all, model_key)
    cov_df = build_dosing_covariates(df_all)

    mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
    if mask.sum() < 50:
        print("  Combined ALT: skip (insufficient data)")
        return {}

    X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
    X["species_rat"] = rat_mask[mask].values
    y = (y_full[mask] == "high")
    g = groups[mask]

    print(f"  Running combined dinucleotide × ALT (n={mask.sum()})...", end=" ")
    result = train_and_evaluate_grouped(X, y, g, make_classifier=spec.make_classifier)
    predictions = {}

    if result:
        print(f"AUC={result['auc']:.3f} (all data)")

        subset_metrics = _eval_on_dose_subset(df_all, y, result)
        if subset_metrics:
            predictions["combined_ALT"] = {**subset_metrics,
                "feature_names": result["feature_names"],
                "fold_importances": result["fold_importances"],
            }
    else:
        print("skip")

    return predictions


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
    print("Hagedorn 2013 — Hepatotoxicity Model Replication")
    print("=" * 60)

    df = load_and_filter()
    groups = assign_groups(df)

    cov_df = build_dosing_covariates(df)
    print(f"\nRunning models (sequence + dosing covariates, eval on 50 mg/kg)...")
    results_df, predictions_dict = run_species_models(
        df, groups, BIOMARKERS, LITERATURE_ULN, cov_df=cov_df,
    )

    cross_species = cross_species_concordance()
    biomarker_corr = mouse_biomarker_correlations()

    # ── Rat models (independent CV on rat data) ──
    print("\n" + "=" * 60)
    print("Rat Hepatotoxicity Models (independent)")
    print("=" * 60)
    rat_df = load_and_filter_rat()
    rat_groups = assign_groups(rat_df)
    rat_cov_df = build_dosing_covariates(rat_df)
    rat_results_df, rat_predictions = run_species_models(
        rat_df, rat_groups, ["ALT"], RAT_LITERATURE_ULN, "rat_",
        cov_df=rat_cov_df,
    )

    # ── Combined (mouse + rat) model ──
    combined_predictions = run_combined_models()

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
            "combined_predictions": combined_predictions,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False, float_format="%.3f"))
    print("=" * 90)

    print("\nDone.")


if __name__ == "__main__":
    main()
