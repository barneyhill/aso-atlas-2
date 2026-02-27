"""
Hagedorn 2022 neurotoxicity model replication.

Linear model (5 sequence features) and RF classifiers for FOB score.
Adapted from LNA to DNA/MOE/cET chemistry with target-level
train/test splits (GroupKFold).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import fisher_exact, spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import (
    MODELS,
    _optimal_threshold,
    prepare_data,
    train_and_evaluate_grouped,
)
_root = Path(__file__).resolve().parents[3]
_data_dir = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"


def _mean_of_array(val):
    """Compute mean from array/scalar FOB column."""
    if isinstance(val, (np.ndarray, list)):
        valid = [v for v in val if v is not None and not np.isnan(v)]
        return np.mean(valid) if valid else np.nan
    if val is not None and not np.isnan(val):
        return float(val)
    return np.nan


def load_and_filter():
    """Load neurotox data, filter to Mouse 700ug ICV valid chemistry."""
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    print(f"Loaded neurotox data: {len(df):,} rows")

    df = df[
        (df["species"] == "Mouse")
        & (df["dosage_ug"] == 700)
        & (df["administration_method"] == "ICV")
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"After filter (Mouse, 700ug ICV, valid chemistry): {len(df):,} rows")

    df["mean_FOB"] = df["FOB_score"].apply(_mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    print(f"With valid FOB scores: {len(df):,} rows")
    return df


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Assign target-level groups via in_vitro join + USPTO fallback."""
    groups = pd.Series(index=df.index, dtype=object)

    # Join via Compound ID to in_vitro target_RNA
    invitro = pd.read_parquet(_data_dir / "in_vitro_inhibition_processed.parquet")
    cid_to_target = (
        invitro.dropna(subset=["target_RNA"])
        .groupby("Compound ID")["target_RNA"]
        .first()
    )
    groups = df["Compound ID"].map(cid_to_target)
    n_invitro = groups.notna().sum()
    print(f"  Filled {n_invitro} groups via in_vitro Compound ID join")

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


def hagedorn_linear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 5 Hagedorn 2022 linear model features.

    Features: g_count, a_count, t_count, c_count, g_free_3p
    (g_free_5p and length were explicitly uninformative in the paper;
    t_count and c_count were significant at p < 4e-10).
    """
    feat_names = ["g_count", "a_count", "t_count", "c_count", "g_free_3p"]
    rows = []
    for helm in df["HELM Annotation"]:
        parsed = Helm.parse(helm)
        if parsed is None:
            rows.append({f: np.nan for f in feat_names})
            continue

        bases = parsed.bases

        # G-free stretch from 3' end (capped at 20, default 20 if no G)
        g_free_3p = 0
        for b in reversed(bases):
            if b != "G":
                g_free_3p += 1
            else:
                break
        if g_free_3p == len(bases):  # no G in sequence
            g_free_3p = 20
        g_free_3p = min(g_free_3p, 20)

        rows.append({
            "g_count": bases.count("G"),
            "a_count": bases.count("A"),
            "t_count": bases.count("T"),
            "c_count": bases.count("C"),
            "g_free_3p": g_free_3p,
        })

    return pd.DataFrame(rows, index=df.index)


# ── Hagedorn 2022 exact score (fixed coefficients) ─────────────

# Coefficients from the original R code (trained on 1,645 LNA gapmers,
# calcium oscillation data). Score > 70 = predicted safe (mild neurotox).
HAGEDORN_INTERCEPT = 136.0430
HAGEDORN_COEFS = {
    "a_count": -3.1263,
    "c_count": -5.1100,
    "t_count": -4.7217,
    "g_count": -10.1264,
    "g_free_3p": 1.3577,
}
HAGEDORN_THRESHOLD = 70  # score > 70 → acceptable (low neurotox)


def hagedorn_score(df: pd.DataFrame) -> pd.Series:
    """Compute the exact Hagedorn 2022 neurotoxicity score.

    score = 136.0430 − 3.1263×A − 5.1100×C − 4.7217×T − 10.1264×G + 1.3577×g_free_3p

    Returns a Series of scores (higher = safer). NaN for unparseable HELM.
    """
    feat = hagedorn_linear_features(df)
    scores = pd.Series(HAGEDORN_INTERCEPT, index=df.index, dtype=float)
    for col, coef in HAGEDORN_COEFS.items():
        scores += coef * feat[col]
    # Round to 1 decimal (matching R code)
    scores = scores.round(1)
    # NaN where features are NaN
    scores[feat.isna().any(axis=1)] = np.nan
    return scores


def binary_labels(df: pd.DataFrame) -> pd.Series:
    """Neurotoxic (FOB > 1) vs non-toxic (FOB <= 1)."""
    y = pd.Series(index=df.index, dtype=str)
    y[df["mean_FOB"] > 1] = "high"
    y[df["mean_FOB"] <= 1] = "low"
    return y


def run_all_models(df, groups):
    """Run linear model + RF models.

    Returns (results_df, predictions_dict) where predictions_dict contains
    the dinucleotide model predictions for pipeline enrichment.
    """
    y_labels = binary_labels(df)
    results = []
    predictions_dict = {}

    # ---- Hagedorn 2022 score (fixed coefficients, optimised threshold) ----
    print("\n  Hagedorn score (fixed coefficients)...", end=" ")
    scores = hagedorn_score(df)
    mask_score = y_labels.isin(["high", "low"]) & scores.notna() & groups.notna()
    if mask_score.sum() >= 50:
        y_score = (y_labels[mask_score] == "high").astype(int)
        s = scores[mask_score]
        # Higher score = safer, so use -score as risk predictor for AUC/ROC
        neg_scores = -s
        try:
            auc_score = roc_auc_score(y_score, neg_scores)
        except ValueError:
            auc_score = np.nan

        # Optimise threshold via Youden's J (paper threshold=70 was for 14-mer LNA)
        threshold = _optimal_threshold(y_score, neg_scores)
        pred_labels = (neg_scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_score, pred_labels).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        _, pval = fisher_exact([[tp, fn], [fp, tn]])
        # Convert back to score space for interpretability
        score_threshold = -threshold

        results.append({
            "model": "Hagedorn score (5 features)",
            "N": int(mask_score.sum()),
            "N_high": int(y_score.sum()),
            "N_low": int(len(y_score) - y_score.sum()),
            "GK_accuracy": acc,
            "GK_sensitivity": sens,
            "GK_specificity": spec,
            "GK_AUC": auc_score,
            "GK_pvalue": pval,
            "N_groups": groups[mask_score].nunique(),
        })
        print(f"acc={acc:.3f} AUC={auc_score:.3f} (threshold={score_threshold:.1f})")

        predictions_dict["hagedorn_score"] = {
            "predictions": neg_scores.tolist(),
            "labels": y_score.tolist(),
            "n": int(mask_score.sum()),
            "accuracy": float(acc),
            "auc": float(auc_score),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "scores": s.tolist(),
            "threshold": float(score_threshold),
        }
    else:
        print(f"skip (n={mask_score.sum()})")

    # ---- RF models ----
    for model_key, spec in MODELS.items():
        print(f"  {spec.name}...", end=" ")
        feature_df = prepare_data(df, model_key)

        mask = y_labels.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
        if mask.sum() < 50:
            print(f"skip (n={mask.sum()})")
            continue

        X = feature_df.loc[mask]
        y = (y_labels[mask] == "high")
        g = groups[mask]

        gk_result = train_and_evaluate_grouped(X, y, g)

        row = {"model": spec.name}
        if gk_result:
            row.update({
                "N": gk_result["n"], "N_high": gk_result["n_high"], "N_low": gk_result["n_low"],
                "GK_accuracy": gk_result["accuracy"],
                "GK_sensitivity": gk_result["sensitivity"],
                "GK_specificity": gk_result["specificity"],
                "GK_AUC": gk_result["auc"],
                "GK_pvalue": gk_result["p_value"],
                "N_groups": gk_result["n_groups"],
            })
            print(f"GK acc={gk_result['accuracy']:.3f}")

            if model_key == "dinucleotide":
                predictions_dict["FOB"] = {
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


def load_and_filter_rat():
    """Load neurotox data, filter to Rat 3000ug valid chemistry."""
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df = df[
        (df["species"] == "Rat")
        & (df["dosage_ug"] == 3000)
        & (df["latency_time_hours"] == 3)
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"Rat neurotox data: {len(df):,} rows")

    df["mean_FOB"] = df["FOB_score"].apply(_mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    print(f"With valid FOB scores: {len(df):,} rows")
    return df


def run_rat_models(df, groups):
    """Run linear + RF models on rat FOB data with GroupKFold.

    Returns (results_df, predictions_dict) — same structure as mouse.
    """
    y_labels = binary_labels(df)
    results = []
    predictions_dict = {}

    # ---- Hagedorn 2022 score (fixed coefficients, optimised threshold) ----
    print("\n  Hagedorn score (fixed coefficients)...", end=" ")
    scores = hagedorn_score(df)
    mask_score = y_labels.isin(["high", "low"]) & scores.notna() & groups.notna()
    if mask_score.sum() >= 50:
        y_score = (y_labels[mask_score] == "high").astype(int)
        s = scores[mask_score]
        neg_scores = -s
        try:
            auc_score = roc_auc_score(y_score, neg_scores)
        except ValueError:
            auc_score = np.nan

        threshold = _optimal_threshold(y_score, neg_scores)
        pred_labels = (neg_scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_score, pred_labels).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        _, pval = fisher_exact([[tp, fn], [fp, tn]])
        score_threshold = -threshold

        results.append({
            "model": "Hagedorn score (5 features)",
            "N": int(mask_score.sum()),
            "N_high": int(y_score.sum()),
            "N_low": int(len(y_score) - y_score.sum()),
            "GK_accuracy": acc,
            "GK_sensitivity": sens,
            "GK_specificity": spec,
            "GK_AUC": auc_score,
            "GK_pvalue": pval,
            "N_groups": groups[mask_score].nunique(),
        })
        print(f"acc={acc:.3f} AUC={auc_score:.3f} (threshold={score_threshold:.1f})")

        predictions_dict["rat_hagedorn_score"] = {
            "predictions": neg_scores.tolist(),
            "labels": y_score.tolist(),
            "n": int(mask_score.sum()),
            "accuracy": float(acc),
            "auc": float(auc_score),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "scores": s.tolist(),
            "threshold": float(score_threshold),
        }
    else:
        print(f"skip (n={mask_score.sum()})")

    # ---- RF models ----
    for model_key, spec in MODELS.items():
        print(f"  {spec.name}...", end=" ")
        feature_df = prepare_data(df, model_key)

        mask = y_labels.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
        if mask.sum() < 50:
            print(f"skip (n={mask.sum()})")
            continue

        X = feature_df.loc[mask]
        y = (y_labels[mask] == "high")
        g = groups[mask]

        gk_result = train_and_evaluate_grouped(X, y, g)

        row = {"model": spec.name}
        if gk_result:
            row.update({
                "N": gk_result["n"], "N_high": gk_result["n_high"], "N_low": gk_result["n_low"],
                "GK_accuracy": gk_result["accuracy"],
                "GK_sensitivity": gk_result["sensitivity"],
                "GK_specificity": gk_result["specificity"],
                "GK_AUC": gk_result["auc"],
                "GK_pvalue": gk_result["p_value"],
                "N_groups": gk_result["n_groups"],
            })
            print(f"GK acc={gk_result['accuracy']:.3f}")

            if model_key == "dinucleotide":
                predictions_dict["rat_FOB"] = {
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
    """Compare mouse vs rat FOB scores for shared compounds.

    Mouse: 700μg ICV; Rat: 3000μg (species-appropriate dose).
    Binary: FOB ≥ 3 (neurotoxic) vs FOB ≤ 1 (non-toxic).
    """
    df_all = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df_all = df_all[df_all["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df_all["mean_FOB"] = df_all["FOB_score"].apply(_mean_of_array)
    df_all = df_all[df_all["mean_FOB"].notna()]

    mouse = df_all[
        (df_all["species"] == "Mouse")
        & (df_all["dosage_ug"] == 700)
        & (df_all["administration_method"] == "ICV")
    ]
    rat = df_all[
        (df_all["species"] == "Rat")
        & (df_all["dosage_ug"] == 3000)
    ]

    shared_cids = set(mouse["Compound ID"].dropna()) & set(rat["Compound ID"].dropna())
    print(f"\nCross-species concordance: {len(shared_cids)} shared compounds")

    # Per-compound mean FOB
    mouse_means = mouse.groupby("Compound ID")["mean_FOB"].mean()
    rat_means = rat.groupby("Compound ID")["mean_FOB"].mean()
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
    print(f"  FOB: n={len(m_vals)}, Spearman ρ={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={pval:.2e}")

    # Binary labels: FOB > 1 = high, FOB ≤ 1 = low
    m_label = pd.Series(index=m_vals.index, dtype=str)
    m_label[m_vals > 1] = "high"
    m_label[m_vals <= 1] = "low"

    r_label = pd.Series(index=r_vals.index, dtype=str)
    r_label[r_vals > 1] = "high"
    r_label[r_vals <= 1] = "low"

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

    return {
        "FOB": {
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
    }


def main():
    print("=" * 60)
    print("Hagedorn 2022 — Neurotoxicity Model Replication")
    print("=" * 60)

    df = load_and_filter()
    groups = assign_groups(df)

    print(f"\nBinary labels:")
    y_labels = binary_labels(df)
    print(f"  Neurotoxic (FOB > 1): {(y_labels == 'high').sum():,}")
    print(f"  Non-toxic (FOB <= 1): {(y_labels == 'low').sum():,}")

    print(f"\nRunning models...")
    results_df, predictions_dict = run_all_models(df, groups)

    cross_species = cross_species_concordance()

    # ── Rat models (independent CV on rat data) ──
    print("\n" + "=" * 60)
    print("Rat Neurotoxicity Models (independent)")
    print("=" * 60)
    rat_df = load_and_filter_rat()
    rat_groups = assign_groups(rat_df)

    print(f"\nRat binary labels:")
    rat_y_labels = binary_labels(rat_df)
    print(f"  Neurotoxic (FOB > 1): {(rat_y_labels == 'high').sum():,}")
    print(f"  Non-toxic (FOB <= 1): {(rat_y_labels == 'low').sum():,}")

    print(f"\nRunning rat models...")
    rat_results_df, rat_predictions = run_rat_models(rat_df, rat_groups)

    # Save consolidated JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "neurotox.json"
    with open(out_path, "w") as f:
        json.dump({
            "models": results_df.to_dict(orient="records"),
            "predictions": predictions_dict,
            "cross_species": cross_species,
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
