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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import fisher_exact, spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import (
    MODELS,
    _optimal_threshold,
    prepare_data,
    train_and_evaluate_grouped,
)
from analyses.logic.models.oligoai_tox import train_and_evaluate as cnn_train_and_evaluate

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

        # G-free stretch from 3' end
        g_free_3p = 0
        for b in reversed(bases):
            if b != "G":
                g_free_3p += 1
            else:
                break

        rows.append({
            "g_count": bases.count("G"),
            "a_count": bases.count("A"),
            "t_count": bases.count("T"),
            "c_count": bases.count("C"),
            "g_free_3p": g_free_3p,
        })

    return pd.DataFrame(rows, index=df.index)


def binary_labels(df: pd.DataFrame) -> pd.Series:
    """Neurotoxic (FOB >= 3) vs non-toxic (FOB <= 1), exclude intermediate."""
    y = pd.Series(index=df.index, dtype=str)
    y[df["mean_FOB"] >= 3] = "high"
    y[df["mean_FOB"] <= 1] = "low"
    return y


def run_linear_model_grouped(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5
) -> dict | None:
    """Logistic regression with GroupKFold."""
    y_binary = y.astype(int)
    n_high, n_low = y_binary.sum(), len(y_binary) - y_binary.sum()
    if n_high < 10 or n_low < 10:
        return None

    n_groups = groups.nunique()
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        return None

    gkf = GroupKFold(n_splits=actual_splits)
    all_preds = pd.Series(index=y_binary.index, dtype=float)

    for train_idx, test_idx in gkf.split(X, y_binary, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_binary.iloc[train_idx]

        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        all_preds.iloc[test_idx] = proba

    # Optimal threshold via Youden's J on pooled OOF predictions
    try:
        threshold = _optimal_threshold(y_binary, all_preds)
    except ValueError:
        threshold = 0.5
    all_pred_labels = (all_preds > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_binary, all_pred_labels).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    _, pval = fisher_exact([[tp, fn], [fp, tn]])

    try:
        auc = roc_auc_score(y_binary, all_preds)
    except ValueError:
        auc = np.nan

    return {
        "n": len(y_binary), "n_high": n_high, "n_low": n_low,
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "threshold": threshold, "p_value": pval, "auc": auc, "n_groups": n_groups,
        "predictions": all_preds,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def run_all_models(df, groups):
    """Run linear model + RF models.

    Returns (results_df, predictions_dict) where predictions_dict contains
    the dinucleotide model predictions for pipeline enrichment.
    """
    y_labels = binary_labels(df)
    results = []
    predictions_dict = {}

    # ---- Hagedorn 2022 linear model ----
    print("\n  Linear model (5 features)...", end=" ")
    linear_feat = hagedorn_linear_features(df)
    mask = y_labels.isin(["high", "low"]) & linear_feat.notna().all(axis=1) & groups.notna()
    if mask.sum() >= 50:
        X = linear_feat.loc[mask]
        y = (y_labels[mask] == "high")
        g = groups[mask]
        lr_result = run_linear_model_grouped(X, y, g)
        if lr_result:
            results.append({
                "model": "Linear (5 features)",
                "N": lr_result["n"], "N_high": lr_result["n_high"], "N_low": lr_result["n_low"],
                "GK_accuracy": lr_result["accuracy"],
                "GK_sensitivity": lr_result["sensitivity"],
                "GK_specificity": lr_result["specificity"],
                "GK_AUC": lr_result["auc"],
                "GK_pvalue": lr_result["p_value"],
                "N_groups": lr_result["n_groups"],
            })
            print(f"GK acc={lr_result['accuracy']:.3f} AUC={lr_result['auc']:.3f}")

            # Save linear model predictions
            predictions_dict["linear"] = {
                "predictions": lr_result["predictions"].tolist(),
                "labels": y.astype(int).tolist(),
                "n": int(lr_result["n"]),
                "accuracy": float(lr_result["accuracy"]),
                "auc": float(lr_result["auc"]),
                "confusion": {k: int(v) for k, v in lr_result["confusion"].items()},
            }
        else:
            print("skip")
    else:
        print(f"skip (n={mask.sum()})")

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

    # ---- Linear model ----
    print("\n  Linear model (5 features)...", end=" ")
    linear_feat = hagedorn_linear_features(df)
    mask = y_labels.isin(["high", "low"]) & linear_feat.notna().all(axis=1) & groups.notna()
    if mask.sum() >= 50:
        X = linear_feat.loc[mask]
        y = (y_labels[mask] == "high")
        g = groups[mask]
        lr_result = run_linear_model_grouped(X, y, g)
        if lr_result:
            results.append({
                "model": "Linear (5 features)",
                "N": lr_result["n"], "N_high": lr_result["n_high"], "N_low": lr_result["n_low"],
                "GK_accuracy": lr_result["accuracy"],
                "GK_sensitivity": lr_result["sensitivity"],
                "GK_specificity": lr_result["specificity"],
                "GK_AUC": lr_result["auc"],
                "GK_pvalue": lr_result["p_value"],
                "N_groups": lr_result["n_groups"],
            })
            print(f"GK acc={lr_result['accuracy']:.3f} AUC={lr_result['auc']:.3f}")

            predictions_dict["rat_linear"] = {
                "predictions": lr_result["predictions"].tolist(),
                "labels": y.astype(int).tolist(),
                "n": int(lr_result["n"]),
                "accuracy": float(lr_result["accuracy"]),
                "auc": float(lr_result["auc"]),
                "confusion": {k: int(v) for k, v in lr_result["confusion"].items()},
            }
        else:
            print("skip")
    else:
        print(f"skip (n={mask.sum()})")

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


def run_cnn_models(df, groups, species_prefix=""):
    """Run OligoAI-tox CNN model on FOB.

    Neurotox data is pre-filtered to fixed dose/route, so no dosing
    covariates — we use n_cov=0 (empty covariate matrix).
    """
    y_labels = binary_labels(df)
    mask = y_labels.isin(["high", "low"]) & groups.notna()
    if mask.sum() < 50:
        print("  CNN: skip (insufficient data)")
        return {}

    helms = df.loc[mask, "HELM Annotation"].values
    y = (y_labels[mask] == "high").astype(int).values
    g = groups[mask].values
    # No dosing covariates for neurotox (fixed dose/route)
    cov = np.zeros((mask.sum(), 0), dtype=np.float32)

    key = f"{species_prefix}FOB_cnn"
    print(f"  OligoAI-tox ({key})...", end=" ")
    result = cnn_train_and_evaluate(helms, y, g, cov)
    if result is None:
        print("skip")
        return {}

    print(f"acc={result['accuracy']:.3f} AUC={result['auc']:.3f}")
    return {key: {
        "predictions": result["predictions"].tolist(),
        "labels": result["labels"].astype(int).tolist(),
        "n": result["n"],
        "accuracy": result["accuracy"],
        "auc": result["auc"],
        "sensitivity": result["sensitivity"],
        "specificity": result["specificity"],
        "confusion": result["confusion"],
        "filters_k2": result["filters_k2"].tolist(),
        "filters_k3": result["filters_k3"].tolist(),
        "hidden_weights": result["hidden_weights"].tolist(),
        "output_weights": result["output_weights"].tolist(),
    }}


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

    # Binary labels: FOB ≥ 3 = high, FOB ≤ 1 = low
    m_label = pd.Series(index=m_vals.index, dtype=str)
    m_label[m_vals >= 3] = "high"
    m_label[m_vals <= 1] = "low"

    r_label = pd.Series(index=r_vals.index, dtype=str)
    r_label[r_vals >= 3] = "high"
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
    print(f"  Neurotoxic (FOB >= 3): {(y_labels == 'high').sum():,}")
    print(f"  Non-toxic (FOB <= 1): {(y_labels == 'low').sum():,}")
    print(f"  Excluded (intermediate): {(~y_labels.isin(['high', 'low'])).sum():,}")

    print(f"\nRunning models...")
    results_df, predictions_dict = run_all_models(df, groups)

    # ── OligoAI-tox CNN models (mouse) ──
    print("\n  OligoAI-tox CNN models (mouse)...")
    cnn_predictions = run_cnn_models(df, groups)

    cross_species = cross_species_concordance()

    # ── Rat models (independent CV on rat data) ──
    print("\n" + "=" * 60)
    print("Rat Neurotoxicity Models (independent)")
    print("=" * 60)
    rat_df = load_and_filter_rat()
    rat_groups = assign_groups(rat_df)

    print(f"\nRat binary labels:")
    rat_y_labels = binary_labels(rat_df)
    print(f"  Neurotoxic (FOB >= 3): {(rat_y_labels == 'high').sum():,}")
    print(f"  Non-toxic (FOB <= 1): {(rat_y_labels == 'low').sum():,}")
    print(f"  Excluded (intermediate): {(~rat_y_labels.isin(['high', 'low'])).sum():,}")

    print(f"\nRunning rat models...")
    rat_results_df, rat_predictions = run_rat_models(rat_df, rat_groups)

    # ── OligoAI-tox CNN models (rat) ──
    print("\n  OligoAI-tox CNN models (rat)...")
    rat_cnn_preds = run_cnn_models(rat_df, rat_groups, species_prefix="rat_")
    cnn_predictions.update(rat_cnn_preds)

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
            "cnn_predictions": cnn_predictions,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False, float_format="%.3f"))
    print("=" * 90)

    print("\nDone.")


if __name__ == "__main__":
    main()
