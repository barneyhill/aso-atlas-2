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
from scipy.stats import fisher_exact

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

    # Save consolidated JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "neurotox.json"
    with open(out_path, "w") as f:
        json.dump({
            "models": results_df.to_dict(orient="records"),
            "predictions": predictions_dict,
        }, f, indent=2)
    print(f"\nSaved {out_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False, float_format="%.3f"))
    print("=" * 90)

    print("\nDone.")


if __name__ == "__main__":
    main()
