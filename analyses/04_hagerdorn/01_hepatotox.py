"""
Hagedorn 2013 hepatotoxicity model replication.

Random forest classifiers for ALT, AST, TBIL using dinucleotide
and other sequence features. Adapted from LNA to DNA/MOE/cET chemistry
with target-level train/test splits (GroupKFold).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.models import (
    MODELS,
    calc_uln,
    prepare_data,
    run_model_grouped,
    valid_chemistry,
)

_script_dir = Path(__file__).parent
_data_dir = _script_dir / "../../data/oligostack/processed"
_fig_dir = _script_dir / "figures"
_fig_dir.mkdir(exist_ok=True)

BIOMARKERS = ["ALT", "AST", "TBIL"]
THRESHOLDS = {
    "ALT": (3, 1),   # 3× ULN high, 1× ULN low
    "AST": (5, 2),   # Hagedorn 2013 original
    "TBIL": (5, 2),
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
        (df["species"] == "mouse") & df["HELM Annotation"].apply(valid_chemistry)
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
    """Run all model x biomarker combinations.

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

                # Collect dinucleotide × ALT predictions for pipeline enrichment
                if model_key == "dinucleotide" and bm == "ALT":
                    preds = gk_result["predictions"]
                    col = f"mean_{bm}"
                    valid = df[col].dropna()
                    uln = calc_uln(valid.values)
                    y_full = pd.Series(index=df.index, dtype=str)
                    y_full[df[col] > high_mult * uln] = "high"
                    y_full[df[col] < low_mult * uln] = "low"
                    mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()

                    predictions_dict["ALT"] = {
                        "predictions": preds.tolist(),
                        "labels": (y_full[mask] == "high").astype(int).tolist(),
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


def plot_model_comparison(results_df: pd.DataFrame):
    """Bar chart comparing models across biomarkers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, bm in enumerate(BIOMARKERS):
        ax = axes[i]
        subset = results_df[results_df["biomarker"] == bm].dropna(subset=["GK_accuracy"])
        if subset.empty:
            ax.set_title(f"{bm} (no data)")
            continue

        x = np.arange(len(subset))
        w = 0.35
        ax.bar(x - w / 2, subset["GK_accuracy"], w, label="GroupKFold", color="#1f77b4")
        if "OOB_accuracy" in subset.columns:
            ax.bar(x + w / 2, subset["OOB_accuracy"], w, label="OOB", color="#ff7f0e")

        ax.set_xticks(x)
        ax.set_xticklabels(subset["model"], rotation=30, ha="right", fontsize=8)
        ax.set_title(bm, fontsize=13)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Accuracy")
            ax.legend(fontsize=9)

    plt.suptitle("Hepatotoxicity Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(_fig_dir / "hepatotox_model_comparison.svg", bbox_inches="tight")
    fig.savefig(_fig_dir / "hepatotox_model_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved hepatotox_model_comparison.svg/png")
    plt.close(fig)


def plot_feature_importance(df, groups, cov_df):
    """Top-20 dinucleotide feature importance (fitted on full data)."""
    from sklearn.ensemble import RandomForestClassifier

    feature_df = prepare_data(df, "dinucleotide")
    col = "mean_ALT"
    valid = df[col].dropna()
    uln = calc_uln(valid.values)

    y_full = pd.Series(index=df.index, dtype=str)
    high_mult, low_mult = THRESHOLDS["ALT"]
    y_full[df[col] > high_mult * uln] = "high"
    y_full[df[col] < low_mult * uln] = "low"
    mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()

    X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
    y = (y_full[mask] == "high").astype(int)

    rf = RandomForestClassifier(
        n_estimators=1000, max_features=8, random_state=42,
        n_jobs=-1, class_weight="balanced",
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top20 = importances.nlargest(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    top20.sort_values().plot.barh(ax=ax, color="#1f77b4")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 20 Features — Dinucleotide Model (ALT)", fontsize=13)
    plt.tight_layout()
    fig.savefig(_fig_dir / "hepatotox_feature_importance.svg", bbox_inches="tight")
    fig.savefig(_fig_dir / "hepatotox_feature_importance.png", dpi=150, bbox_inches="tight")
    print("Saved hepatotox_feature_importance.svg/png")
    plt.close(fig)


def plot_roc_curves(df, groups, cov_df):
    """ROC curves for all models on ALT."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bm = "ALT"
    high_mult, low_mult = THRESHOLDS[bm]
    col = f"mean_{bm}"
    valid = df[col].dropna()
    uln = calc_uln(valid.values)

    for i, (model_key, spec) in enumerate(MODELS.items()):
        feature_df = prepare_data(df, model_key)
        y_full = pd.Series(index=df.index, dtype=str)
        y_full[df[col] > high_mult * uln] = "high"
        y_full[df[col] < low_mult * uln] = "low"
        mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
        if mask.sum() < 50:
            continue

        X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
        y = (y_full[mask] == "high")
        g = groups[mask]
        result = train_and_evaluate_grouped(X, y, g)
        if result and not np.isnan(result["auc"]):
            fpr, tpr, _ = roc_curve(y.astype(int), result["predictions"])
            ax.plot(fpr, tpr, label=f"{spec.name} (AUC={result['auc']:.3f})",
                    color=colors[i % len(colors)], linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Hepatotoxicity — ROC Curves (GroupKFold, ALT)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(_fig_dir / "hepatotox_roc_curves.svg", bbox_inches="tight")
    fig.savefig(_fig_dir / "hepatotox_roc_curves.png", dpi=150, bbox_inches="tight")
    print("Saved hepatotox_roc_curves.svg/png")
    plt.close(fig)


def plot_confusion_matrices(df, groups, cov_df):
    """Confusion matrices for dinucleotide model across biomarkers."""
    feature_df = prepare_data(df, "dinucleotide")
    fig, axes = plt.subplots(1, len(BIOMARKERS), figsize=(5 * len(BIOMARKERS), 4))

    for i, bm in enumerate(BIOMARKERS):
        ax = axes[i]
        col = f"mean_{bm}"
        valid = df[col].dropna()
        if len(valid) < 100:
            ax.set_title(f"{bm} (insufficient data)")
            continue

        uln = calc_uln(valid.values)
        high_mult, low_mult = THRESHOLDS[bm]
        y_full = pd.Series(index=df.index, dtype=str)
        y_full[df[col] > high_mult * uln] = "high"
        y_full[df[col] < low_mult * uln] = "low"
        mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
        if mask.sum() < 50:
            ax.set_title(f"{bm} (insufficient data)")
            continue

        X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
        y = y_full[mask] == "high"
        g = groups[mask]
        result = train_and_evaluate_grouped(X, y, g)
        if result is None:
            ax.set_title(f"{bm} (model failed)")
            continue

        cm = np.array([[result["confusion"]["tn"], result["confusion"]["fp"]],
                       [result["confusion"]["fn"], result["confusion"]["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > cm.max() / 2 else "black"
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", color=color, fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Low", "High"])
        ax.set_yticklabels(["Low", "High"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{bm} (acc={result['accuracy']:.2f})")

    plt.suptitle("Confusion Matrices — Dinucleotide Model (GroupKFold)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(_fig_dir / "hepatotox_confusion.svg", bbox_inches="tight")
    fig.savefig(_fig_dir / "hepatotox_confusion.png", dpi=150, bbox_inches="tight")
    print("Saved hepatotox_confusion.svg/png")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Hagedorn 2013 — Hepatotoxicity Model Replication")
    print("=" * 60)

    df = load_and_filter()
    groups = assign_groups(df)
    cov_df = build_covariates(df)

    print(f"\nRunning models...")
    results_df, predictions_dict = run_all_models(df, groups, cov_df)

    # Save results
    out_csv = _script_dir / "hepatotox_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # Save predictions JSON for pipeline enrichment
    pred_path = _script_dir / "hepatotox_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions_dict, f, indent=2)
    print(f"Saved {pred_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(results_df.to_string(index=False, float_format="%.3f"))
    print("=" * 90)

    print("\nDone.")


if __name__ == "__main__":
    main()
