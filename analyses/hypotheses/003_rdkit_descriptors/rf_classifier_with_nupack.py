#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "scikit-learn>=1.3",
#     "matplotlib>=3.7",
#     "seaborn>=0.12",
# ]
# ///
"""
Random Forest Classification: Before/After NUPACK Features Comparison

Compares model performance with RDKit descriptors alone vs RDKit + NUPACK
thermodynamic metrics for ASO hepatotoxicity prediction (ALT > 100).

Run with: uv run analyses/hypotheses/003_rdkit_descriptors/rf_classifier_with_nupack.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PATHS
# =============================================================================

HYPOTHESIS_DIR = Path(__file__).parent
NUPACK_DIR = HYPOTHESIS_DIR.parent / "001b_wing_homodimerization_nupack"
FIGURES_DIR = HYPOTHESIS_DIR / "figures"

# RDKit descriptor columns
RDKIT_COLS = [
    "MolLogP", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "FractionCSP3", "NumAromaticRings", "MolMR",
    "LabuteASA", "NumHeteroatoms", "BertzCT", "MaxPartialCharge",
    "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "MaxAbsEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "MaxEStateIndex",
    "NumSaturatedRings", "NumSaturatedHeterocycles", "NumAliphaticRings",
    "NumAliphaticHeterocycles", "RingCount", "Chi0", "Chi1", "Chi0n", "Chi1n",
    "Chi0v", "Chi1v", "Kappa1", "Kappa2", "Kappa3", "HallKierAlpha",
    "NHOHCount", "NOCount", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA6",
    "PEOE_VSA7", "PEOE_VSA8", "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3",
    "SlogP_VSA5", "SlogP_VSA6", "SMR_VSA1", "SMR_VSA3", "SMR_VSA5", "SMR_VSA7",
    "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
    "EState_VSA6"
]

# NUPACK thermodynamic metrics
NUPACK_COLS = [
    "homodimer_dG", "monomer_dG", "ddG_dimerization",
    "wing5_homodimer_dG", "wing3_homodimer_dG", "wing5_wing3_dG"
]

ALT_THRESHOLD = 100


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RDKit and NUPACK data."""
    # RDKit data
    rdkit_file = HYPOTHESIS_DIR / "rdkit_descriptor_data.csv"
    df_rdkit = pd.read_csv(rdkit_file)
    print(f"Loaded {len(df_rdkit)} samples from RDKit data")

    # NUPACK data
    nupack_file = NUPACK_DIR / "nupack_metrics.csv"
    df_nupack = pd.read_csv(nupack_file)
    print(f"Loaded {len(df_nupack)} samples from NUPACK data")

    return df_rdkit, df_nupack


def merge_data(df_rdkit: pd.DataFrame, df_nupack: pd.DataFrame) -> pd.DataFrame:
    """Merge RDKit and NUPACK data on Compound ID / HELM."""
    # The NUPACK data has Compound ID, RDKit has HELM
    # We need to link them through the original parquet

    # Load parquet to get HELM -> Compound ID mapping
    parquet_file = HYPOTHESIS_DIR.parent.parent.parent / "data/oligostack/processed/hepatictoxicity_processed.parquet"
    df_link = pd.read_parquet(parquet_file, columns=["HELM Annotation", "Compound ID"])
    df_link = df_link.drop_duplicates()

    # Add Compound ID to RDKit data
    df_rdkit_with_id = df_rdkit.merge(
        df_link.rename(columns={"HELM Annotation": "HELM"}),
        on="HELM",
        how="left"
    )

    # Merge with NUPACK on Compound ID
    merged = df_rdkit_with_id.merge(
        df_nupack[["Compound ID"] + NUPACK_COLS],
        on="Compound ID",
        how="inner"
    )

    print(f"Merged dataset: {len(merged)} samples with both RDKit and NUPACK features")
    return merged


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> dict:
    """Train RF model and evaluate."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

    # Test evaluation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name": model_name,
        "n_features": len(feature_names),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def print_results(results: dict) -> None:
    """Print evaluation results."""
    print(f"\n{'─'*60}")
    print(f"Model: {results['model_name']}")
    print(f"{'─'*60}")
    print(f"Features: {results['n_features']}")
    print(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} +/- {results['cv_accuracy_std']:.3f}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {results['test_accuracy']:.3f}")
    print(f"  Precision: {results['test_precision']:.3f}")
    print(f"  Recall:    {results['test_recall']:.3f}")
    print(f"  F1 Score:  {results['test_f1']:.3f}")
    print(f"  AUC-ROC:   {results['test_auc']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={results['tn']}, FP={results['fp']}")
    print(f"  FN={results['fn']}, TP={results['tp']}")


def plot_comparison(results_before: dict, results_after: dict, output_path: Path) -> None:
    """Plot before/after comparison."""
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    before_vals = [results_before[m] for m in metrics]
    after_vals = [results_after[m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, before_vals, width, label='RDKit Only', color='steelblue')
    bars2 = ax.bar(x + width/2, after_vals, width, label='RDKit + NUPACK', color='coral')

    ax.set_ylabel('Score')
    ax.set_title('Random Forest: Before vs After Adding NUPACK Features')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("RANDOM FOREST: BEFORE/AFTER NUPACK COMPARISON")
    print("=" * 70)

    FIGURES_DIR.mkdir(exist_ok=True)

    # Load and merge data
    print("\n[1] Loading data...")
    df_rdkit, df_nupack = load_data()
    df_merged = merge_data(df_rdkit, df_nupack)

    # Create binary target
    y = (df_merged["ALT_flat"] > ALT_THRESHOLD).astype(int).values
    print(f"\nTarget: {np.sum(y == 1)} positive ({100*np.mean(y):.1f}%), "
          f"{np.sum(y == 0)} negative")

    # Prepare feature sets
    rdkit_cols = [c for c in RDKIT_COLS if c in df_merged.columns]
    nupack_cols = [c for c in NUPACK_COLS if c in df_merged.columns]

    X_rdkit = df_merged[rdkit_cols].values
    X_combined = df_merged[rdkit_cols + nupack_cols].values

    # Handle NaN in NUPACK columns (drop rows with NaN)
    valid_mask = ~np.isnan(X_combined).any(axis=1)
    X_rdkit_clean = X_rdkit[valid_mask]
    X_combined_clean = X_combined[valid_mask]
    y_clean = y[valid_mask]

    print(f"\nAfter removing NaN: {len(y_clean)} samples")
    print(f"RDKit features: {len(rdkit_cols)}")
    print(f"NUPACK features: {len(nupack_cols)}")
    print(f"Combined features: {len(rdkit_cols) + len(nupack_cols)}")

    # Train/test split (same split for both models)
    print("\n[2] Train/test split (80/20, stratified)...")
    X_rdkit_train, X_rdkit_test, y_train, y_test = train_test_split(
        X_rdkit_clean, y_clean,
        test_size=0.2,
        random_state=42,
        stratify=y_clean,
    )

    # Use same indices for combined
    X_comb_train, X_comb_test, _, _ = train_test_split(
        X_combined_clean, y_clean,
        test_size=0.2,
        random_state=42,
        stratify=y_clean,
    )

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # Train and evaluate: RDKit only
    print("\n[3] Training: RDKit Only...")
    results_before = train_and_evaluate(
        X_rdkit_train, X_rdkit_test, y_train, y_test,
        rdkit_cols, "RDKit Only"
    )
    print_results(results_before)

    # Train and evaluate: RDKit + NUPACK
    print("\n[4] Training: RDKit + NUPACK...")
    results_after = train_and_evaluate(
        X_comb_train, X_comb_test, y_train, y_test,
        rdkit_cols + nupack_cols, "RDKit + NUPACK"
    )
    print_results(results_after)

    # Summary comparison
    print("\n" + "=" * 70)
    print("BEFORE/AFTER COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<15} {'RDKit Only':>12} {'+ NUPACK':>12} {'Change':>12}")
    print("-" * 55)

    for metric, label in [
        ("test_accuracy", "Accuracy"),
        ("test_precision", "Precision"),
        ("test_recall", "Recall"),
        ("test_f1", "F1 Score"),
        ("test_auc", "AUC-ROC"),
    ]:
        before = results_before[metric]
        after = results_after[metric]
        change = after - before
        sign = "+" if change >= 0 else ""
        print(f"{label:<15} {before:>12.3f} {after:>12.3f} {sign}{change:>11.3f}")

    # Plot comparison
    print("\n[5] Saving comparison plot...")
    plot_comparison(results_before, results_after, FIGURES_DIR / "nupack_comparison.png")

    # Feature importance for combined model
    print("\n[6] Top 10 features (Combined model):")
    importances = results_after["model"].feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    feature_names = results_after["feature_names"]

    for i, idx in enumerate(indices, 1):
        name = feature_names[idx]
        is_nupack = name in NUPACK_COLS
        marker = " *NUPACK*" if is_nupack else ""
        print(f"  {i:2d}. {name:<25} {importances[idx]:.4f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
