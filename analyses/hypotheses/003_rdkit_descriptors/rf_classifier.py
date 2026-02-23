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
Random Forest Classification for ASO Hepatotoxicity Prediction

Predicts binary ALT > 100 using RDKit molecular descriptors.
Outputs 2x2 confusion matrix and feature importance.

Run with: uv run analyses/hypotheses/003_rdkit_descriptors/rf_classifier.py
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
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PATHS
# =============================================================================

HYPOTHESIS_DIR = Path(__file__).parent
FIGURES_DIR = HYPOTHESIS_DIR / "figures"

# RDKit descriptor columns (from analysis.py)
DESCRIPTOR_COLS = [
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

ALT_THRESHOLD = 100


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data() -> pd.DataFrame:
    """Load pre-computed RDKit descriptor data with ALT values."""
    data_file = HYPOTHESIS_DIR / "rdkit_descriptor_data.csv"
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} samples from {data_file.name}")
    return df


def prepare_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and binary target y.

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary target (1 if ALT > 100, 0 otherwise)
    """
    # Filter to columns that exist in dataframe
    available_cols = [c for c in DESCRIPTOR_COLS if c in df.columns]
    X = df[available_cols].values

    # Create binary target: ALT > 100
    y = (df["ALT_flat"] > ALT_THRESHOLD).astype(int).values

    print(f"Features: {X.shape[1]} descriptors, {X.shape[0]} samples")
    print(f"Target: {np.sum(y == 1)} positive ({100*np.mean(y):.1f}%), "
          f"{np.sum(y == 0)} negative ({100*(1-np.mean(y)):.1f}%)")

    return X, y, available_cols


# =============================================================================
# MODEL
# =============================================================================


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    return rf


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
        "y_test": y_test,
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot 2x2 confusion matrix with counts and percentages."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate percentages
    cm_percent = 100 * cm / cm.sum()

    # Create labels with counts and percentages
    labels = [
        [f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ]

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=["Predicted: ALT<=100", "Predicted: ALT>100"],
        yticklabels=["Actual: ALT<=100", "Actual: ALT>100"],
        ax=ax,
    )

    ax.set_title("Random Forest: ALT > 100 Classification\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 20,
) -> pd.DataFrame:
    """Plot top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color="steelblue",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Features for ALT > 100 Classification")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

    # Return as DataFrame
    return pd.DataFrame({
        "feature": [feature_names[i] for i in indices],
        "importance": importances[indices],
    })


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("RANDOM FOREST CLASSIFICATION: ALT > 100")
    print("=" * 70)

    FIGURES_DIR.mkdir(exist_ok=True)

    # Load data
    print("\n[1] Loading data...")
    df = load_data()

    # Prepare features
    print("\n[2] Preparing features and target...")
    X, y, feature_names = prepare_features_target(df)

    # Train/test split
    print("\n[3] Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale features
    print("\n[4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\n[5] Training Random Forest...")
    model = train_random_forest(X_train_scaled, y_train)

    # Cross-validation
    print("\n[6] Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Evaluate
    print("\n[7] Evaluating on test set...")
    results = evaluate_model(model, X_test_scaled, y_test)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy:  {results['accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1 Score:  {results['f1']:.3f}")

    cm = results["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()

    print("\n2x2 Confusion Matrix:")
    print("                     Predicted")
    print("                  ALT<=100  ALT>100")
    print(f"Actual ALT<=100   [{tn:5d}]   [{fp:5d}]")
    print(f"Actual ALT>100    [{fn:5d}]   [{tp:5d}]")

    print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Save plots
    print("\n[8] Saving plots...")
    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix.png")
    importance_df = plot_feature_importance(
        model, feature_names, FIGURES_DIR / "feature_importance.png"
    )
    importance_df.to_csv(HYPOTHESIS_DIR / "feature_importance.csv", index=False)

    # Full classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        results["y_pred"],
        target_names=["ALT <= 100", "ALT > 100"],
    ))

    print("\nDone.")


if __name__ == "__main__":
    main()
