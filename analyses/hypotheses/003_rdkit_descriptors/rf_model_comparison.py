#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "scikit-learn>=1.3",
#     "pyarrow>=14.0",
# ]
# ///
"""
Random Forest Model Comparison: NUPACK vs RDKit for ASO Hepatotoxicity

Compares model performance using proper compound-level train/test splitting
to prevent data leakage. Outputs practical screening reduction estimates.

Run with: uv run analyses/hypotheses/003_rdkit_descriptors/rf_model_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
NUPACK_DIR = SCRIPT_DIR.parent / "001b_wing_homodimerization_nupack"
PARQUET = PROJECT_ROOT / "data/oligostack/processed/hepatictoxicity_processed.parquet"

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

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

# Core NUPACK metrics (available for all ASOs)
NUPACK_CORE_COLS = ["homodimer_dG", "monomer_dG", "ddG_dimerization"]

# Wing-specific NUPACK metrics (only gapmers with wings)
NUPACK_WING_COLS = ["wing5_homodimer_dG", "wing3_homodimer_dG", "wing5_wing3_dG"]

# All NUPACK columns
NUPACK_COLS = NUPACK_CORE_COLS + NUPACK_WING_COLS

EXP_COLS = ["num_doses", "dosing_period_days", "dosage_mg_per_kg"]

ALT_THRESHOLD = 100


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load and merge RDKit, NUPACK, and experimental data."""
    # RDKit data (has ALT_flat already)
    df_rdkit = pd.read_csv(SCRIPT_DIR / "rdkit_descriptor_data.csv")

    # Parquet for experimental vars and Compound ID
    df_parquet = pd.read_parquet(PARQUET)
    df_parquet = df_parquet[
        (df_parquet["species"] == "mouse") &
        (df_parquet["adminstration_method"] == "subcutaneous")
    ].copy()

    # Get experimental vars + Compound ID
    df_exp = df_parquet[["HELM Annotation", "Compound ID"] + EXP_COLS].copy()
    df_exp = df_exp.rename(columns={"HELM Annotation": "HELM"})

    # Merge RDKit with exp vars
    df_merged = df_rdkit.merge(df_exp, on="HELM", how="inner")

    # NUPACK data
    df_nupack = pd.read_csv(NUPACK_DIR / "nupack_metrics.csv")
    df_merged = df_merged.merge(
        df_nupack[["Compound ID"] + NUPACK_COLS],
        on="Compound ID",
        how="inner"
    )

    return df_merged


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_cols: list[str],
    name: str,
) -> dict:
    """Train and evaluate RF model with grouped CV."""
    X_train = df.loc[train_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values
    y_train = df.loc[train_mask, "toxic"].values
    y_test = df.loc[test_mask, "toxic"].values
    groups_train = df.loc[train_mask, "Compound ID"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Group K-Fold CV (no compound leakage)
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in gkf.split(X_train_s, y_train, groups_train):
        model.fit(X_train_s[train_idx], y_train[train_idx])
        cv_scores.append(model.score(X_train_s[val_idx], y_train[val_idx]))

    # Final model
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    return {
        "name": name,
        "n_features": len(feature_cols),
        "cv_acc": np.mean(cv_scores),
        "cv_std": np.std(cv_scores),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }


def compute_screening_reduction(
    precision: float,
    recall: float,
    toxic_rate: float,
    n_candidates: int = 1000,
) -> dict:
    """Compute practical screening reduction metrics."""
    n_toxic = int(n_candidates * toxic_rate)
    n_safe = n_candidates - n_toxic

    # True positives: toxic ASOs correctly identified
    tp = int(n_toxic * recall)
    # False negatives: toxic ASOs missed
    fn = n_toxic - tp
    # False positives: safe ASOs incorrectly flagged
    fp = int(tp * (1 - precision) / precision) if precision > 0 else 0
    # True negatives: safe ASOs correctly kept
    tn = n_safe - fp

    # ASOs proceeding to testing
    proceed = tn + fn
    tests_saved = n_candidates - proceed
    reduction_pct = 100 * tests_saved / n_candidates

    # Toxicity rate in filtered pipeline
    pipeline_tox_rate = 100 * fn / proceed if proceed > 0 else 0

    return {
        "toxic_caught": tp,
        "toxic_missed": fn,
        "safe_rejected": fp,
        "safe_kept": tn,
        "tests_saved": tests_saved,
        "reduction_pct": reduction_pct,
        "pipeline_tox_rate": pipeline_tox_rate,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("RF MODEL COMPARISON: NUPACK vs RDKit (Compound-Level Split)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = load_data()
    print(f"    Rows: {len(df)}, Compounds: {df['Compound ID'].nunique()}")

    # Get available feature columns
    rdkit_cols = [c for c in RDKIT_COLS if c in df.columns]
    nupack_core_cols = [c for c in NUPACK_CORE_COLS if c in df.columns]
    exp_cols = [c for c in EXP_COLS if c in df.columns]

    # Create target
    df["toxic"] = (df["ALT_flat"] > ALT_THRESHOLD).astype(int)

    # Remove NaN - only require core features (not wing metrics which are sparse)
    required_cols = rdkit_cols + nupack_core_cols + exp_cols
    df_clean = df.dropna(subset=required_cols + ["toxic"])
    toxic_rate = df_clean["toxic"].mean()

    print(f"    After NaN removal: {len(df_clean)} rows, "
          f"{df_clean['Compound ID'].nunique()} compounds")
    print(f"    Toxic (ALT>{ALT_THRESHOLD}): {df_clean['toxic'].sum()} "
          f"({100*toxic_rate:.1f}%)")

    # Compound-level split
    print("\n[2] Compound-level train/test split...")
    compounds = df_clean["Compound ID"].unique()
    np.random.seed(42)
    np.random.shuffle(compounds)
    split_idx = int(len(compounds) * 0.8)
    train_compounds = set(compounds[:split_idx])
    test_compounds = set(compounds[split_idx:])

    train_mask = df_clean["Compound ID"].isin(train_compounds)
    test_mask = df_clean["Compound ID"].isin(test_compounds)

    print(f"    Train: {train_mask.sum()} rows from {len(train_compounds)} compounds")
    print(f"    Test: {test_mask.sum()} rows from {len(test_compounds)} compounds")

    # Evaluate models (using core NUPACK metrics only - available for all ASOs)
    print("\n[3] Training models...")
    models = [
        (nupack_core_cols, "NUPACK only"),
        (nupack_core_cols + exp_cols, "NUPACK + Exp"),
        (rdkit_cols + exp_cols, "RDKit + Exp"),
        (rdkit_cols + nupack_core_cols + exp_cols, "All Features"),
    ]

    results = []
    for cols, name in models:
        r = evaluate_model(df_clean, train_mask, test_mask, cols, name)
        results.append(r)
        print(f"    {name}: AUC={r['auc']:.3f}")

    # Results table
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE (Compound-level split, no leakage)")
    print("=" * 70)
    print(f"{'Model':<20} {'Features':>8} {'CV Acc':>10} {'Test Acc':>10} "
          f"{'F1':>8} {'AUC':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['n_features']:>8} {r['cv_acc']:>10.3f} "
              f"{r['accuracy']:>10.3f} {r['f1']:>8.3f} {r['auc']:>8.3f}")

    # Screening reduction
    print("\n" + "=" * 70)
    print("PRACTICAL SCREENING REDUCTION (per 1000 candidates)")
    print("=" * 70)
    print(f"Baseline toxicity rate: {100*toxic_rate:.1f}%\n")

    print(f"{'Model':<20} {'Tests Saved':>12} {'Reduction':>10} "
          f"{'Toxic Caught':>14} {'Pipeline Tox':>14}")
    print("-" * 70)

    for r in results:
        sr = compute_screening_reduction(
            r["precision"], r["recall"], toxic_rate
        )
        print(f"{r['name']:<20} {sr['tests_saved']:>12} "
              f"{sr['reduction_pct']:>9.0f}% "
              f"{sr['toxic_caught']:>10}/447 "
              f"{sr['pipeline_tox_rate']:>13.1f}%")

    # Best model summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best = max(results, key=lambda x: x["auc"])
    best_sr = compute_screening_reduction(
        best["precision"], best["recall"], toxic_rate
    )

    print(f"""
Best model: {best['name']}
  - AUC-ROC: {best['auc']:.3f}
  - F1 Score: {best['f1']:.3f}

Practical impact (per 1000 ASO candidates):
  - Tests saved: {best_sr['tests_saved']} ({best_sr['reduction_pct']:.0f}% reduction)
  - Toxic ASOs caught before testing: {best_sr['toxic_caught']}/447 ({100*best['recall']:.0f}%)
  - Pipeline toxicity rate: {best_sr['pipeline_tox_rate']:.1f}% (down from {100*toxic_rate:.1f}%)
""")


if __name__ == "__main__":
    main()
