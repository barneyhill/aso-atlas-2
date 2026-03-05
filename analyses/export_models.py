"""
Train final RF toxicity models on ALL labelled data and export for serving.

Produces:
  data/models/hepatotox_alt.joblib   — combined mouse+rat hepatotox RF
  data/models/neurotox_fob.joblib    — combined mouse+rat neurotox RF
  data/models/metadata.json          — feature names, thresholds, CV metrics
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from analyses.logic.models.hepatotox import (
    LITERATURE_ULN,
    RAT_LITERATURE_ULN,
    THRESHOLDS,
    assign_groups as hepato_assign_groups,
    build_dosing_covariates,
)
from analyses.logic.models.neurotox import (
    assign_groups as neuro_assign_groups,
    binary_labels,
)
from analyses.utils.helm import Helm
from analyses.utils.models import (
    MODELS,
    _optimal_threshold,
    mean_of_array,
    prepare_data,
    train_and_evaluate_grouped,
)

_root = Path(__file__).resolve().parents[1]
_data_dir = _root / "data/oligostack/processed"
MODEL_DIR = _root / "data/models"


def _train_hepatotox():
    """Train combined mouse+rat hepatotox ALT model (131 features)."""
    print("=" * 60)
    print("Hepatotox ALT — Combined Model")
    print("=" * 60)

    df_all = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df_all = df_all[
        df_all["species"].isin(["mouse", "rat"])
        & df_all["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"Data: {len(df_all):,} rows")

    for bm in ["ALT", "AST", "TBIL"]:
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

    df_all = df_all.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)
    rat_indicator = (df_all["species"] == "rat").astype(int)
    groups = hepato_assign_groups(df_all)

    model_key = "dinucleotide"
    spec = MODELS[model_key]
    feature_df = prepare_data(df_all, model_key)
    cov_df = build_dosing_covariates(df_all)

    mask = y_full.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
    X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
    X["species_rat"] = rat_indicator[mask].values
    y = (y_full[mask] == "high")
    g = groups[mask]

    print(f"Training samples: {len(y)} ({y.sum()} high, {(~y).sum()} low)")

    # CV metrics via existing grouped evaluation
    cv_result = train_and_evaluate_grouped(X, y, g, make_classifier=spec.make_classifier)
    if cv_result is None:
        raise RuntimeError("CV failed — insufficient data")

    threshold = cv_result["threshold"]
    print(f"CV AUC={cv_result['auc']:.3f}, threshold={threshold:.4f}")

    # Train final model on ALL data
    n_features = X.shape[1]
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features=min(8, n_features),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y.astype(int))

    # Default covariate values (training medians)
    default_num_doses = float(df_all.loc[mask, "num_doses"].median())
    default_dosing_period_days = float(df_all.loc[mask, "dosing_period_days"].median())

    meta = {
        "feature_names": list(X.columns),
        "threshold": threshold,
        "cv_auc": cv_result["auc"],
        "cv_accuracy": cv_result["accuracy"],
        "cv_sensitivity": cv_result["sensitivity"],
        "cv_specificity": cv_result["specificity"],
        "n_train": len(y),
        "n_high": int(y.sum()),
        "n_low": int((~y).sum()),
        "default_num_doses": default_num_doses,
        "default_dosing_period_days": default_dosing_period_days,
        "mouse_uln": mouse_uln,
        "rat_uln": rat_uln,
        "high_mult": high_mult,
        "low_mult": low_mult,
    }

    return rf, meta


def _train_neurotox():
    """Train combined mouse+rat neurotox FOB model (129 features)."""
    print("\n" + "=" * 60)
    print("Neurotox FOB — Combined Model")
    print("=" * 60)

    df_all = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df_all = df_all[df_all["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df_all["mean_FOB"] = df_all["FOB_score"].apply(mean_of_array)
    df_all = df_all[df_all["mean_FOB"].notna()]

    mouse = df_all[
        (df_all["species"] == "Mouse")
        & (df_all["dosage_ug"] == 700)
        & (df_all["administration_method"] == "ICV")
    ].copy()
    rat = df_all[
        (df_all["species"] == "Rat")
        & (df_all["dosage_ug"] == 3000)
        & (df_all["latency_time_hours"] == 3)
    ].copy()

    df_combined = pd.concat([mouse, rat], ignore_index=True)
    print(f"Data: {len(df_combined):,} rows (Mouse: {len(mouse)}, Rat: {len(rat)})")

    is_rat = (df_combined["species"] == "Rat").astype(int)
    y_labels = binary_labels(df_combined)
    groups = neuro_assign_groups(df_combined)

    model_key = "dinucleotide"
    spec = MODELS[model_key]
    feature_df = prepare_data(df_combined, model_key)

    mask = y_labels.isin(["high", "low"]) & feature_df.notna().all(axis=1) & groups.notna()
    X = feature_df.loc[mask].copy()
    X["species_rat"] = is_rat[mask].values
    y = (y_labels[mask] == "high")
    g = groups[mask]

    print(f"Training samples: {len(y)} ({y.sum()} high, {(~y).sum()} low)")

    # CV metrics
    cv_result = train_and_evaluate_grouped(X, y, g, make_classifier=spec.make_classifier)
    if cv_result is None:
        raise RuntimeError("CV failed — insufficient data")

    threshold = cv_result["threshold"]
    print(f"CV AUC={cv_result['auc']:.3f}, threshold={threshold:.4f}")

    # Train final model on ALL data
    n_features = X.shape[1]
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features=min(8, n_features),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y.astype(int))

    meta = {
        "feature_names": list(X.columns),
        "threshold": threshold,
        "cv_auc": cv_result["auc"],
        "cv_accuracy": cv_result["accuracy"],
        "cv_sensitivity": cv_result["sensitivity"],
        "cv_specificity": cv_result["specificity"],
        "n_train": len(y),
        "n_high": int(y.sum()),
        "n_low": int((~y).sum()),
    }

    return rf, meta


def _roundtrip_check(model_path, rf_orig, X_sample):
    """Verify saved model reproduces identical predictions."""
    rf_loaded = load(model_path)
    orig_proba = rf_orig.predict_proba(X_sample)[:, 1]
    loaded_proba = rf_loaded.predict_proba(X_sample)[:, 1]
    if not np.allclose(orig_proba, loaded_proba):
        raise RuntimeError(f"Roundtrip check FAILED for {model_path}")
    print(f"  Roundtrip check passed ({model_path.name})")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Hepatotox
    rf_hepato, meta_hepato = _train_hepatotox()
    hepato_path = MODEL_DIR / "hepatotox_alt.joblib"
    dump(rf_hepato, hepato_path)
    print(f"Saved {hepato_path}")

    # Neurotox
    rf_neuro, meta_neuro = _train_neurotox()
    neuro_path = MODEL_DIR / "neurotox_fob.joblib"
    dump(rf_neuro, neuro_path)
    print(f"Saved {neuro_path}")

    # Roundtrip checks
    print("\nRoundtrip validation:")
    # Build small test samples matching feature order
    dinuc_feats, dinuc_extract = MODELS["dinucleotide"].features, MODELS["dinucleotide"].extractor
    test_helm = "RNA1{{[moe](A)[sp].[moe](C)[sp].[moe](G)[sp].[moe](T)[sp].[moe](A)[sp].d(C)[sp].d(G)[sp].d(T)[sp].d(A)[sp].d(C)[sp].d(G)[sp].d(T)[sp].d(A)[sp].d(C)[sp].d(G)[sp].[moe](T)[sp].[moe](A)[sp].[moe](C)[sp].[moe](G)[sp].[moe](T)}}$$$$V2.0"
    feats = dinuc_extract(test_helm)
    row = [feats[f] for f in dinuc_feats]

    hepato_sample = pd.DataFrame([row + [6.0, 28.0, 0]], columns=meta_hepato["feature_names"])
    _roundtrip_check(hepato_path, rf_hepato, hepato_sample)

    neuro_sample = pd.DataFrame([row + [0]], columns=meta_neuro["feature_names"])
    _roundtrip_check(neuro_path, rf_neuro, neuro_sample)

    # Save metadata
    metadata = {
        "hepatotox_alt": meta_hepato,
        "neurotox_fob": meta_neuro,
        "sklearn_version": sklearn.__version__,
        "training_date": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = MODEL_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
