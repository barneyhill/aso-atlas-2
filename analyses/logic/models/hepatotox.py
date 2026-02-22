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

from analyses.utils.helm import Helm
from analyses.utils.models import (
    MODELS,
    calc_uln,
    prepare_data,
    run_model_grouped,
    train_and_evaluate_hepatotox,
)

_root = Path(__file__).resolve().parents[3]
_data_dir = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"

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
    """Run all model x biomarker combinations.

    Returns (results_df, predictions_dict) where predictions_dict contains
    the dinucleotide model × ALT predictions for pipeline enrichment.

    The dinucleotide × ALT combination uses Hagedorn 2013-style training
    (OOB + Levenshtein-stratified 10-fold CV). All other combinations use
    GroupKFold.
    """
    results = []
    predictions_dict = {}

    for model_key, spec in MODELS.items():
        feature_df = prepare_data(df, model_key)

        for bm in BIOMARKERS:
            print(f"  {spec.name} x {bm}...", end=" ")

            high_mult, low_mult = THRESHOLDS[bm]

            # Dinucleotide × ALT: Hagedorn 2013-style training
            if model_key == "dinucleotide" and bm == "ALT":
                col = f"mean_{bm}"
                valid = df[col].dropna()
                uln = calc_uln(valid.values)
                y_full = pd.Series(index=df.index, dtype=str)
                y_full[df[col] > high_mult * uln] = "high"
                y_full[df[col] < low_mult * uln] = "low"

                mask = (
                    y_full.isin(["high", "low"])
                    & feature_df.notna().all(axis=1)
                    & groups.notna()
                )
                if mask.sum() < 50:
                    print(f"skip (n={mask.sum()})")
                    results.append({"model": spec.name, "biomarker": bm})
                    continue

                X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
                y = (y_full[mask] == "high")
                sequences = df.loc[mask, "HELM Annotation"].apply(
                    lambda h: Helm.parse(h).dna_sequence if Helm.parse(h) else ""
                )

                hep_result = train_and_evaluate_hepatotox(X, y, sequences)

                row = {"model": spec.name, "biomarker": bm}
                if hep_result:
                    row.update({
                        "N": hep_result["n"],
                        "N_high": hep_result["n_high"],
                        "N_low": hep_result["n_low"],
                        "OOB_accuracy": hep_result["oob_accuracy"],
                        "CV_accuracy": hep_result["accuracy"],
                        "CV_sensitivity": hep_result["sensitivity"],
                        "CV_specificity": hep_result["specificity"],
                        "CV_AUC": hep_result["auc"],
                        "CV_pvalue": hep_result["p_value"],
                        "stratum_accuracy": hep_result["stratum_accuracy"],
                    })
                    strata_str = ", ".join(
                        f"{k}: {v:.0%}" for k, v in hep_result["stratum_accuracy"].items()
                    )
                    print(
                        f"OOB={hep_result['oob_accuracy']:.3f} "
                        f"CV={hep_result['accuracy']:.3f} "
                        f"[{strata_str}]"
                    )

                    predictions_dict["ALT"] = {
                        "predictions": hep_result["predictions"].tolist(),
                        "labels": y.astype(int).tolist(),
                        "n": int(hep_result["n"]),
                        "oob_accuracy": float(hep_result["oob_accuracy"]),
                        "accuracy": float(hep_result["accuracy"]),
                        "auc": float(hep_result["auc"]),
                        "sensitivity": float(hep_result["sensitivity"]),
                        "specificity": float(hep_result["specificity"]),
                        "confusion": hep_result["confusion"],
                        "stratum_accuracy": hep_result["stratum_accuracy"],
                    }
                else:
                    print("skip")
                results.append(row)
                continue

            # All other combinations: GroupKFold
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
            else:
                print("GK=skip")

            results.append(row)

    return pd.DataFrame(results), predictions_dict


def main():
    print("=" * 60)
    print("Hagedorn 2013 — Hepatotoxicity Model Replication")
    print("=" * 60)

    df = load_and_filter()
    groups = assign_groups(df)
    cov_df = build_covariates(df)

    print(f"\nRunning models...")
    results_df, predictions_dict = run_all_models(df, groups, cov_df)

    # Save consolidated JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "hepatotox.json"
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
