"""Export paper numbers from processed parquets to typst/data/paper_numbers.json."""

import json
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/oligostack/processed")
OUT_PATH = Path("typst/data/paper_numbers.json")

HAGERDORN_DIR = Path("analyses/04_hagerdorn")
PIPELINE_RESULTS_PATH = Path("data/results/pipeline_results.json")

BIOMARKER_COLS = ["ALB", "ALT", "AST", "BUN", "CREA", "TBIL", "PC_ratio"]


def main() -> None:
    invitro = pd.read_parquet(DATA_DIR / "in_vitro_inhibition_processed.parquet")
    dose_response = pd.read_parquet(DATA_DIR / "dose_response_processed.parquet")
    hepatic = pd.read_parquet(DATA_DIR / "hepatictoxicity_processed.parquet")
    neuro = pd.read_parquet(DATA_DIR / "neurotoxicity_processed.parquet")

    all_dfs = [invitro, dose_response, hepatic, neuro]

    numbers = {
        "in_vitro": {
            "n_measurements": len(invitro),
            "n_asos": int(invitro["Compound ID"].nunique()),
        },
        "dose_response": {
            "n_measurements": len(dose_response),
            "n_asos": int(dose_response["Compound ID"].nunique()),
        },
        "hepatic": {
            "n_records": len(hepatic),
            "n_asos": int(hepatic["Compound ID"].nunique()),
            "n_biomarker_channels": sum(
                1 for col in BIOMARKER_COLS if hepatic[col].notna().any()
            ),
        },
        "neuro": {
            "n_records": len(neuro),
            "n_asos": int(neuro["Compound ID"].nunique()),
            "gene_coverage_pct": int(100 * neuro["target_RNA"].notna().mean()),
        },
        "genes": {
            "n_unique": int(
                pd.concat([df["target_RNA"] for df in all_dfs]).dropna().nunique()
            ),
            "n_total_measurements": int(
                sum(df["target_RNA"].notna().sum() for df in all_dfs)
            ),
        },
    }

    # ── Hagerdorn hepatotoxicity metrics ──
    hepatotox_csv = HAGERDORN_DIR / "hepatotox_results.csv"
    hepatotox_preds = HAGERDORN_DIR / "hepatotox_predictions.json"
    if hepatotox_csv.exists() and hepatotox_preds.exists():
        hep_df = pd.read_csv(hepatotox_csv)
        hep_preds = json.loads(hepatotox_preds.read_text())
        # Dinucleotide model × ALT
        dinuc_alt = hep_df[(hep_df["model"] == "Dinucleotide (288)") & (hep_df["biomarker"] == "ALT")]
        if len(dinuc_alt) > 0:
            row = dinuc_alt.iloc[0]
            numbers["hagerdorn_hepatotox"] = {
                "accuracy": round(float(row["GK_accuracy"]), 3),
                "sensitivity": round(float(row["GK_sensitivity"]), 3),
                "specificity": round(float(row["GK_specificity"]), 3),
                "auc": round(float(row["GK_AUC"]), 3),
                "n": int(row["N"]),
                "n_high": int(row["N_high"]),
                "n_low": int(row["N_low"]),
                "n_groups": int(row["N_groups"]),
            }
            if "ALT" in hep_preds:
                numbers["hagerdorn_hepatotox"]["confusion"] = hep_preds["ALT"]["confusion"]
    else:
        warnings.warn(f"Hagerdorn hepatotox results not found — run `just hagerdorn` first")

    # ── Hagerdorn neurotoxicity metrics ──
    neurotox_csv = HAGERDORN_DIR / "neurotox_results.csv"
    neurotox_preds = HAGERDORN_DIR / "neurotox_predictions.json"
    if neurotox_csv.exists() and neurotox_preds.exists():
        neuro_df = pd.read_csv(neurotox_csv)
        neuro_preds = json.loads(neurotox_preds.read_text())
        # Dinucleotide model
        dinuc = neuro_df[neuro_df["model"] == "Dinucleotide (288)"]
        if len(dinuc) > 0:
            row = dinuc.iloc[0]
            numbers["hagerdorn_neurotox"] = {
                "accuracy": round(float(row["GK_accuracy"]), 3),
                "sensitivity": round(float(row["GK_sensitivity"]), 3),
                "specificity": round(float(row["GK_specificity"]), 3),
                "auc": round(float(row["GK_AUC"]), 3),
                "n": int(row["N"]),
                "n_high": int(row["N_high"]),
                "n_low": int(row["N_low"]),
                "n_groups": int(row["N_groups"]),
            }
            if "FOB" in neuro_preds:
                numbers["hagerdorn_neurotox"]["confusion"] = neuro_preds["FOB"]["confusion"]
    else:
        warnings.warn(f"Hagerdorn neurotox results not found — run `just hagerdorn` first")

    # ── Pipeline cost analysis ──
    if PIPELINE_RESULTS_PATH.exists():
        pipeline_data = json.loads(PIPELINE_RESULTS_PATH.read_text())
        baseline = pipeline_data["baseline"]
        hagerdorn = pipeline_data.get("hagerdorn")
        oligoai = pipeline_data.get("oligoai")
        combined = pipeline_data.get("combined")

        pipeline_numbers = {
            "baseline_n_initial": baseline["n_initial"],
            "baseline_total_cost": baseline["total_cost"],
        }
        if hagerdorn:
            pipeline_numbers["hagerdorn_n_initial"] = hagerdorn["n_initial"]
            pipeline_numbers["hagerdorn_total_cost"] = hagerdorn["total_cost"]
            pipeline_numbers["hagerdorn_savings_pct"] = round(
                (1 - hagerdorn["total_cost"] / baseline["total_cost"]) * 100, 1
            )
        if oligoai:
            pipeline_numbers["oligoai_n_initial"] = oligoai["n_initial"]
            pipeline_numbers["oligoai_total_cost"] = oligoai["total_cost"]
            pipeline_numbers["oligoai_savings_pct"] = round(
                (1 - oligoai["total_cost"] / baseline["total_cost"]) * 100, 1
            )
        if combined:
            pipeline_numbers["combined_n_initial"] = combined["n_initial"]
            pipeline_numbers["combined_total_cost"] = combined["total_cost"]
            pipeline_numbers["combined_savings_pct"] = round(
                (1 - combined["total_cost"] / baseline["total_cost"]) * 100, 1
            )

        numbers["pipeline"] = pipeline_numbers
    else:
        warnings.warn(f"{PIPELINE_RESULTS_PATH} not found — run `just analysis` first")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
