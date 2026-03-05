"""Export paper numbers from processed parquets to typst/data/paper_numbers.json."""

import json
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/oligostack/processed")
OUT_PATH = Path("typst/data/paper_numbers.json")

RESULTS_DIR = Path("data/results")
PIPELINE_RESULTS_PATH = RESULTS_DIR / "pipeline_results.json"

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

    # ── Hagedorn hepatotoxicity metrics ──
    hepatotox_path = RESULTS_DIR / "hepatotox.json"
    if hepatotox_path.exists():
        hep_data = json.loads(hepatotox_path.read_text())
        hep_df = pd.DataFrame(hep_data["models"])
        hep_preds = hep_data["predictions"]
        # OligoAI-tox (dinucleotide RF) × ALT (GroupKFold CV)
        dinuc_alt = hep_df[(hep_df["model"] == "Dinucleotide (128)") & (hep_df["biomarker"] == "ALT")]
        if len(dinuc_alt) > 0:
            row = dinuc_alt.iloc[0]
            numbers["oligoai_tox_hepatotox"] = {
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
                numbers["oligoai_tox_hepatotox"]["confusion"] = hep_preds["ALT"]["confusion"]

        # Cross-species concordance
        if "cross_species" in hep_data:
            cs = hep_data["cross_species"]
            numbers["cross_species_hepatotox"] = {
                bm: {
                    "n_shared": v["n_shared"],
                    "spearman_rho": v["spearman_rho"],
                    "spearman_p": v["spearman_p"],
                    "concordance_rate": v["concordance_rate"],
                    "concordance_n": v["concordance_n"],
                }
                for bm, v in cs.items()
            }

        # Combined (mouse+rat) hepatotoxicity model
        if "combined_predictions" in hep_data:
            cp = hep_data["combined_predictions"]
            if "combined_ALT" in cp:
                p = cp["combined_ALT"]
                numbers["combined_hepatotox"] = {
                    "accuracy": round(float(p["accuracy"]), 3),
                    "sensitivity": round(float(p["sensitivity"]), 3),
                    "specificity": round(float(p["specificity"]), 3),
                    "auc": round(float(p["auc"]), 3),
                    "n": int(p["n"]),
                    "confusion": p["confusion"],
                }

        # Rat hepatotoxicity model (independent CV on rat data)
        if "rat_models" in hep_data:
            rat_df = pd.DataFrame(hep_data["rat_models"])
            rat_dinuc = rat_df[rat_df["model"] == "Dinucleotide (128)"]
            if len(rat_dinuc) > 0:
                row = rat_dinuc.iloc[0]
                numbers["rat_hepatotox"] = {
                    "accuracy": round(float(row["GK_accuracy"]), 3),
                    "sensitivity": round(float(row["GK_sensitivity"]), 3),
                    "specificity": round(float(row["GK_specificity"]), 3),
                    "auc": round(float(row["GK_AUC"]), 3),
                    "n": int(row["N"]),
                    "n_high": int(row["N_high"]),
                    "n_low": int(row["N_low"]),
                    "n_groups": int(row["N_groups"]),
                }
                if "rat_predictions" in hep_data and "rat_ALT" in hep_data["rat_predictions"]:
                    numbers["rat_hepatotox"]["confusion"] = hep_data["rat_predictions"]["rat_ALT"]["confusion"]
    else:
        warnings.warn(f"Hagedorn hepatotox results not found — run `just hagerdorn` first")

    # ── Hagedorn neurotoxicity metrics ──
    neurotox_path = RESULTS_DIR / "neurotox.json"
    if neurotox_path.exists():
        neuro_data = json.loads(neurotox_path.read_text())
        neuro_df = pd.DataFrame(neuro_data["models"])
        neuro_preds = neuro_data["predictions"]
        # OligoAI-tox (dinucleotide RF) × FOB
        dinuc = neuro_df[neuro_df["model"] == "Dinucleotide (128)"]
        if len(dinuc) > 0:
            row = dinuc.iloc[0]
            numbers["oligoai_tox_neurotox"] = {
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
                numbers["oligoai_tox_neurotox"]["confusion"] = neuro_preds["FOB"]["confusion"]

        # Hagedorn et al. 2022 baseline (neurotox only)
        hagedorn_linear = neuro_df[neuro_df["model"] == "Hagedorn score (5 features)"]
        if len(hagedorn_linear) > 0:
            row = hagedorn_linear.iloc[0]
            numbers["hagedorn_linear_neurotox"] = {
                "accuracy": round(float(row["GK_accuracy"]), 3),
                "sensitivity": round(float(row["GK_sensitivity"]), 3),
                "specificity": round(float(row["GK_specificity"]), 3),
                "auc": round(float(row["GK_AUC"]), 3),
                "n": int(row["N"]),
                "n_high": int(row["N_high"]),
                "n_low": int(row["N_low"]),
                "n_groups": int(row["N_groups"]),
            }
            if "hagedorn_score" in neuro_preds:
                numbers["hagedorn_linear_neurotox"]["confusion"] = neuro_preds["hagedorn_score"]["confusion"]

        # Cross-species concordance
        if "cross_species" in neuro_data:
            cs = neuro_data["cross_species"]
            numbers["cross_species_neurotox"] = {
                bm: {
                    "n_shared": v["n_shared"],
                    "spearman_rho": v["spearman_rho"],
                    "spearman_p": v["spearman_p"],
                    "concordance_rate": v["concordance_rate"],
                    "concordance_n": v["concordance_n"],
                }
                for bm, v in cs.items()
            }

        # Rat neurotoxicity model (independent CV on rat data)
        if "rat_models" in neuro_data:
            rat_df = pd.DataFrame(neuro_data["rat_models"])
            rat_dinuc = rat_df[rat_df["model"] == "Dinucleotide (128)"]
            if len(rat_dinuc) > 0:
                row = rat_dinuc.iloc[0]
                numbers["rat_neurotox"] = {
                    "accuracy": round(float(row["GK_accuracy"]), 3),
                    "sensitivity": round(float(row["GK_sensitivity"]), 3),
                    "specificity": round(float(row["GK_specificity"]), 3),
                    "auc": round(float(row["GK_AUC"]), 3),
                    "n": int(row["N"]),
                    "n_high": int(row["N_high"]),
                    "n_low": int(row["N_low"]),
                    "n_groups": int(row["N_groups"]),
                }
                if "rat_predictions" in neuro_data and "rat_FOB" in neuro_data["rat_predictions"]:
                    numbers["rat_neurotox"]["confusion"] = neuro_data["rat_predictions"]["rat_FOB"]["confusion"]

        # Combined (mouse+rat) neurotoxicity model
        if "combined_predictions" in neuro_data:
            cp = neuro_data["combined_predictions"]
            if "combined_FOB" in cp:
                p = cp["combined_FOB"]
                numbers["combined_neurotox"] = {
                    "accuracy": round(float(p["accuracy"]), 3),
                    "sensitivity": round(float(p["sensitivity"]), 3),
                    "specificity": round(float(p["specificity"]), 3),
                    "auc": round(float(p["auc"]), 3),
                    "n": int(p["n"]),
                    "confusion": p["confusion"],
                }

        # Rat Hagedorn et al. 2022 baseline (neurotox only)
        if "rat_models" in neuro_data:
            rat_df_lin = pd.DataFrame(neuro_data["rat_models"])
            rat_hagedorn = rat_df_lin[rat_df_lin["model"] == "Hagedorn score (5 features)"]
            if len(rat_hagedorn) > 0:
                row = rat_hagedorn.iloc[0]
                numbers["hagedorn_linear_rat_neurotox"] = {
                    "accuracy": round(float(row["GK_accuracy"]), 3),
                    "sensitivity": round(float(row["GK_sensitivity"]), 3),
                    "specificity": round(float(row["GK_specificity"]), 3),
                    "auc": round(float(row["GK_AUC"]), 3),
                    "n": int(row["N"]),
                    "n_high": int(row["N_high"]),
                    "n_low": int(row["N_low"]),
                    "n_groups": int(row["N_groups"]),
                }
                if "rat_predictions" in neuro_data and "rat_hagedorn_score" in neuro_data["rat_predictions"]:
                    numbers["hagedorn_linear_rat_neurotox"]["confusion"] = neuro_data["rat_predictions"]["rat_hagedorn_score"]["confusion"]
    else:
        warnings.warn(f"Hagedorn neurotox results not found — run `just hagerdorn` first")

    # ── Pipeline cost analysis ──
    if PIPELINE_RESULTS_PATH.exists():
        pipeline_data = json.loads(PIPELINE_RESULTS_PATH.read_text())
        baseline = pipeline_data["baseline"]
        oligoai_tox_pipeline = pipeline_data.get("oligoai_tox")
        oligoai = pipeline_data.get("oligoai")
        combined = pipeline_data.get("combined")

        pipeline_numbers = {
            "baseline_n_initial": baseline["n_initial"],
            "baseline_total_cost": baseline["total_cost"],
        }
        if oligoai_tox_pipeline:
            pipeline_numbers["oligoai_tox_n_initial"] = oligoai_tox_pipeline["n_initial"]
            pipeline_numbers["oligoai_tox_total_cost"] = oligoai_tox_pipeline["total_cost"]
            pipeline_numbers["oligoai_tox_savings_pct"] = round(
                (1 - oligoai_tox_pipeline["total_cost"] / baseline["total_cost"]) * 100, 1
            )
            # Per-stage enrichment factors
            if "enriched_stages" in oligoai_tox_pipeline:
                stage_labels = {"2": "mouse_ALT", "3": "mouse_FOB", "4": "rat_ALT", "5": "rat_FOB"}
                for stage_idx, info in oligoai_tox_pipeline["enriched_stages"].items():
                    label = stage_labels.get(stage_idx, f"stage_{stage_idx}")
                    pipeline_numbers[f"ef_{label}"] = round(info["enrichment_factor"], 3)
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
