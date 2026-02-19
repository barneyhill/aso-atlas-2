"""Export paper numbers from processed parquets to typst/data/paper_numbers.json."""

import json
import warnings
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/oligostack/processed")
OUT_PATH = Path("typst/data/paper_numbers.json")

ENRICHMENT_PATH = Path("analyses/05_oligoai2/enrichment_metrics.json")
SWEEP_SUMMARY_PATH = Path("analyses/05_oligoai2/sweep_results/summary.json")
ABLATION_SUMMARY_PATH = Path("analyses/05_oligoai2/ablation_results/summary.json")

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

    # ── Model metrics (optional — only present after `just evaluate`) ──
    model: dict = {}
    if SWEEP_SUMMARY_PATH.exists():
        sweep = json.loads(SWEEP_SUMMARY_PATH.read_text())
        best = sweep[0]  # rank 1
        model["n_params"] = best["n_params"]
        model["median_spearman"] = round(best["overall_median_spearman"], 3)
        model["iv_spearman"] = round(best["iv_spearman"], 3)
        model["alt_spearman"] = round(best["alt_spearman"], 3)
        model["ast_spearman"] = round(best["ast_spearman"], 3)
        model["fob_spearman"] = round(best["fob_spearman"], 3)
    else:
        warnings.warn(f"{SWEEP_SUMMARY_PATH} not found — run `just evaluate` first")

    if ENRICHMENT_PATH.exists():
        enrich = json.loads(ENRICHMENT_PATH.read_text())
        model["top_k_fraction"] = enrich["top_k_fraction"]
        for stage in ("inhibition", "ALT", "FOB"):
            if stage in enrich:
                e = enrich[stage]
                model[f"{stage}_base_rate"] = e["base_rate"]
                model[f"{stage}_top_k_pass_rate"] = e["top_k_pass_rate"]
                model[f"{stage}_enrichment_factor"] = e["enrichment_factor"]
                model[f"{stage}_n"] = e["n"]
    else:
        warnings.warn(f"{ENRICHMENT_PATH} not found — run `just evaluate` first")

    if model:
        numbers["model"] = model

    # ── Ablation metrics (optional — only present after `just ablation`) ──
    if ABLATION_SUMMARY_PATH.exists():
        ablation_raw = json.loads(ABLATION_SUMMARY_PATH.read_text())
        ablation = {}
        for condition in ("full", "no_warmup", "vivo_only"):
            if condition in ablation_raw:
                m = ablation_raw[condition]
                ablation[condition] = {
                    "alt_spearman": round(m["alt_spearman"], 3),
                    "ast_spearman": round(m["ast_spearman"], 3),
                    "fob_spearman": round(m["fob_spearman"], 3),
                    "median_vivo_spearman": round(m["median_vivo_spearman"], 3),
                }
        if ablation:
            numbers["ablation"] = ablation
    else:
        warnings.warn(f"{ABLATION_SUMMARY_PATH} not found — run `just ablation` first")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
