"""Export paper numbers from processed parquets to typst/data/paper_numbers.json."""

import json
import math
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr as _spearmanr

from analyses.utils.compounds import (
    compound_ic50s,
    compound_max_inhibition,
    compound_mean_biomarker,
    has_measurement,
)

DATA_DIR = Path("data/oligostack/processed")
OUT_PATH = Path("typst/data/paper_numbers.json")

RESULTS_DIR = Path("data/results")
PIPELINE_RESULTS_PATH = RESULTS_DIR / "pipeline_results.json"

BIOMARKER_COLS = ["ALB", "ALT", "AST", "BUN", "CREA", "TBIL", "PC_ratio"]


def sf(x, n=3):
    if x == 0:
        return 0
    if isinstance(x, int):
        return x
    d = n - 1 - math.floor(math.log10(abs(x)))
    result = round(x, d)
    if d <= 0:
        return int(result)
    return result


def _round_floats(obj, n=3):
    if isinstance(obj, dict):
        return {k: _round_floats(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, n) for v in obj]
    if isinstance(obj, float):
        return sf(obj, n)
    return obj


def _gene_stats(invitro, dose_response, hepatic, neuro, biomarker_cols) -> dict:
    """Gene counts matching plot_fig_atlas.py: resolve target_RNA → gene_symbol."""
    iv_genomic = pd.read_parquet(DATA_DIR / "in_vitro_inhibition_processed_with_genomic_data.parquet")
    dr_genomic = pd.read_parquet(DATA_DIR / "dose_response_with_genomic.parquet")

    gene_map = (
        pd.concat([iv_genomic[["target_RNA", "gene_symbol"]], dr_genomic[["target_RNA", "gene_symbol"]]])
        .dropna(subset=["gene_symbol"])
        .drop_duplicates("target_RNA")
        .set_index("target_RNA")["gene_symbol"]
    )

    def resolve(target_rna):
        if pd.isna(target_rna):
            return None
        return gene_map.get(target_rna, target_rna)

    iv_genes = invitro["target_RNA"].map(resolve)
    dr_genes = dose_response["target_RNA"].map(resolve)
    iv_counts = iv_genes.groupby(iv_genes).size()
    dr_counts = dr_genes.groupby(dr_genes).size()

    hep = hepatic.copy()
    hep["_meas"] = hep[biomarker_cols].apply(
        lambda row: sum(has_measurement(row[c]) for c in biomarker_cols), axis=1
    )
    hep["_gene"] = hep["target_RNA"].map(resolve)
    hep_counts = hep.groupby("_gene")["_meas"].sum()

    neu = neuro.copy()
    neu["_meas"] = neu["FOB_score"].apply(has_measurement)
    neu["_gene"] = neu["target_RNA"].map(resolve)
    neuro_counts = neu.groupby("_gene")["_meas"].sum()

    gene_meas = (
        pd.concat([iv_counts, dr_counts, hep_counts, neuro_counts])
        .groupby(level=0).sum()
        .sort_values(ascending=False)
    )

    total_all = gene_meas.sum()
    cumulative_angle = 0
    cutoff_idx = len(gene_meas)
    for i, count in enumerate(gene_meas.values):
        wedge_angle = 360 * count / total_all
        mid_angle = cumulative_angle + wedge_angle / 2
        if (mid_angle % 360) > 270:
            cutoff_idx = i
            break
        cumulative_angle += wedge_angle
    major = gene_meas.iloc[:cutoff_idx]
    minor = gene_meas.iloc[cutoff_idx:]

    return {
        "n_unique": int(len(gene_meas)),
        "n_total_measurements": int(total_all),
        "n_major": int(len(major)),
        "n_minor": int(len(minor)),
        "major_pct": round(100 * major.sum() / total_all),
    }


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
            "n_measurements": int(
                hepatic[BIOMARKER_COLS].apply(
                    lambda row: sum(has_measurement(row[c]) for c in BIOMARKER_COLS),
                    axis=1,
                ).sum()
            ),
            "n_asos": int(hepatic["Compound ID"].nunique()),
            "n_biomarker_channels": sum(
                1 for col in BIOMARKER_COLS if hepatic[col].notna().any()
            ),
        },
        "neuro": {
            "n_records": len(neuro),
            "n_measurements": int(neuro["FOB_score"].apply(has_measurement).sum()),
            "n_asos": int(neuro["Compound ID"].nunique()),
            "n_mouse": int((neuro["species"] == "Mouse").sum()),
            "n_rat": int((neuro["species"] == "Rat").sum()),
            "gene_coverage_pct": int(100 * neuro["target_RNA"].notna().mean()),
        },
        "genes": _gene_stats(invitro, dose_response, hepatic, neuro, BIOMARKER_COLS),
    }

    # ── Concordance / cross-endpoint statistics ──
    concordance: dict = {}

    mouse_hep_ids = set(hepatic[hepatic["species"] == "mouse"]["Compound ID"].unique())
    rat_hep_ids = set(hepatic[hepatic["species"] == "rat"]["Compound ID"].unique())
    mouse_neuro_ids = set(neuro[neuro["species"] == "Mouse"]["Compound ID"].unique())
    rat_neuro_ids = set(neuro[neuro["species"] == "Rat"]["Compound ID"].unique())

    concordance["rat_hep_overlap_pct"] = (
        round(100 * len(rat_hep_ids & mouse_hep_ids) / len(rat_hep_ids))
        if rat_hep_ids else 0
    )
    concordance["rat_neuro_overlap_pct"] = (
        round(100 * len(rat_neuro_ids & mouse_neuro_ids) / len(rat_neuro_ids))
        if rat_neuro_ids else 0
    )
    all_hep_ids = mouse_hep_ids | rat_hep_ids
    all_neuro_ids = mouse_neuro_ids | rat_neuro_ids
    concordance["hep_neuro_overlap_n"] = int(len(all_hep_ids & all_neuro_ids))

    iv_max = compound_max_inhibition(invitro)
    dr_ic50 = compound_ic50s(dose_response)
    mouse_alt = compound_mean_biomarker(
        hepatic[hepatic["species"] == "mouse"], "ALT",
    )
    mouse_fob = compound_mean_biomarker(
        neuro[(neuro["species"] == "Mouse") & (neuro["dosage_ug"] == 700)],
        "FOB_score",
    )

    cross_assay_pairs = [
        ("inhib_vs_ic50", iv_max, dr_ic50),
        ("inhib_vs_alt", iv_max, mouse_alt),
        ("inhib_vs_bfob", iv_max, mouse_fob),
        ("ic50_vs_alt", dr_ic50, mouse_alt),
        ("ic50_vs_bfob", dr_ic50, mouse_fob),
    ]
    N_CROSS_ASSAY_TESTS = 15  # 6×6 upper triangle
    for tag, s1, s2 in cross_assay_pairs:
        shared = s1.dropna().index.intersection(s2.dropna().index)
        if len(shared) >= 10:
            rho, p = _spearmanr(s1[shared], s2[shared])
            concordance[tag] = {
                "rho": sf(float(rho)),
                "p": sf(float(p)),
                "p_bonf": sf(min(float(p) * N_CROSS_ASSAY_TESTS, 1.0)),
                "n": int(len(shared)),
            }

    if "inhib_vs_alt" in concordance and "inhib_vs_bfob" in concordance:
        min_p = min(concordance["inhib_vs_alt"]["p"], concordance["inhib_vs_bfob"]["p"])
        concordance["cross_endpoint_p_gt"] = round(math.floor(min_p * 10) / 10, 1)
        concordance["cross_endpoint_min_n"] = min(
            concordance["inhib_vs_alt"]["n"], concordance["inhib_vs_bfob"]["n"],
        )

    numbers["concordance"] = concordance

    # ── Hepatotoxicity cross-species concordance ──
    hepatotox_path = RESULTS_DIR / "hepatotox.json"
    if hepatotox_path.exists():
        hep_data = json.loads(hepatotox_path.read_text())

        N_CROSS_SPECIES_TESTS = 4  # ALT, AST, TBIL, FOB
        if "cross_species" in hep_data:
            cs = hep_data["cross_species"]
            numbers["cross_species_hepatotox"] = {
                bm: {
                    "n_shared": v["n_shared"],
                    "spearman_rho": sf(v["spearman_rho"]),
                    "spearman_p": sf(v["spearman_p"]),
                    "spearman_p_bonf": sf(min(v["spearman_p"] * N_CROSS_SPECIES_TESTS, 1.0)),
                    "concordance_rate": sf(v["concordance_rate"]),
                    "concordance_n": v["concordance_n"],
                }
                for bm, v in cs.items()
            }
    else:
        warnings.warn(f"Hepatotox results not found — run `just hagerdorn` first")

    # ── Hagedorn neurotoxicity metrics ──
    neurotox_path = RESULTS_DIR / "neurotox.json"
    if neurotox_path.exists():
        neuro_data = json.loads(neurotox_path.read_text())
        neuro_df = pd.DataFrame(neuro_data["models"])
        neuro_preds = neuro_data["predictions"]
        # Hagedorn et al. 2022 baseline (neurotox only)
        hagedorn_linear = neuro_df[neuro_df["model"] == "Hagedorn score (5 features)"]
        if len(hagedorn_linear) > 0:
            row = hagedorn_linear.iloc[0]
            numbers["hagedorn_linear_neurotox"] = {
                "accuracy": sf(float(row["GK_accuracy"])),
                "sensitivity": sf(float(row["GK_sensitivity"])),
                "specificity": sf(float(row["GK_specificity"])),
                "auc": sf(float(row["GK_AUC"])),
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
                    "spearman_rho": sf(v["spearman_rho"]),
                    "spearman_p": sf(v["spearman_p"]),
                    "spearman_p_bonf": sf(min(v["spearman_p"] * N_CROSS_SPECIES_TESTS, 1.0)),
                    "concordance_rate": sf(v["concordance_rate"]),
                    "concordance_n": v["concordance_n"],
                }
                for bm, v in cs.items()
            }

    else:
        warnings.warn(f"Hagedorn neurotox results not found — run `just hagerdorn` first")

    # ── Pipeline cost analysis ──
    if PIPELINE_RESULTS_PATH.exists():
        pipeline_data = json.loads(PIPELINE_RESULTS_PATH.read_text())
        baseline = pipeline_data["baseline"]

        # Per-stage p_k (pass rate), r_k (cost per ASO), N_k (back-calculated ASOs entering stage)
        stages_list = []
        sample_sizes = pipeline_data.get("sample_sizes", [])
        for i, stage in enumerate(pipeline_data["stages"]):
            entry = {
                "name": stage["name"],
                "p_k": sf(pipeline_data["proportions"][i]),
                "r_k": stage["cost_per_aso"],
                "N_k": baseline["asos_at_stage"][i],
            }
            if i < len(sample_sizes):
                entry["n_pass"] = sample_sizes[i][0]
                entry["n_total"] = sample_sizes[i][1]
            stages_list.append(entry)

        pipeline_numbers = {
            "baseline_n_initial": baseline["n_initial"],
            "baseline_total_cost": baseline["total_cost"],
            "stages": stages_list,
        }

        # OligoAI-tox and combined costs: canonical source is ef_table.json
        ef_table_path = RESULTS_DIR / "ef_table.json"
        if ef_table_path.exists():
            ef_rows = json.loads(ef_table_path.read_text())
            for row in ef_rows:
                if row.get("oligoai_tox"):
                    pipeline_numbers["oligoai_tox_n_initial"] = row["n_initial"]
                    pipeline_numbers["oligoai_tox_total_cost"] = row["total_cost"]
                    pipeline_numbers["oligoai_tox_savings_pct"] = sf(row["delta_pct"])
                    pipeline_numbers["oligoai_tox_savings_std"] = sf(row["delta_std_pct"]) if row.get("delta_std_pct") else None
                    pipeline_numbers["oligoai_tox_source_model"] = row.get("source_model", "")
                if row.get("group") == "combined":
                    pipeline_numbers["combined_n_initial"] = row["n_initial"]
                    pipeline_numbers["combined_total_cost"] = row["total_cost"]
                    pipeline_numbers["combined_savings_pct"] = sf(row["delta_pct"])
                    pipeline_numbers["combined_savings_std"] = sf(row["delta_std_pct"]) if row.get("delta_std_pct") else None
                if row.get("name") == "OligoAI":
                    pipeline_numbers["oligoai_n_initial"] = row["n_initial"]
                    pipeline_numbers["oligoai_total_cost"] = row["total_cost"]
                    pipeline_numbers["oligoai_savings_pct"] = sf(row["delta_pct"])

            # OligoAI-tox EF range across in vivo stages
            for row in ef_rows:
                if row.get("oligoai_tox"):
                    efs = [v for v in row["ef_by_stage"].values() if v is not None]
                    if efs:
                        pipeline_numbers["oligoai_tox_ef_lo"] = sf(min(efs))
                        pipeline_numbers["oligoai_tox_ef_hi"] = sf(max(efs))

        # In vitro cost fraction and per-ASO cost range
        in_vitro_cost = sum(
            s["r_k"] * s["N_k"] for s in stages_list[:2]
        )
        pipeline_numbers["in_vitro_cost_pct"] = sf(
            100 * in_vitro_cost / baseline["total_cost"]
        )
        pipeline_numbers["in_vitro_cost_lo"] = stages_list[0]["r_k"]
        pipeline_numbers["in_vitro_cost_hi"] = stages_list[1]["r_k"]

        # Animal reduction (4 animals per ASO, in vivo stages only: indices 2-6)
        ANIMALS_PER_ASO = 4
        IN_VIVO_STAGES = range(2, 7)
        baseline_animals = sum(
            baseline["asos_at_stage"][i] * ANIMALS_PER_ASO for i in IN_VIVO_STAGES
        )
        if ef_table_path.exists():
            for row in ef_rows:
                if row.get("oligoai_tox"):
                    tox_animals = sum(
                        row["asos_at_stage"][i] * ANIMALS_PER_ASO for i in IN_VIVO_STAGES
                    )
                    pipeline_numbers["baseline_animals"] = baseline_animals
                    pipeline_numbers["oligoai_tox_animals"] = tox_animals
                    pipeline_numbers["animal_reduction_pct"] = sf(
                        (1 - tox_animals / baseline_animals) * 100
                    )
                    break

        numbers["pipeline"] = pipeline_numbers
    else:
        warnings.warn(f"{PIPELINE_RESULTS_PATH} not found — run `just analysis` first")

    # ── Cost sensitivity Monte Carlo ──
    cs_path = RESULTS_DIR / "cost_sensitivity.json"
    if cs_path.exists():
        numbers["cost_sensitivity"] = _round_floats(json.loads(cs_path.read_text()))

    # ── Mouse vs Rat ALT concordance (dose-matched) ──
    mra_path = RESULTS_DIR / "mouse_rat_alt.json"
    if mra_path.exists():
        numbers["mouse_rat_alt"] = _round_floats(json.loads(mra_path.read_text()))

    # ── OligoGym benchmark summary ──
    bench_path = RESULTS_DIR / "oligogym_benchmark.json"
    if bench_path.exists():
        bench_data = json.loads(bench_path.read_text())
        best = bench_data.get("best_per_model", [])
        if best:
            # Best model per dataset
            bench_summary = {}
            for ds in ["mouse_hepatic", "rat_hepatic", "mouse_neuro", "rat_neuro"]:
                ds_rows = [r for r in best if r["dataset"] == ds and r.get("spearman") is not None]
                if ds_rows:
                    top = max(ds_rows, key=lambda r: r["spearman"])
                    bench_summary[ds] = {
                        "best_model": top["model"],
                        "spearman": {
                            "mean": sf(top["spearman"]),
                            "std": sf(top["spearman_std"]) if top.get("spearman_std") is not None else None,
                            "n_folds": top.get("n_folds", 5),
                        },
                        "r2": sf(top["r2"]) if top.get("r2") is not None else None,
                        "rmse": sf(top["rmse"]) if top.get("rmse") is not None else None,
                    }
            numbers["oligogym_benchmark"] = bench_summary

    # ── OligoAI fine-tune on ASO Atlas 2.0 (5-fold patent GroupKFold) ──
    oligoai_paths = [RESULTS_DIR / f"oligoai_fold{i}.json" for i in range(5)]
    oligoai_folds = [json.loads(p.read_text()) for p in oligoai_paths if p.exists()]
    if oligoai_folds:
        import numpy as _np  # local import keeps top-level imports unchanged

        def _agg(values):
            arr = _np.asarray(
                [v for v in values if v is not None], dtype=float
            )
            arr = arr[_np.isfinite(arr)]
            if arr.size == 0:
                return {"mean": None, "std": None, "n_folds": 0}
            return {
                "mean": sf(float(arr.mean())),
                "std": (sf(float(arr.std(ddof=1)))
                        if arr.size > 1 else 0.0),
                "n_folds": int(arr.size),
            }

        def _agg_int_sum(values):
            arr = [v for v in values if v is not None]
            return int(sum(arr)) if arr else None

        def _f(path, key):
            out = []
            for fold in oligoai_folds:
                ref = fold
                for k in path:
                    ref = ref.get(k) if isinstance(ref, dict) else None
                    if ref is None:
                        break
                out.append(ref.get(key) if isinstance(ref, dict) else None)
            return out

        oligoai_numbers = {
            "efficacy": {
                "spearman": _agg(_f(["per_dose", "single_dose"], "spearman")),
                "r2":       _agg(_f(["per_dose", "single_dose"], "r2")),
                "rmse":     _agg(_f(["per_dose", "single_dose"], "rmse")),
                "n":        _agg_int_sum(_f(["per_dose", "single_dose"], "n")),
                "n_tables": _agg_int_sum(
                    _f(["per_dose"], "n_tables_single_dose")),
            },
            "dose_response": {
                "spearman": _agg(_f(["per_dose", "multi_dose"], "spearman")),
                "r2":       _agg(_f(["per_dose", "multi_dose"], "r2")),
                "rmse":     _agg(_f(["per_dose", "multi_dose"], "rmse")),
                "n":        _agg_int_sum(_f(["per_dose", "multi_dose"], "n")),
                "n_tables": _agg_int_sum(
                    _f(["per_dose"], "n_tables_multi_dose")),
            },
            "potency": {
                "spearman_log": _agg(_f(["ic50"], "spearman_log")),
                "log_rmse":     _agg(_f(["ic50"], "log_rmse")),
                "n_valid":      _agg_int_sum(_f(["ic50"], "n_valid")),
                "n_candidates": _agg_int_sum(_f(["ic50"], "n_candidates")),
            },
            "emax": {
                "spearman": _agg(_f(["emax"], "spearman")),
                "n_valid":  _agg_int_sum(_f(["emax"], "n_valid")),
            },
            "per_group_spearman": {
                "mean":     _agg(_f(["per_group_spearman"], "mean")),
                "n_groups": _agg_int_sum(
                    _f(["per_group_spearman"], "n_groups")),
            },
            "n_folds": len(oligoai_folds),
            "split": (
                "patent-level 5-fold GroupKFold with HELM-level dedup "
                f"(seed=42); reported as mean (std) over "
                f"{len(oligoai_folds)} fold(s)"
            ),
        }
        ef_path = RESULTS_DIR / "oligoai_efs.json"
        if ef_path.exists():
            efs = json.loads(ef_path.read_text())
            oligoai_numbers["efficacy"]["ef"] = {
                "mean": sf(efs["efficacy"]["enrichment_factor"]),
                "std":  sf(efs["efficacy"].get("enrichment_factor_std", 0.0)),
                "n_folds": efs["efficacy"].get("n_folds", 1),
            }
            oligoai_numbers["potency"]["ef"] = {
                "mean": sf(efs["potency"]["enrichment_factor"]),
                "std":  sf(efs["potency"].get("enrichment_factor_std", 0.0)),
                "n_folds": efs["potency"].get("n_folds", 1),
            }
        numbers["oligoai"] = oligoai_numbers

    # ── Base-composition × biomarker correlations (fig_concordance D) ──
    bb_path = RESULTS_DIR / "base_biomarker.json"
    if bb_path.exists():
        bb = json.loads(bb_path.read_text())
        numbers["base_biomarker"] = {
            k: {
                "rho": sf(v["rho"]) if v["rho"] is not None else None,
                "p_bonf": sf(v["p_bonf"]) if v.get("p_bonf") is not None else None,
                "significant": v["significant"],
                "n": v["n"],
            }
            for k, v in bb.items()
        }
    else:
        warnings.warn(f"{bb_path} not found — run `just plots` first")

    # ── Mouse within-biomarker correlations (fig_concordance C) ──
    if hepatotox_path.exists():
        hep_data = json.loads(hepatotox_path.read_text())
        if "mouse_biomarker_correlations" in hep_data:
            mc = hep_data["mouse_biomarker_correlations"]
            bm_names = mc["biomarkers"]
            rho_mat = mc["rho"]
            p_mat = mc["p_values"]
            n_mat = mc["n_pairs"]
            n_bm = len(bm_names)
            n_pairs = n_bm * (n_bm - 1) // 2  # Bonferroni factor
            mouse_bm = {}
            for i in range(n_bm):
                for j in range(i + 1, n_bm):
                    key = f"{bm_names[i]}_{bm_names[j]}"
                    raw_p = p_mat[i][j]
                    mouse_bm[key] = {
                        "rho": sf(rho_mat[i][j]),
                        "p_bonf": sf(min(raw_p * n_pairs, 1.0)),
                        "n": n_mat[i][j],
                    }
            numbers["mouse_biomarker"] = mouse_bm

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(numbers, indent=2) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
