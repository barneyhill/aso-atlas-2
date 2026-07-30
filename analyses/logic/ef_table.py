"""Build cost-based strategy leaderboard.

For each strategy (OligoAI, Hagedorn-Linear, each OligoGym architecture):
  1. Compute per-stage precision@K and Spearman ρ.
  2. Call `back_calculate_enriched` to get total pipeline cost.
Emit both a JSON payload (for the export script) and a Typst table with
grouped In vitro / In vivo super-headers, stacked P@K / ρ cells.

Label the lowest-cost OligoGym row as "(OligoAI-tox)".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analyses.logic.enrichment import (
    DATASET_TO_STAGE,
    enrichment_at_top_k,
    stage_for_dataset,
    stratified_enrichment_at_top_k,
)
from analyses.logic.pipeline import (
    OLIGOAI_ENRICHMENT,
    PIPELINE_STAGES,
    back_calculate_enriched,
    compute_savings_with_uncertainty,
)

def _sf(x, n=3):
    if x == 0:
        return 0
    if isinstance(x, int):
        return x
    d = n - 1 - math.floor(math.log10(abs(x)))
    result = round(x, d)
    if d <= 0:
        return int(result)
    return result


_root = Path(__file__).resolve().parents[2]
RESULTS_DIR = _root / "data/results"
TYPST_DATA_DIR = _root / "typst/data"

COLUMN_STAGES = [
    (0, "Efficacy"),
    (1, "Potency"),
    (2, "μALT"),
    (3, "μFOB"),
    (4, "rALT"),
    (5, "rFOB"),
]

OLIGOGYM_MODEL_ORDER = ["Linear", "Random Forest", "XGBoost", "KNN", "CatBoost", "MLP"]
OLIGOAI_TOX_MODEL = "XGBoost"

DATASET_STRATA_KEY: dict[str, str] = {
    "in_vitro_inhibition": "dosage_nm",
    "mouse_hepatic": "dosage_mg_per_kg",
    "rat_hepatic": "dosage_mg_per_kg",
}


# ---------------------------------------------------------------------------
# Per-strategy enrichment + precision assembly
# ---------------------------------------------------------------------------

def _best_config_per_dataset(bench: dict) -> dict:
    """Return {(dataset, model): row} from benchmark results."""
    best = {}
    for row in bench["all_results"]:
        if row.get("spearman") is None or (isinstance(row.get("spearman"), float) and math.isnan(row["spearman"])):
            continue
        key = (row["dataset"], row["model"])
        if key not in best or row["spearman"] > best[key]["spearman"]:
            best[key] = row
    return best


def _concat_fold_preds(row: dict) -> tuple[np.ndarray, np.ndarray]:
    yt, yp = [], []
    for fm in row.get("fold_metrics", []):
        if "y_true" in fm and "y_pred" in fm:
            yt.extend(fm["y_true"])
            yp.extend(fm["y_pred"])
    return np.asarray(yt, dtype=float), np.asarray(yp, dtype=float)


def per_fold_ef_and_prec(fold_metrics: list[dict], stage,
                         strata_key: str | None = None,
                         k: float | None = None) -> tuple[list[float], list[float]]:
    """Return (fold_efs, fold_precs) from per-fold y_true/y_pred.

    When ``strata_key`` is set, uses stratified within-dosage-group ranking.
    When ``k`` is set, uses that fixed selection fraction instead of base rate.
    """
    efs, precs = [], []
    for fm in fold_metrics:
        if "y_true" not in fm or "y_pred" not in fm:
            continue
        strata = fm.get("conditions", {}).get(strata_key) if strata_key else None
        if strata is not None:
            r = stratified_enrichment_at_top_k(
                fm["y_true"], fm["y_pred"], stage, np.asarray(strata), k=k)
        else:
            r = enrichment_at_top_k(fm["y_true"], fm["y_pred"], stage, k=k)
        ef = r.get("enrichment_factor")
        pr = r.get("selected_pass_rate")
        if ef is not None and ef == ef:
            efs.append(ef)
        if pr is not None and pr == pr:
            precs.append(pr)
    return efs, precs


def oligogym_strategy_data(bench: dict) -> dict[str, dict[str, dict]]:
    """Return {model: {dataset: {enrichment + spearman + precision data}}}.

    For datasets with dosage conditions (in vitro inhibition, hepatic),
    enrichment is stratified by dosage to control for dose confounding.
    """
    best = _best_config_per_dataset(bench)
    out: dict[str, dict[str, dict]] = {m: {} for m in OLIGOGYM_MODEL_ORDER}
    for (ds, model), row in best.items():
        if model not in out or ds not in DATASET_TO_STAGE:
            continue
        y_true, y_pred = _concat_fold_preds(row)
        if len(y_true) == 0:
            continue
        stage = stage_for_dataset(ds)
        strata_key = DATASET_STRATA_KEY.get(ds)

        # Pooled EF (stratified when conditions are available)
        strata_all = None
        if strata_key:
            fold_metrics = row.get("fold_metrics", [])
            parts = []
            for fm in fold_metrics:
                cond = fm.get("conditions", {}).get(strata_key)
                if cond is not None:
                    parts.append(np.asarray(cond, dtype=float))
            if parts and sum(len(s) for s in parts) == len(y_true):
                strata_all = np.concatenate(parts)

        if strata_all is not None:
            ef = stratified_enrichment_at_top_k(y_true, y_pred, stage, strata_all)
        else:
            ef = enrichment_at_top_k(y_true, y_pred, stage)

        fold_efs, fold_precs = per_fold_ef_and_prec(
            row.get("fold_metrics", []), stage, strata_key=strata_key)
        if fold_efs:
            arr = np.asarray(fold_efs, dtype=float)
            ef["fold_efs"] = fold_efs
            ef["ef_mean"] = float(arr.mean())
            ef["ef_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        if fold_precs:
            arr = np.asarray(fold_precs, dtype=float)
            ef["fold_precs"] = fold_precs
            ef["prec_mean"] = float(arr.mean())
            ef["prec_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        if row.get("spearman") is not None:
            ef["spearman"] = row["spearman"]
            ef["spearman_std"] = row.get("spearman_std")

        out[model][ds] = ef
    return out


def load_oligoai_spearmans() -> dict[str, dict]:
    """Load OligoAI per-fold Spearmans from fold JSONs."""
    fold_jsons = [RESULTS_DIR / f"oligoai_fold{i}.json" for i in range(5)]
    available = [p for p in fold_jsons if p.exists()]
    if not available:
        return {}

    eff_rhos: list[float] = []
    pot_rhos: list[float] = []
    for p in available:
        d = json.loads(p.read_text())
        sd = d.get("per_dose", {}).get("single_dose", {})
        if sd.get("spearman") is not None:
            eff_rhos.append(sd["spearman"])
        ic = d.get("ic50", {})
        if ic.get("spearman_log") is not None:
            pot_rhos.append(ic["spearman_log"])

    out: dict[str, dict] = {}
    if eff_rhos:
        arr = np.asarray(eff_rhos, dtype=float)
        out["in_vitro_inhibition"] = {
            "spearman": _sf(float(arr.mean())),
            "spearman_std": _sf(float(arr.std(ddof=1))) if len(arr) > 1 else None,
        }
    if pot_rhos:
        arr = np.asarray(pot_rhos, dtype=float)
        out["potency"] = {
            "spearman": _sf(float(arr.mean())),
            "spearman_std": _sf(float(arr.std(ddof=1))) if len(arr) > 1 else None,
        }
    return out


def hagedorn_linear_enrichment(neuro: dict, n_splits: int = 5) -> dict[str, dict]:
    """Compute Hagedorn-Linear enrichment using the same continuous EF as OligoGym.

    The Hagedorn score is negated (higher score = safer = lower FOB) so that
    enrichment_at_top_k selects the lowest predicted-FOB compounds, consistent
    with the FOB ≤ 1 threshold direction.
    """
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    src_map = {
        "mouse_neuro": neuro.get("predictions", {}).get("hagedorn_score"),
        "rat_neuro": neuro.get("rat_predictions", {}).get("rat_hagedorn_score"),
    }

    out: dict[str, dict] = {}
    for ds, raw in src_map.items():
        if raw is None or "fob_values" not in raw or "scores" not in raw:
            continue
        fob = np.asarray(raw["fob_values"], dtype=float)
        scores = np.asarray(raw["scores"], dtype=float)
        grp = np.asarray(raw.get("groups", []))

        # Negate scores: higher Hagedorn = safer = lower FOB
        neg_scores = -scores
        stage = stage_for_dataset(ds)

        # Pooled EF
        ef = enrichment_at_top_k(fob, neg_scores, stage)

        # Per-fold EF and Spearman via GroupKFold
        valid_grp = grp[~np.array([g is None for g in grp])] if len(grp) > 0 else grp
        n_grp = len(np.unique(valid_grp)) if len(valid_grp) > 0 else 0
        actual_splits = min(n_splits, n_grp)

        if actual_splits >= 2 and len(grp) == len(fob):
            gkf = GroupKFold(n_splits=actual_splits)
            fold_efs, fold_precs, fold_rhos = [], [], []
            for _, test_idx in gkf.split(fob, fob, grp):
                if len(test_idx) < 10:
                    continue
                r_ef = enrichment_at_top_k(fob[test_idx], neg_scores[test_idx], stage)
                e = r_ef.get("enrichment_factor")
                p = r_ef.get("selected_pass_rate")
                if e is not None and e == e:
                    fold_efs.append(e)
                if p is not None and p == p:
                    fold_precs.append(p)
                rho, _ = spearmanr(neg_scores[test_idx], fob[test_idx])
                if rho == rho:
                    fold_rhos.append(rho)

            if len(fold_efs) > 1:
                arr = np.asarray(fold_efs, dtype=float)
                ef["fold_efs"] = fold_efs
                ef["ef_mean"] = float(arr.mean())
                ef["ef_std"] = float(arr.std(ddof=1))
            if len(fold_precs) > 1:
                arr_p = np.asarray(fold_precs, dtype=float)
                ef["fold_precs"] = fold_precs
                ef["prec_mean"] = float(arr_p.mean())
                ef["prec_std"] = float(arr_p.std(ddof=1))
            if len(fold_rhos) > 1:
                arr_r = np.asarray(fold_rhos, dtype=float)
                ef["spearman"] = _sf(float(arr_r.mean()))
                ef["spearman_std"] = _sf(float(arr_r.std(ddof=1)))

        out[ds] = ef
    return out


# ---------------------------------------------------------------------------
# Strategy row construction
# ---------------------------------------------------------------------------

def _row(
    name: str,
    enrichment: dict[str, dict],
    proportions: list[float],
    baseline_cost: float,
    spearman_data: dict[str, dict] | None = None,
) -> dict:
    """Compose a single strategy row."""
    stage_map = {ds: idx for ds, idx in DATASET_TO_STAGE.items() if ds in enrichment}
    result = back_calculate_enriched(proportions, enrichment, stage_map)

    efs: dict[int, float | None] = {idx: None for idx, _ in COLUMN_STAGES}
    ef_stds: dict[int, float | None] = {idx: None for idx, _ in COLUMN_STAGES}
    precs: dict[int, float | None] = {idx: None for idx, _ in COLUMN_STAGES}
    prec_stds: dict[int, float | None] = {idx: None for idx, _ in COLUMN_STAGES}
    rhos: dict[int, float | None] = {idx: None for idx, _ in COLUMN_STAGES}
    rho_stds: dict[int, float | None] = {idx: None for idx, _ in COLUMN_STAGES}

    for ds, ef_entry in enrichment.items():
        idx = DATASET_TO_STAGE.get(ds)
        if idx is None:
            continue
        efs[idx] = ef_entry.get("enrichment_factor")
        if "ef_std" in ef_entry:
            ef_stds[idx] = ef_entry["ef_std"]
        if "selected_pass_rate" in ef_entry:
            precs[idx] = ef_entry["selected_pass_rate"]
        elif "base_rate" in ef_entry and ef_entry.get("enrichment_factor") is not None:
            precs[idx] = min(ef_entry["base_rate"] * ef_entry["enrichment_factor"], 1.0)
        if "prec_std" in ef_entry:
            prec_stds[idx] = ef_entry["prec_std"]
        elif "prec_mean" in ef_entry:
            prec_stds[idx] = ef_entry.get("prec_std")
        if ef_entry.get("spearman") is not None:
            rhos[idx] = ef_entry["spearman"]
            rho_stds[idx] = ef_entry.get("spearman_std")

    if spearman_data:
        for ds, sp in spearman_data.items():
            idx = DATASET_TO_STAGE.get(ds)
            if idx is None:
                continue
            if sp.get("spearman") is not None:
                rhos[idx] = sp["spearman"]
                rho_stds[idx] = sp.get("spearman_std")

    savings, savings_std = compute_savings_with_uncertainty(
        precs, prec_stds, proportions, baseline_cost)

    return {
        "name": name,
        "ef_by_stage": efs,
        "ef_std_by_stage": ef_stds,
        "prec_by_stage": precs,
        "prec_std_by_stage": prec_stds,
        "rho_by_stage": rhos,
        "rho_std_by_stage": rho_stds,
        "n_initial": result["n_initial"],
        "total_cost": result["total_cost"],
        "savings_pct": savings,
        "savings_std_pct": savings_std,
        "costs_per_stage": result["costs_per_stage"],
        "asos_at_stage": result["asos_at_stage"],
        "proportions": result["proportions"],
    }


def _extract_base_rates(gym_data: dict[str, dict[str, dict]]) -> dict[int, float]:
    """Get base_rate per stage from the first complete OligoGym model."""
    base: dict[int, float] = {}
    for model_data in gym_data.values():
        for ds, ef_entry in model_data.items():
            idx = DATASET_TO_STAGE.get(ds)
            if idx is not None and "base_rate" in ef_entry and idx not in base:
                base[idx] = ef_entry["base_rate"]
    return base


def _add_oligoai_precision(enrichment: dict[str, dict], base_rates: dict[int, float]) -> None:
    """Ensure OligoAI entries contain precision, with a legacy fallback.

    Current OligoAI result files contain pooled OOF selected-pass rates and
    fold-level precisions directly.  The EF × reference-base-rate calculation is
    retained only so older result files remain readable.
    """
    for ds, ef_entry in enrichment.items():
        idx = DATASET_TO_STAGE.get(ds)
        if idx is None or idx not in base_rates:
            continue
        if ef_entry.get("selected_pass_rate") is not None:
            continue
        br = base_rates[idx]
        ef = ef_entry.get("enrichment_factor")
        if ef is not None:
            ef_entry["selected_pass_rate"] = min(_sf(br * ef), 1.0)
            ef_entry["base_rate"] = br
            fold_efs = ef_entry.get("fold_efs")
            if fold_efs:
                fold_precs = [min(br * fe, 1.0) for fe in fold_efs]
                arr = np.asarray(fold_precs, dtype=float)
                ef_entry["fold_precs"] = fold_precs
                ef_entry["prec_mean"] = float(arr.mean())
                ef_entry["prec_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def build_rows(bench: dict, neuro: dict, baseline: dict) -> list[dict]:
    proportions = baseline["proportions"]
    baseline_cost = baseline["baseline"]["total_cost"]
    rows: list[dict] = []

    oligoai_sp = load_oligoai_spearmans()
    gym_data = oligogym_strategy_data(bench)
    base_rates = _extract_base_rates(gym_data)

    # --- Baseline row (no screening) ---
    baseline_precs = {idx: base_rates.get(idx) for idx, _ in COLUMN_STAGES}
    baseline_row = {
        "name": "Baseline",
        "ef_by_stage": {idx: None for idx, _ in COLUMN_STAGES},
        "ef_std_by_stage": {idx: None for idx, _ in COLUMN_STAGES},
        "prec_by_stage": baseline_precs,
        "prec_std_by_stage": {idx: None for idx, _ in COLUMN_STAGES},
        "rho_by_stage": {idx: None for idx, _ in COLUMN_STAGES},
        "rho_std_by_stage": {idx: None for idx, _ in COLUMN_STAGES},
        "n_initial": baseline["baseline"]["n_initial"],
        "total_cost": float(baseline_cost),
        "costs_per_stage": baseline["baseline"]["costs_per_stage"],
        "asos_at_stage": baseline["baseline"]["asos_at_stage"],
        "proportions": proportions,
        "group": "baseline",
        "is_baseline": True,
    }
    rows.append(baseline_row)

    # --- OligoAI ---
    if OLIGOAI_ENRICHMENT.get("inhibition") is not None:
        oligoai_en: dict = {
            "in_vitro_inhibition": dict(OLIGOAI_ENRICHMENT["inhibition"]),
        }
        if "potency" in OLIGOAI_ENRICHMENT:
            oligoai_en["potency"] = dict(OLIGOAI_ENRICHMENT["potency"])
        _add_oligoai_precision(oligoai_en, base_rates)
        rows.append({**_row("OligoAI", oligoai_en, proportions, baseline_cost, oligoai_sp), "group": "external"})

    # --- Hagedorn-Linear ---
    hl_en = hagedorn_linear_enrichment(neuro)
    if hl_en:
        rows.append({**_row("Hagedorn-Linear", hl_en, proportions, baseline_cost), "group": "hagedorn"})

    # --- OligoGym architectures ---
    gym_rows = []
    for model in OLIGOGYM_MODEL_ORDER:
        data = gym_data.get(model, {})
        if not data:
            continue
        gym_rows.append({**_row(model, data, proportions, baseline_cost), "group": "oligogym"})
    rows.extend(gym_rows)

    # --- Tox model (in vivo only) + Combined ---
    IN_VIVO_DATASETS = {"mouse_hepatic", "mouse_neuro", "rat_hepatic", "rat_neuro"}
    tox_data = gym_data.get(OLIGOAI_TOX_MODEL, {})
    tox_invivo = {ds: d for ds, d in tox_data.items() if ds in IN_VIVO_DATASETS}
    tox_invivo_sp = {ds: {"spearman": d.get("spearman"), "spearman_std": d.get("spearman_std")}
                     for ds, d in tox_data.items() if d.get("spearman") is not None and ds in IN_VIVO_DATASETS}
    if tox_invivo:
        rows.append({
            **_row("XGBoost (in vivo)", tox_invivo, proportions, baseline_cost, tox_invivo_sp),
            "group": "tox_only",
            "oligoai_tox": True,
            "source_model": OLIGOAI_TOX_MODEL,
        })

    if OLIGOAI_ENRICHMENT.get("inhibition") is not None and tox_invivo:
        combined_en: dict = {
            "in_vitro_inhibition": dict(OLIGOAI_ENRICHMENT["inhibition"]),
        }
        if "potency" in OLIGOAI_ENRICHMENT:
            combined_en["potency"] = dict(OLIGOAI_ENRICHMENT["potency"])
        _add_oligoai_precision(combined_en, base_rates)
        combined_en.update(tox_invivo)
        combined_sp = {ds: sp for ds, sp in oligoai_sp.items() if ds not in IN_VIVO_DATASETS}
        combined_sp.update(tox_invivo_sp)
        rows.append({
            **_row("OligoAI + XGBoost (in vivo)", combined_en, proportions, baseline_cost, combined_sp),
            "group": "combined",
            "typst_name": "OligoAI +\\ XGBoost\\ (in vivo)",
            "source_model": OLIGOAI_TOX_MODEL,
        })

    for r in rows:
        r["delta_pct"] = r.get("savings_pct", _sf(100.0 * (baseline_cost - r["total_cost"]) / baseline_cost))
        r["delta_std_pct"] = r.get("savings_std_pct")

    return rows


# ---------------------------------------------------------------------------
# Typst rendering
# ---------------------------------------------------------------------------

def _fmt_prec(prec: float | None, std: float | None = None) -> str:
    if prec is None:
        return "—"
    pct = prec * 100
    if std is None:
        return f"{pct:.1f}%"
    std_pct = std * 100
    return f"{pct:.1f}±{std_pct:.1f}%"


def _fmt_rho(rho: float | None, std: float | None = None) -> str:
    if rho is None:
        return "—"
    if std is None:
        return f".{int(round(rho * 1000)):03d}"
    return f".{int(round(rho * 1000)):03d}±.{int(round(std * 1000)):03d}"


def _fmt_savings(c: float, baseline: float, std: float | None = None) -> str:
    pct = 100.0 * (baseline - c) / baseline
    if std is None:
        return f"{pct:.1f}%"
    std_pct = 100.0 * std / baseline
    return f"{pct:.1f}±{std_pct:.1f}%"


def _best_per_column(rows: list[dict], key: str, skip_baseline: bool = True) -> dict[int, float]:
    best: dict[int, float] = {}
    for idx, _ in COLUMN_STAGES:
        vals = []
        for r in rows:
            if skip_baseline and r.get("is_baseline"):
                continue
            if r.get("group") == "combined":
                continue
            v = r[key].get(idx)
            if v is not None and v == v:
                vals.append(v)
        if vals:
            best[idx] = max(vals)
    return best


def _fmt_prec_cell(prec: float | None, prec_std: float | None, bold: bool) -> str:
    if prec is None:
        return "[—]"
    txt = _fmt_prec(prec, prec_std)
    if bold:
        txt = f"*{txt}*"
    return f"[{txt}]"


def _fmt_rho_cell(rho: float | None, rho_std: float | None, bold: bool) -> str:
    if rho is None:
        return "[—]"
    txt = _fmt_rho(rho, rho_std)
    if bold:
        txt = f"*{txt}*"
    return f"[{txt}]"


def _row_separator(r: dict, next_r: dict) -> str:
    if next_r.get("group") == "combined":
        return "  table.hline(stroke: 0.8pt),"
    elif r.get("group") != next_r.get("group"):
        return "  table.hline(stroke: 0.5pt),"
    else:
        return "  table.hline(stroke: (paint: gray, thickness: 0.3pt)),"


def render_typst(rows: list[dict], baseline_cost: float) -> str:
    """Main table: precision@K + savings."""
    rows = [r for r in rows if r.get("group") != "tox_only"]
    lines = [
        "#text(size: 8pt)[#table(",
        "  columns: (6em, auto, auto, auto, auto, auto, auto, auto),",
        "  align: center + horizon,",
        "  stroke: (left: 0.8pt, right: 0.8pt, top: none, bottom: none),",
        "  inset: (x: 6pt, y: 4pt),",
        "  table.vline(x: 1, stroke: 0.5pt),",
        "  table.vline(x: 3, stroke: 0.5pt),",
        "  table.vline(x: 5, stroke: 0.5pt),",
        "  table.vline(x: 7, stroke: 0.5pt),",
        "  table.vline(x: 2, stroke: (paint: gray, thickness: 0.3pt)),",
        "  table.vline(x: 4, stroke: (paint: gray, thickness: 0.3pt)),",
        "  table.vline(x: 6, stroke: (paint: gray, thickness: 0.3pt)),",
        "  table.hline(stroke: 0.8pt),",
        "  table.header(",
        "    table.cell(rowspan: 4)[*Model*],",
        "    table.cell(colspan: 6)[*Precision*],",
        "    table.cell(rowspan: 4)[*Savings*],",
        "    table.hline(start: 1, end: 7, stroke: 0.4pt),",
        "    table.cell(colspan: 2)[*In vitro*],",
        "    table.cell(colspan: 4)[*In vivo*],",
        "    table.hline(start: 1, end: 3, stroke: 0.4pt),",
        "    table.hline(start: 3, end: 7, stroke: 0.4pt),",
        "    table.cell(rowspan: 2)[Efficacy], table.cell(rowspan: 2)[Potency],",
        "    table.cell(colspan: 2)[Mouse], table.cell(colspan: 2)[Rat],",
        "    table.hline(start: 3, end: 5, stroke: 0.4pt),",
        "    table.hline(start: 5, end: 7, stroke: 0.4pt),",
        "    [Hepatic], [Neuro], [Hepatic], [Neuro],",
        "  ),",
        "  table.hline(stroke: 0.5pt),",
    ]

    non_combined = [r for r in rows if r.get("group") != "combined" and not r.get("is_baseline")]
    best_prec = _best_per_column(rows, "prec_by_stage")
    best_savings_pct = max(
        r.get("savings_pct", 0.0) for r in non_combined
    ) if non_combined else 0

    for i, r in enumerate(rows):
        display_name = r.get("typst_name") or r["name"].replace(" ", "\\ ")
        is_combined = r.get("group") == "combined"
        is_baseline = r.get("is_baseline", False)

        cells = [f"  [{display_name}]"]
        for idx, _ in COLUMN_STAGES:
            prec = r["prec_by_stage"][idx]
            prec_std = r["prec_std_by_stage"].get(idx) if not is_baseline else None
            bold = (not is_baseline and not is_combined and prec is not None
                    and idx in best_prec and prec == best_prec[idx])
            cells.append("  " + _fmt_prec_cell(prec, prec_std, bold))
        if is_baseline:
            cells.append("  [0.0%]")
        else:
            sav = r.get("savings_pct", 0.0)
            sav_std = r.get("savings_std_pct")
            if sav_std is not None:
                cost_txt = f"{sav:.1f}±{sav_std:.1f}%"
            else:
                cost_txt = f"{sav:.1f}%"
            if is_combined:
                cost_txt = f"*{cost_txt}*"
            cells.append(f"  [{cost_txt}]")

        lines.append(", ".join(cells) + ",")
        if i < len(rows) - 1:
            lines.append(_row_separator(r, rows[i + 1]))
    lines.append("  table.hline(stroke: 0.8pt),")
    lines.append(")]")
    return "\n".join(lines) + "\n"


def render_spearman_typst(rows: list[dict]) -> str:
    """Appendix table: Spearman rho per model per endpoint."""
    rows = [r for r in rows
            if r.get("group") != "tox_only"
            and not r.get("is_baseline")
            and r.get("group") != "combined"]
    lines = [
        "#text(size: 8pt)[#table(",
        "  columns: (6em, auto, auto, auto, auto, auto, auto),",
        "  align: center + horizon,",
        "  stroke: (left: 0.8pt, right: 0.8pt, top: none, bottom: none),",
        "  inset: (x: 6pt, y: 4pt),",
        "  table.vline(x: 1, stroke: 0.5pt),",
        "  table.vline(x: 3, stroke: 0.5pt),",
        "  table.vline(x: 5, stroke: 0.5pt),",
        "  table.vline(x: 2, stroke: (paint: gray, thickness: 0.3pt)),",
        "  table.vline(x: 4, stroke: (paint: gray, thickness: 0.3pt)),",
        "  table.vline(x: 6, stroke: (paint: gray, thickness: 0.3pt)),",
        "  table.hline(stroke: 0.8pt),",
        "  table.header(",
        "    table.cell(rowspan: 4)[*Model*],",
        "    table.cell(colspan: 6)[*Spearman $rho$*],",
        "    table.hline(start: 1, end: 7, stroke: 0.4pt),",
        "    table.cell(colspan: 2)[*In vitro*],",
        "    table.cell(colspan: 4)[*In vivo*],",
        "    table.hline(start: 1, end: 3, stroke: 0.4pt),",
        "    table.hline(start: 3, end: 7, stroke: 0.4pt),",
        "    table.cell(rowspan: 2)[Efficacy], table.cell(rowspan: 2)[Potency],",
        "    table.cell(colspan: 2)[Mouse], table.cell(colspan: 2)[Rat],",
        "    table.hline(start: 3, end: 5, stroke: 0.4pt),",
        "    table.hline(start: 5, end: 7, stroke: 0.4pt),",
        "    [Hepatic], [Neuro], [Hepatic], [Neuro],",
        "  ),",
        "  table.hline(stroke: 0.5pt),",
    ]

    best_rho = _best_per_column(rows, "rho_by_stage")

    for i, r in enumerate(rows):
        display_name = r["name"].replace(" ", "\\ ")
        cells = [f"  [{display_name}]"]
        for idx, _ in COLUMN_STAGES:
            rho = r["rho_by_stage"][idx]
            rho_std = r["rho_std_by_stage"].get(idx)
            bold = rho is not None and idx in best_rho and rho == best_rho[idx]
            cells.append("  " + _fmt_rho_cell(rho, rho_std, bold))
        lines.append(", ".join(cells) + ",")
        if i < len(rows) - 1:
            lines.append(_row_separator(r, rows[i + 1]))

    lines.append("  table.hline(stroke: 0.8pt),")
    lines.append(")]")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bench = json.loads((RESULTS_DIR / "oligogym_benchmark.json").read_text())
    neuro = json.loads((RESULTS_DIR / "neurotox.json").read_text())
    baseline = json.loads((RESULTS_DIR / "pipeline_results.json").read_text())

    rows = build_rows(bench, neuro, baseline)

    out_rows = [
        {
            "name": r["name"],
            "group": r["group"],
            "ef_by_stage": {str(k): v for k, v in r["ef_by_stage"].items()},
            "ef_std_by_stage": {str(k): v for k, v in r["ef_std_by_stage"].items()},
            "prec_by_stage": {str(k): v for k, v in r["prec_by_stage"].items()},
            "prec_std_by_stage": {str(k): v for k, v in r["prec_std_by_stage"].items()},
            "rho_by_stage": {str(k): v for k, v in r["rho_by_stage"].items()},
            "rho_std_by_stage": {str(k): v for k, v in r["rho_std_by_stage"].items()},
            "n_initial": int(r["n_initial"]),
            "total_cost": float(r["total_cost"]),
            "costs_per_stage": [float(c) for c in r["costs_per_stage"]],
            "asos_at_stage": [int(a) if isinstance(a, (int, float)) and a != float("inf") else a for a in r["asos_at_stage"]],
            "proportions": [float(p) for p in r["proportions"]],
            "delta_pct": r["delta_pct"],
            "delta_std_pct": r.get("delta_std_pct"),
            "savings_pct": r.get("savings_pct"),
            "savings_std_pct": r.get("savings_std_pct"),
            "oligoai_tox": bool(r.get("oligoai_tox", False)),
            "is_baseline": bool(r.get("is_baseline", False)),
            **({"source_model": r["source_model"]} if "source_model" in r else {}),
        }
        for r in rows
    ]
    (RESULTS_DIR / "ef_table.json").write_text(json.dumps(out_rows, indent=2))
    print(f"Wrote {RESULTS_DIR / 'ef_table.json'} ({len(out_rows)} rows)")

    TYPST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    typst = render_typst(rows, baseline_cost=baseline["baseline"]["total_cost"])
    (TYPST_DATA_DIR / "ef_table.typ").write_text(typst)
    print(f"Wrote {TYPST_DATA_DIR / 'ef_table.typ'}")

    spearman_typst = render_spearman_typst(rows)
    (TYPST_DATA_DIR / "spearman_table.typ").write_text(spearman_typst)
    print(f"Wrote {TYPST_DATA_DIR / 'spearman_table.typ'}")


if __name__ == "__main__":
    main()
