"""
Figure 5: Selection bias, cross-assay and cross-biomarker correlations,
base composition, and cross-species concordance.

(A) Selection bias KDEs: biomarker distributions for compounds that do vs
    don't advance to the next pipeline stage (4 transitions).
(B) Cross-assay Spearman ρ (4×4): pairwise correlations between per-compound
    metrics across pipeline stages (BH-corrected).
(C) Cross-biomarker Spearman ρ: mouse hepatotoxicity biomarker correlations
    (Bonferroni-corrected).
(D) Base × biomarker Spearman ρ (4×4): nucleotide base composition vs
    toxicity biomarkers (BH-corrected).
(E) Cross-species bFOB concordance: mouse vs rat FOB score heatmap.

Reads: data/results/{hepatotox,neurotox}.json
        data/oligostack/processed/{hepatictoxicity,neurotoxicity}_processed.parquet
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyses.utils.models import mean_of_array

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
HEPATOTOX_JSON = _root / "data/results/hepatotox.json"
NEUROTOX_JSON = _root / "data/results/neurotox.json"
OUT_DIR = _root / "typst/plots/fig5"


def draw_cross_species_rho(ax, biomarkers):
    """Draw dot + 95% CI plot of cross-species Spearman ρ.

    biomarkers: list of (label, rho, ci_lo, ci_hi, pval, n) tuples.
    Only significant correlations (Bonferroni-corrected) are shown.
    """
    n_tests = len(biomarkers)
    alpha = 0.05 / n_tests

    labels, rhos, ci_los, ci_his, pvals, ns = zip(*biomarkers)
    y = np.arange(len(labels))

    for i in range(len(labels)):
        sig = pvals[i] < alpha
        color = "#4878A8" if sig else "#cccccc"
        ax.errorbar(rhos[i], y[i],
                    xerr=[[rhos[i] - ci_los[i]], [ci_his[i] - rhos[i]]],
                    fmt="o", color=color, ecolor=color, capsize=4,
                    markersize=6, elinewidth=1.5)
        # n annotation
        ax.text(ci_his[i] + 0.02, y[i], f"n={ns[i]}",
                va="center", fontsize=7, color="#666666")
        if not sig:
            ax.text(rhos[i], y[i] + 0.25, "n.s.", ha="center",
                    fontsize=7, color="#999999")

    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Spearman \u03c1 (mouse vs rat)", fontsize=9)
    ax.set_xlim(-0.1, 1.0)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)


def _load_compound_metrics():
    """Load per-compound metrics and progression ID sets (shared by panels A & B).

    Returns (metrics, id_sets) where:
      metrics = {"iv_max": Series, "dr_ic50": Series, "mouse_alt": Series, "mouse_fob": Series}
      id_sets = {"dr_ids": set, "tox_ids": set, "rat_hep_ids": set, "rat_neuro_ids": set}
    """
    from analyses.logic.pipeline import fit_ic50_for_compound

    _data_dir = _root / "data/oligostack/processed"

    invitro = pd.read_parquet(_data_dir / "in_vitro_inhibition_processed.parquet")
    dose_resp = pd.read_parquet(_data_dir / "dose_response_processed.parquet")
    hep = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    neuro = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")

    iv_max = invitro.groupby("Compound ID")["Inhibition_pct"].max()

    dr_elec = dose_resp[dose_resp["transfection_method"] == "Electroporation"]
    ic50_results = []
    for cid, grp in dr_elec.groupby("Compound ID"):
        ic50 = fit_ic50_for_compound(
            grp["dosage_nm"].values, grp["Inhibition_pct"].values,
        )
        if not np.isnan(ic50):
            ic50_results.append({"Compound ID": cid, "ic50_nm": ic50})
    dr_ic50 = pd.DataFrame(ic50_results).set_index("Compound ID")["ic50_nm"]

    hep["ALT_mean"] = hep["ALT"].apply(mean_of_array)
    mouse_alt = (
        hep[hep["species"] == "mouse"]
        .groupby("Compound ID")["ALT_mean"].mean().dropna()
    )

    neuro["FOB_mean"] = neuro["FOB_score"].apply(mean_of_array)
    mouse_fob = (
        neuro[(neuro["species"] == "Mouse") & (neuro["dosage_ug"] == 700)]
        .groupby("Compound ID")["FOB_mean"].mean().dropna()
    )

    metrics = {
        "iv_max": iv_max,
        "dr_ic50": dr_ic50,
        "mouse_alt": mouse_alt,
        "mouse_fob": mouse_fob,
    }
    id_sets = {
        "dr_ids": set(dose_resp["Compound ID"].unique()),
        "tox_ids": set(hep["Compound ID"].unique()) | set(neuro["Compound ID"].unique()),
        "rat_hep_ids": set(hep[hep["species"] == "rat"]["Compound ID"].unique()),
        "rat_neuro_ids": set(neuro[neuro["species"] == "Rat"]["Compound ID"].unique()),
    }
    return metrics, id_sets


def draw_selection_bias(axes, metrics, id_sets):
    """Draw 4 KDE panels showing selection bias at each pipeline gate.

    axes: list of 4 axes, one per transition.
    metrics: dict from _load_compound_metrics().
    id_sets: dict from _load_compound_metrics().
    """
    from scipy.stats import gaussian_kde

    iv_max = metrics["iv_max"]
    dr_ic50 = metrics["dr_ic50"]
    mouse_alt = metrics["mouse_alt"]
    mouse_fob = metrics["mouse_fob"]
    dr_ids = id_sets["dr_ids"]
    tox_ids = id_sets["tox_ids"]
    rat_hep_ids = id_sets["rat_hep_ids"]
    rat_neuro_ids = id_sets["rat_neuro_ids"]

    transitions = [
        dict(
            dest="Dose-response",
            values=iv_max, prog_ids=dr_ids,
            xlabel="In vitro max inhibition (%)", log_scale=False,
        ),
        dict(
            dest="In vivo tox",
            values=dr_ic50, prog_ids=tox_ids,
            xlabel="Electroporation IC50 (nM)", log_scale=True,
            log_ticks=[10, 100, 1000, 10000],
        ),
        dict(
            dest="Rat hepatotox",
            values=mouse_alt, prog_ids=rat_hep_ids,
            xlabel="Mouse mean ALT (IU/L)", log_scale=True,
            log_ticks=[10, 100, 1000, 10000],
        ),
        dict(
            dest="Rat neurotox",
            values=mouse_fob, prog_ids=rat_neuro_ids,
            xlabel="Mouse mean bFOB score", log_scale=False,
            xticks=list(range(8)),
        ),
    ]

    c_prog = "#4878A8"
    c_nonprog = "#aaaaaa"

    for ax, t in zip(axes, transitions):
        vals_all = t["values"]
        prog = vals_all[vals_all.index.isin(t["prog_ids"])]
        nonprog = vals_all[~vals_all.index.isin(t["prog_ids"])]

        if t["log_scale"]:
            lo_log = np.log10(max(vals_all.quantile(0.005), 1))
            hi_log = np.log10(vals_all.quantile(0.995))
            x_range = np.linspace(lo_log - 0.2, hi_log + 0.2, 300)

            # Gray baseline (no outline)
            log_np = np.log10(np.clip(nonprog.values, 10**lo_log, None))
            kde_np = gaussian_kde(log_np, bw_method=0.3)
            ax.fill_between(x_range, kde_np(x_range), alpha=0.25, color=c_nonprog)
            # Coloured "present" distribution
            log_p = np.log10(np.clip(prog.values, 10**lo_log, None))
            kde_p = gaussian_kde(log_p, bw_method=0.3)
            ax.fill_between(x_range, kde_p(x_range), alpha=0.35, color=c_prog)
            ax.plot(x_range, kde_p(x_range), color=c_prog, lw=1.5)

            tick_vals = t.get("log_ticks", [10, 100, 1000])
            ax.set_xticks([np.log10(v) for v in tick_vals])
            ax.set_xticklabels(tick_vals, fontsize=7)
        else:
            lo = max(vals_all.quantile(0.001), -1)
            hi = vals_all.quantile(0.999)
            margin = (hi - lo) * 0.05
            x_range = np.linspace(lo - margin, hi + margin, 300)

            # Gray baseline (no outline)
            clipped_np = np.clip(nonprog.values, x_range[0], x_range[-1])
            kde_np = gaussian_kde(clipped_np, bw_method=0.3)
            ax.fill_between(x_range, kde_np(x_range), alpha=0.25, color=c_nonprog)
            # Coloured "present" distribution
            clipped_p = np.clip(prog.values, x_range[0], x_range[-1])
            kde_p = gaussian_kde(clipped_p, bw_method=0.3)
            ax.fill_between(x_range, kde_p(x_range), alpha=0.35, color=c_prog)
            ax.plot(x_range, kde_p(x_range), color=c_prog, lw=1.5)

            if "xticks" in t:
                ax.set_xticks(t["xticks"])
            ax.tick_params(axis="x", labelsize=7)

        ax.set_title(f"Present in {t['dest']}", fontsize=9,
                     fontweight="bold", color=c_prog)

        ax.set_xlabel(t["xlabel"], fontsize=8)
        ax.set_yticks([])
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)


def draw_cross_assay_corr(ax, metrics):
    """Draw 4×4 Spearman correlation matrix across assay stages.

    metrics: dict from _load_compound_metrics() with keys
      iv_max, dr_ic50, mouse_alt, mouse_fob (each a per-compound Series).
    """
    from statsmodels.stats.multitest import multipletests

    labels = ["Max inhib.", "IC50", "ALT", "bFOB"]
    series = [metrics["iv_max"], metrics["dr_ic50"],
              metrics["mouse_alt"], metrics["mouse_fob"]]
    n = len(labels)

    rho_matrix = np.full((n, n), np.nan)
    pval_matrix = np.full((n, n), np.nan)

    for i in range(n):
        rho_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            shared = series[i].index.intersection(series[j].index)
            si = series[i].reindex(shared).dropna()
            sj = series[j].reindex(shared).dropna()
            shared_valid = si.index.intersection(sj.index)
            if len(shared_valid) < 10:
                continue
            rho, p = spearmanr(si[shared_valid], sj[shared_valid])
            rho_matrix[i, j] = rho_matrix[j, i] = float(rho)
            pval_matrix[i, j] = pval_matrix[j, i] = float(p)

    # BH correction on upper-triangle p-values
    tri_idx = np.triu_indices(n, k=1)
    flat_p = pval_matrix[tri_idx]
    valid_mask = ~np.isnan(flat_p)
    reject_flat = np.zeros_like(flat_p, dtype=bool)
    if valid_mask.any():
        reject_flat[valid_mask], _, _, _ = multipletests(
            flat_p[valid_mask], alpha=0.05, method="fdr_bh",
        )
    sig_matrix = np.eye(n, dtype=bool)  # diagonal always "significant"
    for k, (i, j) in enumerate(zip(*tri_idx)):
        if reject_flat[k]:
            sig_matrix[i, j] = sig_matrix[j, i] = True

    # Gray background for non-significant cells
    ax.imshow(
        np.ones((n, n)),
        cmap=matplotlib.colors.ListedColormap(["#e0e0e0"]),
        aspect="equal",
    )
    display = np.where(sig_matrix, np.where(np.isnan(rho_matrix), 0, rho_matrix), np.nan)
    ax.imshow(display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            if sig_matrix[i, j] and not np.isnan(rho_matrix[i, j]):
                val = rho_matrix[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
            elif not sig_matrix[i, j]:
                ax.text(j, i, "n.s.", ha="center", va="center",
                        fontsize=7, color="#888888")

    # Cell borders (clip_on=False so edge lines aren't clipped)
    for i in range(-1, n):
        ax.axhline(i + 0.5, color="black", linewidth=0.3, clip_on=False)
        ax.axvline(i + 0.5, color="black", linewidth=0.3, clip_on=False)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_fob_heatmap(ax, mouse_vals, rat_vals, rho, pval, n):
    """Draw FOB integer-grid heatmap."""
    import pandas as pd

    mouse_int = np.round(mouse_vals).astype(int)
    rat_int = np.round(rat_vals).astype(int)

    paired = pd.DataFrame({"Mouse": mouse_int, "Rat": rat_int})
    count_matrix = paired.groupby(["Mouse", "Rat"]).size().unstack(fill_value=0)
    full_index = range(8)
    count_matrix = count_matrix.reindex(index=full_index, columns=full_index, fill_value=0)

    log_counts = np.where(
        count_matrix.values > 0,
        np.log10(count_matrix.values),
        np.nan,
    )
    ax.imshow(
        log_counts, origin="lower", cmap="Reds", aspect="equal",
        vmin=0, vmax=np.nanmax(log_counts),
    )

    for i in range(8):
        for j in range(8):
            val = count_matrix.values[i, j]
            if val > 0:
                lv = np.log10(val)
                color = "white" if lv > np.nanmax(log_counts) * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    for i in range(-1, 8):
        ax.axhline(i + 0.5, color="black", linewidth=0.5)
        ax.axvline(i + 0.5, color="black", linewidth=0.5)

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("Mouse mean bFOB score", fontsize=9)
    ax.set_ylabel("Rat mean mFOB score", fontsize=9)

    # Remove outer spines (cell borders define the boundary)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_corr_matrix(ax, corr_data):
    """Draw inter-biomarker Spearman correlation matrix."""
    biomarkers = corr_data["biomarkers"]
    rho = np.array(corr_data["rho"])
    pvals = np.array(corr_data["p_values"])
    n = len(biomarkers)

    n_pairs = n * (n - 1) // 2
    alpha = 0.05 / n_pairs

    sig = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j or (not np.isnan(pvals[i, j]) and pvals[i, j] < alpha):
                sig[i, j] = True

    # Gray background for non-significant cells
    ax.imshow(
        np.ones((n, n)),
        cmap=matplotlib.colors.ListedColormap(["#e0e0e0"]),
        aspect="equal",
    )
    # Overlay significant + diagonal in RdBu_r
    display = np.where(sig, np.where(np.isnan(rho), 0, rho), np.nan)
    ax.imshow(display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            if sig[i, j] and not np.isnan(rho[i, j]):
                val = rho[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
            elif not sig[i, j]:
                ax.text(j, i, "n.s.", ha="center", va="center",
                        fontsize=7, color="#888888")

    # Cell borders (clip_on=False so edge lines aren't clipped)
    for i in range(-1, n):
        ax.axhline(i + 0.5, color="black", linewidth=0.3, clip_on=False)
        ax.axvline(i + 0.5, color="black", linewidth=0.3, clip_on=False)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(biomarkers, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(biomarkers, fontsize=8)

    # Remove outer spines
    for spine in ax.spines.values():
        spine.set_visible(False)



BASES = ["A", "C", "G", "T"]


def _count_bp(helm_str):
    from analyses.utils.helm import Helm
    parsed = Helm.parse(helm_str)
    base_pairs = [f"{b1}{b2}" for b1 in BASES for b2 in BASES]
    if parsed is None:
        return {bp: 0 for bp in base_pairs}
    counts = {bp: 0 for bp in base_pairs}
    for i in range(parsed.length - 1):
        bp = f"{parsed.bases[i]}{parsed.bases[i + 1]}"
        if bp in counts:
            counts[bp] += 1
    return counts


def _compute_spearman_with_pvals(df, biomarker_col):
    base_pairs = [f"{b1}{b2}" for b1 in BASES for b2 in BASES]
    bp_df = pd.DataFrame(df["HELM Annotation"].apply(_count_bp).tolist(), index=df.index)
    rhos, pvals = {}, {}
    for bp in base_pairs:
        valid = bp_df[bp].notna() & df[biomarker_col].notna()
        if valid.sum() < 20:
            rhos[bp], pvals[bp] = np.nan, np.nan
            continue
        rho, p = spearmanr(bp_df.loc[valid, bp], df.loc[valid, biomarker_col])
        rhos[bp] = float(rho)
        pvals[bp] = float(p)
    return rhos, pvals


def draw_base_biomarker_corr(ax):
    """Draw 4×4 heatmap: base composition (A,T,G,C) vs toxicity biomarkers."""
    from analyses.utils.helm import Helm
    from statsmodels.stats.multitest import multipletests

    _data_dir = _root / "data/oligostack/processed"

    hep = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    hep = hep[hep["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    hep["mean_ALT"] = hep["ALT"].apply(mean_of_array)

    neuro = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    neuro = neuro[neuro["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    neuro["mean_FOB"] = neuro["FOB_score"].apply(mean_of_array)
    neuro = neuro[neuro["mean_FOB"].notna()]

    contexts = [
        ("Mouse\nALT", hep[hep["species"] == "mouse"], "mean_ALT"),
        ("Rat\nALT", hep[hep["species"] == "rat"], "mean_ALT"),
        ("Mouse\nbFOB", neuro[
            (neuro["species"] == "Mouse") & (neuro["dosage_ug"] == 700)
            & (neuro["administration_method"] == "ICV")
        ], "mean_FOB"),
        ("Rat\nmFOB", neuro[
            (neuro["species"] == "Rat") & (neuro["dosage_ug"] == 3000)
        ], "mean_FOB"),
    ]

    bases = BASES  # ["A", "C", "G", "T"]

    def _count_bases(helm_str):
        parsed = Helm.parse(helm_str)
        if parsed is None:
            return {b: 0 for b in bases}
        counts = {b: 0 for b in bases}
        for base in parsed.bases:
            if base in counts:
                counts[base] += 1
        return counts

    rho_matrix = np.zeros((4, 4))  # bases × biomarkers
    pval_matrix = np.ones((4, 4))

    for j, (label, df, col) in enumerate(contexts):
        base_df = pd.DataFrame(
            df["HELM Annotation"].apply(_count_bases).tolist(), index=df.index,
        )
        for i, base in enumerate(bases):
            valid = base_df[base].notna() & df[col].notna()
            if valid.sum() < 20:
                rho_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan
                continue
            rho, p = spearmanr(base_df.loc[valid, base], df.loc[valid, col])
            rho_matrix[i, j] = float(rho)
            pval_matrix[i, j] = float(p)

    # BH correction across all 16 tests
    flat_p = pval_matrix.ravel()
    valid_mask = ~np.isnan(flat_p)
    reject = np.zeros_like(flat_p, dtype=bool)
    if valid_mask.any():
        reject[valid_mask], _, _, _ = multipletests(
            flat_p[valid_mask], alpha=0.05, method="fdr_bh",
        )
    sig_matrix = reject.reshape(pval_matrix.shape)

    # Gray background for non-significant
    ax.imshow(
        np.ones((4, 4)),
        cmap=matplotlib.colors.ListedColormap(["#e0e0e0"]),
        aspect="equal",
    )
    display = np.where(sig_matrix, rho_matrix, np.nan)
    ax.imshow(display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    for i in range(4):
        for j in range(4):
            if sig_matrix[i, j]:
                val = rho_matrix[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)
            else:
                ax.text(j, i, "n.s.", ha="center", va="center",
                        fontsize=7, color="#888888")

    # Cell borders (clip_on=False so edge lines aren't clipped)
    for i in range(-1, 4):
        ax.axhline(i + 0.5, color="black", linewidth=0.3, clip_on=False)
        ax.axvline(i + 0.5, color="black", linewidth=0.3, clip_on=False)

    biomarker_labels = [c[0] for c in contexts]
    ax.set_xticks(range(4))
    ax.set_xticklabels(biomarker_labels, fontsize=8, ha="center")
    ax.set_yticks(range(4))
    ax.set_yticklabels(bases, fontsize=9)
    ax.set_ylabel("Nucleotide base", fontsize=9)

    # Remove outer spines (cell borders define the boundary)
    for spine in ax.spines.values():
        spine.set_visible(False)



def main():
    for p in [HEPATOTOX_JSON, NEUROTOX_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `just hagerdorn` first.")

    with open(HEPATOTOX_JSON) as f:
        hep_data = json.load(f)
    with open(NEUROTOX_JSON) as f:
        neuro_data = json.load(f)

    neuro_cs = neuro_data["cross_species"]
    corr_data = hep_data["mouse_biomarker_correlations"]
    fob_cs = neuro_cs["FOB"]

    # Shared compound metrics (used by panels A and B)
    metrics, id_sets = _load_compound_metrics()

    # 2-row layout: A (4 selection-bias KDEs), B+C+D+E+colorbar
    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = gridspec.GridSpec(2, 1, figure=fig,
                           height_ratios=[0.5, 1], hspace=0.05,
                           top=0.92)

    # Row 1: A — 4 selection-bias KDE panels
    gs_a = gs[0].subgridspec(1, 4, wspace=0.35)
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(4)]
    draw_selection_bias(axes_a, metrics, id_sets)
    axes_a[0].text(-0.15, 1.15, "A", transform=axes_a[0].transAxes,
                   fontsize=14, fontweight="bold")

    # Row 2: B (cross-assay corr), C (FOB heatmap), D (biomarker corr), E (base corr)
    gs_bot = gs[1].subgridspec(
        1, 4, width_ratios=[1, 1, 1, 1], wspace=0.4,
    )

    ax_b = fig.add_subplot(gs_bot[0, 0])
    draw_cross_assay_corr(ax_b, metrics)
    ax_b.set_title("Cross-assay Spearman \u03c1", fontsize=9, pad=6)

    ax_c = fig.add_subplot(gs_bot[0, 1])
    draw_corr_matrix(ax_c, corr_data)
    ax_c.set_title("Cross-biomarker Spearman \u03c1", fontsize=9, pad=6)

    ax_d = fig.add_subplot(gs_bot[0, 2])
    draw_base_biomarker_corr(ax_d)
    ax_d.set_title("Base \u00d7 biomarker Spearman \u03c1", fontsize=9, pad=6)

    ax_e = fig.add_subplot(gs_bot[0, 3])
    draw_fob_heatmap(
        ax_e, np.array(fob_cs["mouse_values"]), np.array(fob_cs["rat_values"]),
        rho=fob_cs["spearman_rho"], pval=fob_cs["spearman_p"], n=fob_cs["n_shared"],
    )
    ax_e.set_title("Cross-species bFOB concordance", fontsize=9, pad=6)

    # Align B/C/D/E labels at the same figure-y (highest axis top)
    fig.canvas.draw()
    label_y = max(ax.get_position().y1 for ax in [ax_b, ax_c, ax_d, ax_e])
    for ax, letter in [(ax_b, "B"), (ax_c, "C"), (ax_d, "D"), (ax_e, "E")]:
        x = ax.get_position().x0 - 0.02
        fig.text(x, label_y + 0.02, letter, fontsize=14, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig5.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # ── Panel E standalone ──
    fig_e, ax_e2 = plt.subplots(figsize=(5, 5), dpi=300)
    draw_fob_heatmap(
        ax_e2, np.array(fob_cs["mouse_values"]), np.array(fob_cs["rat_values"]),
        rho=fob_cs["spearman_rho"], pval=fob_cs["spearman_p"], n=fob_cs["n_shared"],
    )
    out_e = OUT_DIR / "fig5-E.svg"
    fig_e.savefig(out_e, format="svg", bbox_inches="tight")
    fig_e.savefig(out_e.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig_e)
    print(f"Saved {out_e}")


if __name__ == "__main__":
    main()
