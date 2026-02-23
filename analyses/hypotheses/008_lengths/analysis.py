#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "scipy>=1.10",
#     "matplotlib>=3.7",
#     "pyarrow>=14.0",
# ]
# ///
"""
Hypothesis 008: ASO 5' Positional Bias

Tests whether gapmer ASO efficacy decreases with distance from the 5' end
of transcripts, consistent with a co-transcriptional accessibility model.

Panel A: Metatranscript plot (40 bins, all ASOs)
Panel B: Meta-intron plot (30 bins, intronic ASOs only)
Panel C: Gene length stratification (5' vs 3' half by pre-mRNA length)

Run with: uv run analyses/hypotheses/008_lengths/analysis.py
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data/oligostack/processed/in_vitro_inhibition_processed_with_genomic_data.parquet"
TRANSCRIPT_CACHE_PATH = PROJECT_ROOT / "data/oligostack/processed/transcript_cache.json"
GTF_PATH = PROJECT_ROOT / "kinetics_model/data/reference/canonical_transcripts.gtf"
FIGURES_DIR = SCRIPT_DIR / "figures"


# =============================================================================
# GTF PARSING
# =============================================================================

def parse_gtf(gtf_path, transcript_ids):
    """Parse GTF for transcript spans and exon coordinates.

    Returns:
        transcripts: {tx_id: {"start": int, "end": int, "strand": str}}
        exons: {tx_id: [(start, end), ...]}  sorted by genomic position
    """
    transcripts = {}
    exons = {}

    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature = parts[2]
            if feature not in ("transcript", "exon"):
                continue

            m = re.search(r'transcript_id "([^"]+)"', parts[8])
            if not m:
                continue
            tx_id = m.group(1)

            if tx_id not in transcript_ids:
                continue

            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]

            if feature == "transcript":
                transcripts[tx_id] = {"start": start, "end": end, "strand": strand}
            elif feature == "exon":
                if tx_id not in exons:
                    exons[tx_id] = []
                exons[tx_id].append((start, end))

    # Sort exons by start position
    for tx_id in exons:
        exons[tx_id].sort()

    return transcripts, exons


def classify_exon_intron(aso_start, aso_end, tx_info, exon_list):
    """Classify ASO as exonic or intronic and compute intron-relative position.

    aso_start, aso_end: 0-based positions along pre-mRNA
    tx_info: {"start": genomic_start, "end": genomic_end, "strand": "+"/"-"}
    exon_list: [(genomic_start, genomic_end), ...] sorted
    """
    tx_start = tx_info["start"]
    tx_end = tx_info["end"]
    strand = tx_info["strand"]

    # Convert ASO pre-mRNA midpoint to genomic coordinate
    aso_mid = (aso_start + aso_end) // 2
    if strand == "+":
        aso_genomic_mid = tx_start + aso_mid
    else:
        aso_genomic_mid = tx_end - aso_mid

    # Check if midpoint falls in any exon
    for ex_start, ex_end in exon_list:
        if ex_start <= aso_genomic_mid <= ex_end:
            return {"region": "exonic"}

    # Find which intron it falls in
    for i in range(len(exon_list) - 1):
        intron_start = exon_list[i][1] + 1
        intron_end = exon_list[i + 1][0] - 1
        if intron_start <= aso_genomic_mid <= intron_end:
            intron_len = intron_end - intron_start + 1
            if intron_len > 0:
                if strand == "+":
                    rel = (aso_genomic_mid - intron_start) / intron_len
                    bp_from_5ss = aso_genomic_mid - intron_start
                else:
                    rel = (intron_end - aso_genomic_mid) / intron_len
                    bp_from_5ss = intron_end - aso_genomic_mid
                return {"region": "intronic", "intron_rel_pos": rel, "bp_from_5ss": bp_from_5ss}

    return {"region": "unknown"}


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HYPOTHESIS 008: ASO 5' POSITIONAL BIAS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # [1] Load in vitro data
    # -------------------------------------------------------------------------
    print("\n[1] Loading in vitro inhibition data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"    Total rows: {len(df):,}")

    # Filter to rows with valid matches and inhibition
    df = df[
        df["aso_transcript_matches"].notna()
        & (df["aso_match_count"] > 0)
        & df["Inhibition_pct"].notna()
        & df["canonical_transcript_id"].notna()
    ].copy()
    print(f"    After filtering (valid matches + inhibition): {len(df):,}")

    # Clip inhibition to [0, 100]
    df["Inhibition_pct"] = df["Inhibition_pct"].clip(0, 100)

    # Parse aso_start and aso_end from JSON (take first match)
    def parse_first_match(json_str):
        try:
            matches = json.loads(json_str)
            if matches and len(matches) > 0:
                return matches[0].get("start"), matches[0].get("end")
        except (json.JSONDecodeError, TypeError):
            pass
        return None, None

    positions = df["aso_transcript_matches"].apply(parse_first_match)
    df["aso_start"] = positions.apply(lambda x: x[0])
    df["aso_end"] = positions.apply(lambda x: x[1])
    df = df[df["aso_start"].notna() & df["aso_end"].notna()].copy()
    df["aso_start"] = df["aso_start"].astype(int)
    df["aso_end"] = df["aso_end"].astype(int)
    print(f"    With parsed ASO positions: {len(df):,}")

    # -------------------------------------------------------------------------
    # [2] Load transcript lengths (pre-mRNA = genomic span)
    # -------------------------------------------------------------------------
    print("\n[2] Loading transcript lengths from cache...")
    with open(TRANSCRIPT_CACHE_PATH) as f:
        tx_cache = json.load(f)

    tx_lengths = {tx_id: len(info["sequence"]) for tx_id, info in tx_cache.items()}
    print(f"    Transcripts in cache: {len(tx_lengths):,}")

    df["transcript_length"] = df["canonical_transcript_id"].map(tx_lengths)
    df = df[df["transcript_length"].notna() & (df["transcript_length"] > 0)].copy()
    df["transcript_length"] = df["transcript_length"].astype(int)
    print(f"    With transcript length: {len(df):,}")

    # Compute relative position along pre-mRNA
    df["rel_pos"] = df["aso_start"] / df["transcript_length"]
    df["rel_pos"] = df["rel_pos"].clip(0, 1)

    # -------------------------------------------------------------------------
    # [3] Aggregate to one row per Compound ID
    # -------------------------------------------------------------------------
    print("\n[3] Aggregating per compound...")
    agg = df.groupby("Compound ID").agg(
        Inhibition_pct=("Inhibition_pct", "median"),
        rel_pos=("rel_pos", "first"),
        aso_start=("aso_start", "first"),
        aso_end=("aso_end", "first"),
        transcript_length=("transcript_length", "first"),
        canonical_transcript_id=("canonical_transcript_id", "first"),
    ).reset_index()
    print(f"    Unique ASOs: {len(agg):,}")
    print(f"    Unique transcripts: {agg['canonical_transcript_id'].nunique():,}")
    print(f"    Rel position range: {agg['rel_pos'].min():.4f} - {agg['rel_pos'].max():.4f}")
    print(f"    Inhibition_pct range: {agg['Inhibition_pct'].min():.1f} - {agg['Inhibition_pct'].max():.1f}")

    # -------------------------------------------------------------------------
    # [4] Parse GTF for exon/intron classification
    # -------------------------------------------------------------------------
    print("\n[4] Parsing GTF for exon/intron classification...")
    target_tx_ids = set(agg["canonical_transcript_id"].unique())
    tx_spans, tx_exons = parse_gtf(GTF_PATH, target_tx_ids)
    print(f"    Transcripts found in GTF: {len(tx_spans):,}")
    print(f"    Transcripts with exons: {len(tx_exons):,}")

    # Classify each ASO
    classifications = []
    for _, row in agg.iterrows():
        tx_id = row["canonical_transcript_id"]
        if tx_id in tx_spans and tx_id in tx_exons:
            result = classify_exon_intron(
                row["aso_start"], row["aso_end"],
                tx_spans[tx_id], tx_exons[tx_id],
            )
        else:
            result = {"region": "unknown"}
        classifications.append(result)

    agg["region"] = [c["region"] for c in classifications]
    agg["intron_rel_pos"] = [c.get("intron_rel_pos", np.nan) for c in classifications]
    agg["bp_from_5ss"] = [c.get("bp_from_5ss", np.nan) for c in classifications]

    n_exonic = (agg["region"] == "exonic").sum()
    n_intronic = (agg["region"] == "intronic").sum()
    n_unknown = (agg["region"] == "unknown").sum()
    print(f"    Exonic ASOs: {n_exonic:,}")
    print(f"    Intronic ASOs: {n_intronic:,}")
    print(f"    Unknown: {n_unknown:,}")

    # -------------------------------------------------------------------------
    # [5] Statistical tests
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STATISTICAL RESULTS")
    print("=" * 70)

    # Panel A: Spearman correlation (rel_pos vs Inhibition_pct)
    rho_a, p_a = stats.spearmanr(agg["rel_pos"], agg["Inhibition_pct"])
    print(f"\nPanel A - Transcript (all ASOs, n={len(agg):,}):")
    print(f"  Spearman rho = {rho_a:.4f}, p = {p_a:.2e}")
    if rho_a < 0:
        print("  Direction: 5' ASOs have HIGHER knockdown (supports co-transcriptional model)")
    else:
        print("  Direction: 5' ASOs have LOWER knockdown (opposes co-transcriptional model)")

    # Panel B: Spearman correlation (intron_rel_pos vs Inhibition_pct)
    intronic = agg[agg["region"] == "intronic"].dropna(subset=["intron_rel_pos"])
    if len(intronic) > 10:
        rho_b, p_b = stats.spearmanr(intronic["intron_rel_pos"], intronic["Inhibition_pct"])
        print(f"\nPanel B - Intron (intronic ASOs, n={len(intronic):,}):")
        print(f"  Spearman rho = {rho_b:.4f}, p = {p_b:.2e}")
    else:
        rho_b, p_b = np.nan, np.nan
        print(f"\nPanel B - Intron: insufficient intronic ASOs (n={len(intronic)})")

    # Panel C: Gene-length stratified 5' vs 3' comparison
    print(f"\nPanel C - Gene-length stratification:")
    length_bins = [0, 10_000, 30_000, 60_000, 100_000, float("inf")]
    length_labels = ["<10kb", "10-30kb", "30-60kb", "60-100kb", ">100kb"]
    agg["length_bin"] = pd.cut(
        agg["transcript_length"], bins=length_bins, labels=length_labels, right=False,
    )
    agg["half"] = np.where(agg["rel_pos"] < 0.5, "5' half", "3' half")

    for label in length_labels:
        subset = agg[agg["length_bin"] == label]
        five = subset[subset["half"] == "5' half"]["Inhibition_pct"]
        three = subset[subset["half"] == "3' half"]["Inhibition_pct"]
        if len(five) > 5 and len(three) > 5:
            u, p = stats.mannwhitneyu(five, three, alternative="two-sided")
            delta = five.mean() - three.mean()
            print(
                f"  {label:>8s}: 5' mean={five.mean():.1f}% (n={len(five):,}), "
                f"3' mean={three.mean():.1f}% (n={len(three):,}), "
                f"delta={delta:+.1f}%, MWU p={p:.2e}"
            )
        else:
            print(f"  {label:>8s}: insufficient data (5'={len(five)}, 3'={len(three)})")

    # -------------------------------------------------------------------------
    # [6] Generate figures
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    ax_tl, ax_tr = axes[0]  # transcript relative, transcript absolute
    ax_bl, ax_br = axes[1]  # intron relative, intron absolute

    rng = np.random.default_rng(42)

    # Compute absolute transcript position in kb
    agg["aso_start_kb"] = agg["aso_start"] / 1000

    n_scatter = min(2000, len(agg))
    scatter_idx = rng.choice(len(agg), n_scatter, replace=False)
    scatter_data = agg.iloc[scatter_idx]
    cap_tx_kb = agg["aso_start_kb"].quantile(0.99)
    agg_abs = agg[agg["aso_start_kb"] <= cap_tx_kb]
    rho_a_abs, p_a_abs = stats.spearmanr(agg["aso_start"], agg["Inhibition_pct"])

    # Helper to plot binned median + IQR ribbon + scatter
    def plot_panel(ax, x_scatter, y_scatter, x_all, y_all, n_bins, color, x_range=None):
        ax.scatter(x_scatter, y_scatter, alpha=0.08, s=8, c=color, edgecolors="none", rasterized=True)
        bins = pd.cut(x_all, bins=n_bins, labels=False)
        stats_df = y_all.groupby(bins).agg(
            ["median", lambda v: v.quantile(0.25), lambda v: v.quantile(0.75)],
        )
        stats_df.columns = ["median", "q25", "q75"]
        if x_range is None:
            x_range = (x_all.min(), x_all.max())
        centers = np.linspace(
            x_range[0] + (x_range[1] - x_range[0]) / (2 * n_bins),
            x_range[1] - (x_range[1] - x_range[0]) / (2 * n_bins),
            n_bins,
        )
        valid = stats_df.dropna()
        c = centers[valid.index.astype(int)]
        ax.fill_between(c, valid["q25"], valid["q75"], alpha=0.3, color=color)
        ax.plot(c, valid["median"], color=color, linewidth=2)

    # ----- Top left: Transcript relative -----
    plot_panel(ax_tl, scatter_data["rel_pos"], scatter_data["Inhibition_pct"],
               agg["rel_pos"], agg["Inhibition_pct"], 40, "steelblue", x_range=(0, 1))
    ax_tl.set_xlabel("Relative position (5' -> 3')")
    ax_tl.set_ylabel("Inhibition (%)")
    ax_tl.set_title(f"Transcript-level relative (n = {len(agg):,}, rho = {rho_a:.3f}, p = {p_a:.2e})")
    ax_tl.set_xlim(0, 1)

    # ----- Top right: Transcript absolute -----
    scatter_abs = scatter_data[scatter_data["aso_start_kb"] <= cap_tx_kb]
    plot_panel(ax_tr, scatter_abs["aso_start_kb"], scatter_abs["Inhibition_pct"],
               agg_abs["aso_start_kb"], agg_abs["Inhibition_pct"], 40, "steelblue")
    ax_tr.set_xlabel("Distance from 5' end (kb)")
    ax_tr.set_ylabel("Inhibition (%)")
    ax_tr.set_title(f"Transcript-level absolute (rho = {rho_a_abs:.3f}, p = {p_a_abs:.2e})")

    # ----- Prepare intronic data -----
    intronic = intronic.copy()
    intronic = intronic.dropna(subset=["bp_from_5ss"])
    intronic["bp_from_5ss_kb"] = intronic["bp_from_5ss"] / 1000
    cap_int_kb = intronic["bp_from_5ss_kb"].quantile(0.99)
    intronic_abs = intronic[intronic["bp_from_5ss_kb"] <= cap_int_kb]

    n_scatter_b = min(2000, len(intronic))
    scatter_idx_b = rng.choice(len(intronic), n_scatter_b, replace=False)
    scatter_int = intronic.iloc[scatter_idx_b]
    scatter_int_abs = scatter_int[scatter_int["bp_from_5ss_kb"] <= cap_int_kb]

    rho_b_abs, p_b_abs = stats.spearmanr(intronic["bp_from_5ss"], intronic["Inhibition_pct"])

    # ----- Bottom left: Intron relative -----
    plot_panel(ax_bl, scatter_int["intron_rel_pos"], scatter_int["Inhibition_pct"],
               intronic["intron_rel_pos"], intronic["Inhibition_pct"], 30, "darkorange", x_range=(0, 1))
    ax_bl.set_xlabel("Relative position within intron (5' -> 3')")
    ax_bl.set_ylabel("Inhibition (%)")
    ax_bl.set_title(f"Intron-level relative (n = {len(intronic):,}, rho = {rho_b:.3f}, p = {p_b:.2e})")
    ax_bl.set_xlim(0, 1)

    # ----- Bottom right: Intron absolute -----
    plot_panel(ax_br, scatter_int_abs["bp_from_5ss_kb"], scatter_int_abs["Inhibition_pct"],
               intronic_abs["bp_from_5ss_kb"], intronic_abs["Inhibition_pct"], 30, "darkorange")
    ax_br.set_xlabel("Distance from 5' splice site (kb)")
    ax_br.set_ylabel("Inhibition (%)")
    ax_br.set_title(f"Intron-level absolute (rho = {rho_b_abs:.3f}, p = {p_b_abs:.2e})")

    plt.savefig(FIGURES_DIR / "positional_bias.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "positional_bias.svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'positional_bias.png'}")
    print(f"Saved: {FIGURES_DIR / 'positional_bias.svg'}")

    # -------------------------------------------------------------------------
    # [7] Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if rho_a < 0 and p_a < 0.05:
        print(f"\nVerdict: SUPPORTED")
        print(f"  ASOs targeting 5' regions show higher knockdown (rho={rho_a:.4f}, p={p_a:.2e})")
    elif rho_a > 0 and p_a < 0.05:
        print(f"\nVerdict: OPPOSITE")
        print(f"  ASOs targeting 3' regions show higher knockdown (rho={rho_a:.4f}, p={p_a:.2e})")
    else:
        print(f"\nVerdict: NOT SUPPORTED")
        print(f"  No significant positional bias detected (rho={rho_a:.4f}, p={p_a:.2e})")

    print(f"\n  Panel A (transcript): rho = {rho_a:.4f}, p = {p_a:.2e}")
    if not np.isnan(rho_b):
        print(f"  Panel B (intron):    rho = {rho_b:.4f}, p = {p_b:.2e}")
    print(f"\nAnalysis complete.")


if __name__ == "__main__":
    main()
