"""
Hypothesis 002: Sequence motifs drive ASO hepatotoxicity.

Find significant 3-mer motifs associated with elevated ALT.
Control for confounding variables (dose, gapmer design, GC content).
Correct for multiple testing (Bonferroni and FDR).

Prior work: Burdick et al. (2014) found TCC and TGC motifs associated with
hepatotoxicity in LNA-modified ASOs. This analysis tests whether similar
patterns exist in our MOE gapmer dataset.
"""

import re
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from itertools import product


# =============================================================================
# HELM PARSING FUNCTIONS
# =============================================================================

def normalize_helm(helm: str) -> str:
    """Normalize HELM format: convert double braces to single braces."""
    return helm.replace('{{', '{').replace('}}', '}')


def parse_helm_to_sequence(helm: str) -> str | None:
    """Extract base sequence from HELM annotation."""
    if not helm or not isinstance(helm, str):
        return None

    helm = normalize_helm(helm)
    match = re.search(r'\{(.+?)\}', helm)
    if not match:
        return None

    nucleotides = match.group(1).split('.')
    bases = []
    for nuc in nucleotides:
        base_match = re.search(r'\(([^)]+)\)', nuc)
        if base_match:
            base = base_match.group(1)
            # Normalize 5meC to C
            if base.lower() in ('5mec', '5-mec', '5-methylc'):
                base = 'C'
            bases.append(base[0] if len(base) > 1 else base)
    return ''.join(bases) if bases else None


def parse_helm_wings(helm: str) -> tuple[int, int, int] | None:
    """Parse HELM to get (5' wing, gap, 3' wing) lengths."""
    if not helm or not isinstance(helm, str):
        return None

    match = re.search(r'\{\{(.+?)\}\}', helm)
    if not match:
        return None

    seq = match.group(1)
    nucleotides = seq.split('.')

    sugar_types = []
    for nuc in nucleotides:
        nuc = nuc.strip()
        if not nuc:
            continue
        if nuc.startswith('[moe]'):
            sugar_types.append('MOE')
        elif nuc.startswith('d('):
            sugar_types.append('DNA')
        else:
            sugar_types.append('OTHER')

    if not sugar_types:
        return None

    # Count 5' wing (MOE from start)
    five_prime = 0
    for s in sugar_types:
        if s == 'MOE':
            five_prime += 1
        else:
            break

    # Count 3' wing (MOE from end)
    three_prime = 0
    for s in reversed(sugar_types):
        if s == 'MOE':
            three_prime += 1
        else:
            break

    # Gap in middle
    if five_prime + three_prime >= len(sugar_types):
        gap = 0
    else:
        middle = sugar_types[five_prime:len(sugar_types) - three_prime if three_prime > 0 else len(sugar_types)]
        gap = sum(1 for s in middle if s == 'DNA')

    return (five_prime, gap, three_prime)


# =============================================================================
# MOTIF ANALYSIS FUNCTIONS
# =============================================================================

def get_all_3mers(seq: str) -> list[str]:
    """Extract all 3-mer subsequences from a sequence."""
    if not seq or len(seq) < 3:
        return []
    return [seq[i:i+3] for i in range(len(seq) - 2)]


def count_motif(seq: str, motif: str) -> int:
    """Count occurrences of a motif in a sequence."""
    if not seq or not motif:
        return 0
    count = 0
    start = 0
    while True:
        pos = seq.find(motif, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def has_motif(seq: str, motif: str) -> bool:
    """Check if sequence contains motif."""
    return motif in seq if seq and motif else False


def calc_gc_content(seq: str) -> float:
    """Calculate GC content as fraction."""
    if not seq:
        return 0.0
    gc = sum(1 for b in seq if b in 'GC')
    return gc / len(seq)


def flatten_alt(alt_value):
    """Convert ALT value (possibly array) to single float (mean)."""
    if alt_value is None:
        return np.nan
    if isinstance(alt_value, (int, float)):
        return float(alt_value)
    if isinstance(alt_value, np.ndarray):
        if len(alt_value) == 0:
            return np.nan
        return float(np.mean(alt_value))
    try:
        return float(alt_value)
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    # Load data
    df = pd.read_parquet('/Users/barneyh/dphil/paper2/data/oligostack/processed/hepatictoxicity_processed.parquet')
    print(f"Total records: {len(df):,}")

    # Filter to mice, subcutaneous
    filtered = df[
        (df['species'] == 'mouse') &
        (df['adminstration_method'] == 'subcutaneous')
    ].copy()
    print(f"Mouse, subcutaneous: {len(filtered):,}")

    # Process ALT and HELM
    filtered['ALT_mean'] = filtered['ALT'].apply(flatten_alt)
    filtered['sequence'] = filtered['HELM Annotation'].apply(parse_helm_to_sequence)
    filtered['wings'] = filtered['HELM Annotation'].apply(parse_helm_wings)

    # Filter valid data
    valid = filtered[
        filtered['ALT_mean'].notna() &
        filtered['sequence'].notna() &
        filtered['wings'].notna()
    ].copy()
    print(f"With valid ALT, sequence, wings: {len(valid):,}")

    # Extract features for controlling
    valid['seq_length'] = valid['sequence'].str.len()
    valid['gc_content'] = valid['sequence'].apply(calc_gc_content)
    valid['wing5_len'] = valid['wings'].apply(lambda x: x[0] if x else None)
    valid['gap_len'] = valid['wings'].apply(lambda x: x[1] if x else None)
    valid['wing3_len'] = valid['wings'].apply(lambda x: x[2] if x else None)

    # Handle dosage - extract numeric value
    def parse_dosage(d):
        if pd.isna(d):
            return np.nan
        if isinstance(d, (int, float)):
            return float(d)
        return np.nan

    valid['dose'] = valid['dosage_mg_per_kg'].apply(parse_dosage)

    # Aggregate by Compound ID (mean ALT, first for sequence features)
    agg_cols = {
        'ALT_mean': 'mean',
        'sequence': 'first',
        'seq_length': 'first',
        'gc_content': 'first',
        'wing5_len': 'first',
        'gap_len': 'first',
        'wing3_len': 'first',
        'dose': 'mean',
        'HELM Annotation': 'first'
    }
    agg = valid.groupby('Compound ID').agg(agg_cols).reset_index()
    agg.rename(columns={'ALT_mean': 'ALT'}, inplace=True)
    print(f"\nUnique ASOs: {len(agg):,}")

    # Generate all possible 3-mers
    bases = ['A', 'T', 'G', 'C']
    all_3mers = [''.join(p) for p in product(bases, repeat=3)]
    print(f"Testing {len(all_3mers)} possible 3-mers")

    # Calculate presence/count of each 3-mer for each ASO
    for motif in all_3mers:
        agg[f'has_{motif}'] = agg['sequence'].apply(lambda s: has_motif(s, motif))
        agg[f'count_{motif}'] = agg['sequence'].apply(lambda s: count_motif(s, motif))

    # ==========================================================================
    # STATISTICAL TESTS
    # ==========================================================================

    print("\n" + "="*80)
    print("UNIVARIATE ANALYSIS: 3-mer presence vs ALT")
    print("="*80)

    results = []
    for motif in all_3mers:
        has_col = f'has_{motif}'
        present = agg[agg[has_col]]['ALT']
        absent = agg[~agg[has_col]]['ALT']

        if len(present) < 5 or len(absent) < 5:
            continue

        # Mann-Whitney U
        try:
            U, p = stats.mannwhitneyu(present, absent, alternative='two-sided')
        except ValueError:
            continue

        # Effect size: difference in medians
        median_present = present.median()
        median_absent = absent.median()
        median_diff = median_present - median_absent

        # Spearman correlation with count
        count_col = f'count_{motif}'
        rho, p_spearman = stats.spearmanr(agg[count_col], agg['ALT'])

        results.append({
            'motif': motif,
            'n_present': len(present),
            'n_absent': len(absent),
            'median_present': median_present,
            'median_absent': median_absent,
            'median_diff': median_diff,
            'fold_change': median_present / median_absent if median_absent > 0 else np.nan,
            'U': U,
            'p_mw': p,
            'spearman_rho': rho,
            'p_spearman': p_spearman
        })

    results_df = pd.DataFrame(results)

    # Multiple testing correction
    results_df['p_mw_bonf'] = results_df['p_mw'] * len(results_df)
    results_df['p_mw_bonf'] = results_df['p_mw_bonf'].clip(upper=1.0)

    _, p_fdr, _, _ = multipletests(results_df['p_mw'], method='fdr_bh')
    results_df['p_mw_fdr'] = p_fdr

    # Sort by p-value
    results_df = results_df.sort_values('p_mw')

    print("\nTop 20 motifs by Mann-Whitney p-value:")
    print("-"*100)
    print(f"{'Motif':<8} {'n_pres':>8} {'n_abs':>8} {'med_pres':>10} {'med_abs':>10} {'fold':>8} {'p_raw':>12} {'p_FDR':>12}")
    print("-"*100)
    for _, row in results_df.head(20).iterrows():
        print(f"{row['motif']:<8} {row['n_present']:>8} {row['n_absent']:>8} "
              f"{row['median_present']:>10.1f} {row['median_absent']:>10.1f} "
              f"{row['fold_change']:>8.2f} {row['p_mw']:>12.2e} {row['p_mw_fdr']:>12.2e}")

    # Significant after FDR correction
    sig_fdr = results_df[results_df['p_mw_fdr'] < 0.05]
    print(f"\n\nSignificant after FDR correction (q < 0.05): {len(sig_fdr)}")
    if len(sig_fdr) > 0:
        print("\nSignificant motifs:")
        for _, row in sig_fdr.iterrows():
            direction = "↑ ALT" if row['median_diff'] > 0 else "↓ ALT"
            print(f"  {row['motif']}: {direction}, fold={row['fold_change']:.2f}, p_FDR={row['p_mw_fdr']:.2e}")

    # ==========================================================================
    # CONTROLLED ANALYSIS: Partial correlation controlling for confounders
    # ==========================================================================

    print("\n" + "="*80)
    print("CONTROLLED ANALYSIS: Adjusting for dose, GC content, sequence length")
    print("="*80)

    # Use linear regression to control for confounders
    from scipy.stats import spearmanr

    # Subset with complete covariate data
    covars = ['dose', 'gc_content', 'seq_length']
    complete = agg.dropna(subset=covars + ['ALT']).copy()
    print(f"ASOs with complete covariate data: {len(complete)}")

    # For each motif, run partial correlation
    # Residualize ALT against covariates, then correlate with motif
    from numpy.linalg import lstsq

    def partial_correlation(motif_col, alt_col, covar_cols, data):
        """Calculate partial Spearman correlation controlling for covariates."""
        # Get residuals of ALT after regressing on covariates
        X = data[covar_cols].values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept
        y_alt = data[alt_col].values

        # Residualize ALT
        coeffs, _, _, _ = lstsq(X, y_alt, rcond=None)
        alt_resid = y_alt - X @ coeffs

        # Residualize motif count
        y_motif = data[motif_col].values
        coeffs, _, _, _ = lstsq(X, y_motif, rcond=None)
        motif_resid = y_motif - X @ coeffs

        # Spearman on residuals
        rho, p = spearmanr(motif_resid, alt_resid)
        return rho, p

    controlled_results = []
    for motif in all_3mers:
        count_col = f'count_{motif}'
        if count_col not in complete.columns:
            continue

        try:
            rho, p = partial_correlation(count_col, 'ALT', covars, complete)
            controlled_results.append({
                'motif': motif,
                'partial_rho': rho,
                'partial_p': p
            })
        except Exception:
            continue

    controlled_df = pd.DataFrame(controlled_results)
    _, p_fdr_ctrl, _, _ = multipletests(controlled_df['partial_p'], method='fdr_bh')
    controlled_df['partial_p_fdr'] = p_fdr_ctrl
    controlled_df = controlled_df.sort_values('partial_p')

    print("\nTop 20 motifs by partial correlation (controlling for dose, GC, length):")
    print("-"*70)
    print(f"{'Motif':<8} {'partial_rho':>12} {'p_raw':>12} {'p_FDR':>12}")
    print("-"*70)
    for _, row in controlled_df.head(20).iterrows():
        print(f"{row['motif']:<8} {row['partial_rho']:>12.3f} {row['partial_p']:>12.2e} {row['partial_p_fdr']:>12.2e}")

    sig_ctrl = controlled_df[controlled_df['partial_p_fdr'] < 0.05]
    print(f"\nSignificant after controlling + FDR: {len(sig_ctrl)}")

    # ==========================================================================
    # CHECK KNOWN TOXIC MOTIFS (TCC, TGC from Burdick et al.)
    # ==========================================================================

    print("\n" + "="*80)
    print("VALIDATION: Known toxic motifs from Burdick et al. (2014)")
    print("="*80)

    for motif in ['TCC', 'TGC']:
        has_col = f'has_{motif}'
        if has_col not in agg.columns:
            print(f"{motif}: Not found in data")
            continue

        present = agg[agg[has_col]]['ALT']
        absent = agg[~agg[has_col]]['ALT']

        print(f"\n{motif}:")
        print(f"  Present: n={len(present)}, median ALT={present.median():.1f}")
        print(f"  Absent:  n={len(absent)}, median ALT={absent.median():.1f}")

        if len(present) >= 5 and len(absent) >= 5:
            U, p = stats.mannwhitneyu(present, absent, alternative='two-sided')
            print(f"  Mann-Whitney U={U:.0f}, p={p:.2e}")

            # Get FDR-corrected p-value from results
            motif_row = results_df[results_df['motif'] == motif]
            if len(motif_row) > 0:
                print(f"  FDR-corrected p={motif_row['p_mw_fdr'].values[0]:.2e}")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================

    print("\n" + "="*80)
    print("Generating visualizations...")

    # Plot 1: Volcano plot of all 3-mers
    fig, ax = plt.subplots(figsize=(10, 8))

    results_df['neg_log_p'] = -np.log10(results_df['p_mw'])
    results_df['log_fold'] = np.log2(results_df['fold_change'])

    # Color by significance
    colors = ['red' if p < 0.05 else 'gray' for p in results_df['p_mw_fdr']]

    ax.scatter(results_df['log_fold'], results_df['neg_log_p'], c=colors, alpha=0.6, s=50)

    # Label significant points
    for _, row in results_df[results_df['p_mw_fdr'] < 0.05].iterrows():
        ax.annotate(row['motif'], (row['log_fold'], row['neg_log_p']),
                   fontsize=9, ha='center', va='bottom')

    # Highlight known toxic motifs
    for motif in ['TCC', 'TGC']:
        motif_row = results_df[results_df['motif'] == motif]
        if len(motif_row) > 0:
            ax.scatter(motif_row['log_fold'], motif_row['neg_log_p'],
                      c='blue', s=100, marker='s', label=f'{motif} (literature)')

    ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel('log2(Fold Change in median ALT)', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title('3-mer Motifs vs Hepatotoxicity (ALT)\nVolcano Plot', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/figures/volcano_plot.png', dpi=150)
    plt.close()

    # Plot 2: Boxplots for top motifs
    top_motifs = results_df.head(6)['motif'].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, motif in enumerate(top_motifs):
        ax = axes[idx]
        has_col = f'has_{motif}'

        present = agg[agg[has_col]]['ALT']
        absent = agg[~agg[has_col]]['ALT']

        data = [absent, present]
        labels = [f'Absent\n(n={len(absent)})', f'Present\n(n={len(present)})']

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')

        motif_row = results_df[results_df['motif'] == motif].iloc[0]
        ax.set_title(f'{motif}\np_FDR={motif_row["p_mw_fdr"]:.2e}', fontsize=11)
        ax.set_ylabel('ALT (IU/L)')
        ax.set_yscale('log')
        ax.axhline(50, color='red', linestyle='--', alpha=0.3)

    plt.suptitle('Top 6 3-mer Motifs Associated with ALT', fontsize=14)
    plt.tight_layout()
    plt.savefig('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/figures/top_motifs_boxplot.png', dpi=150)
    plt.close()

    # Plot 3: Heatmap of motif effects
    fig, ax = plt.subplots(figsize=(12, 10))

    # Reshape for heatmap: first two bases as rows, third base as columns within groups
    pivot_data = results_df.copy()
    pivot_data['first2'] = pivot_data['motif'].str[:2]
    pivot_data['third'] = pivot_data['motif'].str[2]

    # Use log fold change for heatmap
    heatmap_df = pivot_data.pivot(index='first2', columns='third', values='log_fold')

    sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'log2(Fold Change ALT)'})
    ax.set_xlabel('3rd Base', fontsize=12)
    ax.set_ylabel('First 2 Bases', fontsize=12)
    ax.set_title('3-mer Motif Effects on ALT\n(positive = higher ALT when motif present)', fontsize=14)

    plt.tight_layout()
    plt.savefig('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/figures/motif_heatmap.png', dpi=150)
    plt.close()

    print("Figures saved to figures/")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nDataset: {len(agg)} unique ASOs (mouse, subcutaneous)")
    print(f"3-mers tested: {len(results_df)}")
    print(f"Significant (FDR < 0.05): {len(sig_fdr)}")
    print(f"Significant after controlling for confounders: {len(sig_ctrl)}")

    if len(sig_fdr) > 0:
        print("\nSignificant motifs (univariate, FDR < 0.05):")
        for _, row in sig_fdr.iterrows():
            direction = "↑" if row['median_diff'] > 0 else "↓"
            print(f"  {row['motif']}: {direction} ALT, fold={row['fold_change']:.2f}, p_FDR={row['p_mw_fdr']:.2e}")

    # Save results to CSV
    results_df.to_csv('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/motif_results.csv', index=False)
    controlled_df.to_csv('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/motif_results_controlled.csv', index=False)
    print("\nResults saved to CSV files.")


if __name__ == "__main__":
    main()
