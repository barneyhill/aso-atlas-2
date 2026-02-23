"""
Extended motif analysis: Test all 3-mers against ALL biomarkers.

Biomarkers:
- ALT: Alanine aminotransferase (liver)
- AST: Aspartate aminotransferase (liver/muscle)
- BUN: Blood urea nitrogen (kidney)
- CREA: Creatinine (kidney)
- TBIL: Total bilirubin (liver)
- ALB: Albumin (liver synthetic function)
- PC_ratio: Protein/creatinine ratio (kidney)
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
from itertools import product


BIOMARKERS = ['ALT', 'AST', 'BUN', 'CREA', 'TBIL', 'ALB', 'PC_ratio']


def normalize_helm(helm: str) -> str:
    return helm.replace('{{', '{').replace('}}', '}')


def parse_helm_to_sequence(helm: str) -> str | None:
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
            if base.lower() in ('5mec', '5-mec', '5-methylc'):
                base = 'C'
            bases.append(base[0] if len(base) > 1 else base)
    return ''.join(bases) if bases else None


def has_motif(seq: str, motif: str) -> bool:
    return motif in seq if seq and motif else False


def flatten_value(val):
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, np.ndarray):
        if len(val) == 0:
            return np.nan
        return float(np.mean(val))
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def main():
    # Load data
    df = pd.read_parquet('/Users/barneyh/dphil/paper2/data/oligostack/processed/hepatictoxicity_processed.parquet')
    print(f"Total records: {len(df):,}")

    # Filter
    filtered = df[
        (df['species'] == 'mouse') &
        (df['adminstration_method'] == 'subcutaneous')
    ].copy()
    print(f"Mouse, subcutaneous: {len(filtered):,}")

    # Parse sequences
    filtered['sequence'] = filtered['HELM Annotation'].apply(parse_helm_to_sequence)
    filtered = filtered[filtered['sequence'].notna()].copy()
    print(f"With valid sequence: {len(filtered):,}")

    # Flatten all biomarkers
    for bm in BIOMARKERS:
        filtered[f'{bm}_mean'] = filtered[bm].apply(flatten_value)

    # Generate all 3-mers
    bases = ['A', 'T', 'G', 'C']
    all_3mers = [''.join(p) for p in product(bases, repeat=3)]

    # Precompute motif presence
    motif_presence = {}
    for motif in all_3mers:
        motif_presence[motif] = filtered['sequence'].apply(lambda s: has_motif(s, motif))

    # Aggregate by Compound ID
    agg_cols = {'sequence': 'first', 'HELM Annotation': 'first'}
    for bm in BIOMARKERS:
        agg_cols[f'{bm}_mean'] = 'mean'

    agg = filtered.groupby('Compound ID').agg(agg_cols).reset_index()
    print(f"\nUnique ASOs: {len(agg):,}")

    # Recompute motif presence on aggregated data
    for motif in all_3mers:
        agg[f'has_{motif}'] = agg['sequence'].apply(lambda s: has_motif(s, motif))

    # Test each motif against each biomarker
    print("\n" + "="*100)
    print("TESTING ALL 3-MERS AGAINST ALL BIOMARKERS")
    print("="*100)

    all_results = []

    for bm in BIOMARKERS:
        bm_col = f'{bm}_mean'
        valid_bm = agg[agg[bm_col].notna()].copy()
        n_valid = len(valid_bm)

        if n_valid < 100:
            print(f"\n{bm}: Skipping (only {n_valid} valid values)")
            continue

        print(f"\n{bm}: Testing {len(all_3mers)} motifs (n={n_valid} ASOs)")

        for motif in all_3mers:
            has_col = f'has_{motif}'
            present = valid_bm[valid_bm[has_col]][bm_col]
            absent = valid_bm[~valid_bm[has_col]][bm_col]

            if len(present) < 5 or len(absent) < 5:
                continue

            try:
                U, p = stats.mannwhitneyu(present, absent, alternative='two-sided')
            except ValueError:
                continue

            median_present = present.median()
            median_absent = absent.median()

            if median_absent > 0:
                fold_change = median_present / median_absent
            else:
                fold_change = np.nan

            all_results.append({
                'biomarker': bm,
                'motif': motif,
                'n_present': len(present),
                'n_absent': len(absent),
                'median_present': median_present,
                'median_absent': median_absent,
                'fold_change': fold_change,
                'p_raw': p
            })

    results_df = pd.DataFrame(all_results)

    # FDR correction within each biomarker
    results_df['p_fdr'] = np.nan
    for bm in BIOMARKERS:
        mask = results_df['biomarker'] == bm
        if mask.sum() > 0:
            _, p_fdr, _, _ = multipletests(results_df.loc[mask, 'p_raw'], method='fdr_bh')
            results_df.loc[mask, 'p_fdr'] = p_fdr

    # Also do global FDR across all tests
    _, p_fdr_global, _, _ = multipletests(results_df['p_raw'], method='fdr_bh')
    results_df['p_fdr_global'] = p_fdr_global

    # Sort by fold change (extreme values)
    results_df['abs_log_fold'] = np.abs(np.log2(results_df['fold_change']))
    results_df = results_df.sort_values('abs_log_fold', ascending=False)

    # Print top results by fold change
    print("\n" + "="*100)
    print("TOP 30 MOTIF-BIOMARKER COMBINATIONS BY FOLD CHANGE")
    print("="*100)
    print(f"{'Biomarker':<10} {'Motif':<8} {'n_pres':>8} {'med_pres':>12} {'med_abs':>12} {'Fold':>8} {'p_raw':>12} {'p_FDR':>12}")
    print("-"*100)

    for _, row in results_df.head(30).iterrows():
        print(f"{row['biomarker']:<10} {row['motif']:<8} {row['n_present']:>8} "
              f"{row['median_present']:>12.2f} {row['median_absent']:>12.2f} "
              f"{row['fold_change']:>8.2f} {row['p_raw']:>12.2e} {row['p_fdr']:>12.2e}")

    # Summary by biomarker
    print("\n" + "="*100)
    print("SUMMARY BY BIOMARKER")
    print("="*100)

    summary_data = []
    for bm in BIOMARKERS:
        bm_results = results_df[results_df['biomarker'] == bm]
        if len(bm_results) == 0:
            continue

        sig = bm_results[bm_results['p_fdr'] < 0.05]
        max_fold = bm_results.loc[bm_results['fold_change'].idxmax()] if len(bm_results) > 0 else None
        min_fold = bm_results.loc[bm_results['fold_change'].idxmin()] if len(bm_results) > 0 else None

        print(f"\n{bm}:")
        print(f"  Significant motifs (FDR<0.05): {len(sig)}/64")
        if max_fold is not None:
            print(f"  Highest fold: {max_fold['motif']} = {max_fold['fold_change']:.2f}x (p_FDR={max_fold['p_fdr']:.2e})")
        if min_fold is not None:
            print(f"  Lowest fold:  {min_fold['motif']} = {min_fold['fold_change']:.2f}x (p_FDR={min_fold['p_fdr']:.2e})")

        summary_data.append({
            'biomarker': bm,
            'n_sig': len(sig),
            'max_motif': max_fold['motif'] if max_fold is not None else None,
            'max_fold': max_fold['fold_change'] if max_fold is not None else None,
            'min_motif': min_fold['motif'] if min_fold is not None else None,
            'min_fold': min_fold['fold_change'] if min_fold is not None else None
        })

    # Find MASSIVE fold changes (>5x or <0.2x) that are significant
    print("\n" + "="*100)
    print("MASSIVE FOLD CHANGES (>5x or <0.2x, FDR<0.05)")
    print("="*100)

    massive = results_df[
        (results_df['p_fdr'] < 0.05) &
        ((results_df['fold_change'] > 5) | (results_df['fold_change'] < 0.2))
    ].sort_values('fold_change', ascending=False)

    if len(massive) > 0:
        print(f"\nFound {len(massive)} massive effects:")
        print(f"{'Biomarker':<10} {'Motif':<8} {'Fold':>10} {'p_FDR':>12} {'n_present':>10}")
        print("-"*60)
        for _, row in massive.iterrows():
            print(f"{row['biomarker']:<10} {row['motif']:<8} {row['fold_change']:>10.2f} {row['p_fdr']:>12.2e} {row['n_present']:>10}")
    else:
        print("No massive fold changes (>5x or <0.2x) with FDR<0.05")

    # Visualization: Heatmap of fold changes for significant results
    print("\n" + "="*100)
    print("Generating visualizations...")

    # Create pivot table for heatmap
    sig_results = results_df[results_df['p_fdr'] < 0.05].copy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, bm in enumerate(BIOMARKERS):
        ax = axes[idx]
        bm_data = results_df[results_df['biomarker'] == bm].copy()

        if len(bm_data) == 0:
            ax.text(0.5, 0.5, f'{bm}\nNo data', ha='center', va='center')
            ax.set_title(bm)
            continue

        # Create mini heatmap
        bm_data['first2'] = bm_data['motif'].str[:2]
        bm_data['third'] = bm_data['motif'].str[2]
        pivot = bm_data.pivot(index='first2', columns='third', values='fold_change')

        # Log transform for visualization
        pivot_log = np.log2(pivot)

        sns.heatmap(pivot_log, annot=False, cmap='RdBu_r', center=0, ax=ax,
                   cbar_kws={'label': 'log2(FC)'}, vmin=-2, vmax=2)
        ax.set_title(f'{bm}\n(n_sig={len(sig_results[sig_results["biomarker"]==bm])})', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Remove extra subplot
    axes[-1].axis('off')

    plt.suptitle('3-mer Motif Effects Across All Biomarkers\n(log2 Fold Change)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/figures/all_biomarkers_heatmap.png', dpi=150)
    plt.close()

    # Bar plot of number of significant motifs per biomarker
    fig, ax = plt.subplots(figsize=(10, 6))

    bm_counts = []
    for bm in BIOMARKERS:
        n_sig = len(results_df[(results_df['biomarker'] == bm) & (results_df['p_fdr'] < 0.05)])
        bm_counts.append({'biomarker': bm, 'n_significant': n_sig})

    counts_df = pd.DataFrame(bm_counts)
    bars = ax.bar(counts_df['biomarker'], counts_df['n_significant'], color='steelblue')
    ax.axhline(y=64, color='red', linestyle='--', alpha=0.5, label='Max possible (64)')
    ax.set_ylabel('Number of Significant 3-mers (FDR < 0.05)')
    ax.set_xlabel('Biomarker')
    ax.set_title('Significant Sequence Motifs by Biomarker')
    ax.legend()

    for bar, count in zip(bars, counts_df['n_significant']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/figures/biomarker_sig_counts.png', dpi=150)
    plt.close()

    # Save full results
    results_df.to_csv('/Users/barneyh/dphil/paper2/analyses/hypotheses/002_sequence_motifs/all_biomarkers_results.csv', index=False)
    print("\nResults saved to all_biomarkers_results.csv")
    print("Figures saved to figures/")


if __name__ == "__main__":
    main()
