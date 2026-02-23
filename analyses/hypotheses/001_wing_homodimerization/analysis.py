"""
Comprehensive self-complementarity analysis for ASO hepatotoxicity.

Tests multiple dimerization modes:
1. 5' wing self-complementarity (wing homodimer)
2. 3' wing self-complementarity (wing homodimer)
3. 5' to 3' wing complementarity (inter-wing heterodimer)
4. Full ASO antiparallel self-complementarity (how molecules actually bind)
5. Maximum internal palindrome
6. Terminal complementarity (5' to 3' end)

Results: All metrics show significant NEGATIVE correlation with ALT.
Higher self-complementarity → LOWER hepatotoxicity (opposite of hypothesis).
"""

import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'modxna/src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modxna.helm_parser import parse_helm, get_sequence
from utils.helm import parse_helm_wings

COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def reverse_complement(seq: str) -> str:
    """Get reverse complement of a sequence."""
    return ''.join(COMPLEMENT.get(b, 'N') for b in reversed(seq))


def max_contiguous_match(seq1: str, seq2: str) -> int:
    """
    Find maximum contiguous matching bases between two sequences
    across all possible alignments (sliding one over the other).
    """
    if not seq1 or not seq2:
        return 0

    max_match = 0
    n1, n2 = len(seq1), len(seq2)

    # Try all alignments
    for offset in range(-n2 + 1, n1):
        contiguous = 0
        best = 0
        for i in range(n1):
            j = i - offset
            if 0 <= j < n2 and seq1[i] == seq2[j]:
                contiguous += 1
                best = max(best, contiguous)
            else:
                contiguous = 0
        max_match = max(max_match, best)

    return max_match


def wing_self_complementarity(wing: str) -> int:
    """Max contiguous bp when wing dimerizes with itself (antiparallel)."""
    if not wing or len(wing) < 2:
        return 0
    rev_comp = reverse_complement(wing)
    return max_contiguous_match(wing, rev_comp)


def inter_wing_complementarity(wing5: str, wing3: str) -> int:
    """
    Max contiguous bp between 5' wing and 3' wing.
    Models one molecule's 5' wing binding to another's 3' wing.
    The 3' wing would align antiparallel to the 5' wing.
    """
    if not wing5 or not wing3:
        return 0
    # 5' wing (5'→3') pairs with reverse complement of 3' wing
    wing3_rc = reverse_complement(wing3)
    return max_contiguous_match(wing5, wing3_rc)


def full_aso_self_complementarity(seq: str) -> int:
    """
    Max contiguous bp when full ASO dimerizes with itself (antiparallel).

    Models two identical ASOs binding:
    5'--------------------3'
       ||||||||||||||||
    3'--------------------5'

    Checks all possible staggered alignments.
    """
    if not seq or len(seq) < 2:
        return 0
    rev_comp = reverse_complement(seq)
    return max_contiguous_match(seq, rev_comp)


def max_internal_palindrome(seq: str) -> int:
    """
    Find the longest internal palindromic region in the sequence.
    A palindrome reads the same 5'→3' as its complement reads 3'→5'.
    e.g., GAATTC is palindromic (complement is CTTAAG, reversed = GAATTC)
    """
    if not seq or len(seq) < 2:
        return 0

    max_len = 0
    n = len(seq)

    # Check all possible palindrome centers
    for center in range(n):
        # Odd length palindromes (center on a base)
        for radius in range(1, n):
            left_idx = center - radius
            right_idx = center + radius
            if left_idx < 0 or right_idx >= n:
                break
            left = seq[left_idx]
            right = seq[right_idx]
            if COMPLEMENT.get(left) == right:
                max_len = max(max_len, 2 * radius + 1)
            else:
                break

        # Even length palindromes (center between bases)
        for radius in range(1, n):
            left_idx = center - radius + 1
            right_idx = center + radius
            if left_idx < 0 or right_idx >= n:
                break
            left = seq[left_idx]
            right = seq[right_idx]
            if COMPLEMENT.get(left) == right:
                max_len = max(max_len, 2 * radius)
            else:
                break

    return max_len


def terminal_complementarity(seq: str, window: int = 5) -> int:
    """
    Check if the terminal regions (first and last N bases) are complementary.
    This would allow end-to-end dimerization.
    """
    if not seq or len(seq) < 2 * window:
        return 0

    five_prime = seq[:window]
    three_prime = seq[-window:]
    three_prime_rc = reverse_complement(three_prime)

    return max_contiguous_match(five_prime, three_prime_rc)


def normalize_helm(helm: str) -> str:
    """Normalize HELM format: convert double braces to single braces."""
    return helm.replace('{{', '{').replace('}}', '}')


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


def extract_sequences(helm: str) -> dict | None:
    """Extract full sequence and wing sequences from HELM."""
    if not helm or not isinstance(helm, str):
        return None

    helm_norm = normalize_helm(helm)
    wings = parse_helm_wings(helm)

    if wings is None:
        return None

    five_len, gap_len, three_len = wings

    try:
        nucleotides = parse_helm(helm_norm)
        full_seq = get_sequence(nucleotides)
    except Exception:
        return None

    if five_len == 0 and three_len == 0:
        return None

    wing5 = full_seq[:five_len] if five_len > 0 else ""
    wing3 = full_seq[-three_len:] if three_len > 0 else ""

    return {
        'full_seq': full_seq,
        'wing5': wing5,
        'wing3': wing3,
        'gap': full_seq[five_len:len(full_seq)-three_len] if three_len > 0 else full_seq[five_len:]
    }


def compute_all_metrics(row: dict) -> dict:
    """Compute all complementarity metrics for a single ASO."""
    seqs = row.get('sequences')
    if seqs is None:
        return {k: np.nan for k in [
            'wing5_self', 'wing3_self', 'wing5_to_wing3',
            'full_aso_self', 'max_palindrome', 'terminal_comp'
        ]}

    return {
        'wing5_self': wing_self_complementarity(seqs['wing5']),
        'wing3_self': wing_self_complementarity(seqs['wing3']),
        'wing5_to_wing3': inter_wing_complementarity(seqs['wing5'], seqs['wing3']),
        'full_aso_self': full_aso_self_complementarity(seqs['full_seq']),
        'max_palindrome': max_internal_palindrome(seqs['full_seq']),
        'terminal_comp': terminal_complementarity(seqs['full_seq'], window=5),
    }


def run_statistics(agg: pd.DataFrame, metric_col: str, metric_name: str) -> dict:
    """Run statistical tests for a single metric."""
    valid = agg[agg[metric_col].notna()].copy()

    if len(valid) < 10:
        return {'name': metric_name, 'n': len(valid), 'error': 'Insufficient data'}

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(valid[metric_col], valid['ALT'])

    # Mann-Whitney at threshold 3
    high3 = valid[valid[metric_col] >= 3]['ALT']
    low3 = valid[valid[metric_col] < 3]['ALT']

    if len(high3) >= 3 and len(low3) >= 3:
        mw3_stat, mw3_p = stats.mannwhitneyu(high3, low3, alternative='two-sided')
        mw3_result = {
            'n_high': len(high3), 'median_high': high3.median(),
            'n_low': len(low3), 'median_low': low3.median(),
            'U': mw3_stat, 'p': mw3_p
        }
    else:
        mw3_result = None

    # Mann-Whitney at threshold 4
    high4 = valid[valid[metric_col] >= 4]['ALT']
    low4 = valid[valid[metric_col] < 4]['ALT']

    if len(high4) >= 3 and len(low4) >= 3:
        mw4_stat, mw4_p = stats.mannwhitneyu(high4, low4, alternative='two-sided')
        mw4_result = {
            'n_high': len(high4), 'median_high': high4.median(),
            'n_low': len(low4), 'median_low': low4.median(),
            'U': mw4_stat, 'p': mw4_p
        }
    else:
        mw4_result = None

    return {
        'name': metric_name,
        'n': len(valid),
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mw_3bp': mw3_result,
        'mw_4bp': mw4_result,
        'distribution': valid[metric_col].value_counts().sort_index().to_dict()
    }


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

    # Convert ALT
    filtered['ALT_mean'] = filtered['ALT'].apply(flatten_alt)
    filtered = filtered[
        filtered['HELM Annotation'].notna() &
        filtered['ALT_mean'].notna()
    ].copy()
    print(f"With HELM and ALT: {len(filtered):,}")

    # Extract sequences
    filtered['sequences'] = filtered['HELM Annotation'].apply(extract_sequences)
    filtered = filtered[filtered['sequences'].notna()].copy()
    print(f"With valid sequences: {len(filtered):,}")

    # Compute all metrics
    print("\nComputing complementarity metrics...")
    metrics_df = filtered.apply(lambda row: compute_all_metrics(row), axis=1, result_type='expand')
    for col in metrics_df.columns:
        filtered[col] = metrics_df[col]

    # Aggregate by Compound ID
    agg_cols = {
        'ALT_mean': 'mean',
        'wing5_self': 'first',
        'wing3_self': 'first',
        'wing5_to_wing3': 'first',
        'full_aso_self': 'first',
        'max_palindrome': 'first',
        'terminal_comp': 'first',
        'HELM Annotation': 'first',
        'sequences': 'first'
    }
    agg = filtered.groupby('Compound ID').agg(agg_cols).reset_index()
    agg.rename(columns={'ALT_mean': 'ALT'}, inplace=True)
    print(f"\nUnique ASOs: {len(agg):,}")

    # Define metrics to test
    metrics = [
        ('wing5_self', "5' Wing Self-Complementarity"),
        ('wing3_self', "3' Wing Self-Complementarity"),
        ('wing5_to_wing3', "5' to 3' Wing Complementarity"),
        ('full_aso_self', "Full ASO Self-Complementarity (antiparallel)"),
        ('max_palindrome', "Max Internal Palindrome"),
        ('terminal_comp', "Terminal Complementarity (5bp window)"),
    ]

    # Run statistics for each metric
    print("\n" + "="*80)
    print("STATISTICAL RESULTS")
    print("="*80)

    results = []
    for col, name in metrics:
        result = run_statistics(agg, col, name)
        results.append(result)

        print(f"\n{'─'*80}")
        print(f"METRIC: {name}")
        print(f"{'─'*80}")

        if 'error' in result:
            print(f"  {result['error']}")
            continue

        print(f"  N = {result['n']}")
        print(f"  Spearman: ρ = {result['spearman_r']:.4f}, p = {result['spearman_p']:.2e}")

        if result['mw_3bp']:
            mw = result['mw_3bp']
            print(f"  Mann-Whitney (≥3bp vs <3bp):")
            print(f"    High: n={mw['n_high']}, median ALT={mw['median_high']:.1f}")
            print(f"    Low:  n={mw['n_low']}, median ALT={mw['median_low']:.1f}")
            print(f"    U={mw['U']:.0f}, p={mw['p']:.2e}")

        if result['mw_4bp']:
            mw = result['mw_4bp']
            print(f"  Mann-Whitney (≥4bp vs <4bp):")
            print(f"    High: n={mw['n_high']}, median ALT={mw['median_high']:.1f}")
            print(f"    Low:  n={mw['n_low']}, median ALT={mw['median_low']:.1f}")
            print(f"    U={mw['U']:.0f}, p={mw['p']:.2e}")

        # Show distribution
        print(f"  Distribution: {result['distribution']}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Metric':<45} {'Spearman ρ':>12} {'p-value':>12} {'Direction':>12}")
    print("-"*80)
    for r in results:
        if 'spearman_r' in r:
            direction = "↑ ALT" if r['spearman_r'] > 0 else "↓ ALT"
            sig = "*" if r['spearman_p'] < 0.05 else ""
            print(f"{r['name']:<45} {r['spearman_r']:>11.3f} {r['spearman_p']:>11.2e}{sig} {direction:>11}")

    # Generate multi-panel boxplot
    print("\n" + "="*80)
    print("Generating boxplots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(metrics):
        ax = axes[idx]
        valid = agg[agg[col].notna()].copy()

        if len(valid) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
            continue

        # Group by score
        score_groups = valid.groupby(col)['ALT'].apply(list).to_dict()
        scores = sorted(score_groups.keys())

        # Limit to reasonable number of categories
        if len(scores) > 8:
            # Bin high values together
            binned = {}
            for s in scores:
                if s >= 6:
                    binned.setdefault('6+', []).extend(score_groups[s])
                else:
                    binned[s] = score_groups[s]
            score_groups = binned
            scores = sorted([s for s in score_groups.keys() if isinstance(s, (int, float))]) + (['6+'] if '6+' in binned else [])

        data = [score_groups[s] for s in scores]
        labels = [f"{s}\n(n={len(score_groups[s])})" for s in scores]

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)

        ax.set_ylabel('ALT (IU/L)')
        ax.set_title(name, fontsize=10)
        ax.set_yscale('log')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', labelsize=8)

    plt.suptitle("ASO Self-Complementarity Metrics vs Hepatotoxicity (ALT)\nMice, Subcutaneous", fontsize=14)
    plt.tight_layout()

    plot_path = '/Users/barneyh/dphil/paper2/analyses/wing_complementarity_vs_alt.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Boxplot saved to: {plot_path}")
    plt.close()

    print("\n" + "="*80)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
