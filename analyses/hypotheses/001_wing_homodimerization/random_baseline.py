"""
Compare self-complementarity metrics between real ASOs and random sequences.

Tests whether ASOs are enriched or depleted for self-complementarity
compared to random sequences of the same length distribution.
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
BASES = ['A', 'T', 'G', 'C']


def reverse_complement(seq: str) -> str:
    return ''.join(COMPLEMENT.get(b, 'N') for b in reversed(seq))


def max_contiguous_match(seq1: str, seq2: str) -> int:
    if not seq1 or not seq2:
        return 0
    max_match = 0
    n1, n2 = len(seq1), len(seq2)
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
    if not wing or len(wing) < 2:
        return 0
    return max_contiguous_match(wing, reverse_complement(wing))


def inter_wing_complementarity(wing5: str, wing3: str) -> int:
    if not wing5 or not wing3:
        return 0
    return max_contiguous_match(wing5, reverse_complement(wing3))


def full_aso_self_complementarity(seq: str) -> int:
    if not seq or len(seq) < 2:
        return 0
    return max_contiguous_match(seq, reverse_complement(seq))


def terminal_complementarity(seq: str, window: int = 5) -> int:
    if not seq or len(seq) < 2 * window:
        return 0
    five_prime = seq[:window]
    three_prime = seq[-window:]
    return max_contiguous_match(five_prime, reverse_complement(three_prime))


def normalize_helm(helm: str) -> str:
    return helm.replace('{{', '{').replace('}}', '}')


def extract_sequences(helm: str) -> dict | None:
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
        'five_len': five_len,
        'three_len': three_len,
        'gap_len': gap_len
    }


def generate_random_sequence(length: int) -> str:
    """Generate a random DNA sequence of given length."""
    return ''.join(np.random.choice(BASES) for _ in range(length))


def compute_metrics(full_seq: str, wing5: str, wing3: str) -> dict:
    """Compute all complementarity metrics."""
    return {
        'wing5_self': wing_self_complementarity(wing5),
        'wing3_self': wing_self_complementarity(wing3),
        'wing5_to_wing3': inter_wing_complementarity(wing5, wing3),
        'full_aso_self': full_aso_self_complementarity(full_seq),
        'terminal_comp': terminal_complementarity(full_seq, window=5),
    }


def flatten_alt(alt_value):
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


def main():
    np.random.seed(42)

    # Load real data
    df = pd.read_parquet('/Users/barneyh/dphil/paper2/data/oligostack/processed/hepatictoxicity_processed.parquet')
    filtered = df[
        (df['species'] == 'mouse') &
        (df['adminstration_method'] == 'subcutaneous')
    ].copy()
    filtered['ALT_mean'] = filtered['ALT'].apply(flatten_alt)
    filtered = filtered[
        filtered['HELM Annotation'].notna() &
        filtered['ALT_mean'].notna()
    ].copy()
    filtered['sequences'] = filtered['HELM Annotation'].apply(extract_sequences)
    filtered = filtered[filtered['sequences'].notna()].copy()

    # Aggregate by Compound ID
    agg = filtered.groupby('Compound ID').agg({
        'ALT_mean': 'mean',
        'HELM Annotation': 'first',
        'sequences': 'first'
    }).reset_index()
    agg.rename(columns={'ALT_mean': 'ALT'}, inplace=True)

    print(f"Real ASOs: {len(agg):,}")

    # Compute metrics for real ASOs
    real_metrics = []
    for _, row in agg.iterrows():
        seqs = row['sequences']
        metrics = compute_metrics(seqs['full_seq'], seqs['wing5'], seqs['wing3'])
        metrics['source'] = 'Real ASO'
        real_metrics.append(metrics)
    real_df = pd.DataFrame(real_metrics)

    # Generate random sequences with matching architecture
    n_random = 10000  # Generate many random sequences

    # Get the architecture distribution from real ASOs
    architectures = [(s['five_len'], s['gap_len'], s['three_len']) for s in agg['sequences']]
    arch_counts = pd.Series(architectures).value_counts()
    print(f"\nTop ASO architectures (5'-gap-3'):")
    for arch, count in arch_counts.head(10).items():
        print(f"  {arch[0]}-{arch[1]}-{arch[2]}: {count}")

    # Sample architectures proportionally
    arch_probs = arch_counts / arch_counts.sum()
    sampled_archs = np.random.choice(
        len(arch_counts),
        size=n_random,
        p=arch_probs.values
    )

    random_metrics = []
    for idx in sampled_archs:
        five_len, gap_len, three_len = arch_counts.index[idx]
        total_len = five_len + gap_len + three_len
        full_seq = generate_random_sequence(total_len)
        wing5 = full_seq[:five_len] if five_len > 0 else ""
        wing3 = full_seq[-three_len:] if three_len > 0 else ""

        metrics = compute_metrics(full_seq, wing5, wing3)
        metrics['source'] = 'Random'
        random_metrics.append(metrics)

    random_df = pd.DataFrame(random_metrics)
    print(f"Random sequences generated: {len(random_df):,}")

    # Compare distributions
    print("\n" + "="*80)
    print("COMPARISON: REAL ASOs vs RANDOM SEQUENCES")
    print("="*80)

    metric_names = {
        'wing5_self': "5' Wing Self-Complementarity",
        'wing3_self': "3' Wing Self-Complementarity",
        'wing5_to_wing3': "5' to 3' Wing Complementarity",
        'full_aso_self': "Full ASO Self-Complementarity",
        'terminal_comp': "Terminal Complementarity",
    }

    comparison_results = []
    for col, name in metric_names.items():
        real_vals = real_df[col].dropna()
        rand_vals = random_df[col].dropna()

        # Mann-Whitney U test
        U, p = stats.mannwhitneyu(real_vals, rand_vals, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(real_vals), len(rand_vals)
        r = 1 - (2*U)/(n1*n2)  # rank-biserial correlation

        result = {
            'metric': name,
            'real_mean': real_vals.mean(),
            'real_median': real_vals.median(),
            'random_mean': rand_vals.mean(),
            'random_median': rand_vals.median(),
            'U': U,
            'p': p,
            'effect_size': r
        }
        comparison_results.append(result)

        print(f"\n{name}:")
        print(f"  Real ASOs:  mean={real_vals.mean():.2f}, median={real_vals.median():.0f}")
        print(f"  Random:     mean={rand_vals.mean():.2f}, median={rand_vals.median():.0f}")
        print(f"  Mann-Whitney U: {U:.0f}, p={p:.2e}")
        if real_vals.mean() > rand_vals.mean():
            print(f"  → Real ASOs are ENRICHED (effect size r={r:.3f})")
        else:
            print(f"  → Real ASOs are DEPLETED (effect size r={r:.3f})")

    # Distribution comparison
    print("\n" + "="*80)
    print("VALUE DISTRIBUTIONS")
    print("="*80)

    for col, name in metric_names.items():
        print(f"\n{name}:")
        real_dist = real_df[col].value_counts().sort_index()
        rand_dist = random_df[col].value_counts().sort_index()

        all_vals = sorted(set(real_dist.index) | set(rand_dist.index))
        print(f"  {'Value':>6} | {'Real':>8} ({'%':>5}) | {'Random':>8} ({'%':>5})")
        print(f"  {'-'*6}-+-{'-'*15}-+-{'-'*15}")
        for v in all_vals[:10]:  # Show first 10 values
            real_n = real_dist.get(v, 0)
            rand_n = rand_dist.get(v, 0)
            real_pct = 100 * real_n / len(real_df)
            rand_pct = 100 * rand_n / len(random_df)
            print(f"  {v:>6} | {real_n:>8} ({real_pct:>5.1f}%) | {rand_n:>8} ({rand_pct:>5.1f}%)")

    # Generate comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(metric_names.items()):
        ax = axes[idx]

        real_vals = real_df[col].dropna()
        rand_vals = random_df[col].dropna()

        # Get value counts
        max_val = max(real_vals.max(), rand_vals.max())
        bins = np.arange(-0.5, min(max_val + 1.5, 12.5), 1)

        ax.hist(rand_vals, bins=bins, alpha=0.5, label='Random', density=True, color='gray')
        ax.hist(real_vals, bins=bins, alpha=0.7, label='Real ASOs', density=True, color='steelblue')

        ax.set_xlabel('Complementarity Score (bp)')
        ax.set_ylabel('Density')
        ax.set_title(name, fontsize=10)
        ax.legend()

        # Add stats annotation
        p = [r['p'] for r in comparison_results if r['metric'] == name][0]
        direction = "enriched" if real_vals.mean() > rand_vals.mean() else "depleted"
        ax.annotate(f'p = {p:.2e}\nASOs {direction}',
                   xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Use last subplot for summary
    ax = axes[5]
    ax.axis('off')
    summary_text = "Summary:\n\n"
    for r in comparison_results:
        direction = "↑ enriched" if r['real_mean'] > r['random_mean'] else "↓ depleted"
        sig = "***" if r['p'] < 0.001 else ("**" if r['p'] < 0.01 else ("*" if r['p'] < 0.05 else ""))
        summary_text += f"{r['metric'][:25]:25} {direction} {sig}\n"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace')

    plt.suptitle("ASO Self-Complementarity: Real vs Random Sequences\n(n=832 real ASOs, n=10,000 random)", fontsize=14)
    plt.tight_layout()

    plot_path = '/Users/barneyh/dphil/paper2/analyses/hypotheses/001_wing_homodimerization/figures/real_vs_random.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.close()

    # Final summary
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    enriched = sum(1 for r in comparison_results if r['real_mean'] > r['random_mean'] and r['p'] < 0.05)
    depleted = sum(1 for r in comparison_results if r['real_mean'] < r['random_mean'] and r['p'] < 0.05)
    print(f"Metrics where ASOs are significantly enriched: {enriched}/5")
    print(f"Metrics where ASOs are significantly depleted: {depleted}/5")


if __name__ == "__main__":
    main()
