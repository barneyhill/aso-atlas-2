"""
NUPACK-based thermodynamic analysis for wing homodimerization hypothesis.

Uses NUPACK to compute rigorous thermodynamic free energy (dG) for ASO
homodimer formation and tests correlation with hepatotoxicity (ALT).

Key metrics:
- homodimer_dG: Free energy of full ASO homodimer formation
- ddG_dimerization: Dimerization propensity (homodimer_dG - 2*monomer_dG)
- wing5_homodimer_dG: Free energy of 5' wing homodimer (matches 6YCS structure)
- wing3_homodimer_dG: Free energy of 3' wing homodimer
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add paths for imports
ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT / 'modxna/src'))
sys.path.insert(0, str(ROOT))

from modxna.helm_parser import parse_helm, get_sequence
from utils.helm import parse_helm_wings
# Import from local module
from nupack_metrics import create_model, compute_all_metrics, NupackMetrics

# Constants
DATA_PATH = ROOT / 'data/oligostack/processed/hepatictoxicity_processed.parquet'
OUTPUT_DIR = Path(__file__).parent / 'figures'


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


def clean_sequence(seq: str) -> str:
    """Remove non-ATGC characters from sequence (modification markers like [])."""
    return ''.join(c for c in seq.upper() if c in 'ATGC')


def extract_sequences(helm: str) -> dict | None:
    """Extract full sequence and wing sequences from HELM.

    Now extracts sequences for ALL ASOs, not just MOE gapmers.
    Wing sequences are only populated for gapmers with wings.
    """
    if not helm or not isinstance(helm, str):
        return None

    helm_norm = normalize_helm(helm)

    # Try to parse the sequence
    try:
        nucleotides = parse_helm(helm_norm)
        full_seq_raw = get_sequence(nucleotides)
    except Exception:
        return None

    # Clean the sequence - remove modification markers, convert 5meC -> C
    full_seq = clean_sequence(full_seq_raw)

    # Validate we got a reasonable sequence
    if len(full_seq) < 10:
        return None

    # Try to get wing structure (may be None or (0,X,0) for non-gapmers)
    wings = parse_helm_wings(helm)

    wing5 = ""
    wing3 = ""
    gap = full_seq  # Default: entire sequence is the "gap"

    if wings is not None:
        five_len, gap_len, three_len = wings
        # Only set wing sequences if there are actual wings
        if five_len > 0 and five_len <= len(full_seq):
            wing5 = full_seq[:five_len]
        if three_len > 0 and three_len <= len(full_seq):
            wing3 = full_seq[-three_len:]
        # Calculate gap
        if five_len > 0 or three_len > 0:
            end_idx = len(full_seq) - three_len if three_len > 0 else len(full_seq)
            gap = full_seq[five_len:end_idx]

    return {
        'full_seq': full_seq,
        'wing5': wing5,
        'wing3': wing3,
        'gap': gap
    }


def run_spearman_test(data: pd.DataFrame, metric_col: str, alt_col: str = 'ALT') -> dict:
    """Run Spearman correlation test."""
    valid = data[[metric_col, alt_col]].dropna()
    # Also remove infinite values
    valid = valid[np.isfinite(valid[metric_col]) & np.isfinite(valid[alt_col])]
    if len(valid) < 10:
        return {'n': len(valid), 'error': 'Insufficient data'}

    # Check for constant values
    if valid[metric_col].std() == 0 or valid[alt_col].std() == 0:
        return {'n': len(valid), 'error': 'Constant values - no correlation possible'}

    # Use .values to ensure we pass numpy arrays
    x = valid[metric_col].astype(float).values
    y = valid[alt_col].astype(float).values
    rho, p = stats.spearmanr(x, y)

    return {
        'n': len(valid),
        'rho': float(rho),
        'p': float(p)
    }


def run_mannwhitney_by_median(data: pd.DataFrame, metric_col: str, alt_col: str = 'ALT') -> dict:
    """Run Mann-Whitney U test splitting by median of metric."""
    valid = data[[metric_col, alt_col]].dropna()
    if len(valid) < 10:
        return {'error': 'Insufficient data'}

    median_val = valid[metric_col].median()
    high = valid[valid[metric_col] <= median_val][alt_col]  # More negative dG = more stable
    low = valid[valid[metric_col] > median_val][alt_col]

    if len(high) < 3 or len(low) < 3:
        return {'error': 'Insufficient group sizes'}

    stat, p = stats.mannwhitneyu(high, low, alternative='two-sided')
    return {
        'threshold': f'<= {median_val:.2f} (median)',
        'n_stable': len(high),
        'median_alt_stable': high.median(),
        'n_unstable': len(low),
        'median_alt_unstable': low.median(),
        'U': stat,
        'p': p
    }


def run_linear_regression(data: pd.DataFrame, metric_col: str, alt_col: str = 'ALT') -> dict:
    """Run linear regression of metric vs log(ALT)."""
    valid = data[[metric_col, alt_col]].dropna()
    valid = valid[valid[alt_col] > 0]  # Need positive ALT for log

    if len(valid) < 10:
        return {'error': 'Insufficient data'}

    log_alt = np.log10(valid[alt_col])
    result = linregress(valid[metric_col], log_alt)

    return {
        'n': len(valid),
        'slope': result.slope,
        'intercept': result.intercept,
        'r_squared': result.rvalue**2,
        'p': result.pvalue,
        'stderr': result.stderr
    }


def plot_scatter_with_regression(data: pd.DataFrame, metric_col: str, metric_name: str,
                                  output_path: Path):
    """Create scatter plot of metric vs ALT with regression line."""
    valid = data[[metric_col, 'ALT']].dropna()
    valid = valid[valid['ALT'] > 0]
    valid = valid[np.isfinite(valid[metric_col])]

    if len(valid) < 10:
        print(f"Insufficient data for {metric_name} scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(valid[metric_col], valid['ALT'], alpha=0.5, s=30, c='steelblue')

    # Regression line on log scale
    log_alt = np.log10(valid['ALT'])
    result = linregress(valid[metric_col], log_alt)

    x_range = np.linspace(valid[metric_col].min(), valid[metric_col].max(), 100)
    y_pred = 10**(result.intercept + result.slope * x_range)
    ax.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Regression (R²={result.rvalue**2:.3f})')

    ax.set_xlabel(f'{metric_name} (kcal/mol)', fontsize=12)
    ax.set_ylabel('ALT (IU/L)', fontsize=12)
    ax.set_yscale('log')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Normal ALT threshold')
    ax.legend()

    # Add Spearman correlation annotation
    rho, p = stats.spearmanr(valid[metric_col], valid['ALT'])
    ax.annotate(f'Spearman ρ = {rho:.3f}\np = {p:.2e}\nn = {len(valid)}',
                xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_title(f'{metric_name} vs Hepatotoxicity (ALT)\nMice, Subcutaneous', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_boxplot_by_quartiles(data: pd.DataFrame, metric_col: str, metric_name: str,
                               output_path: Path):
    """Create boxplot of ALT by metric quartiles."""
    valid = data[[metric_col, 'ALT']].dropna().copy()
    valid = valid[np.isfinite(valid[metric_col])]

    if len(valid) < 20:
        print(f"Insufficient data for {metric_name} boxplot")
        return

    # Create quartile bins - handle duplicate edges
    try:
        valid['quartile'] = pd.qcut(valid[metric_col], q=4,
                                     labels=['Q1\n(most stable)', 'Q2', 'Q3', 'Q4\n(least stable)'],
                                     duplicates='drop')
    except ValueError:
        # If quartiles fail, try median split
        median = valid[metric_col].median()
        valid['quartile'] = valid[metric_col].apply(
            lambda x: 'Below median\n(more stable)' if x <= median else 'Above median\n(less stable)'
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    groups = valid['quartile'].unique()
    groups = sorted(groups, key=lambda x: str(x))  # Sort for consistency
    data_by_group = [valid[valid['quartile'] == g]['ALT'].values for g in groups]
    labels = [f"{g}\n(n={len(d)})" for g, d in zip(groups, data_by_group)]

    bp = ax.boxplot(data_by_group, tick_labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)

    ax.set_ylabel('ALT (IU/L)', fontsize=12)
    ax.set_xlabel(f'{metric_name} Groups', fontsize=12)
    ax.set_yscale('log')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7)

    ax.set_title(f'ALT by {metric_name} Groups\nMice, Subcutaneous', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
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
    print("Extracting sequences from HELM...")
    filtered['sequences'] = filtered['HELM Annotation'].apply(extract_sequences)
    filtered = filtered[filtered['sequences'].notna()].copy()
    print(f"With valid sequences: {len(filtered):,}")

    # Aggregate by Compound ID first (get unique ASOs)
    agg_cols = {
        'ALT_mean': 'mean',
        'HELM Annotation': 'first',
        'sequences': 'first'
    }
    agg = filtered.groupby('Compound ID').agg(agg_cols).reset_index()
    agg.rename(columns={'ALT_mean': 'ALT'}, inplace=True)
    print(f"\nUnique ASOs: {len(agg):,}")

    # Compute NUPACK metrics
    print("\nComputing NUPACK thermodynamic metrics (this may take a few minutes)...")
    model = create_model(celsius=37, sodium=0.15, magnesium=0.001)

    nupack_results = []
    for idx, row in tqdm(agg.iterrows(), total=len(agg), desc="NUPACK"):
        seqs = row['sequences']
        try:
            metrics = compute_all_metrics(seqs['full_seq'], seqs['wing5'], seqs['wing3'], model)
            nupack_results.append(metrics.to_dict())
        except Exception as e:
            print(f"Error computing metrics for {row['Compound ID']}: {e}")
            nupack_results.append({k: np.nan for k in NupackMetrics.__dataclass_fields__.keys()})

    # Add NUPACK metrics to dataframe
    metrics_df = pd.DataFrame(nupack_results)
    for col in metrics_df.columns:
        agg[col] = metrics_df[col].values

    # Handle infinite values (replace with NaN for statistics)
    for col in ['homodimer_dG', 'monomer_dG', 'ddG_dimerization',
                'wing5_homodimer_dG', 'wing3_homodimer_dG', 'wing5_wing3_dG']:
        if col in agg.columns:
            agg[col] = agg[col].replace([np.inf, -np.inf], np.nan)

    # Define metrics to analyze
    metrics = [
        ('homodimer_dG', 'Full ASO Homodimer ΔG'),
        ('ddG_dimerization', 'ΔΔG Dimerization'),
        ('wing5_homodimer_dG', "5' Wing Homodimer ΔG"),
        ('wing3_homodimer_dG', "3' Wing Homodimer ΔG"),
        ('wing5_wing3_dG', "Wing5-Wing3 Heterodimer ΔG"),
    ]

    # Run statistical tests
    print("\n" + "="*80)
    print("STATISTICAL RESULTS (NUPACK Thermodynamic Metrics)")
    print("="*80)

    results = []
    for col, name in metrics:
        result = {'name': name, 'column': col}

        # Spearman
        spearman = run_spearman_test(agg, col)
        result.update({f'spearman_{k}': v for k, v in spearman.items()})

        # Mann-Whitney by median
        mw = run_mannwhitney_by_median(agg, col)
        result.update({f'mw_{k}': v for k, v in mw.items()})

        # Linear regression
        lr = run_linear_regression(agg, col)
        result.update({f'lr_{k}': v for k, v in lr.items()})

        results.append(result)

        print(f"\n{'─'*80}")
        print(f"METRIC: {name}")
        print(f"{'─'*80}")

        if 'spearman_n' in result:
            print(f"  N = {result['spearman_n']}")
            print(f"  Spearman: ρ = {result.get('spearman_rho', np.nan):.4f}, p = {result.get('spearman_p', np.nan):.2e}")
        elif 'spearman_error' in result:
            print(f"  {result.get('spearman_error', 'Unknown error')}")

        if 'mw_threshold' in result:
            print(f"  Mann-Whitney ({result['mw_threshold']}):")
            print(f"    Stable (more neg dG): n={result['mw_n_stable']}, median ALT={result['mw_median_alt_stable']:.1f}")
            print(f"    Unstable: n={result['mw_n_unstable']}, median ALT={result['mw_median_alt_unstable']:.1f}")
            print(f"    U={result['mw_U']:.0f}, p={result['mw_p']:.2e}")

        if 'lr_r_squared' in result:
            print(f"  Linear Regression (log ALT):")
            print(f"    R² = {result['lr_r_squared']:.4f}, p = {result['lr_p']:.2e}")
            print(f"    Slope = {result['lr_slope']:.4f} ± {result['lr_stderr']:.4f}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Metric':<35} {'Spearman ρ':>12} {'p-value':>12} {'R²':>10} {'Direction':>12}")
    print("-"*80)
    for r in results:
        if 'spearman_rho' in r:
            rho = r['spearman_rho']
            direction = "↑ ALT" if rho < 0 else "↓ ALT"  # More negative dG → effect on ALT
            sig = "*" if r.get('spearman_p', 1) < 0.05 else ""
            r2 = r.get('lr_r_squared', np.nan)
            print(f"{r['name']:<35} {rho:>11.3f} {r.get('spearman_p', np.nan):>11.2e}{sig} {r2:>9.3f} {direction:>11}")


    # Generate plots
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)

    # Individual scatter plots
    for col, name in metrics:
        safe_name = col.replace('_', '-')
        plot_scatter_with_regression(agg, col, name, OUTPUT_DIR / f'{safe_name}_vs_alt.png')
        plot_boxplot_by_quartiles(agg, col, name, OUTPUT_DIR / f'{safe_name}_boxplot.png')

    # Multi-panel summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (col, name) in enumerate(metrics):
        if idx >= 5:
            break
        ax = axes[idx]
        valid = agg[[col, 'ALT']].dropna()
        valid = valid[np.isfinite(valid[col])]

        if len(valid) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name, fontsize=10)
            continue

        ax.scatter(valid[col], valid['ALT'], alpha=0.4, s=20, c='steelblue')
        ax.set_xlabel('ΔG (kcal/mol)')
        ax.set_ylabel('ALT (IU/L)')
        ax.set_yscale('log')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5)

        rho, p = stats.spearmanr(valid[col], valid['ALT'])
        ax.set_title(f'{name}\nρ={rho:.3f}, p={p:.1e}', fontsize=9)

    # Hide extra subplot
    axes[5].axis('off')

    plt.suptitle("NUPACK Thermodynamic Metrics vs Hepatotoxicity (ALT)\nMice, Subcutaneous", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'nupack_summary.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'nupack_summary.png'}")

    # Save results to CSV
    results_df = agg[['Compound ID', 'ALT', 'homodimer_dG', 'monomer_dG', 'ddG_dimerization',
                      'wing5_homodimer_dG', 'wing3_homodimer_dG', 'wing5_wing3_dG']].copy()
    results_df.to_csv(OUTPUT_DIR.parent / 'nupack_metrics.csv', index=False)
    print(f"Saved metrics to: {OUTPUT_DIR.parent / 'nupack_metrics.csv'}")

    print("\n" + "="*80)
    print("Analysis complete.")
    print("="*80)


if __name__ == "__main__":
    main()
