"""
Hypothesis 006: mRNA Production Rate vs ASO Knockdown Efficacy

Tests whether genes with higher mRNA production rates show different knockdown
efficacy with ASOs.

Mathematical derivation:
At steady state: d[mRNA]/dt = 0 = k_prod - k_deg × [mRNA]
Therefore: k_prod = k_deg × [mRNA] = (ln(2) / half_life) × TPM

With real half-lives from RNAdecayCafe, the production rate formula simplifies:
    k_deg = ln(2) / halflife_hours
    log(k_deg) = donorm_log_kdeg  (directly from RNAdecayCafe)

Production rate calculation (with TPM stored as log(TPM+1)):
    Production = k_deg × TPM
    log(Production) = log(k_deg) + log(TPM)
                    = donorm_log_kdeg + log_tpm

Key insight: Use donorm_log_kdeg directly - no conversion needed.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
KINETICS_DIR = BASE_DIR / 'kinetics_model' / 'data'
OUTPUT_DIR = Path(__file__).parent / 'figures'


def load_tpm_data() -> dict[tuple[str, str], float]:
    """Load TPM data into (gene_symbol, model_id) -> log_tpm lookup."""
    path = KINETICS_DIR / 'OmicsExpressionTPMLogp1HumanAllGenes.csv.gz'
    df = pd.read_csv(path, compression='gzip')

    # Parse gene columns: "GENE_NAME (ENTREZ_ID)" -> "GENE_NAME"
    gene_cols = []
    gene_symbols = []
    for col in df.columns:
        if ' (' in col and col.endswith(')'):
            gene_symbol = col.split(' (')[0].strip()
            gene_cols.append(col)
            gene_symbols.append(gene_symbol)

    # Build lookup dictionary
    lookup: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        model_id = row['ModelID']
        if pd.isna(model_id):
            continue
        for col, gene_symbol in zip(gene_cols, gene_symbols):
            value = row[col]
            if pd.notna(value):
                lookup[(gene_symbol.upper(), model_id)] = float(value)

    return lookup


def load_halflife_data() -> dict[str, tuple[float, float]]:
    """Load RNAdecayCafe half-life data, aggregated to gene level.

    Returns: gene_symbol -> (median_log_kdeg, median_halflife_hours)

    Higher log_kdeg = faster degradation = shorter half-life.
    """
    path = DATA_DIR / 'RNAdecayCafe_v1_onetable.csv.gz'
    df = pd.read_csv(path, compression='gzip')

    # Filter for quality (low uncertainty in degradation rate)
    df = df[df['uncertainty_log_kdeg'] < 0.3].copy()

    # Aggregate to gene level (median across all cell lines and samples)
    agg = df.groupby('feature_ID').agg({
        'donorm_log_kdeg': 'median',
        'donorm_halflife': 'median',
    }).reset_index()

    # Build lookup: gene -> (log_kdeg, halflife_hours)
    lookup = {}
    for _, row in agg.iterrows():
        gene = row['feature_ID'].upper()
        lookup[gene] = (row['donorm_log_kdeg'], row['donorm_halflife'])

    return lookup


def load_inhibition_data() -> pd.DataFrame:
    """Load in vitro inhibition data, filter to human with CCLE mapping."""
    path = DATA_DIR / 'oligostack' / 'processed' / 'in_vitro_inhibition_processed_with_ccle.parquet'
    df = pd.read_parquet(path)

    # Filter to human cells with CCLE mapping
    df = df[
        (df['cell_line_species'] == 'human') &
        df['ccle_model_id'].notna() &
        df['gene_symbol'].notna() &
        df['Inhibition_pct'].notna()
    ].copy()

    # Clean transfection_method - treat None/NaN as 'Unknown'
    df['transfection_method'] = df['transfection_method'].fillna('Unknown')
    df.loc[df['transfection_method'] == 'None', 'transfection_method'] = 'Unknown'

    return df


def merge_all_data(
    inhibition_df: pd.DataFrame,
    tpm_lookup: dict[tuple[str, str], float],
    halflife_lookup: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Merge inhibition with TPM and half-life data."""

    # Add TPM values (normalize gene symbol to uppercase for case-insensitive matching)
    inhibition_df['log_tpm'] = inhibition_df.apply(
        lambda row: tpm_lookup.get((row['gene_symbol'].upper(), row['ccle_model_id'])),
        axis=1
    )

    # Add half-life values (gene-level, from RNAdecayCafe)
    def get_halflife(gene):
        return halflife_lookup.get(gene.upper() if gene else None, (None, None))

    hl_data = inhibition_df['gene_symbol'].apply(get_halflife)
    inhibition_df['log_kdeg'] = hl_data.apply(lambda x: x[0])
    inhibition_df['halflife_hours'] = hl_data.apply(lambda x: x[1])

    # Filter to rows with all data
    merged = inhibition_df[
        inhibition_df['log_tpm'].notna() &
        inhibition_df['log_kdeg'].notna()
    ].copy()

    return merged


def aggregate_to_gene_cell_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to mean inhibition per gene-cell line-transfection method combination."""
    agg = df.groupby(['gene_symbol', 'ccle_model_id', 'transfection_method']).agg({
        'Inhibition_pct': 'mean',
        'log_tpm': 'first',  # Same for all rows in group
        'log_kdeg': 'first',  # From RNAdecayCafe
        'halflife_hours': 'first',  # Keep for reference
        'dosage_nm': 'mean',
        'ccle_cell_line_name': 'first'
    }).reset_index()

    agg.rename(columns={'Inhibition_pct': 'mean_inhibition'}, inplace=True)

    # Calculate log production rate with real values from RNAdecayCafe:
    # log(production) = log(k_deg) + log(TPM)
    #                 = donorm_log_kdeg + log_tpm
    agg['log_production'] = agg['log_kdeg'] + agg['log_tpm']

    return agg


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run correlation and group comparison tests."""
    results = {}

    # 1. Spearman: TPM vs inhibition
    rho, p = stats.spearmanr(df['log_tpm'], df['mean_inhibition'])
    results['tpm_spearman'] = {'rho': rho, 'p': p}
    print(f"TPM vs Inhibition: Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 2. Spearman: half-life (hours) vs inhibition
    rho, p = stats.spearmanr(df['halflife_hours'], df['mean_inhibition'])
    results['halflife_spearman'] = {'rho': rho, 'p': p}
    print(f"Half-life (hours) vs Inhibition: Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 3. Spearman: log_kdeg vs inhibition
    rho, p = stats.spearmanr(df['log_kdeg'], df['mean_inhibition'])
    results['kdeg_spearman'] = {'rho': rho, 'p': p}
    print(f"log(k_deg) vs Inhibition: Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 4. Spearman: log production rate vs inhibition
    rho, p = stats.spearmanr(df['log_production'], df['mean_inhibition'])
    results['production_spearman'] = {'rho': rho, 'p': p}
    print(f"log(Production) vs Inhibition: Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 5. Mann-Whitney: high vs low TPM (median split)
    median_tpm = df['log_tpm'].median()
    high_tpm = df[df['log_tpm'] >= median_tpm]['mean_inhibition']
    low_tpm = df[df['log_tpm'] < median_tpm]['mean_inhibition']
    U, p = stats.mannwhitneyu(high_tpm, low_tpm, alternative='two-sided')
    results['tpm_mannwhitney'] = {'U': U, 'p': p,
                                  'high_median': high_tpm.median(),
                                  'low_median': low_tpm.median()}
    print(f"Mann-Whitney (TPM): U = {U:.0f}, p = {p:.2e}")
    print(f"  High TPM median inhibition: {high_tpm.median():.1f}%")
    print(f"  Low TPM median inhibition: {low_tpm.median():.1f}%")

    # 6. Mann-Whitney: long vs short half-life (median split)
    median_hl = df['halflife_hours'].median()
    long_hl = df[df['halflife_hours'] >= median_hl]['mean_inhibition']
    short_hl = df[df['halflife_hours'] < median_hl]['mean_inhibition']
    U, p = stats.mannwhitneyu(long_hl, short_hl, alternative='two-sided')
    results['halflife_mannwhitney'] = {'U': U, 'p': p,
                                        'long_median': long_hl.median(),
                                        'short_median': short_hl.median()}
    print(f"Mann-Whitney (half-life): U = {U:.0f}, p = {p:.2e}")
    print(f"  Long half-life (>={median_hl:.1f}h) median inhibition: {long_hl.median():.1f}%")
    print(f"  Short half-life (<{median_hl:.1f}h) median inhibition: {short_hl.median():.1f}%")

    # 7. OLS regression with predictors + transfection method
    try:
        import statsmodels.api as sm
        # Prepare data - drop rows with NaN in predictors
        ols_df = df[['log_tpm', 'log_kdeg', 'dosage_nm', 'transfection_method', 'mean_inhibition']].dropna()

        # Create dummy variables for transfection method (drop first to avoid multicollinearity)
        transfection_dummies = pd.get_dummies(ols_df['transfection_method'], prefix='transfection', drop_first=True, dtype=float)

        X = pd.concat([ols_df[['log_tpm', 'log_kdeg', 'dosage_nm']].astype(float), transfection_dummies], axis=1)
        X = sm.add_constant(X)
        y = ols_df['mean_inhibition']
        model = sm.OLS(y, X).fit()
        results['ols'] = {
            'r2': model.rsquared,
            'tpm_coef': model.params.get('log_tpm', np.nan),
            'tpm_pval': model.pvalues.get('log_tpm', np.nan),
            'kdeg_coef': model.params.get('log_kdeg', np.nan),
            'kdeg_pval': model.pvalues.get('log_kdeg', np.nan),
        }
        print(f"\nOLS Regression (R² = {model.rsquared:.3f}):")
        print(f"  log(TPM): coef = {model.params['log_tpm']:.2f}, p = {model.pvalues['log_tpm']:.2e}")
        print(f"  log(k_deg): coef = {model.params['log_kdeg']:.2f}, p = {model.pvalues['log_kdeg']:.2e}")
        print(f"  dosage_nm: coef = {model.params['dosage_nm']:.4f}, p = {model.pvalues['dosage_nm']:.2e}")
        # Print transfection method effects
        for col in transfection_dummies.columns:
            print(f"  {col}: coef = {model.params[col]:.2f}, p = {model.pvalues[col]:.2e}")
    except ImportError:
        print("statsmodels not available, skipping OLS")
        results['ols'] = None

    # 8. Transfection method comparison
    print(f"\nInhibition by transfection method:")
    for method in df['transfection_method'].unique():
        subset = df[df['transfection_method'] == method]['mean_inhibition']
        print(f"  {method}: n={len(subset)}, median={subset.median():.1f}%, mean={subset.mean():.1f}%")

    return results


def plot_tpm_vs_inhibition(df: pd.DataFrame, output_path: Path):
    """Scatter plot of TPM vs inhibition, colored by half-life."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df['log_tpm'],
        df['mean_inhibition'],
        c=df['halflife_hours'],
        cmap='RdYlBu_r',  # Red = long half-life, Blue = short
        alpha=0.6,
        s=40
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Half-life (hours)', fontsize=10)

    ax.set_xlabel('log(TPM + 1)', fontsize=12)
    ax.set_ylabel('Mean Inhibition (%)', fontsize=12)
    ax.set_title('Target Expression vs ASO Knockdown Efficacy', fontsize=14)

    # Add regression line
    z = np.polyfit(df['log_tpm'], df['mean_inhibition'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['log_tpm'].min(), df['log_tpm'].max(), 100)
    ax.plot(x_range, p(x_range), 'k--', alpha=0.5, label='Linear fit')

    ax.legend(loc='upper right')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% inhibition')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_inhibition_by_tpm_tertile(df: pd.DataFrame, output_path: Path):
    """Boxplot of inhibition by TPM tertile."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create TPM tertiles
    df = df.copy()
    df['tpm_tertile'] = pd.qcut(df['log_tpm'], 3, labels=['Low', 'Medium', 'High'])

    # Boxplot
    tertile_order = ['Low', 'Medium', 'High']
    data = [df[df['tpm_tertile'] == t]['mean_inhibition'].values for t in tertile_order]

    bp = ax.boxplot(data, labels=tertile_order, patch_artist=True)

    colors = ['#3498db', '#f1c40f', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('TPM Tertile', fontsize=12)
    ax.set_ylabel('Mean Inhibition (%)', fontsize=12)
    ax.set_title('ASO Knockdown Efficacy by Target Expression Level', fontsize=14)

    # Add sample sizes
    for i, tertile in enumerate(tertile_order):
        n = len(df[df['tpm_tertile'] == tertile])
        median = df[df['tpm_tertile'] == tertile]['mean_inhibition'].median()
        ax.annotate(f'n={n}\nmed={median:.1f}%',
                    xy=(i+1, ax.get_ylim()[1] - 5),
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_halflife_vs_inhibition(df: pd.DataFrame, output_path: Path):
    """Scatter plot of half-life vs inhibition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df['halflife_hours'],
        df['mean_inhibition'],
        c=df['log_tpm'],
        cmap='viridis',
        alpha=0.6,
        s=40
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log(TPM + 1)', fontsize=10)

    ax.set_xlabel('mRNA Half-life (hours)', fontsize=12)
    ax.set_ylabel('Mean Inhibition (%)', fontsize=12)
    ax.set_title('mRNA Stability vs ASO Knockdown Efficacy', fontsize=14)

    # Add regression line
    z = np.polyfit(df['halflife_hours'], df['mean_inhibition'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['halflife_hours'].min(), df['halflife_hours'].max(), 100)
    ax.plot(x_range, p(x_range), 'k--', alpha=0.5, label='Linear fit')

    ax.legend(loc='upper right')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_tpm_halflife_heatmap(df: pd.DataFrame, output_path: Path):
    """2D heatmap of mean inhibition by TPM and half-life bins."""
    fig, ax = plt.subplots(figsize=(10, 8))

    df = df.copy()

    # Create bins
    df['tpm_bin'] = pd.qcut(df['log_tpm'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    df['halflife_bin'] = pd.qcut(df['halflife_hours'], 4, labels=['Q1 (short)', 'Q2', 'Q3', 'Q4 (long)'])

    # Create pivot table
    pivot = df.pivot_table(
        values='mean_inhibition',
        index='halflife_bin',
        columns='tpm_bin',
        aggfunc='mean'
    )

    # Reorder index so short half-life (fast degradation) is at bottom
    pivot = pivot.reindex(['Q4 (long)', 'Q3', 'Q2', 'Q1 (short)'])

    # Plot heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Inhibition (%)', fontsize=10)

    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    ax.set_xlabel('TPM Quartile', fontsize=12)
    ax.set_ylabel('Half-life Quartile (hours)', fontsize=12)
    ax.set_title('ASO Knockdown Efficacy by Expression Level and mRNA Stability', fontsize=14)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                # Count samples in this bin
                n = len(df[(df['halflife_bin'] == pivot.index[i]) &
                           (df['tpm_bin'] == pivot.columns[j])])
                text_color = 'white' if val < 40 or val > 60 else 'black'
                ax.text(j, i, f'{val:.1f}%\n(n={n})', ha='center', va='center',
                        fontsize=9, color=text_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_production_vs_inhibition(df: pd.DataFrame, output_path: Path):
    """Scatter plot of log(production rate) vs inhibition with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df['log_production'],
        df['mean_inhibition'],
        alpha=0.6,
        s=40,
        c='#3498db',
        edgecolors='white',
        linewidth=0.5
    )

    ax.set_xlabel('log(Production Rate) = log(k_deg) + log(TPM)', fontsize=12)
    ax.set_ylabel('Mean Inhibition (%)', fontsize=12)
    ax.set_title('mRNA Production Rate vs ASO Knockdown Efficacy', fontsize=14)

    # Add regression line
    z = np.polyfit(df['log_production'], df['mean_inhibition'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['log_production'].min(), df['log_production'].max(), 100)
    ax.plot(x_range, p(x_range), 'r-', linewidth=2, label='Linear fit')

    # Calculate and annotate correlation
    rho, pval = stats.spearmanr(df['log_production'], df['mean_inhibition'])
    ax.annotate(
        f'Spearman ρ = {rho:.3f}\np = {pval:.2e}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% inhibition')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_inhibition_by_transfection(df: pd.DataFrame, output_path: Path):
    """Boxplot of inhibition by transfection method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order by median inhibition
    method_order = df.groupby('transfection_method')['mean_inhibition'].median().sort_values(ascending=False).index.tolist()
    data = [df[df['transfection_method'] == m]['mean_inhibition'].values for m in method_order]

    bp = ax.boxplot(data, labels=method_order, patch_artist=True)

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(method_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Transfection Method', fontsize=12)
    ax.set_ylabel('Mean Inhibition (%)', fontsize=12)
    ax.set_title('ASO Knockdown Efficacy by Transfection Method', fontsize=14)

    # Add sample sizes and medians
    for i, method in enumerate(method_order):
        subset = df[df['transfection_method'] == method]['mean_inhibition']
        ax.annotate(f'n={len(subset)}\nmed={subset.median():.1f}%',
                    xy=(i+1, ax.get_ylim()[1] - 5),
                    ha='center', fontsize=9)

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Hypothesis 006: mRNA Production Rate vs ASO Knockdown Efficacy")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    tpm_lookup = load_tpm_data()
    print(f"  TPM lookup: {len(tpm_lookup):,} (gene, cell) pairs")

    halflife_lookup = load_halflife_data()
    print(f"  Half-life lookup (RNAdecayCafe): {len(halflife_lookup):,} genes")

    inhibition_df = load_inhibition_data()
    print(f"  Inhibition data: {len(inhibition_df):,} rows (human, CCLE-mapped)")
    print(f"    Unique genes: {inhibition_df['gene_symbol'].nunique()}")
    print(f"    Unique cell lines: {inhibition_df['ccle_model_id'].nunique()}")
    print()

    # Merge data
    print("Merging datasets...")
    merged = merge_all_data(inhibition_df, tpm_lookup, halflife_lookup)
    print(f"  After merging: {len(merged):,} rows with TPM and half-life data")
    print(f"    Unique genes: {merged['gene_symbol'].nunique()}")
    print(f"    Unique cell lines: {merged['ccle_model_id'].nunique()}")
    print()

    # Aggregate to gene-cell-transfection level
    print("Aggregating to gene-cell-transfection level...")
    agg = aggregate_to_gene_cell_level(merged)
    print(f"  Final dataset: {len(agg):,} gene-cell-transfection combinations")
    print(f"  Transfection methods: {agg['transfection_method'].value_counts().to_dict()}")
    print()

    # Print summary statistics
    print("Summary statistics:")
    print(f"  log(TPM+1): mean={agg['log_tpm'].mean():.2f}, std={agg['log_tpm'].std():.2f}")
    print(f"  Half-life (hours): mean={agg['halflife_hours'].mean():.1f}, std={agg['halflife_hours'].std():.1f}")
    print(f"  log(k_deg): mean={agg['log_kdeg'].mean():.2f}, std={agg['log_kdeg'].std():.2f}")
    print(f"  Mean inhibition: mean={agg['mean_inhibition'].mean():.1f}%, std={agg['mean_inhibition'].std():.1f}%")
    print()

    # Run statistical tests
    print("Statistical tests:")
    print("-" * 40)
    results = run_statistical_tests(agg)
    print()

    # Generate figures
    print("Generating figures...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_production_vs_inhibition(agg, OUTPUT_DIR / 'production_vs_inhibition.png')
    plot_tpm_vs_inhibition(agg, OUTPUT_DIR / 'tpm_vs_inhibition.png')
    plot_inhibition_by_tpm_tertile(agg, OUTPUT_DIR / 'inhibition_by_tpm_tertile.png')
    plot_halflife_vs_inhibition(agg, OUTPUT_DIR / 'halflife_vs_inhibition.png')
    plot_tpm_halflife_heatmap(agg, OUTPUT_DIR / 'tpm_halflife_heatmap.png')
    plot_inhibition_by_transfection(agg, OUTPUT_DIR / 'inhibition_by_transfection.png')

    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return agg, results


if __name__ == '__main__':
    agg, results = main()
