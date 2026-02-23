"""
Hypothesis 006b: mRNA Production Rate vs ASO IC50

Tests whether genes with higher mRNA production rates require higher ASO
concentrations (IC50) for knockdown.

Rationale: If mRNA is being produced faster, more ASO may be needed to
achieve the same level of knockdown.

IC50 is calculated by fitting a 4-parameter logistic curve to dose-response data.
Uses same fitting approach as analyses/summarise_dose_response.py.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
KINETICS_DIR = BASE_DIR / 'kinetics_model' / 'data'
OUTPUT_DIR = Path(__file__).parent / 'figures'


def hill_equation(dose, bottom, top, ic50, hill):
    """4-parameter logistic (Hill) equation for dose-response curves."""
    return bottom + (top - bottom) / (1 + (ic50 / dose) ** hill)


def fit_ic50(doses, responses):
    """
    Fit IC50 from dose-response data using 4-parameter logistic model.
    Same approach as analyses/summarise_dose_response.py.

    Returns dict with IC50, hill coefficient, bottom, top, r_squared, and success flag.
    """
    # Remove NaN values
    mask = ~(np.isnan(doses) | np.isnan(responses))
    doses = np.array(doses)[mask]
    responses = np.array(responses)[mask]

    if len(doses) < 4:
        return {'ic50': np.nan, 'r_squared': np.nan, 'success': False}

    # Initial parameter guesses
    bottom_init = np.min(responses)
    top_init = np.max(responses)
    ic50_init = np.median(doses)
    hill_init = 1.0

    # Bounds
    min_dose = np.min(doses[doses > 0]) if np.any(doses > 0) else 1e-6
    max_dose = np.max(doses)

    bounds = (
        [0, 0, min_dose / 100, 0.1],  # Lower bounds
        [100, 100, max_dose * 100, 10]  # Upper bounds
    )

    try:
        popt, _ = curve_fit(
            hill_equation,
            doses,
            responses,
            p0=[bottom_init, top_init, ic50_init, hill_init],
            bounds=bounds,
            maxfev=5000
        )

        bottom, top, ic50, hill = popt

        # Calculate R-squared
        predicted = hill_equation(doses, *popt)
        ss_res = np.sum((responses - predicted) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Validate IC50 is within reasonable range
        if ic50 < min_dose / 10 or ic50 > max_dose * 10:
            return {'ic50': np.nan, 'r_squared': np.nan, 'success': False}

        return {'ic50': ic50, 'r_squared': r_squared, 'success': True}

    except (RuntimeError, ValueError):
        return {'ic50': np.nan, 'r_squared': np.nan, 'success': False}


def load_dose_response_data() -> pd.DataFrame:
    """Load dose-response data with CCLE mapping."""
    path = DATA_DIR / 'oligostack' / 'processed' / 'dose_response_with_genomic_and_ccle.parquet'
    df = pd.read_parquet(path)

    # Filter to human cells with CCLE mapping
    df = df[
        (df['cell_line_species'] == 'human') &
        df['ccle_model_id'].notna() &
        df['gene_symbol'].notna() &
        df['Inhibition_pct'].notna() &
        df['dosage_nm'].notna()
    ].copy()

    # Clean transfection_method
    df['transfection_method'] = df['transfection_method'].fillna('Unknown')
    df.loc[df['transfection_method'] == 'None', 'transfection_method'] = 'Unknown'

    return df


def calculate_ic50_per_experiment(df: pd.DataFrame, min_r2: float = 0.7) -> pd.DataFrame:
    """Calculate IC50 for each compound-cell line-gene-transfection combination.

    Args:
        df: Dose-response data
        min_r2: Minimum R² for quality filter (default 0.7, same as summarise_dose_response.py)
    """
    results = []

    # Group by experiment (same grouping as summarise_dose_response.py)
    groups = df.groupby(
        ['USPTO ID', 'Table Number', 'Compound ID', 'cell_line', 'gene_symbol',
         'ccle_model_id', 'transfection_method'],
        dropna=False
    )

    print(f"Fitting IC50 for {len(groups)} experiments...")

    for keys, group in groups:
        if len(group) < 4:
            continue

        doses = group['dosage_nm'].values
        responses = group['Inhibition_pct'].values

        result = fit_ic50(doses, responses)

        if result['success'] and result['r_squared'] >= min_r2:
            results.append({
                'USPTO ID': keys[0],
                'Table Number': keys[1],
                'Compound ID': keys[2],
                'cell_line': keys[3],
                'gene_symbol': keys[4],
                'ccle_model_id': keys[5],
                'transfection_method': keys[6],
                'IC50_nM': result['ic50'],
                'fit_r2': result['r_squared'],
                'n_doses': len(doses),
                'ccle_cell_line_name': group['ccle_cell_line_name'].iloc[0] if 'ccle_cell_line_name' in group.columns else None
            })

    return pd.DataFrame(results)


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

    # Build lookup dictionary (uppercase for case-insensitive matching)
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
    """Load RNAdecayCafe half-life data, aggregated to gene level."""
    path = DATA_DIR / 'RNAdecayCafe_v1_onetable.csv.gz'
    df = pd.read_csv(path, compression='gzip')

    # Filter for quality
    df = df[df['uncertainty_log_kdeg'] < 0.3].copy()

    # Aggregate to gene level
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


def merge_ic50_with_kinetics(
    ic50_df: pd.DataFrame,
    tpm_lookup: dict[tuple[str, str], float],
    halflife_lookup: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    """Merge IC50 data with TPM and half-life."""

    # Add TPM values (uppercase for case-insensitive matching)
    ic50_df['log_tpm'] = ic50_df.apply(
        lambda row: tpm_lookup.get((row['gene_symbol'].upper(), row['ccle_model_id'])),
        axis=1
    )

    # Add half-life values
    def get_halflife(gene):
        return halflife_lookup.get(gene.upper() if gene else None, (None, None))

    hl_data = ic50_df['gene_symbol'].apply(get_halflife)
    ic50_df['log_kdeg'] = hl_data.apply(lambda x: x[0])
    ic50_df['halflife_hours'] = hl_data.apply(lambda x: x[1])

    # Calculate log production rate
    ic50_df['log_production'] = ic50_df['log_kdeg'] + ic50_df['log_tpm']

    # Filter to rows with all data
    merged = ic50_df[
        ic50_df['log_tpm'].notna() &
        ic50_df['log_kdeg'].notna() &
        ic50_df['IC50_nM'].notna()
    ].copy()

    # Add log IC50 for analysis
    merged['log_IC50'] = np.log10(merged['IC50_nM'])

    return merged


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run correlation and regression tests for IC50."""
    results = {}

    print(f"Analyzing {len(df)} experiments with valid IC50")
    print()

    # 1. Spearman: TPM vs log(IC50)
    rho, p = stats.spearmanr(df['log_tpm'], df['log_IC50'])
    results['tpm_spearman'] = {'rho': rho, 'p': p}
    print(f"TPM vs log(IC50): Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 2. Spearman: half-life vs log(IC50)
    rho, p = stats.spearmanr(df['halflife_hours'], df['log_IC50'])
    results['halflife_spearman'] = {'rho': rho, 'p': p}
    print(f"Half-life vs log(IC50): Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 3. Spearman: log_kdeg vs log(IC50)
    rho, p = stats.spearmanr(df['log_kdeg'], df['log_IC50'])
    results['kdeg_spearman'] = {'rho': rho, 'p': p}
    print(f"log(k_deg) vs log(IC50): Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 4. Spearman: log production rate vs log(IC50)
    rho, p = stats.spearmanr(df['log_production'], df['log_IC50'])
    results['production_spearman'] = {'rho': rho, 'p': p}
    print(f"log(Production) vs log(IC50): Spearman rho = {rho:.3f}, p = {p:.2e}")

    # 5. OLS regression with all predictors
    try:
        import statsmodels.api as sm

        ols_df = df[['log_tpm', 'log_kdeg', 'transfection_method', 'log_IC50']].dropna()

        # Create dummy variables for transfection method
        transfection_dummies = pd.get_dummies(
            ols_df['transfection_method'], prefix='transfection', drop_first=True, dtype=float
        )

        X = pd.concat([ols_df[['log_tpm', 'log_kdeg']].astype(float), transfection_dummies], axis=1)
        X = sm.add_constant(X)
        y = ols_df['log_IC50']
        model = sm.OLS(y, X).fit()

        results['ols'] = {
            'r2': model.rsquared,
            'tpm_coef': model.params.get('log_tpm', np.nan),
            'tpm_pval': model.pvalues.get('log_tpm', np.nan),
            'kdeg_coef': model.params.get('log_kdeg', np.nan),
            'kdeg_pval': model.pvalues.get('log_kdeg', np.nan),
        }
        print(f"\nOLS Regression (R² = {model.rsquared:.3f}):")
        print(f"  log(TPM): coef = {model.params['log_tpm']:.3f}, p = {model.pvalues['log_tpm']:.2e}")
        print(f"  log(k_deg): coef = {model.params['log_kdeg']:.3f}, p = {model.pvalues['log_kdeg']:.2e}")
        for col in transfection_dummies.columns:
            print(f"  {col}: coef = {model.params[col]:.3f}, p = {model.pvalues[col]:.2e}")
    except ImportError:
        print("statsmodels not available, skipping OLS")
        results['ols'] = None

    # 6. IC50 by transfection method
    print(f"\nIC50 by transfection method:")
    for method in df['transfection_method'].unique():
        subset = df[df['transfection_method'] == method]['IC50_nM']
        print(f"  {method}: n={len(subset)}, median={subset.median():.1f} nM, mean={subset.mean():.1f} nM")

    return results


def plot_tpm_vs_ic50(df: pd.DataFrame, output_path: Path):
    """Scatter plot of TPM vs IC50."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df['log_tpm'],
        df['log_IC50'],
        c=df['halflife_hours'],
        cmap='RdYlBu_r',
        alpha=0.6,
        s=40
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Half-life (hours)', fontsize=10)

    ax.set_xlabel('log(TPM + 1)', fontsize=12)
    ax.set_ylabel('log10(IC50 nM)', fontsize=12)
    ax.set_title('Target Expression vs ASO IC50', fontsize=14)

    # Add regression line
    z = np.polyfit(df['log_tpm'], df['log_IC50'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['log_tpm'].min(), df['log_tpm'].max(), 100)
    ax.plot(x_range, p(x_range), 'k--', alpha=0.5, label='Linear fit')

    # Add correlation
    rho, pval = stats.spearmanr(df['log_tpm'], df['log_IC50'])
    ax.annotate(
        f'Spearman rho = {rho:.3f}\np = {pval:.2e}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_production_vs_ic50(df: pd.DataFrame, output_path: Path):
    """Scatter plot of production rate vs IC50."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df['log_production'],
        df['log_IC50'],
        alpha=0.6,
        s=40,
        c='#3498db',
        edgecolors='white',
        linewidth=0.5
    )

    ax.set_xlabel('log(Production Rate) = log(k_deg) + log(TPM)', fontsize=12)
    ax.set_ylabel('log10(IC50 nM)', fontsize=12)
    ax.set_title('mRNA Production Rate vs ASO IC50', fontsize=14)

    # Add regression line
    z = np.polyfit(df['log_production'], df['log_IC50'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['log_production'].min(), df['log_production'].max(), 100)
    ax.plot(x_range, p(x_range), 'r-', linewidth=2, label='Linear fit')

    # Add correlation
    rho, pval = stats.spearmanr(df['log_production'], df['log_IC50'])
    ax.annotate(
        f'Spearman rho = {rho:.3f}\np = {pval:.2e}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_halflife_vs_ic50(df: pd.DataFrame, output_path: Path):
    """Scatter plot of half-life vs IC50."""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        df['halflife_hours'],
        df['log_IC50'],
        c=df['log_tpm'],
        cmap='viridis',
        alpha=0.6,
        s=40
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log(TPM + 1)', fontsize=10)

    ax.set_xlabel('mRNA Half-life (hours)', fontsize=12)
    ax.set_ylabel('log10(IC50 nM)', fontsize=12)
    ax.set_title('mRNA Stability vs ASO IC50', fontsize=14)

    # Add regression line
    z = np.polyfit(df['halflife_hours'], df['log_IC50'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['halflife_hours'].min(), df['halflife_hours'].max(), 100)
    ax.plot(x_range, p(x_range), 'k--', alpha=0.5, label='Linear fit')

    # Add correlation
    rho, pval = stats.spearmanr(df['halflife_hours'], df['log_IC50'])
    ax.annotate(
        f'Spearman rho = {rho:.3f}\np = {pval:.2e}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ic50_by_transfection(df: pd.DataFrame, output_path: Path):
    """Boxplot of IC50 by transfection method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order by median IC50
    method_order = df.groupby('transfection_method')['IC50_nM'].median().sort_values().index.tolist()
    data = [df[df['transfection_method'] == m]['log_IC50'].values for m in method_order]

    bp = ax.boxplot(data, labels=method_order, patch_artist=True)

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors[:len(method_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Transfection Method', fontsize=12)
    ax.set_ylabel('log10(IC50 nM)', fontsize=12)
    ax.set_title('ASO IC50 by Transfection Method', fontsize=14)

    # Add sample sizes
    for i, method in enumerate(method_order):
        subset = df[df['transfection_method'] == method]
        median_ic50 = subset['IC50_nM'].median()
        ax.annotate(f'n={len(subset)}\nmed={median_ic50:.0f}nM',
                    xy=(i+1, ax.get_ylim()[1] - 0.2),
                    ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ic50_distribution(df: pd.DataFrame, output_path: Path):
    """Histogram of IC50 values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df['log_IC50'], bins=30, edgecolor='white', alpha=0.7, color='#3498db')

    ax.set_xlabel('log10(IC50 nM)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of IC50 Values (n={len(df)})', fontsize=14)

    # Add summary stats
    median = df['IC50_nM'].median()
    mean = df['IC50_nM'].mean()
    ax.axvline(np.log10(median), color='red', linestyle='--', label=f'Median: {median:.0f} nM')
    ax.axvline(np.log10(mean), color='orange', linestyle=':', label=f'Mean: {mean:.0f} nM')

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Hypothesis 006b: mRNA Production Rate vs ASO IC50")
    print("=" * 60)
    print()

    # Load data
    print("Loading dose-response data...")
    dr_df = load_dose_response_data()
    print(f"  Dose-response data: {len(dr_df):,} rows (human, CCLE-mapped)")
    print(f"    Unique compounds: {dr_df['Compound ID'].nunique()}")
    print(f"    Unique genes: {dr_df['gene_symbol'].nunique()}")
    print(f"    Unique cell lines: {dr_df['ccle_model_id'].nunique()}")
    print()

    # Calculate IC50 for each experiment
    print("Calculating IC50 values...")
    ic50_df = calculate_ic50_per_experiment(dr_df)
    print(f"  Successful IC50 fits: {len(ic50_df):,}")
    print(f"    Unique genes: {ic50_df['gene_symbol'].nunique()}")
    print(f"    Unique cell lines: {ic50_df['ccle_model_id'].nunique()}")
    print()

    # Load kinetics data
    print("Loading kinetics data...")
    tpm_lookup = load_tpm_data()
    print(f"  TPM lookup: {len(tpm_lookup):,} (gene, cell) pairs")

    halflife_lookup = load_halflife_data()
    print(f"  Half-life lookup: {len(halflife_lookup):,} genes")
    print()

    # Merge data
    print("Merging datasets...")
    merged = merge_ic50_with_kinetics(ic50_df, tpm_lookup, halflife_lookup)
    print(f"  Final dataset: {len(merged):,} experiments with IC50, TPM, and half-life")
    print(f"    Unique genes: {merged['gene_symbol'].nunique()}")
    print(f"    Transfection methods: {merged['transfection_method'].value_counts().to_dict()}")
    print()

    # Summary statistics
    print("Summary statistics:")
    print(f"  IC50 (nM): median={merged['IC50_nM'].median():.1f}, mean={merged['IC50_nM'].mean():.1f}")
    print(f"  log(TPM+1): mean={merged['log_tpm'].mean():.2f}, std={merged['log_tpm'].std():.2f}")
    print(f"  Half-life (hours): mean={merged['halflife_hours'].mean():.1f}, std={merged['halflife_hours'].std():.1f}")
    print()

    # Run statistical tests
    print("Statistical tests:")
    print("-" * 40)
    results = run_statistical_tests(merged)
    print()

    # Generate figures
    print("Generating figures...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_ic50_distribution(merged, OUTPUT_DIR / 'ic50_distribution.png')
    plot_tpm_vs_ic50(merged, OUTPUT_DIR / 'tpm_vs_ic50.png')
    plot_production_vs_ic50(merged, OUTPUT_DIR / 'production_vs_ic50.png')
    plot_halflife_vs_ic50(merged, OUTPUT_DIR / 'halflife_vs_ic50.png')
    plot_ic50_by_transfection(merged, OUTPUT_DIR / 'ic50_by_transfection.png')

    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    return merged, results


if __name__ == '__main__':
    merged, results = main()
