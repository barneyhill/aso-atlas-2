"""
Hypothesis 007: CNS Knockdown vs Neurotoxicity

Investigates relationship between knockdown efficacy (% UTC) across brain regions
and neurotoxicity (FOB score) in mouse and rat models.

Key questions:
- Does better knockdown correlate with higher neurotoxicity?
- Does this relationship vary by brain region?
- Does this relationship vary by target gene?
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"

# Distinctive colors for targets
DISTINCTIVE_COLORS = [
    '#e6194B',  # red
    '#3cb44b',  # green
    '#4363d8',  # blue
    '#f58231',  # orange
    '#911eb4',  # purple
    '#42d4f4',  # cyan
    '#f032e6',  # magenta
    '#bfef45',  # lime
    '#fabed4',  # pink
    '#469990',  # teal
    '#dcbeff',  # lavender
    '#9A6324',  # brown
]

BRAIN_REGIONS = ['Cortex', 'Hippocampus', 'Spinal Cord', 'Cerebellum', 'Brain Stem']


def parse_fob(val):
    """Parse FOB score from various formats."""
    if val is None:
        return np.nan
    if isinstance(val, np.ndarray):
        return float(val[0]) if len(val) > 0 else np.nan
    if isinstance(val, list):
        return float(val[0]) if len(val) > 0 else np.nan
    try:
        return float(val)
    except:
        return np.nan


def load_data():
    """Load and prepare rodent activity and neurotoxicity data."""
    rodent = pd.read_csv(DATA_DIR / "moeacuterodent_activity.csv")
    neuro = pd.read_parquet(DATA_DIR / "oligostack/processed/neurotoxicity_processed.parquet")

    rodent['Dose_ug'] = rodent['Dose'].str.replace('ug', '').astype(float)
    neuro['FOB_score_parsed'] = neuro['FOB_score'].apply(parse_fob)

    return rodent, neuro


def prepare_neuro_by_species(neuro):
    """Aggregate neurotoxicity data by species."""
    neuro_mouse = neuro[neuro['species'] == 'Mouse'].groupby('Compound ID').agg({
        'FOB_score_parsed': 'mean',
        'dosage_ug': 'first',
        'administration_method': 'first'
    }).reset_index()

    neuro_rat = neuro[neuro['species'] == 'Rat'].groupby('Compound ID').agg({
        'FOB_score_parsed': 'mean',
        'dosage_ug': 'first',
        'administration_method': 'first'
    }).reset_index()

    return neuro_mouse, neuro_rat


def fit_model(merged, region):
    """Fit linear model: FOB ~ UTC (+ Dose if variable)."""
    if len(merged) < 3:
        return None, None, None

    if merged['Dose_ug'].nunique() > 1:
        X = sm.add_constant(merged[[region, 'Dose_ug']])
        model = sm.OLS(merged['FOB_score_parsed'], X).fit()
    else:
        X = sm.add_constant(merged[region])
        model = sm.OLS(merged['FOB_score_parsed'], X).fit()

    pval = model.pvalues[region]
    beta = model.params[region]

    return model, beta, pval


def fit_model_inhibition(merged):
    """Fit linear model: FOB ~ Inhibition + Dose."""
    if len(merged) < 3:
        return None, None, None

    # Use Inhibition + Dose if dose varies, otherwise just Inhibition
    if merged['Dose_ug'].nunique() > 1:
        X = sm.add_constant(merged[['Inhibition', 'Dose_ug']])
        model = sm.OLS(merged['FOB_score_parsed'], X).fit()
        pval = model.pvalues['Inhibition']
        beta = model.params['Inhibition']
    else:
        X = sm.add_constant(merged['Inhibition'])
        model = sm.OLS(merged['FOB_score_parsed'], X).fit()
        pval = model.pvalues['Inhibition']
        beta = model.params['Inhibition']

    return model, beta, pval


def create_region_plots(rodent, neuro_mouse, neuro_rat, target_colors):
    """Create 2xN grid of plots for all brain regions."""
    fig, axes = plt.subplots(2, len(BRAIN_REGIONS), figsize=(20, 9))
    all_handles = {}

    for col, region in enumerate(BRAIN_REGIONS):
        for row, (species, neuro_data, admin, fob_type) in enumerate([
            ('Mouse', neuro_mouse, 'ICV', 'Behavioral'),
            ('Rat', neuro_rat, 'IT', 'Body Parts')
        ]):
            ax = axes[row, col]
            rodent_species = rodent[rodent['Species'] == species].rename(
                columns={'Compound': 'Compound ID'}
            )

            merged = pd.merge(
                rodent_species[['Compound ID', region, 'Target', 'Dose_ug']],
                neuro_data,
                on='Compound ID',
                how='inner'
            ).dropna(subset=[region, 'FOB_score_parsed'])

            # Convert UTC to Inhibition and clip to 0-100
            merged['Inhibition'] = (100 - merged[region]).clip(0, 100)

            if len(merged) < 3:
                ax.text(0.5, 0.5, f'No data\n(n={len(merged)})', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_xlabel(f'{region} Inhibition (%)')
                ax.set_ylabel(f'{fob_type} FOB ({admin})')
                if row == 0:
                    ax.set_title(f'{region}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                continue

            _, beta, pval = fit_model_inhibition(merged)

            for target in merged['Target'].dropna().unique():
                subset = merged[merged['Target'] == target]
                scatter = ax.scatter(
                    subset['Inhibition'], subset['FOB_score_parsed'],
                    alpha=0.7, c=target_colors[target], s=35,
                    edgecolors='white', linewidth=0.5, label=target
                )
                if target not in all_handles:
                    all_handles[target] = scatter

            ax.set_xlabel(f'{region} Inhibition (%)')
            ax.set_ylabel(f'{fob_type} FOB ({admin})')
            ax.set_xlim(-5, 105)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(f'{region}', fontsize=12, fontweight='bold')

            ns_marker = ' (n.s.)' if pval > 0.05 else ''
            ax.text(0.05, 0.95, f'n={len(merged)}\nβ={beta:.1e}\np={pval:.1e}{ns_marker}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.text(0.02, 0.70, 'Mouse\nTox: 700ug ICV', ha='center', va='center',
             fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.28, 'Rat\nTox: 3000ug IT', ha='center', va='center',
             fontsize=12, fontweight='bold', rotation=90)

    fig.legend(all_handles.values(), all_handles.keys(), loc='upper center',
              ncol=len(all_handles), fontsize=10, title='Target', bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0.03, 0.02, 1, 0.93])
    return fig


def analyse_by_target(rodent, neuro_mouse, neuro_rat):
    """Analyse relationship broken down by target gene."""
    results = []

    for species, neuro_data, admin in [
        ('Mouse', neuro_mouse, 'ICV'),
        ('Rat', neuro_rat, 'IT')
    ]:
        rodent_species = rodent[rodent['Species'] == species].rename(
            columns={'Compound': 'Compound ID'}
        )

        for region in BRAIN_REGIONS:
            merged = pd.merge(
                rodent_species[['Compound ID', region, 'Target', 'Dose_ug']],
                neuro_data,
                on='Compound ID',
                how='inner'
            ).dropna(subset=[region, 'FOB_score_parsed'])

            # Convert UTC to Inhibition and clip to 0-100
            merged['Inhibition'] = (100 - merged[region]).clip(0, 100)

            for target in merged['Target'].dropna().unique():
                target_data = merged[merged['Target'] == target]

                if len(target_data) < 5:
                    continue

                # FOB ~ Inhibition + Dose (if dose varies)
                try:
                    if target_data['Dose_ug'].nunique() > 1:
                        X = sm.add_constant(target_data[['Inhibition', 'Dose_ug']])
                        model = sm.OLS(target_data['FOB_score_parsed'], X).fit()
                        pval = model.pvalues['Inhibition']
                        beta = model.params['Inhibition']
                    else:
                        X = sm.add_constant(target_data['Inhibition'])
                        model = sm.OLS(target_data['FOB_score_parsed'], X).fit()
                        pval = model.pvalues['Inhibition']
                        beta = model.params['Inhibition']
                except:
                    continue

                results.append({
                    'Species': species,
                    'Region': region,
                    'Target': target,
                    'n': len(target_data),
                    'beta': beta,
                    'p_value': pval,
                    'significant': pval < 0.05
                })

    return pd.DataFrame(results)


def print_target_table(df):
    """Print formatted table of per-target results."""
    print("\n" + "=" * 90)
    print("KNOCKDOWN vs NEUROTOXICITY BY TARGET")
    print("=" * 90)

    for species in ['Mouse', 'Rat']:
        species_df = df[df['Species'] == species]
        if len(species_df) == 0:
            continue

        print(f"\n{species.upper()}")
        print("-" * 90)
        print(f"{'Region':<15} {'Target':<12} {'n':>6} {'beta':>12} {'p-value':>12} {'Sig':>6}")
        print("-" * 90)

        for region in BRAIN_REGIONS:
            region_df = species_df[species_df['Region'] == region].sort_values('p_value')
            for _, row in region_df.iterrows():
                sig = '*' if row['significant'] else 'n.s.'
                print(f"{row['Region']:<15} {row['Target']:<12} {row['n']:>6} "
                      f"{row['beta']:>12.2e} {row['p_value']:>12.2e} {sig:>6}")
        print()


def main():
    print("Loading data...")
    rodent, neuro = load_data()
    neuro_mouse, neuro_rat = prepare_neuro_by_species(neuro)

    # Get all targets and assign colors
    all_targets = list(rodent['Target'].dropna().unique())
    target_colors = dict(zip(all_targets, DISTINCTIVE_COLORS[:len(all_targets)]))

    # Create and save plots
    print("Creating region plots...")
    fig = create_region_plots(rodent, neuro_mouse, neuro_rat, target_colors)
    fig.savefig(PLOT_DIR / 'utc_vs_neurotox_all_regions.png', dpi=150, bbox_inches='tight')
    fig.savefig(PLOT_DIR / 'utc_vs_neurotox_all_regions.svg', bbox_inches='tight')
    print(f"Saved: {PLOT_DIR / 'utc_vs_neurotox_all_regions.png'}")

    # Analyse by target
    print("\nAnalysing by target...")
    target_results = analyse_by_target(rodent, neuro_mouse, neuro_rat)
    print_target_table(target_results)

    # Save results
    output_path = Path(__file__).parent / 'target_results.csv'
    target_results.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # Summary of significant findings
    sig_results = target_results[target_results['significant']]
    if len(sig_results) > 0:
        print("\n" + "=" * 90)
        print("SIGNIFICANT FINDINGS (p < 0.05)")
        print("=" * 90)
        for _, row in sig_results.iterrows():
            direction = "positive" if row['beta'] > 0 else "negative"
            print(f"  {row['Species']} {row['Region']} - {row['Target']}: "
                  f"β={row['beta']:.2e} ({direction}), p={row['p_value']:.2e}, n={row['n']}")


if __name__ == "__main__":
    main()
