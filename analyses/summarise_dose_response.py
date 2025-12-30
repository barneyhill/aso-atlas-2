import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Dose-Response IC50 Analysis

    Calculate IC50 values from dose-response curves using a 4-parameter logistic model (Hill equation).
    Valid IC50s require at least 4 measurements within the same Table Number.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from scipy.optimize import curve_fit
    from pathlib import Path
    import warnings

    # Get the directory where this script is located
    _script_dir = Path(__file__).parent

    # Load dose-response data
    df = pd.read_parquet(_script_dir / "../data/oligostack/processed/dose_response_processed.parquet")

    print(f"Loaded {len(df):,} dose-response measurements")
    print(f"Unique compounds: {df['Compound ID'].nunique():,}")
    print(f"Transfection methods: {df['transfection_method'].dropna().unique().tolist()}")
    return Path, curve_fit, df, np, pd, warnings


@app.cell
def _(df):
    # Group data by Table Number, Compound ID, cell_line, and transfection_method
    # Each group represents a single dose-response curve

    grouped = df.groupby(
        ['Table Number', 'Compound ID', 'cell_line', 'transfection_method'],
        dropna=False
    )

    # Count measurements per curve
    curve_counts = grouped.size().reset_index(name='n_measurements')

    # Filter for curves with 4+ measurements (valid for IC50 fitting)
    valid_curves = curve_counts[curve_counts['n_measurements'] >= 4].copy()

    print(f"Total dose-response curves: {len(curve_counts):,}")
    print(f"Valid curves (4+ measurements): {len(valid_curves):,}")
    print(f"\nMeasurements per curve distribution (valid curves only):")
    print(valid_curves['n_measurements'].describe())
    return


@app.cell
def _(curve_fit, np):
    # Define 4-parameter logistic (Hill) function for dose-response
    # Response = Bottom + (Top - Bottom) / (1 + (IC50/Dose)^Hill)

    def hill_equation(dose, bottom, top, ic50, hill):
        """4-parameter logistic (Hill) equation for dose-response curves."""
        return bottom + (top - bottom) / (1 + (ic50 / dose) ** hill)

    def fit_ic50(doses, responses):
        """
        Fit IC50 from dose-response data using 4-parameter logistic model.

        Returns dict with IC50, hill coefficient, bottom, top, r_squared, and success flag.
        """
        # Remove NaN values
        mask = ~(np.isnan(doses) | np.isnan(responses))
        doses = np.array(doses)[mask]
        responses = np.array(responses)[mask]

        if len(doses) < 4:
            return {'ic50': np.nan, 'hill': np.nan, 'bottom': np.nan,
                    'top': np.nan, 'r_squared': np.nan, 'success': False}

        # Initial parameter guesses
        bottom_init = np.min(responses)
        top_init = np.max(responses)
        ic50_init = np.median(doses)
        hill_init = 1.0

        # Bounds: bottom [0, 100], top [0, 100], IC50 [min_dose/100, max_dose*100], hill [0.1, 10]
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
                return {'ic50': np.nan, 'hill': np.nan, 'bottom': np.nan,
                        'top': np.nan, 'r_squared': np.nan, 'success': False}

            return {
                'ic50': ic50,
                'hill': hill,
                'bottom': bottom,
                'top': top,
                'r_squared': r_squared,
                'success': True
            }

        except (RuntimeError, ValueError):
            return {'ic50': np.nan, 'hill': np.nan, 'bottom': np.nan,
                    'top': np.nan, 'r_squared': np.nan, 'success': False}
    return (fit_ic50,)


@app.cell
def _(df, fit_ic50, pd, warnings):
    from joblib import Parallel, delayed

    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # Prepare curve data
    _grouped = df.groupby(
        ['USPTO ID', 'Table Number', 'Compound ID', 'cell_line', 'transfection_method'],
        dropna=False
    )

    curves = [
        (key, g['dosage_nm'].values, g['Inhibition_pct'].values)
        for key, g in _grouped if len(g) >= 4
    ]

    def _fit_one(key, doses, responses):
        r = fit_ic50(doses, responses)
        return {
            'USPTO ID': key[0], 'Table Number': key[1], 'Compound ID': key[2],
            'cell_line': key[3], 'transfection_method': key[4],
            'n_measurements': len(doses), 'ic50_nm': r['ic50'],
            'hill_coefficient': r['hill'], 'bottom': r['bottom'],
            'top': r['top'], 'r_squared': r['r_squared'], 'fit_success': r['success']
        }

    ic50_results = Parallel(n_jobs=16)(
        delayed(_fit_one)(k, d, r) for k, d, r in curves
    )

    ic50_df = pd.DataFrame(ic50_results)

    # Summary statistics
    successful_fits = ic50_df[ic50_df['fit_success']]
    print(f"IC50 fitting complete:")
    print(f"  Total curves attempted: {len(ic50_df):,}")
    print(f"  Successful fits: {len(successful_fits):,} ({100*len(successful_fits)/len(ic50_df):.1f}%)")
    print(f"\nIC50 distribution (nM) for successful fits:")
    print(successful_fits['ic50_nm'].describe())
    return (ic50_df,)


@app.cell
def _(ic50_df):
    # Filter to high-quality fits (R² > 0.7) for distribution analysis
    quality_ic50 = ic50_df[
        (ic50_df['fit_success']) &
        (ic50_df['r_squared'] > 0.7)
    ].copy()

    # Exclude curves with no transfection method
    quality_ic50 = quality_ic50[quality_ic50['transfection_method'].notna()]

    print(f"High-quality IC50 fits (R² > 0.7): {len(quality_ic50):,}")
    print(f"\nBreakdown by transfection method:")
    print(quality_ic50['transfection_method'].value_counts())
    return (quality_ic50,)


@app.cell
def _(Path, np, quality_ic50):
    import matplotlib.pyplot as plt

    # Plot IC50 distributions by transfection method (3 stacked on top of each other, shared x-axis)
    methods_order = ['Lipofection', 'Gymnosis', 'Electroporation']
    methods_present = [m for m in methods_order if m in quality_ic50['transfection_method'].unique()]
    n_methods = len(methods_present)

    # Color palette for transfection methods
    colors = {
        'Electroporation': '#1f77b4',
        'Gymnosis': '#ff7f0e',
        'Lipofection': '#2ca02c'
    }

    fig, axes = plt.subplots(
        n_methods, 1, figsize=(7, 2.5 * n_methods), sharex=True, dpi=150
    )
    if n_methods == 1:
        axes = [axes]

    for idx, (_ax, _method) in enumerate(zip(axes, methods_present)):
        _subset = quality_ic50[quality_ic50['transfection_method'] == _method]
        _ic50_values = _subset['ic50_nm']
        # Use log scale for IC50 values
        _log_ic50 = np.log10(_ic50_values[_ic50_values > 0])

        _ax.hist(
            _log_ic50,
            bins=30,
            color=colors.get(_method, 'gray'),
            edgecolor='black',
            alpha=0.7
        )

        _median_ic50 = np.median(_ic50_values)
        if np.isfinite(_median_ic50) and _median_ic50 > 0:
            _ax.axvline(
                np.log10(_median_ic50),
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Median: {_median_ic50:.1f} nM'
            )

        _ax.set_ylabel('Count', fontsize=12)
        _ax.set_title(f'{_method}\n(n={len(_subset):,})', fontsize=13)
        _ax.legend(loc='upper right')
        _ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('log₁₀(IC50, nM)', fontsize=12)

    plt.tight_layout()
    fig.savefig(Path(__file__).parent / "../plots/ic50_distributions_stacked_by_transfection.svg", format="svg")
    fig
    return (plt,)


@app.cell
def _(Path, ic50_df, np, pd, plt):
    # Load hepatic data and filter to mice only
    _hepatic = pd.read_parquet(Path(__file__).parent / "../data/oligostack/processed/hepatictoxicity_processed.parquet")
    _hepatic = _hepatic[_hepatic['species'] == 'mouse']

    _alt_rows = []
    for _, _row in _hepatic.iterrows():
        _alt = _row['ALT']
        if isinstance(_alt, (np.ndarray, list)):
            _valid = [v for v in _alt if v is not None and not np.isnan(v)]
            _alt_mean = np.mean(_valid) if _valid else np.nan
        elif _alt is not None and not np.isnan(_alt):
            _alt_mean = _alt
        else:
            _alt_mean = np.nan
        if not np.isnan(_alt_mean) and _alt_mean > 0:
            _alt_rows.append({'Compound ID': _row['Compound ID'], 'ALT': _alt_mean})

    _alt_df = pd.DataFrame(_alt_rows).groupby('Compound ID')['ALT'].mean().reset_index()

    # Filter IC50 to electroporation with good fits
    _elec_ic50 = ic50_df[
        (ic50_df['transfection_method'] == 'Electroporation') &
        (ic50_df['fit_success']) &
        (ic50_df['r_squared'] > 0.7)
    ][['Compound ID', 'ic50_nm']].groupby('Compound ID')['ic50_nm'].median().reset_index()

    # Merge
    _merged = _elec_ic50.merge(_alt_df, on='Compound ID')
    print(f"Matched {len(_merged):,} compounds with both IC50 and ALT data (mice only)")

    # Plot
    _fig, _ax = plt.subplots(figsize=(7, 6), dpi=150)
    _ax.scatter(_merged['ic50_nm'], _merged['ALT'], alpha=0.5, s=20, edgecolors='none')
    _ax.set_xscale('log')
    _ax.set_yscale('log')
    # Spearman correlation
    from scipy.stats import spearmanr
    _rho, _pval = spearmanr(_merged['ic50_nm'], _merged['ALT'])

    _ax.set_xlabel('IC50 (nM)', fontsize=12)
    _ax.set_ylabel('ALT (U/L)', fontsize=12)
    _ax.set_title(f'IC50 vs ALT (Electroporation only, mice only, n={len(_merged):,})\nSpearman ρ={_rho:.3f}, p={_pval:.2e}', fontsize=14)
    _ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    _fig.savefig(Path(__file__).parent / "../plots/ic50_vs_ALT_electroporation.svg", format="svg")
    _fig
    return


@app.cell
def _(Path, ic50_df, np, pd):
    import statsmodels.formula.api as smf

    # Load hepatic data (all species)
    _hep = pd.read_parquet(Path(__file__).parent / "../data/oligostack/processed/hepatictoxicity_processed.parquet").copy()

    # Calculate mean ALT and mg/kg/day per Compound ID, keep Hepatic Table
    _rows = []
    for _, _r in _hep.iterrows():
        _alt = _r['ALT']
        if isinstance(_alt, (np.ndarray, list)):
            _valid = [v for v in _alt if v is not None and not np.isnan(v)]
            _alt_mean = np.mean(_valid) if _valid else np.nan
        elif _alt is not None and not np.isnan(_alt):
            _alt_mean = _alt
        else:
            _alt_mean = np.nan

        _dose = _r['dosage_mg_per_kg']
        _ndays = _r['dosing_period_days']
        _ndoses = _r['num_doses']
        if pd.notna(_dose) and pd.notna(_ndays) and _ndays > 0 and pd.notna(_ndoses):
            _mg_kg_day = (_dose * _ndoses) / _ndays
        else:
            _mg_kg_day = np.nan

        if not np.isnan(_alt_mean) and _alt_mean > 0 and not np.isnan(_mg_kg_day) and _mg_kg_day > 0:
            _rows.append({
                'Compound ID': _r['Compound ID'],
                'ALT': _alt_mean,
                'mg_kg_day': _mg_kg_day,
                'species': _r['species']
            })

    _hep_df = pd.DataFrame(_rows).groupby(['Compound ID', 'species']).agg({'ALT': 'mean', 'mg_kg_day': 'mean'}).reset_index()

    # Get IC50 with transfection method and USPTO ID
    _ic50 = ic50_df[
        (ic50_df['fit_success']) & (ic50_df['r_squared'] > 0.7)
    ][['Compound ID', 'USPTO ID', 'ic50_nm', 'transfection_method']].copy()
    _ic50 = _ic50.rename(columns={'USPTO ID': 'IC50_USPTO'})
    _ic50 = _ic50.groupby(['Compound ID', 'IC50_USPTO', 'transfection_method'])['ic50_nm'].median().reset_index()

    # Merge and filter to mice only (avoids survivorship bias)
    _model_df = _ic50.merge(_hep_df, on='Compound ID').dropna()
    _model_df = _model_df[_model_df['species'] == 'mouse']
    _model_df['log_ALT'] = np.log10(_model_df['ALT'])
    _model_df['log_IC50'] = np.log10(_model_df['ic50_nm'])
    _model_df['log_mg_kg_day'] = np.log10(_model_df['mg_kg_day'])

    print(f"Model data: {len(_model_df):,} observations")
    print(f"IC50 USPTOs: {_model_df['IC50_USPTO'].nunique()}")
    print(f"Transfection: {_model_df['transfection_method'].value_counts().to_dict()}")
    print(f"Species: {_model_df['species'].value_counts().to_dict()}")

    # Mixed effects: random intercept for USPTO ID (anchors IC50 by patent/study)
    _model = smf.mixedlm(
        'log_ALT ~ log_IC50 * C(transfection_method) + log_mg_kg_day',
        data=_model_df,
        groups=_model_df['IC50_USPTO']
    ).fit()

    # Extract results with scientific notation for p-values
    _summary = pd.DataFrame({
        'Coef': _model.fe_params,
        'Std.Err': _model.bse_fe,
        'z': _model.tvalues,
        'P>|z|': _model.pvalues
    }).drop(index='Group Var', errors='ignore')
    _summary['P>|z|'] = _summary['P>|z|'].apply(lambda x: f'{x:.2e}')

    # Calculate R² (Nakagawa & Schielzeth method)
    _var_fixed = _model.fittedvalues.var()
    _var_random = _model.cov_re.iloc[0, 0]
    _var_resid = _model.scale
    _var_total = _var_fixed + _var_random + _var_resid
    _r2_marginal = _var_fixed / _var_total  # Fixed effects only
    _r2_conditional = (_var_fixed + _var_random) / _var_total  # Fixed + random

    print(f"Mixed Linear Model (n={len(_model_df):,}, groups={_model_df['IC50_USPTO'].nunique()} USPTO IDs)")
    print(f"R² marginal: {_r2_marginal:.3f} (fixed effects only)")
    print(f"R² conditional: {_r2_conditional:.3f} (fixed + random)")
    print(f"Group Var: {_model.cov_re.iloc[0,0]:.4f}")
    print(_summary.round(4).to_string())

    # Diagnostics
    from scipy import stats as _stats
    import matplotlib.pyplot as _plt

    _resid = _model.resid
    _fitted = _model.fittedvalues
    _re = _model.random_effects

    _fig, _axes = _plt.subplots(2, 2, figsize=(10, 8), dpi=150)

    # 1. Residuals vs Fitted
    _axes[0, 0].scatter(_fitted, _resid, alpha=0.3, s=10)
    _axes[0, 0].axhline(0, color='red', linestyle='--')
    _axes[0, 0].set_xlabel('Fitted values')
    _axes[0, 0].set_ylabel('Residuals')
    _axes[0, 0].set_title('Residuals vs Fitted')

    # 2. Q-Q plot of residuals
    _stats.probplot(_resid, dist="norm", plot=_axes[0, 1])
    _axes[0, 1].set_title('Q-Q Plot (Residuals)')

    # 3. Histogram of residuals
    _axes[1, 0].hist(_resid, bins=40, edgecolor='black', alpha=0.7)
    _axes[1, 0].set_xlabel('Residuals')
    _axes[1, 0].set_ylabel('Count')
    _sw_stat, _sw_p = _stats.shapiro(_resid[:5000] if len(_resid) > 5000 else _resid)
    _axes[1, 0].set_title(f'Residual Distribution (Shapiro p={_sw_p:.2e})')

    # 4. Q-Q plot of random effects
    _re_vals = [v.iloc[0] for v in _re.values()]
    _stats.probplot(_re_vals, dist="norm", plot=_axes[1, 1])
    _axes[1, 1].set_title('Q-Q Plot (Random Intercepts)')

    _plt.tight_layout()
    _fig.savefig(Path(__file__).parent / "../plots/model_diagnostics.svg", format="svg")
    _fig
    return


@app.cell
def _(Path, ic50_df, np, pd):
    import statsmodels.formula.api as smf
    from scipy import stats as sp_stats

    # Prepare data (same as previous cell)
    _hep = pd.read_parquet(Path(__file__).parent / "../data/oligostack/processed/hepatictoxicity_processed.parquet").copy()

    _rows = []
    for _, _r in _hep.iterrows():
        _alt = _r['ALT']
        if isinstance(_alt, (np.ndarray, list)):
            _valid = [v for v in _alt if v is not None and not np.isnan(v)]
            _alt_mean = np.mean(_valid) if _valid else np.nan
        elif _alt is not None and not np.isnan(_alt):
            _alt_mean = _alt
        else:
            _alt_mean = np.nan

        _dose = _r['dosage_mg_per_kg']
        _ndays = _r['dosing_period_days']
        _ndoses = _r['num_doses']
        if pd.notna(_dose) and pd.notna(_ndays) and _ndays > 0 and pd.notna(_ndoses):
            _mg_kg_day = (_dose * _ndoses) / _ndays
        else:
            _mg_kg_day = np.nan

        if not np.isnan(_alt_mean) and _alt_mean > 0 and not np.isnan(_mg_kg_day) and _mg_kg_day > 0:
            _rows.append({
                'Compound ID': _r['Compound ID'],
                'ALT': _alt_mean,
                'mg_kg_day': _mg_kg_day,
                'species': _r['species']
            })

    _hep_df = pd.DataFrame(_rows).groupby(['Compound ID', 'species']).agg({'ALT': 'mean', 'mg_kg_day': 'mean'}).reset_index()

    _ic50 = ic50_df[
        (ic50_df['fit_success']) & (ic50_df['r_squared'] > 0.7)
    ][['Compound ID', 'USPTO ID', 'ic50_nm', 'transfection_method']].copy()
    _ic50 = _ic50.rename(columns={'USPTO ID': 'IC50_USPTO'})
    _ic50 = _ic50.groupby(['Compound ID', 'IC50_USPTO', 'transfection_method'])['ic50_nm'].median().reset_index()

    _model_df = _ic50.merge(_hep_df, on='Compound ID').dropna()
    _model_df = _model_df[_model_df['species'] == 'mouse']
    _model_df['log_ALT'] = np.log10(_model_df['ALT'])
    _model_df['log_IC50'] = np.log10(_model_df['ic50_nm'])
    _model_df['log_mg_kg_day'] = np.log10(_model_df['mg_kg_day'])

    # Define model specifications
    _models = {
        'M1: Base (IC50×Transfection + Dose)':
            'log_ALT ~ log_IC50 * C(transfection_method) + log_mg_kg_day',
        'M2: + IC50×Dose':
            'log_ALT ~ log_IC50 * C(transfection_method) + log_IC50 * log_mg_kg_day',
        'M3: Three-way (IC50×Transfection×Dose)':
            'log_ALT ~ log_IC50 * C(transfection_method) * log_mg_kg_day',
    }

    # Fit all models
    _results = {}
    for _name, _formula in _models.items():
        _fit = smf.mixedlm(
            _formula,
            data=_model_df,
            groups=_model_df['IC50_USPTO']
        ).fit()
        _results[_name] = _fit

    # Model comparison table
    print("=" * 80)
    print("MODEL COMPARISON: IC50×Dose Interaction")
    print("=" * 80)
    print(f"Data: n={len(_model_df):,} observations, {_model_df['IC50_USPTO'].nunique()} USPTO groups (mice only)")
    print()

    _comparison = []
    for _name, _fit in _results.items():
        _var_fixed = _fit.fittedvalues.var()
        _var_random = _fit.cov_re.iloc[0, 0]
        _var_resid = _fit.scale
        _var_total = _var_fixed + _var_random + _var_resid
        _r2_marg = _var_fixed / _var_total
        _r2_cond = (_var_fixed + _var_random) / _var_total

        _comparison.append({
            'Model': _name,
            'AIC': _fit.aic,
            'BIC': _fit.bic,
            'Log-Lik': _fit.llf,
            'R²_marg': _r2_marg,
            'R²_cond': _r2_cond,
            'n_params': len(_fit.fe_params) + 1  # +1 for random effect variance
        })

    _comp_df = pd.DataFrame(_comparison)
    print("Model Fit Statistics:")
    print(_comp_df.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print()

    # Likelihood ratio tests (nested models)
    print("Likelihood Ratio Tests:")

    # M1 vs M2
    _lr_12 = 2 * (_results['M2: + IC50×Dose'].llf - _results['M1: Base (IC50×Transfection + Dose)'].llf)
    _df_12 = _comparison[1]['n_params'] - _comparison[0]['n_params']
    _p_12 = 1 - sp_stats.chi2.cdf(_lr_12, _df_12)
    print(f"  M1 vs M2 (adding IC50×Dose): LR={_lr_12:.2f}, df={_df_12}, p={_p_12:.2e}")

    # M2 vs M3
    _lr_23 = 2 * (_results['M3: Three-way (IC50×Transfection×Dose)'].llf - _results['M2: + IC50×Dose'].llf)
    _df_23 = _comparison[2]['n_params'] - _comparison[1]['n_params']
    _p_23 = 1 - sp_stats.chi2.cdf(_lr_23, _df_23)
    print(f"  M2 vs M3 (adding three-way): LR={_lr_23:.2f}, df={_df_23}, p={_p_23:.2e}")

    # M1 vs M3
    _lr_13 = 2 * (_results['M3: Three-way (IC50×Transfection×Dose)'].llf - _results['M1: Base (IC50×Transfection + Dose)'].llf)
    _df_13 = _comparison[2]['n_params'] - _comparison[0]['n_params']
    _p_13 = 1 - sp_stats.chi2.cdf(_lr_13, _df_13)
    print(f"  M1 vs M3 (full comparison): LR={_lr_13:.2f}, df={_df_13}, p={_p_13:.2e}")
    print()

    # Print coefficients for best model (lowest AIC)
    _best_name = _comp_df.loc[_comp_df['AIC'].idxmin(), 'Model']
    _best_fit = _results[_best_name]
    print(f"Best Model by AIC: {_best_name}")
    print("-" * 60)

    _summary = pd.DataFrame({
        'Coef': _best_fit.fe_params,
        'Std.Err': _best_fit.bse_fe,
        'z': _best_fit.tvalues,
        'P>|z|': _best_fit.pvalues
    }).drop(index='Group Var', errors='ignore')
    _summary['P>|z|'] = _summary['P>|z|'].apply(lambda x: f'{x:.2e}')
    _summary['Sig'] = _best_fit.pvalues.apply(lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))
    _summary = _summary.drop(index='Group Var', errors='ignore')

    print(_summary.round(4).to_string())
    print()
    print(f"Group Variance (USPTO): {_best_fit.cov_re.iloc[0,0]:.4f}")
    print(f"Residual Variance: {_best_fit.scale:.4f}")
    return


if __name__ == "__main__":
    app.run()
