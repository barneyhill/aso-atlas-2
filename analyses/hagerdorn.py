import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from models import MODELS, valid_chemistry, prepare_data, calc_uln, run_model
    return MODELS, calc_uln, mo, np, pd, prepare_data, run_model, valid_chemistry


@app.cell
def _(mo):
    mo.md(r"""
    # Hagedorn et al. 2013 Hepatotoxicity Model

    Comparing models for predicting hepatotoxicity from oligonucleotide chemistry.

    **Models:**
    - **Dinucleotide**: 288 features (12×12×2 sugar-base-linkage pairs)
    - **Position**: 480 features (nucleotide type at each position from 5'/3' ends)

    **Species:** Mouse only
    """)
    return


@app.cell
def _(pd, valid_chemistry):
    # Load data, filter to mouse + valid chemistry
    df = pd.read_parquet("../data/oligostack/processed/hepatictoxicity_processed.parquet")
    df = df[(df['species'] == 'mouse') & df['HELM Annotation'].apply(valid_chemistry)].copy()
    print(f"Records: {len(df):,}")
    return (df,)


@app.cell
def _(MODELS, df, pd, prepare_data):
    # Extract features for each model
    feature_dfs = {key: prepare_data(df, key) for key in MODELS}
    for key, fdf in feature_dfs.items():
        print(f"{MODELS[key].name}: {fdf.shape[1]} features")
    return (feature_dfs,)


@app.cell
def _(df, pd):
    # Covariates
    cov_df = pd.DataFrame(index=df.index)
    cov_df['dosage_mg_per_kg'] = df['dosage_mg_per_kg'].fillna(df['dosage_mg_per_kg'].median())
    cov_df['num_doses'] = df['num_doses'].fillna(df['num_doses'].median())
    cov_df['dosing_period_days'] = df['dosing_period_days'].fillna(df['dosing_period_days'].median())
    cov_df['admin_subcut'] = (df['adminstration_method'] == 'subcutaneous').astype(float)
    return (cov_df,)


@app.cell
def _(df, np, pd):
    # Aggregate biomarkers
    BIOMARKERS = ['ALT', 'AST', 'TBIL']

    def mean_val(arr):
        if arr is None or (isinstance(arr, float) and np.isnan(arr)):
            return np.nan
        if isinstance(arr, (list, np.ndarray)):
            valid = [x for x in arr if x is not None and not np.isnan(x)]
            return np.mean(valid) if valid else np.nan
        return float(arr)

    for bm in BIOMARKERS:
        df[f'mean_{bm}'] = df[bm].apply(mean_val)
    return BIOMARKERS, mean_val


@app.cell
def _(BIOMARKERS, MODELS, cov_df, df, feature_dfs, pd, run_model):
    # Run all models on all biomarkers
    results = []
    for model_key in MODELS:
        for bm in BIOMARKERS:
            r = run_model(df, feature_dfs[model_key], cov_df, bm, MODELS[model_key].name)
            if r:
                results.append(r)

    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'biomarker', 'n', 'n_high', 'n_low', 'accuracy', 'sensitivity', 'specificity', 'p_value']]
    return results, results_df


@app.cell
def _(pd, results_df):
    # Format for display
    display_df = results_df.copy()
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.1%}")
    display_df['sensitivity'] = display_df['sensitivity'].apply(lambda x: f"{x:.1%}")
    display_df['specificity'] = display_df['specificity'].apply(lambda x: f"{x:.1%}")
    display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.1e}")
    display_df.columns = ['Model', 'Biomarker', 'N', 'High', 'Low', 'Acc', 'Sens', 'Spec', 'P-value']
    return (display_df,)


@app.cell
def _(display_df, mo):
    mo.md(f"## Results\n\n{display_df.to_markdown(index=False)}")
    return


@app.cell
def _(display_df):
    display_df
    return


if __name__ == "__main__":
    app.run()
