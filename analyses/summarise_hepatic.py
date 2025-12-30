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
    ## Hepatic Toxicity: ALT vs AST

    Scatter plot of ALT vs AST biomarkers (log scale), colored by species.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # Load hepatic toxicity data
    hepatic_df = pd.read_parquet("../data/oligostack/processed/hepatictoxicity_processed.parquet")

    print(f"Loaded {len(hepatic_df):,} rows")
    print(f"Species: {hepatic_df['species'].unique().tolist()}")

    # Print non-NaN counts table: biomarkers (rows) x species (columns)
    _biomarkers = ['ALT', 'AST', 'ALB', 'TBIL', 'BUN', 'CREA', 'PC_ratio']
    _name_map = {
        'ALT': 'Alanine aminotransferase',
        'AST': 'Aspartate aminotransferase',
        'BUN': 'Blood urea nitrogen',
        'ALB': 'Albumin',
        'CREA': 'Creatinine',
        'TBIL': 'Total bilirubin',
        'PC_ratio': 'Protein/creatinine ratio',
    }
    _organ_map = {
        'ALT': 'Liver',
        'AST': 'Liver',
        'BUN': 'Kidney',
        'ALB': 'Liver',
        'CREA': 'Kidney',
        'TBIL': 'Liver',
        'PC_ratio': 'Kidney',
    }
    _unit_map = {
        'ALT': 'IU/L',
        'AST': 'IU/L',
        'BUN': 'mg/dL',
        'ALB': 'g/dL',
        'CREA': 'mg/dL',
        'TBIL': 'mg/dL',
        'PC_ratio': '',
    }
    _species_list = ['mouse', 'rat', 'monkey']

    def _count_non_nan(series):
        _count = 0
        for _val in series:
            if isinstance(_val, (np.ndarray, list)):
                if any(_v is not None and not np.isnan(_v) for _v in _val):
                    _count += 1
            elif _val is not None and not np.isnan(_val):
                _count += 1
        return _count

    # Build counts table
    _counts = {_s: [] for _s in _species_list}
    _counts['Total'] = []
    for _bm in _biomarkers:
        _row_total = 0
        for _s in _species_list:
            _subset = hepatic_df[hepatic_df['species'] == _s][_bm]
            _c = _count_non_nan(_subset)
            _counts[_s].append(_c)
            _row_total += _c
        _counts['Total'].append(_row_total)

    # Add column totals
    _col_totals = {_s: sum(_counts[_s]) for _s in _species_list}
    _col_totals['Total'] = sum(_counts['Total'])

    # Print table
    _header = ['Biomarker', 'Short', 'Organ', 'Unit'] + _species_list + ['Total']
    _col_widths = [28, 8, 8, 6] + [8] * len(_species_list) + [8]
    print('\n' + '  '.join(_h.rjust(_w) for _h, _w in zip(_header, _col_widths)))
    print('  '.join('-' * _w for _w in _col_widths))
    for _i, _bm in enumerate(_biomarkers):
        _row = [_name_map[_bm], _bm, _organ_map[_bm], _unit_map[_bm]] + [str(_counts[_s][_i]) for _s in _species_list] + [str(_counts['Total'][_i])]
        print('  '.join(_r.rjust(_w) for _r, _w in zip(_row, _col_widths)))
    print('  '.join('-' * _w for _w in _col_widths))
    _total_row = ['Total', '', '', ''] + [str(_col_totals[_s]) for _s in _species_list] + [str(_col_totals['Total'])]
    print('  '.join(_r.rjust(_w) for _r, _w in zip(_total_row, _col_widths)))
    print()
    return hepatic_df, np, pd


@app.cell
def _(hepatic_df, np, pd):
    # Take mean ALT and AST per row (ignoring NaN/None), keep species and compound_id

    rows = []
    for _, row in hepatic_df.iterrows():
        alt_vals = row['ALT']
        ast_vals = row['AST']
        species = row['species']
        compound_id = row['Compound ID']

        alt_mean = np.nan
        ast_mean = np.nan

        # Calculate mean if array/sequence, else if scalar copy as is
        if isinstance(alt_vals, (np.ndarray, list)):
            valid_alt = [v for v in alt_vals if v is not None and not np.isnan(v)]
            alt_mean = np.mean(valid_alt) if valid_alt else np.nan
        elif alt_vals is not None and not np.isnan(alt_vals):
            alt_mean = alt_vals

        if isinstance(ast_vals, (np.ndarray, list)):
            valid_ast = [v for v in ast_vals if v is not None and not np.isnan(v)]
            ast_mean = np.mean(valid_ast) if valid_ast else np.nan
        elif ast_vals is not None and not np.isnan(ast_vals):
            ast_mean = ast_vals

        # Only include rows with both ALT and AST available
        if not np.isnan(alt_mean) and not np.isnan(ast_mean):
            rows.append({
                'ALT': alt_mean,
                'AST': ast_mean,
                'species': species,
                'Compound ID': compound_id
            })

    plot_df = pd.DataFrame(rows)

    print(f"Created {len(plot_df):,} mean ALT-AST pairs for plotting")
    print(f"Species counts:")
    print(plot_df['species'].value_counts())
    return (plot_df,)


@app.cell
def _(plot_df):
    import matplotlib.pyplot as plt

    # Define species colors
    species_colors = {
        'mouse': 'red',
        'rat': 'blue',
        'monkey': 'green',
    }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # Specify species plotting order: monkey on top, then rat, then mouse
    plot_order = ['mouse', 'rat', 'monkey']

    for sp in plot_order:
        if sp not in plot_df['species'].unique():
            continue
        subset = plot_df[plot_df['species'] == sp]
        ax.scatter(
            subset['ALT'],
            subset['AST'],
            c=species_colors.get(sp, 'gray'),
            label=f"{sp} (n={len(subset):,})",
            alpha=.5,
            s=20,
            edgecolors='none'
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ALT (U/L)', fontsize=12)
    ax.set_ylabel('AST (U/L)', fontsize=12)
    ax.set_title('ALT vs AST', fontsize=14)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig("../plots/ALTvsAST.svg", format="svg")

    fig
    return (plt,)


@app.cell
def _(hepatic_df, np, pd):
    # Filter to mouse only and calculate total_dosage
    mouse_df = hepatic_df[hepatic_df['species'] == 'mouse'].copy()

    mouse_rows = []
    for _, r in mouse_df.iterrows():
        alt_v = r['ALT']
        ast_v = r['AST']
        dosage = r['dosage_mg_per_kg']
        n_doses = r['num_doses']

        # Calculate total dosage
        if pd.notna(dosage) and pd.notna(n_doses) and n_doses > 0:
            tot_dosage = dosage * n_doses
        else:
            continue

        # Calculate mean ALT
        alt_m = np.nan
        if isinstance(alt_v, (np.ndarray, list)):
            valid = [v for v in alt_v if v is not None and not np.isnan(v)]
            alt_m = np.mean(valid) if valid else np.nan
        elif alt_v is not None and not np.isnan(alt_v):
            alt_m = alt_v

        # Calculate mean AST
        ast_m = np.nan
        if isinstance(ast_v, (np.ndarray, list)):
            valid = [v for v in ast_v if v is not None and not np.isnan(v)]
            ast_m = np.mean(valid) if valid else np.nan
        elif ast_v is not None and not np.isnan(ast_v):
            ast_m = ast_v

        mouse_rows.append({
            'total_dosage': tot_dosage,
            'ALT': alt_m,
            'AST': ast_m,
            'Compound ID': r['Compound ID']
        })

    mouse_plot_df = pd.DataFrame(mouse_rows)
    mouse_plot_df = mouse_plot_df[mouse_plot_df['total_dosage'] > 0]

    print(f"Mouse data: {len(mouse_plot_df):,} rows with total_dosage")
    return (mouse_plot_df,)


@app.cell
def _(mouse_plot_df, plt):
    from matplotlib.ticker import FuncFormatter

    # Format numbers with commas (no scientific notation)
    def fmt_number(x, pos):
        if x >= 1:
            return f'{int(x):,}'
        return f'{x:g}'

    # Filter to valid values
    alt_df = mouse_plot_df[mouse_plot_df['ALT'].notna() & (mouse_plot_df['ALT'] > 0)]
    ast_df = mouse_plot_df[mouse_plot_df['AST'].notna() & (mouse_plot_df['AST'] > 0)]

    fig_dose, (ax_alt, ax_ast) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Choose fixed y-axis limits for ALT and AST (adjust as appropriate)
    alt_ymin, alt_ymax = alt_df['ALT'].min(), alt_df['ALT'].max()
    ast_ymin, ast_ymax = ast_df['AST'].min(), ast_df['AST'].max()

    # ALT plot
    ax_alt.scatter(
        alt_df['total_dosage'],
        alt_df['ALT'],
        c='#1f77b4',
        alpha=0.3,
        s=20,
        edgecolors='none'
    )
    ax_alt.set_xscale('log')
    ax_alt.set_yscale('log')
    ax_alt.set_ylim(alt_ymin * 0.9 if alt_ymin > 0 else 0.1, alt_ymax * 1.1)
    ax_alt.xaxis.set_major_formatter(FuncFormatter(fmt_number))
    ax_alt.yaxis.set_major_formatter(FuncFormatter(fmt_number))
    ax_alt.set_xlabel('Total administered dose (mg/kg)', fontsize=12)
    ax_alt.set_ylabel('ALT (U/L)', fontsize=12)
    ax_alt.set_title(f'Total administered dose vs ALT (n={len(alt_df):,})', fontsize=12)
    ax_alt.grid(True, alpha=0.3, which='both')

    # AST plot
    ax_ast.scatter(
        ast_df['total_dosage'],
        ast_df['AST'],
        c='#ff7f0e',
        alpha=0.3,
        s=20,
        edgecolors='none'
    )
    ax_ast.set_xscale('log')
    ax_ast.set_yscale('log')
    ax_ast.set_ylim(ast_ymin * 0.9 if ast_ymin > 0 else 0.1, ast_ymax * 1.1)
    ax_ast.xaxis.set_major_formatter(FuncFormatter(fmt_number))
    ax_ast.yaxis.set_major_formatter(FuncFormatter(fmt_number))
    ax_ast.set_xlabel('Total administered dose (mg/kg)', fontsize=12)
    ax_ast.set_ylabel('AST (U/L)', fontsize=12)
    ax_ast.set_title(f'Total administered dose vs AST (n={len(ast_df):,})', fontsize=12)
    ax_ast.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig_dose.savefig("../plots/hepatic_dose_vs_ALT_AST.svg", format="svg")
    fig_dose
    return


if __name__ == "__main__":
    app.run()
