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
    ## Neurotoxicity Summary

    Summary statistics and FOB score distributions from neurotoxicity data.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # Load neurotoxicity data
    neuro_df = pd.read_parquet("../data/oligostack/processed/neurotoxicity_processed.parquet")

    print(f"Loaded {len(neuro_df):,} rows")
    print(f"Species: {neuro_df['species'].unique().tolist()}")
    print(f"Administration methods: {neuro_df['administration_method'].unique().tolist()}")
    print(f"Test types: {neuro_df['tolerability_score_type'].unique().tolist()}")

    # Print non-NaN counts table: columns (rows) x species (columns)
    _columns = ['FOB_score', 'dosage_ug', 'latency_time_hours', 'administration_method', 'tolerability_score_type']
    _name_map = {
        'FOB_score': 'FOB Score',
        'dosage_ug': 'Dosage (ug)',
        'latency_time_hours': 'Latency Time (hrs)',
        'administration_method': 'Admin Method',
        'tolerability_score_type': 'Test Type',
    }
    _species_list = ['Mouse', 'Rat']

    def _count_non_nan(series):
        _count = 0
        for _val in series:
            if isinstance(_val, (np.ndarray, list)):
                if len(_val) > 0 and any(_v is not None and not np.isnan(_v) for _v in _val):
                    _count += 1
            elif _val is not None and not pd.isna(_val):
                _count += 1
        return _count

    # Build counts table
    _counts = {_s: [] for _s in _species_list}
    _counts['Total'] = []
    for _col in _columns:
        _row_total = 0
        for _s in _species_list:
            _subset = neuro_df[neuro_df['species'] == _s][_col]
            _c = _count_non_nan(_subset)
            _counts[_s].append(_c)
            _row_total += _c
        _counts['Total'].append(_row_total)

    # Add column totals
    _col_totals = {_s: sum(_counts[_s]) for _s in _species_list}
    _col_totals['Total'] = sum(_counts['Total'])

    # Print table
    _header = ['Field', 'Description'] + _species_list + ['Total']
    _col_widths = [25, 20] + [8] * len(_species_list) + [8]
    print('\n' + '  '.join(_h.rjust(_w) for _h, _w in zip(_header, _col_widths)))
    print('  '.join('-' * _w for _w in _col_widths))
    for _i, _col in enumerate(_columns):
        _row = [_col, _name_map[_col]] + [str(_counts[_s][_i]) for _s in _species_list] + [str(_counts['Total'][_i])]
        print('  '.join(_r.rjust(_w) for _r, _w in zip(_row, _col_widths)))
    print('  '.join('-' * _w for _w in _col_widths))
    _total_row = ['Total', ''] + [str(_col_totals[_s]) for _s in _species_list] + [str(_col_totals['Total'])]
    print('  '.join(_r.rjust(_w) for _r, _w in zip(_total_row, _col_widths)))
    print()
    return neuro_df, np, pd


@app.cell
def _(neuro_df, np, pd):
    # Filter to standard conditions:
    # Mouse: 700ug, 3h, ICV, Behavioral
    # Rat: 3000ug, 3h, IT, Body parts
    _mouse_filter = (
        (neuro_df['species'] == 'Mouse') &
        (neuro_df['dosage_ug'] == 700) &
        (neuro_df['latency_time_hours'] == 3)
    )
    _rat_filter = (
        (neuro_df['species'] == 'Rat') &
        (neuro_df['dosage_ug'] == 3000) &
        (neuro_df['latency_time_hours'] == 3)
    )
    _filtered_df = neuro_df[_mouse_filter | _rat_filter]

    # Calculate mean FOB score per row and create plotting dataframe
    _rows = []
    for _, _row in _filtered_df.iterrows():
        _fob_vals = _row['FOB_score']
        _species = _row['species']
        _dosage = _row['dosage_ug']
        _compound_id = _row['Compound ID']

        _fob_mean = np.nan

        # Calculate mean if array/sequence
        if isinstance(_fob_vals, (np.ndarray, list)):
            _valid_fob = [v for v in _fob_vals if v is not None and not np.isnan(v)]
            _fob_mean = np.mean(_valid_fob) if _valid_fob else np.nan
        elif _fob_vals is not None and not np.isnan(_fob_vals):
            _fob_mean = _fob_vals

        if not np.isnan(_fob_mean):
            # Create group label based on confounded factors
            _group = 'Behavioral 3h after 700μg ICV in mice' if _species == 'Mouse' else 'Body parts 3h after 3000μg IT in rats'
            _rows.append({
                'FOB_mean': _fob_mean,
                'species': _species,
                'group': _group,
                'dosage_ug': _dosage,
                'Compound ID': _compound_id
            })

    plot_df = pd.DataFrame(_rows)

    print(f"Filtered to standard conditions: {len(_filtered_df):,} rows")
    print(f"Created {len(plot_df):,} mean FOB scores for plotting")
    print(f"\nGroup counts:")
    print(plot_df['group'].value_counts())
    return (plot_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## FOB Score Distribution by Group

    Behavioral FOB 3h after 700μg ICV in mice vs Body parts FOB 3h after 3000μg IT in rats.
    """)
    return


@app.cell
def _(np, plot_df):
    import matplotlib.pyplot as plt

    _group_colors = {
        'Behavioral 3h after 700μg ICV in mice': '#1f77b4',
        'Body parts 3h after 3000μg IT in rats': '#ff7f0e',
    }
    _groups = ['Behavioral 3h after 700μg ICV in mice', 'Body parts 3h after 3000μg IT in rats']

    fig_group, _ax = plt.subplots(figsize=(10, 5), dpi=150)

    _scores = np.arange(8)
    _bar_width = 0.35

    for _idx, _g in enumerate(_groups):
        _subset = plot_df[plot_df['group'] == _g]
        _rounded = _subset['FOB_mean'].round().astype(int)
        _counts = _rounded.value_counts().reindex(_scores, fill_value=0)
        _proportions = _counts / len(_rounded)
        _offset = (_idx - 0.5) * _bar_width
        _ax.bar(
            _scores + _offset,
            _proportions,
            width=_bar_width,
            label=f"{_g} (n={len(_subset):,})",
            color=_group_colors[_g],
            edgecolor='white'
        )

    _ax.set_xlabel('FOB Score (rounded)', fontsize=12)
    _ax.set_ylabel('Proportion', fontsize=12)
    _ax.set_title('FOB Score Distribution by Group', fontsize=14)
    _ax.set_xticks(range(8))
    _ax.legend()
    _ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_group.savefig("../plots/neuro_fob_by_group.svg", format="svg")
    fig_group
    return (plt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Mouse vs Rat FOB Scores

    Compounds with FOB scores in both groups.
    """)
    return


@app.cell
def _(pd, plot_df, plt):
    from scipy.stats import spearmanr

    # Get mean FOB per compound per group
    _mouse_df = plot_df[plot_df['group'] == 'Behavioral 3h after 700μg ICV in mice'].groupby('Compound ID')['FOB_mean'].mean()
    _rat_df = plot_df[plot_df['group'] == 'Body parts 3h after 3000μg IT in rats'].groupby('Compound ID')['FOB_mean'].mean()

    # Find compounds in both groups
    _shared_ids = _mouse_df.index.intersection(_rat_df.index)
    _paired_df = pd.DataFrame({
        'Mouse': _mouse_df.loc[_shared_ids].round().astype(int),
        'Rat': _rat_df.loc[_shared_ids].round().astype(int)
    })

    print(f"Compounds with scores in both groups: {len(_paired_df):,}")

    # Create count matrix for heatmap
    _counts = _paired_df.groupby(['Mouse', 'Rat']).size().unstack(fill_value=0)
    # Ensure all integers 0-7 are represented
    _full_index = range(8)
    _counts = _counts.reindex(index=_full_index, columns=_full_index, fill_value=0)

    fig_compare, _ax = plt.subplots(figsize=(7, 6), dpi=150)

    _ax.imshow(_counts.values, origin='lower', cmap='Blues', aspect='equal')

    # Add count annotations
    for _i in range(8):
        for _j in range(8):
            _val = _counts.values[_i, _j]
            if _val > 0:
                _color = 'white' if _val > _counts.values.max() / 2 else 'black'
                _ax.text(_j, _i, str(_val), ha='center', va='center', fontsize=9, color=_color)

    # Spearman correlation on original (non-rounded) values
    _mouse_orig = _mouse_df.loc[_shared_ids]
    _rat_orig = _rat_df.loc[_shared_ids]
    _rho, _p = spearmanr(_mouse_orig, _rat_orig)
    _p_str = f'p={_p:.2e}' if _p < 0.001 else f'p={_p:.3f}'

    _ax.set_xticks(range(8))
    _ax.set_yticks(range(8))
    # Add grid lines between cells
    for _i in range(-1, 8):
        _ax.axhline(_i + 0.5, color='black', linewidth=0.5)
        _ax.axvline(_i + 0.5, color='black', linewidth=0.5)
    _ax.set_xlabel('Behavioral FOB 3h after 700μg ICV in mice', fontsize=10)
    _ax.set_ylabel('Body parts FOB 3h after 3000μg IT in rats', fontsize=10)
    _ax.set_title(f'Comparison between FOB tests (n={len(_paired_df):,}, ρ={_rho:.2f}, {_p_str})', fontsize=12)

    plt.tight_layout()
    fig_compare.savefig("../plots/neuro_fob_mouse_vs_rat.svg", format="svg")
    fig_compare
    return


if __name__ == "__main__":
    app.run()
