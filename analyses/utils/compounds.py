"""Shared per-compound aggregation functions.

Eliminates duplicated logic for computing compound-level metrics
(max inhibition, mean biomarkers, IC50) across pipeline, plotting,
and export scripts.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from analyses.utils.models import mean_of_array


def compound_max_inhibition(invitro_df: pd.DataFrame) -> pd.Series:
    """Per-compound maximum inhibition percentage."""
    return invitro_df.groupby("Compound ID")["Inhibition_pct"].max()


def compound_mean_biomarker(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-compound mean of an array-valued biomarker column.

    Pre-filter df to the desired species/conditions before calling.
    """
    means = df[col].apply(mean_of_array)
    return means.groupby(df["Compound ID"]).mean().dropna()


def hill_equation(dose, bottom, top, ic50, hill):
    return bottom + (top - bottom) / (1 + (ic50 / dose) ** hill)


def fit_ic50_for_compound(doses, responses):
    mask = ~(np.isnan(doses) | np.isnan(responses))
    doses = np.array(doses)[mask]
    responses = np.array(responses)[mask]

    if len(doses) < 4:
        return np.nan

    pos_mask = doses > 0
    doses = doses[pos_mask]
    responses = responses[pos_mask]

    if len(doses) < 4:
        return np.nan

    try:
        min_dose, max_dose = np.min(doses), np.max(doses)
        bounds = ([0, 0, min_dose / 100, 0.1], [100, 100, max_dose * 100, 10])
        popt, _ = curve_fit(
            hill_equation, doses, responses,
            p0=[np.min(responses), np.max(responses), np.median(doses), 1.0],
            bounds=bounds, maxfev=5000,
        )
        ic50 = popt[2]
        predicted = hill_equation(doses, *popt)
        ss_res = np.sum((responses - predicted) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if r_squared > 0.5 and min_dose / 10 <= ic50 <= max_dose * 10:
            return ic50
        return np.nan
    except (RuntimeError, ValueError):
        return np.nan


def compound_ic50s(dose_response_df: pd.DataFrame) -> pd.Series:
    """Per-compound IC50 from electroporation dose-response data."""
    dr_elec = dose_response_df[
        dose_response_df["transfection_method"] == "Electroporation"
    ]
    results = []
    for cid, grp in dr_elec.groupby("Compound ID"):
        ic50 = fit_ic50_for_compound(
            grp["dosage_nm"].values, grp["Inhibition_pct"].values,
        )
        if not np.isnan(ic50):
            results.append({"Compound ID": cid, "ic50_nm": ic50})
    if not results:
        return pd.Series(dtype=float, name="ic50_nm")
    return pd.DataFrame(results).set_index("Compound ID")["ic50_nm"]


def has_measurement(val):
    """Check if a biomarker cell contains a valid measurement."""
    if val is None:
        return 0
    if isinstance(val, (list, np.ndarray)):
        return 1 if any(
            x is not None and not (isinstance(x, float) and np.isnan(x))
            for x in val
        ) else 0
    return 0
