"""Shared utilities for hepatotoxicity and neurotoxicity models."""

import numpy as np
from sklearn.metrics import roc_curve


def mean_of_array(val):
    """Compute mean from array/scalar biomarker column."""
    if isinstance(val, (np.ndarray, list)):
        valid = [v for v in val if v is not None and not np.isnan(v)]
        return np.mean(valid) if valid else np.nan
    if val is not None and not np.isnan(val):
        return float(val)
    return np.nan


def calc_uln(values: np.ndarray) -> float:
    """Calculate upper limit of normal: median + 3 * MAD."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return med + 3 * mad


def _optimal_threshold(y_true, y_proba):
    """Find threshold maximising Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    best_idx = np.argmax(j)
    return float(thresholds[best_idx])
