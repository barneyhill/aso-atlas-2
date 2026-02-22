"""
Hepatotoxicity prediction models based on oligonucleotide chemistry.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import fisher_exact
from dataclasses import dataclass
from typing import Callable

from analyses.utils.helm import Helm


# =============================================================================
# Feature Extractors
# =============================================================================

SUGARS = ['DNA', 'MOE', 'cEt']
BASES = ['A', 'C', 'G', 'T']
NUC_TYPES = [f"{s}_{b}" for s in SUGARS for b in BASES]


def dinucleotide_features() -> tuple[list[str], Callable]:
    """
    Hagedorn-style dinucleotide features with collapsed sugar types.

    MOE and cEt never co-occur in the same ASO, so we collapse them into
    a single 'MOD' category: DNA vs MOD (MOE|cEt).
    2 sugars × 4 bases = 8 monomer types → 8² × 2 linkages = 128 features.
    """
    _DINUC_SUGAR_MAP = {'MOE': 'MOD', 'cEt': 'MOD', 'DNA': 'DNA'}
    _DINUC_SUGARS = ['DNA', 'MOD']
    LINKS = ['PS', 'PO']
    dinuc_nuc_types = [f"{s}_{b}" for s in _DINUC_SUGARS for b in BASES]
    features = [f"{n1}_{l}_{n2}" for n1 in dinuc_nuc_types for n2 in dinuc_nuc_types for l in LINKS]

    def extract(helm: str) -> dict:
        counts = {f: 0 for f in features}
        parsed = Helm.parse(helm)
        if parsed is None:
            return counts

        for i in range(parsed.length - 1):
            s1 = _DINUC_SUGAR_MAP.get(parsed.sugars[i], parsed.sugars[i])
            b1 = parsed.bases[i]
            lnk = parsed.backbones[i]
            s2 = _DINUC_SUGAR_MAP.get(parsed.sugars[i + 1], parsed.sugars[i + 1])
            b2 = parsed.bases[i + 1]
            feat = f"{s1}_{b1}_{lnk}_{s2}_{b2}"
            if feat in counts:
                counts[feat] += 1
        return counts

    return features, extract


def position_features(max_pos: int = 20) -> tuple[list[str], Callable]:
    """
    Position-specific single nucleotide features.
    Features encode nucleotide type at each position from 5' and 3' ends.
    """
    features = []
    for nuc in NUC_TYPES:
        for pos in range(1, max_pos + 1):
            features.append(f"{nuc}_5p{pos}")
            features.append(f"{nuc}_3p{pos}")

    def extract(helm: str) -> dict:
        counts = {f: 0 for f in features}
        parsed = Helm.parse(helm)
        if parsed is None:
            return counts

        for i, (sugar, base) in enumerate(zip(parsed.sugars, parsed.bases)):
            nuc_type = f"{sugar}_{base}"
            pos_5p = i + 1
            pos_3p = parsed.length - i

            if pos_5p <= max_pos:
                feat = f"{nuc_type}_5p{pos_5p}"
                if feat in counts:
                    counts[feat] = 1
            if pos_3p <= max_pos:
                feat = f"{nuc_type}_3p{pos_3p}"
                if feat in counts:
                    counts[feat] = 1
        return counts

    return features, extract


# =============================================================================
# Model Registry
# =============================================================================

@dataclass
class ModelSpec:
    name: str
    features: list[str]
    extractor: Callable


def baseline_features() -> tuple[list[str], Callable]:
    """Baseline: just has_modification (any MOE/cEt present)."""
    features = ['has_mod', 'n_moe', 'n_cet', 'n_dna', 'length']

    def extract(helm: str) -> dict:
        parsed = Helm.parse(helm)
        if parsed is None:
            return {f: 0 for f in features}
        return {
            'has_mod': 1 if any(s in ['MOE', 'cEt'] for s in parsed.sugars) else 0,
            'n_moe': sum(1 for s in parsed.sugars if s == 'MOE'),
            'n_cet': sum(1 for s in parsed.sugars if s == 'cEt'),
            'n_dna': sum(1 for s in parsed.sugars if s == 'DNA'),
            'length': parsed.length,
        }

    return features, extract


def counts_features() -> tuple[list[str], Callable]:
    """Baseline + counts of each sugar-base combo + PS count."""
    features = ['has_mod', 'length', 'n_ps']
    for s in SUGARS:
        for b in BASES:
            features.append(f'n_{s}_{b}')

    def extract(helm: str) -> dict:
        counts = {f: 0 for f in features}
        parsed = Helm.parse(helm)
        if parsed is None:
            return counts

        counts['n_ps'] = parsed.ps_count
        counts['length'] = parsed.length
        counts['has_mod'] = 1 if any(s in ['MOE', 'cEt'] for s in parsed.sugars) else 0

        for sugar, base in zip(parsed.sugars, parsed.bases):
            key = f'n_{sugar}_{base}'
            if key in counts:
                counts[key] += 1

        return counts

    return features, extract


def _make_models():
    base_feats, base_extract = baseline_features()
    counts_feats, counts_extract = counts_features()
    dinuc_feats, dinuc_extract = dinucleotide_features()
    pos_feats, pos_extract = position_features(max_pos=20)
    return {
        'baseline': ModelSpec('Baseline (5)', base_feats, base_extract),
        'counts': ModelSpec('Counts (15)', counts_feats, counts_extract),
        'dinucleotide': ModelSpec('Dinucleotide (128)', dinuc_feats, dinuc_extract),
        'position': ModelSpec('Position (480)', pos_feats, pos_extract),
    }

MODELS = _make_models()


# =============================================================================
# Training
# =============================================================================

def prepare_data(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """Extract features for a given model."""
    spec = MODELS[model_key]
    features = df['HELM Annotation'].apply(spec.extractor)
    return pd.DataFrame(features.tolist(), index=df.index)


def calc_uln(values: np.ndarray) -> float:
    """Calculate upper limit of normal: median + 3 * MAD."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return med + 3 * mad


def _optimal_threshold(y_true, y_proba):
    """Find threshold maximising Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr  # Youden's J = sensitivity + specificity - 1
    best_idx = np.argmax(j)
    return float(thresholds[best_idx])


def train_and_evaluate_hepatotox(
    X: pd.DataFrame,
    y: pd.Series,
    sequences: pd.Series,
    n_splits: int = 10,
    n_estimators: int = 5000,
) -> dict | None:
    """Hagedorn 2013-style training: OOB + Levenshtein-stratified 10-fold CV.

    Steps:
    1. Train a single RF on all data with oob_score=True, report OOB accuracy.
    2. Compute pairwise min Levenshtein edit distances, bin into strata,
       run StratifiedKFold with distance bins as strata.
    3. Apply Youden's J on pooled OOF predictions.
    """
    from rapidfuzz.distance import Levenshtein
    from sklearn.model_selection import StratifiedKFold

    y_binary = y.astype(int)
    n_high, n_low = int(y_binary.sum()), int(len(y_binary) - y_binary.sum())

    if n_high < 10 or n_low < 10:
        return None

    # --- Step 1: OOB ---
    rf_oob = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=min(8, X.shape[1]),
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        oob_score=True,
    )
    rf_oob.fit(X, y_binary)
    oob_accuracy = rf_oob.oob_score_

    # --- Step 2: Levenshtein-stratified CV ---
    seq_list = sequences.values
    n = len(seq_list)

    # Compute min edit distance to any other ASO for each sequence
    min_dists = np.zeros(n, dtype=int)
    for i in range(n):
        best = 999
        for j in range(n):
            if i == j:
                continue
            d = Levenshtein.distance(seq_list[i], seq_list[j])
            if d < best:
                best = d
                if best == 1:
                    break  # can't do better
        min_dists[i] = best

    # Bin into strata matching the paper's table
    dist_bins = np.where(
        min_dists <= 2, 0,           # 1-2 edits
        np.where(min_dists <= 5, 1,  # 3-5 edits
                 2)                   # 6+ edits
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds = pd.Series(index=y_binary.index, dtype=float)

    for train_idx, test_idx in skf.split(X, dist_bins):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_binary.iloc[train_idx]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=min(8, X.shape[1]),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
        )
        rf.fit(X_train, y_train)
        all_preds.iloc[test_idx] = rf.predict_proba(X_test)[:, 1]

    # Per-stratum accuracy (at 0.5 threshold first, for reporting)
    stratum_labels = {0: "1-2 edits", 1: "3-5 edits", 2: "6+ edits"}
    stratum_accuracy = {}
    for bin_val, label in stratum_labels.items():
        mask = dist_bins == bin_val
        if mask.sum() == 0:
            continue
        pred_bin = (all_preds.values[mask] > 0.5).astype(int)
        true_bin = y_binary.values[mask]
        stratum_accuracy[label] = float(np.mean(pred_bin == true_bin))

    # --- Step 3: Youden's J ---
    try:
        auc = roc_auc_score(y_binary, all_preds)
    except ValueError:
        auc = np.nan

    try:
        threshold = _optimal_threshold(y_binary, all_preds)
    except ValueError:
        threshold = 0.5
    all_pred_labels = (all_preds > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_binary, all_pred_labels).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    _, pval = fisher_exact([[tp, fn], [fp, tn]])

    return {
        'n': len(y_binary),
        'n_high': n_high,
        'n_low': n_low,
        'oob_accuracy': oob_accuracy,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'threshold': threshold,
        'p_value': pval,
        'auc': auc,
        'stratum_accuracy': stratum_accuracy,
        'predictions': all_preds,
        'confusion': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
    }


def train_and_evaluate_grouped(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    n_estimators: int = 1000,
) -> dict | None:
    """Train Random Forest with GroupKFold CV (no same-target leakage)."""
    y_binary = y.astype(int)
    n_high, n_low = y_binary.sum(), len(y_binary) - y_binary.sum()

    if n_high < 10 or n_low < 10:
        return None

    n_groups = groups.nunique()
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        return None

    gkf = GroupKFold(n_splits=actual_splits)
    all_preds = pd.Series(index=y_binary.index, dtype=float)
    all_pred_labels = pd.Series(index=y_binary.index, dtype=int)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y_binary, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=min(8, X.shape[1]),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)
        all_preds.iloc[test_idx] = proba
        all_pred_labels.iloc[test_idx] = pred

        if y_test.nunique() == 2:
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            fold_metrics.append({
                'fold': fold_i,
                'n': len(y_test),
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            })

    try:
        auc = roc_auc_score(y_binary, all_preds)
    except ValueError:
        auc = np.nan

    # Optimal threshold via Youden's J on pooled OOF predictions
    try:
        threshold = _optimal_threshold(y_binary, all_preds)
    except ValueError:
        threshold = 0.5
    all_pred_labels = (all_preds > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_binary, all_pred_labels).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    _, pval = fisher_exact([[tp, fn], [fp, tn]])

    return {
        'n': len(y_binary),
        'n_high': n_high,
        'n_low': n_low,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'threshold': threshold,
        'p_value': pval,
        'auc': auc,
        'n_splits': actual_splits,
        'n_groups': n_groups,
        'fold_metrics': fold_metrics,
        'predictions': all_preds,
        'confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
    }


def run_model_grouped(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    groups: pd.Series,
    biomarker: str,
    model_name: str,
    n_splits: int = 5,
    high_mult: float = 5,
    low_mult: float = 2,
) -> dict | None:
    """Run a grouped model on a single biomarker."""
    col = f'mean_{biomarker}'
    valid = df[col].dropna()
    if len(valid) < 100:
        return None

    uln = calc_uln(valid.values)
    y_full = pd.Series(index=df.index, dtype=str)
    y_full[df[col] > high_mult * uln] = 'high'
    y_full[df[col] < low_mult * uln] = 'low'

    mask = y_full.isin(['high', 'low']) & feature_df.notna().all(axis=1) & groups.notna()
    if mask.sum() < 50:
        return None

    X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
    y = (y_full[mask] == 'high')
    g = groups[mask]

    result = train_and_evaluate_grouped(X, y, g, n_splits=n_splits)
    if result is None:
        return None

    result['biomarker'] = biomarker
    result['model'] = model_name
    return result
