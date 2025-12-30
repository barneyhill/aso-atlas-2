"""
Hepatotoxicity prediction models based on oligonucleotide chemistry.
"""
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import fisher_exact
from dataclasses import dataclass
from typing import Callable


# =============================================================================
# HELM Parsing
# =============================================================================

def parse_helm(helm: str) -> list[tuple[str, str]] | None:
    """
    Parse HELM annotation to list of (sugar, base) tuples.
    Returns None if parsing fails.
    """
    if pd.isna(helm) or helm == 'None':
        return None
    match = re.search(r'\{\{(.+?)\}\}', helm) or re.search(r'\{(.+?)\}', helm)
    if not match:
        return None

    nucs = []
    for sug, base, _ in re.findall(r'(\[[a-zA-Z0-9]+\]|d)\(([A-Za-z0-9]+)\)(\[sp\])?\.?', match.group(1)):
        sugar = {'d': 'DNA', '[moe]': 'MOE', '[cet]': 'cET'}.get(sug.lower())
        if not sugar:
            continue
        b = base.upper()
        b = 'C' if b in ['5MEC', '5METHYLC', 'MC'] else ('T' if b == 'U' else b)
        if b in 'ACGT':
            nucs.append((sugar, b))
    return nucs if nucs else None


def valid_chemistry(helm: str) -> bool:
    """Check if HELM contains only DNA/MOE/cET sugars."""
    if pd.isna(helm) or helm == 'None':
        return False
    for ex in ['[lna]', '[LNA]', '[fR]', '[FR]', '[am]', '[AM]', '[?]']:
        if ex in helm:
            return False
    return not re.search(r'\[m\]\(', helm)


# =============================================================================
# Feature Extractors
# =============================================================================

SUGARS = ['DNA', 'MOE', 'cET']
BASES = ['A', 'C', 'G', 'T']
NUC_TYPES = [f"{s}_{b}" for s in SUGARS for b in BASES]


def dinucleotide_features() -> tuple[list[str], Callable]:
    """
    Hagedorn-style dinucleotide features.
    288 features: 12 nucleotide types × 12 × 2 linkages (PS/PO).
    """
    LINKS = ['PS', 'PO']
    features = [f"{n1}_{l}_{n2}" for n1 in NUC_TYPES for n2 in NUC_TYPES for l in LINKS]

    def extract(helm: str) -> dict:
        counts = {f: 0 for f in features}
        match = re.search(r'\{\{(.+?)\}\}', helm) or re.search(r'\{(.+?)\}', helm)
        if not match:
            return counts

        nucs = []
        for sug, base, link in re.findall(r'(\[[a-zA-Z0-9]+\]|d)\(([A-Za-z0-9]+)\)(\[sp\])?\.?', match.group(1)):
            sugar = {'d': 'DNA', '[moe]': 'MOE', '[cet]': 'cET'}.get(sug.lower())
            if not sugar:
                continue
            b = base.upper()
            b = 'C' if b in ['5MEC', '5METHYLC', 'MC'] else ('T' if b == 'U' else b)
            if b in 'ACGT':
                nucs.append((sugar, b, 'PS' if link == '[sp]' else 'PO'))

        for i in range(len(nucs) - 1):
            s1, b1, lnk = nucs[i]
            s2, b2, _ = nucs[i + 1]
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
            features.append(f"{nuc}_5p{pos}")  # position from 5' end
            features.append(f"{nuc}_3p{pos}")  # position from 3' end

    def extract(helm: str) -> dict:
        counts = {f: 0 for f in features}
        nucs = parse_helm(helm)
        if not nucs:
            return counts

        n = len(nucs)
        for i, (sugar, base) in enumerate(nucs):
            nuc_type = f"{sugar}_{base}"
            pos_5p = i + 1  # 1-indexed from 5' end
            pos_3p = n - i  # 1-indexed from 3' end

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
    """
    Baseline: just has_modification (any MOE/cET present).
    """
    features = ['has_mod', 'n_moe', 'n_cet', 'n_dna', 'length']

    def extract(helm: str) -> dict:
        nucs = parse_helm(helm)
        if not nucs:
            return {f: 0 for f in features}
        sugars = [s for s, b in nucs]
        return {
            'has_mod': 1 if any(s in ['MOE', 'cET'] for s in sugars) else 0,
            'n_moe': sum(1 for s in sugars if s == 'MOE'),
            'n_cet': sum(1 for s in sugars if s == 'cET'),
            'n_dna': sum(1 for s in sugars if s == 'DNA'),
            'length': len(nucs),
        }

    return features, extract


def counts_features() -> tuple[list[str], Callable]:
    """
    Baseline + counts of each sugar-base combo + PS count.
    """
    features = ['has_mod', 'length', 'n_ps']
    for s in SUGARS:
        for b in BASES:
            features.append(f'n_{s}_{b}')

    def extract(helm: str) -> dict:
        counts = {f: 0 for f in features}
        nucs = parse_helm(helm)
        if not nucs:
            return counts

        # Count PS linkages from HELM
        match = re.search(r'\{\{(.+?)\}\}', helm) or re.search(r'\{(.+?)\}', helm)
        if match:
            counts['n_ps'] = match.group(1).count('[sp]')

        counts['length'] = len(nucs)
        counts['has_mod'] = 1 if any(s in ['MOE', 'cET'] for s, b in nucs) else 0

        for sugar, base in nucs:
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
        'dinucleotide': ModelSpec('Dinucleotide (288)', dinuc_feats, dinuc_extract),
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


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 1000
) -> dict | None:
    """Train Random Forest and return OOB metrics."""
    y_binary = y.astype(int)
    n_high, n_low = y_binary.sum(), len(y_binary) - y_binary.sum()

    if n_high < 10 or n_low < 10:
        return None

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=8,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X, y_binary)

    pred = (model.oob_decision_function_[:, 1] > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_binary, pred).ravel()

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
        'p_value': pval,
        'model': model
    }


def run_model(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    biomarker: str,
    model_name: str
) -> dict | None:
    """Run a single model on a single biomarker."""
    col = f'mean_{biomarker}'
    valid = df[col].dropna()
    if len(valid) < 100:
        return None

    uln = calc_uln(valid.values)
    y_full = pd.Series(index=df.index, dtype=str)
    y_full[df[col] > 5 * uln] = 'high'
    y_full[df[col] < 2 * uln] = 'low'

    mask = y_full.isin(['high', 'low']) & feature_df.notna().all(axis=1)
    if mask.sum() < 50:
        return None

    X = pd.concat([feature_df.loc[mask], cov_df.loc[mask]], axis=1)
    y = (y_full[mask] == 'high')

    result = train_and_evaluate(X, y)
    if result is None:
        return None

    result['biomarker'] = biomarker
    result['model'] = model_name
    return result
