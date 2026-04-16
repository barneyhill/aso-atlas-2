"""
Bridge between project data (parquet, double-brace HELM) and OligoGym models/featurizers.

Per-species training with HELM-level dedup. Hepatic models include dosage covariates.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

_root = Path(__file__).resolve().parents[2]
_data_dir = _root / "data/oligostack/processed"


# ---------------------------------------------------------------------------
# Mock missing OligoGym optional deps so `oligogym.models` can be imported
# ---------------------------------------------------------------------------

def _mock_missing_deps():
    """Insert lightweight stubs for packages OligoGym imports at module level."""

    class _AutoMock(types.ModuleType):
        """Module whose every attribute is a dynamically-created class."""
        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            return type(name, (), {})

    for mod in [
        "torch_geometric", "torch_geometric.nn",
        "torch_geometric.data", "torch_geometric.utils",
    ]:
        sys.modules.setdefault(mod, _AutoMock(mod))

    if "tabpfn" not in sys.modules:
        m = types.ModuleType("tabpfn")
        m.TabPFNRegressor = type("TabPFNRegressor", (), {})
        m.TabPFNClassifier = type("TabPFNClassifier", (), {})
        sys.modules["tabpfn"] = m


_mock_missing_deps()

from oligogym.features import KMersCounts, OneHotEncoder  # noqa: E402
from oligogym.models import (  # noqa: E402
    MLP,
    CatBoostModel,
    LinearModel,
    NearestNeighborsModel,
    RandomForestModel,
    XGBoostModel,
)

# ---------------------------------------------------------------------------
# HELM conversion
# ---------------------------------------------------------------------------

def convert_helm(helm: str) -> str:
    """Convert project HELM (multi-braces) to OligoGym format (single braces)."""
    import re
    helm = re.sub(r"\{+", "{", helm)
    helm = re.sub(r"\}+(?=\$\$\$\$)", "}", helm)
    return helm


# ---------------------------------------------------------------------------
# Chemistry features from HELM
# ---------------------------------------------------------------------------

CHEM_FEATURES = ["n_MOE", "n_cEt", "n_DNA", "n_PS", "n_PO"]


def _extract_chemistry(helms: np.ndarray) -> np.ndarray:
    """Extract sugar/backbone counts from HELM strings.

    Returns (n_samples, 5) array: [n_MOE, n_cEt, n_DNA, n_PS, n_PO].
    """
    from collections import Counter
    rows = []
    for h in helms:
        parsed = Helm.parse(h)
        if parsed is None:
            rows.append([0] * 5)
            continue
        sc = Counter(parsed.sugars)
        bc = Counter(parsed.backbones)
        rows.append([sc.get("MOE", 0), sc.get("cEt", 0), sc.get("DNA", 0),
                     bc.get("PS", 0), bc.get("PO", 0)])
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset loaders (per-species, HELM-level dedup, dosage + chemistry covariates)
# ---------------------------------------------------------------------------

def _load_hepatic(species: str, species_label: str) -> dict:
    """Load hepatic data for one species with dosage + chemistry covariates."""
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == species) & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna()].reset_index(drop=True)

    groups = df["HELM Annotation"].copy()
    mask = groups.notna()

    helms_raw = df.loc[mask, "HELM Annotation"].values
    x = np.array([convert_helm(h) for h in helms_raw])

    # Dosage covariates (median-fill NaN)
    dose_cols = ["dosage_mg_per_kg", "num_doses", "dosing_period_days"]
    dose_cov = np.column_stack([
        df.loc[mask, col].fillna(df[col].median()).values.astype(np.float32)
        for col in dose_cols
    ])

    # Chemistry covariates from original HELM (before conversion)
    chem_cov = _extract_chemistry(helms_raw)

    covariates = np.column_stack([dose_cov, chem_cov])

    return {
        "x": x,
        "y": df.loc[mask, "mean_ALT"].values.astype(float),
        "groups": groups[mask].values,
        "covariates": covariates,
        "name": f"{species_label} Hepatic (ALT)",
    }


def load_mouse_hepatic() -> dict:
    return _load_hepatic("mouse", "Mouse")


def load_rat_hepatic() -> dict:
    return _load_hepatic("rat", "Rat")


def _load_neuro(species: str, species_label: str, dosage_ug: int,
                admin_method: str = None, latency: int = None) -> dict:
    """Load neuro data for one species with chemistry covariates."""
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    filt = (df["species"] == species) & (df["dosage_ug"] == dosage_ug) & df["HELM Annotation"].apply(Helm.valid_chemistry)
    if admin_method:
        filt = filt & (df["administration_method"] == admin_method)
    if latency:
        filt = filt & (df["latency_time_hours"] == latency)
    df = df[filt].copy()
    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    groups = df["HELM Annotation"].copy()
    mask = groups.notna()

    helms_raw = df.loc[mask, "HELM Annotation"].values
    x = np.array([convert_helm(h) for h in helms_raw])
    chem_cov = _extract_chemistry(helms_raw)

    return {
        "x": x,
        "y": df.loc[mask, "mean_FOB"].values.astype(float),
        "groups": groups[mask].values,
        "covariates": chem_cov,
        "name": f"{species_label} Neuro (FOB)",
    }


def load_mouse_neuro() -> dict:
    return _load_neuro("Mouse", "Mouse", dosage_ug=700, admin_method="ICV")


def load_rat_neuro() -> dict:
    return _load_neuro("Rat", "Rat", dosage_ug=3000, latency=3)


DATASETS = {
    "mouse_hepatic": load_mouse_hepatic,
    "rat_hepatic": load_rat_hepatic,
    "mouse_neuro": load_mouse_neuro,
    "rat_neuro": load_rat_neuro,
}


# ---------------------------------------------------------------------------
# Featurizer configs
# ---------------------------------------------------------------------------

FEATURIZER_CONFIGS = {
    "OneHot_base": (OneHotEncoder, {"encode_components": ["base"]}),
    "KMer_1_2": (KMersCounts, {"k": [1, 2]}),
}


def is_compatible(feat_name: str, model_name: str) -> bool:
    """Check featurizer-model compatibility. All remaining models are tabular."""
    return True


# ---------------------------------------------------------------------------
# Model configs (tabular only — sequential models dropped)
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, tuple[type, list[dict]]] = {
    "Linear": (LinearModel, [{"type": "standard"}, {"type": "ridge"}]),
    "Random Forest": (RandomForestModel, [
        {"n_estimators": 100, "max_depth": 20},
        {"n_estimators": 500, "max_depth": 20},
    ]),
    "XGBoost": (XGBoostModel, [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 500, "max_depth": 10},
    ]),
    "KNN": (NearestNeighborsModel, [
        {"n_neighbors": 5},
        {"n_neighbors": 10},
    ]),
    "CatBoost": (CatBoostModel, [
        {"iterations": 100},
        {"iterations": 500},
    ]),
    "MLP": (MLP, [
        {"hidden_dims": [128], "dropout": 0.25},
        {"hidden_dims": [128, 128], "dropout": 0.25},
    ]),
}
