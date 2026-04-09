"""
Bridge between project data (parquet, double-brace HELM) and OligoGym models/featurizers.
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
    CNN,
    GRU,
    MLP,
    CatBoostModel,
    CausalCNN,
    GaussianProcessModel,
    LinearModel,
    NearestNeighborsModel,
    RandomForestModel,
    Transformer,
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
# Dataset loaders
# ---------------------------------------------------------------------------

def load_mouse_hepatic() -> dict:
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == "mouse") & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "x": np.array([convert_helm(h) for h in df.loc[mask, "HELM Annotation"]]),
        "y": df.loc[mask, "mean_ALT"].values.astype(float),
        "groups": groups[mask].values,
        "name": "Mouse Hepatic (ALT)",
    }


def load_rat_hepatic() -> dict:
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == "rat") & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "x": np.array([convert_helm(h) for h in df.loc[mask, "HELM Annotation"]]),
        "y": df.loc[mask, "mean_ALT"].values.astype(float),
        "groups": groups[mask].values,
        "name": "Rat Hepatic (ALT)",
    }


def load_mouse_neuro() -> dict:
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df = df[
        (df["species"] == "Mouse")
        & (df["dosage_ug"] == 700)
        & (df["administration_method"] == "ICV")
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "x": np.array([convert_helm(h) for h in df.loc[mask, "HELM Annotation"]]),
        "y": df.loc[mask, "mean_FOB"].values.astype(float),
        "groups": groups[mask].values,
        "name": "Mouse Neuro (FOB)",
    }


def load_rat_neuro() -> dict:
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df = df[
        (df["species"] == "Rat")
        & (df["dosage_ug"] == 3000)
        & (df["latency_time_hours"] == 3)
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "x": np.array([convert_helm(h) for h in df.loc[mask, "HELM Annotation"]]),
        "y": df.loc[mask, "mean_FOB"].values.astype(float),
        "groups": groups[mask].values,
        "name": "Rat Neuro (FOB)",
    }


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

# Sequential models need 3D input (OneHot only); tabular models can use either
_SEQUENTIAL_MODELS = {"CNN", "GRU", "CausalCNN", "Transformer"}


def is_compatible(feat_name: str, model_name: str) -> bool:
    """Check featurizer-model compatibility."""
    if model_name in _SEQUENTIAL_MODELS:
        return feat_name.startswith("OneHot")
    return True


# ---------------------------------------------------------------------------
# Model configs  (name -> list of kwarg dicts to try)
# ---------------------------------------------------------------------------

_MAX_GP_SAMPLES = 1000

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
    "Gaussian Process": (GaussianProcessModel, [{}]),
    "CatBoost": (CatBoostModel, [
        {"iterations": 100},
        {"iterations": 500},
    ]),
    "CNN": (CNN, [{"depth": 1, "hidden_dim": 64, "kernel_size": 5}]),
    "MLP": (MLP, [
        {"hidden_dims": [128], "dropout": 0.25},
        {"hidden_dims": [128, 128], "dropout": 0.25},
    ]),
    "GRU": (GRU, [{"hidden_dim": 64, "num_layers": 1}]),
    "CausalCNN": (CausalCNN, [{"depth": 2, "hidden_dim": 64}]),
    "Transformer": (Transformer, [
        {"d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.25},
    ]),
}
