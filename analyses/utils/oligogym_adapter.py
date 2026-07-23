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
    """Load hepatic data for one species with dosage + chemistry covariates.

    GroupKFold groups are patent IDs (matches OligoAI splitting).
    Rows without a USPTO ID are dropped.
    """
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == species) & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna()].reset_index(drop=True)
    df = _earliest_patent(df)

    # Group by cv_group (= HELM's earliest patent) so every row of a given
    # HELM lands in the same fold, regardless of which patent it was
    # actually published in.
    groups = df["cv_group"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
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
        "conditions": {
            "dosage_mg_per_kg": df.loc[mask, "dosage_mg_per_kg"].values.astype(float),
        },
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
    df = _earliest_patent(df)
    groups = df["cv_group"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
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


def _earliest_patent(df: pd.DataFrame, helm_col: str = "HELM Annotation") -> pd.DataFrame:
    """Add a cv_group column = each HELM's earliest-filed patent.

    USPTO IDs sort chronologically as strings, so the earliest patent is
    the first one alphabetically per HELM. cv_group is the GroupKFold
    grouping key downstream — it guarantees that every row of a given
    HELM lands in the same fold (no sequence leakage across train/test)
    *without dropping any rows*. The original USPTO ID column is left
    untouched for provenance.
    """
    df = df[df[helm_col].notna() & df["USPTO ID"].notna()].copy()
    df = df.sort_values(["USPTO ID"], kind="stable")
    first_map = df.drop_duplicates(helm_col, keep="first").set_index(helm_col)["USPTO ID"]
    df["cv_group"] = df[helm_col].map(first_map)
    return df


def load_in_vitro_inhibition() -> dict:
    """Load in vitro inhibition per (HELM, dosage) with patent grouping.

    Each row is a unique (HELM, dosage_nm) combination; replicates across
    cell lines and transfection methods are collapsed by taking the max
    inhibition. Dosage is included as a covariate so the model can learn
    dose-dependent inhibition (consistent with OligoAI's feature set).

    HELMs appearing in multiple patents are assigned to their earliest
    patent — GroupKFold on patent then guarantees sequence-level
    disjointness across folds.
    """
    df = pd.read_parquet(_data_dir / "in_vitro_inhibition_processed.parquet")
    df = df[df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df = df[df["Inhibition_pct"].between(-1000, 100)]
    df = _earliest_patent(df)

    agg = (
        df.groupby(["HELM Annotation", "dosage_nm"])
        .agg(Inhibition_pct=("Inhibition_pct", "max"),
             target_RNA=("target_RNA", "first"),
             cv_group=("cv_group", "first"))
        .reset_index()
    )
    agg = agg[agg["Inhibition_pct"].notna() & agg["cv_group"].notna()]

    helms_raw = agg["HELM Annotation"].values
    x = np.array([convert_helm(h) for h in helms_raw])
    chem_cov = _extract_chemistry(helms_raw)

    dose_cov = agg["dosage_nm"].fillna(agg["dosage_nm"].median()).values.astype(np.float32).reshape(-1, 1)
    covariates = np.column_stack([dose_cov, chem_cov])

    return {
        "x": x,
        "y": agg["Inhibition_pct"].values.astype(float),
        "groups": agg["cv_group"].apply(lambda u: f"patent_{u}").values,
        "covariates": covariates,
        "conditions": {
            "dosage_nm": agg["dosage_nm"].values.astype(float),
        },
        "name": "In vitro inhibition (%)",
    }


def load_potency() -> dict:
    """Load in vitro potency (IC50) per ASO by fitting Hill curves to
    electroporation dose-response. Grouped by patent (earliest) with
    HELM-level dedup so no sequence appears in both train and test.
    """
    from analyses.utils.compounds import fit_ic50_for_compound

    df = pd.read_parquet(_data_dir / "dose_response_processed.parquet")
    df = df[
        df["HELM Annotation"].apply(Helm.valid_chemistry)
        & (df["transfection_method"] == "Electroporation")
    ].copy()
    df = _earliest_patent(df)

    # Group by HELM only (not HELM × USPTO ID) so every dose-response point
    # for that HELM — across all patents — feeds one Hill fit. cv_group
    # provides the GroupKFold key (HELM's earliest patent), so multi-patent
    # HELMs still land in a single fold.
    rows = []
    for helm, g in df.groupby("HELM Annotation"):
        ic50 = fit_ic50_for_compound(g["dosage_nm"].values, g["Inhibition_pct"].values)
        if np.isnan(ic50):
            continue
        target = g["target_RNA"].dropna().iloc[0] if g["target_RNA"].notna().any() else None
        cv_group = g["cv_group"].iloc[0]  # HELM-unique
        rows.append({"HELM Annotation": helm, "ic50_nm": ic50, "target_RNA": target,
                     "cv_group": cv_group})
    fit_df = pd.DataFrame(rows)

    helms_raw = fit_df["HELM Annotation"].values
    x = np.array([convert_helm(h) for h in helms_raw])
    chem_cov = _extract_chemistry(helms_raw)

    return {
        "x": x,
        "y": fit_df["ic50_nm"].values.astype(float),
        "groups": fit_df["cv_group"].apply(lambda u: f"patent_{u}").values,
        "covariates": chem_cov,
        "name": "Potency (IC50 nM)",
    }


DATASETS = {
    "in_vitro_inhibition": load_in_vitro_inhibition,
    "potency": load_potency,
    "mouse_hepatic": load_mouse_hepatic,
    "rat_hepatic": load_rat_hepatic,
    "mouse_neuro": load_mouse_neuro,
    "rat_neuro": load_rat_neuro,
}


# ---------------------------------------------------------------------------
# Featurizer configs
# ---------------------------------------------------------------------------

FEATURIZER_CONFIGS = {
    "OneHot_full": (OneHotEncoder, {"encode_components": ["base", "sugar", "phosphate"]}),
}


def is_compatible(feat_name: str, model_name: str) -> bool:
    return True


# ---------------------------------------------------------------------------
# Model configs — one fixed hyperparameter config per model, selected a
# priori (no selection against the held-out test metric). These values
# were picked once and held constant for every dataset so all OligoGym
# rows are directly comparable.
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, tuple[type, list[dict]]] = {
    "Linear": (LinearModel, [{"type": "ridge"}]),
    "Random Forest": (RandomForestModel, [{"n_estimators": 500, "max_depth": 20}]),
    "XGBoost": (XGBoostModel, [{"n_estimators": 500, "max_depth": 10}]),
    "KNN": (NearestNeighborsModel, [{"n_neighbors": 10}]),
    "CatBoost": (CatBoostModel, [{"iterations": 500}]),
    "MLP": (MLP, [{"hidden_dims": [128, 128], "dropout": 0.25}]),
}
