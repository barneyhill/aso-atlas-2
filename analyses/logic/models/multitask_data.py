"""Data loading and HELM encoding for multi-task benchmark.

Uses analyses/utils/helm.py for parsing, maps to OligoAI2 token vocabulary.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyses.utils.helm import Helm
from analyses.utils.models import mean_of_array

from .multitask_model import BACKBONE_TO_IDX, BASE_TO_IDX, SUGAR_TO_IDX

_root = Path(__file__).resolve().parents[3]
_data_dir = _root / "data/oligostack/processed"


# ── HELM Encoding ───────────────────────────────────────────────────────

# Map from analyses/utils/helm.py sugar names to OligoAI2 vocab
_SUGAR_MAP = {"MOE": "MOE", "DNA": "DNA", "LNA": "LNA", "cEt": "cEt",
              "fR": "fR", "OMe": "OMe", "RNA": "RNA"}


def encode_helms(helm_strings: np.ndarray, max_len: int = 30) -> dict[str, torch.Tensor]:
    """Encode HELM strings to padded token tensors.

    Uses Helm.parse() from analyses/utils/helm.py, then maps to the
    OligoAI2 vocabulary indices (BASE_TO_IDX, SUGAR_TO_IDX, BACKBONE_TO_IDX).
    """
    # Deduplicate
    unique_helms = list(set(helm_strings))
    parsed = {h: Helm.parse(h) for h in unique_helms}
    helm_to_idx = {}
    valid_pairs = [(h, p) for h, p in parsed.items() if p is not None]
    for i, (h, _) in enumerate(valid_pairs):
        helm_to_idx[h] = i

    N = len(valid_pairs)
    base = np.zeros((N, max_len), dtype=np.int64)
    sugar = np.zeros((N, max_len), dtype=np.int64)
    bb = np.full((N, max_len), BACKBONE_TO_IDX["PAD"], dtype=np.int64)
    mask = np.zeros((N, max_len), dtype=np.bool_)

    for row, (_, p) in enumerate(valid_pairs):
        n = min(p.length, max_len)
        for i in range(n):
            bmod = p.base_mods[i]
            if bmod == "5meC":
                base[row, i] = BASE_TO_IDX["5meC"]
            else:
                base[row, i] = BASE_TO_IDX.get(p.bases[i], 0)
            sugar_name = _SUGAR_MAP.get(p.sugars[i], "RNA")
            sugar[row, i] = SUGAR_TO_IDX.get(sugar_name, SUGAR_TO_IDX["RNA"])
            mask[row, i] = True
        for i in range(min(n - 1, max_len)):
            bb[row, i] = BACKBONE_TO_IDX.get(p.backbones[i], BACKBONE_TO_IDX["PO"])

    tensors = {
        "base_idx": torch.from_numpy(base),
        "sugar_idx": torch.from_numpy(sugar),
        "backbone_idx": torch.from_numpy(bb),
        "mask": torch.from_numpy(mask),
    }

    # Build per-sample index array for the full input
    sample_idx = np.array([helm_to_idx.get(h, -1) for h in helm_strings])
    return tensors, helm_to_idx, sample_idx


def gather(tensors: dict[str, torch.Tensor], indices: np.ndarray) -> dict[str, torch.Tensor]:
    """Index into encoded tensors for a subset of samples."""
    idx = torch.from_numpy(indices.astype(np.int64))
    return {k: v[idx] for k, v in tensors.items()}


# ── Data Loaders ────────────────────────────────────────────────────────

def load_mouse_hepatic() -> dict:
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == "mouse") & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "helms": df.loc[mask, "HELM Annotation"].values,
        "y": df.loc[mask, "mean_ALT"].values.astype(np.float32),
        "groups": groups[mask].values,
    }


def load_rat_hepatic() -> dict:
    df = pd.read_parquet(_data_dir / "hepatictoxicity_processed.parquet")
    df = df[(df["species"] == "rat") & df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df["mean_ALT"] = df["ALT"].apply(mean_of_array)
    df = df[df["mean_ALT"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "helms": df.loc[mask, "HELM Annotation"].values,
        "y": df.loc[mask, "mean_ALT"].values.astype(np.float32),
        "groups": groups[mask].values,
    }


def load_mouse_neuro() -> dict:
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df = df[
        (df["species"] == "Mouse") & (df["dosage_ug"] == 700)
        & (df["administration_method"] == "ICV")
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "helms": df.loc[mask, "HELM Annotation"].values,
        "y": df.loc[mask, "mean_FOB"].values.astype(np.float32),
        "groups": groups[mask].values,
    }


def load_rat_neuro() -> dict:
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df = df[
        (df["species"] == "Rat") & (df["dosage_ug"] == 3000)
        & (df["latency_time_hours"] == 3)
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    return {
        "helms": df.loc[mask, "HELM Annotation"].values,
        "y": df.loc[mask, "mean_FOB"].values.astype(np.float32),
        "groups": groups[mask].values,
    }


def load_invitro() -> dict:
    df = pd.read_parquet(_data_dir / "in_vitro_inhibition_processed.parquet")
    df = df[df["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df = df[df["Inhibition_pct"].notna()].reset_index(drop=True)
    groups = df["USPTO ID"].apply(lambda x: f"patent_{x}" if pd.notna(x) else None)
    mask = groups.notna()
    y = df.loc[mask, "Inhibition_pct"].values.astype(np.float32)
    y = np.clip(y / 100.0, 0.0, 1.0)  # normalise to [0, 1]
    return {
        "helms": df.loc[mask, "HELM Annotation"].values,
        "y": y,
        "groups": groups[mask].values,
    }


VIVO_LOADERS = {
    "mouse_hepatic": load_mouse_hepatic,
    "rat_hepatic": load_rat_hepatic,
    "mouse_neuro": load_mouse_neuro,
    "rat_neuro": load_rat_neuro,
}
