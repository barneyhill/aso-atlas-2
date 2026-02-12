"""Data pipeline for OligoAI2 multi-task ASO regression model.

Handles HELM tokenization, data loading from processed parquet files,
target-group-based train/val/test splitting, and PyTorch Dataset classes.
"""

import ast
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import helm directly to avoid kinetics_model __init__ chain (needs polars)
_helm_path = Path(__file__).resolve().parents[2] / "kinetics_model" / "src" / "data" / "helm.py"
_spec = importlib.util.spec_from_file_location("helm", _helm_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_helm, ParsedHelm = _mod.parse_helm, _mod.ParsedHelm

_DATA = Path(__file__).resolve().parents[2] / "data" / "oligostack" / "processed"

# ── Vocabulary ────────────────────────────────────────────────────────

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 4, "5meC": 5}
SUGAR_TO_IDX = {"MOE": 0, "DNA": 1, "LNA": 2, "cEt": 3, "fR": 4, "OMe": 5, "RNA": 6}
BACKBONE_TO_IDX = {"PS": 0, "PO": 1, "PAD": 2}
N_BASES, N_SUGARS, N_BACKBONES = 6, 7, 3

TASK_NAMES = ["inhibition", "ALT", "AST", "FOB"]
TASK_ID = {name: i for i, name in enumerate(TASK_NAMES)}
N_TASKS = len(TASK_NAMES)

HEPATIC_BIOMARKERS = ["ALT", "AST"]
LOG_TRANSFORM = {"ALT", "AST"}


# ── HELM Encoder ─────────────────────────────────────────────────────

class HelmEncoder:
    """Parse + encode unique HELM strings once. Datasets index into results.

    Avoids re-parsing the same strings across train/val/test splits.
    ~10x fewer parse_helm calls than per-row encoding.
    """

    def __init__(self, helm_strings: list[str] | np.ndarray, max_len: int = 40):
        seen: dict[str, ParsedHelm | None] = {}
        for h in helm_strings:
            if h not in seen:
                seen[h] = parse_helm(h)

        self.valid = frozenset(h for h, p in seen.items() if p is not None)
        valid_pairs = [(h, p) for h, p in seen.items() if p is not None]
        self.helm_to_idx = {h: i for i, (h, _) in enumerate(valid_pairs)}

        N = len(valid_pairs)
        base = np.zeros((N, max_len), dtype=np.int64)
        sugar = np.zeros((N, max_len), dtype=np.int64)
        bb = np.full((N, max_len), BACKBONE_TO_IDX["PAD"], dtype=np.int64)
        mask = np.zeros((N, max_len), dtype=np.bool_)

        for row, (_, p) in enumerate(valid_pairs):
            n = min(p.length, max_len)
            for i in range(n):
                base[row, i] = 5 if p.base_mods[i] == "5meC" else BASE_TO_IDX.get(p.bases[i], 0)
                sugar[row, i] = SUGAR_TO_IDX.get(p.sugars[i], SUGAR_TO_IDX["RNA"])
                mask[row, i] = True
            for i in range(min(n - 1, max_len)):
                bb[row, i] = BACKBONE_TO_IDX.get(p.backbones[i], BACKBONE_TO_IDX["PO"])

        self.base_idx = torch.from_numpy(base)
        self.sugar_idx = torch.from_numpy(sugar)
        self.backbone_idx = torch.from_numpy(bb)
        self.mask = torch.from_numpy(mask)
        print(f"  HelmEncoder: {len(helm_strings):,} strings → {N:,} unique valid")

    def gather(self, helm_strings: np.ndarray | list[str]) -> dict[str, torch.Tensor]:
        """Index into encoded tensors for a list of HELM strings."""
        idx = torch.tensor([self.helm_to_idx[h] for h in helm_strings], dtype=torch.long)
        return {
            "base_idx": self.base_idx[idx],
            "sugar_idx": self.sugar_idx[idx],
            "backbone_idx": self.backbone_idx[idx],
            "mask": self.mask[idx],
        }


# ── Normalization ────────────────────────────────────────────────────

@dataclass
class NormStats:
    """Per-task normalization statistics (from training data only)."""
    mean: dict[int, float]
    std: dict[int, float]

    def denormalize_std(self, value: float, task_id: int) -> float:
        return value * self.std.get(task_id, 1.0)


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_array(val) -> list[float]:
    """Parse a biomarker value (list, string repr, scalar, or None) to floats."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, np.ndarray)):
        return [float(v) for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple)):
                return [float(v) for v in parsed if v is not None]
            return [float(parsed)] if parsed is not None else []
        except (ValueError, SyntaxError):
            return []
    return [float(val)]


def assign_groups(df: pd.DataFrame, invitro_df: pd.DataFrame | None = None) -> pd.Series:
    """3-tier target group: target_RNA → Compound ID join → USPTO proxy."""
    groups = df.get("target_RNA", pd.Series(dtype=object, index=df.index)).copy()

    missing = groups.isna()
    if missing.any() and invitro_df is not None and "Compound ID" in df.columns:
        cid_map = invitro_df.dropna(subset=["target_RNA"]).groupby("Compound ID")["target_RNA"].first()
        groups.loc[missing] = df.loc[missing, "Compound ID"].map(cid_map)

    still = groups.isna()
    if still.any() and "USPTO ID" in df.columns:
        groups.loc[still] = df.loc[still, "USPTO ID"].apply(
            lambda x: f"patent_{x}" if pd.notna(x) else None,
        )
    return groups


# ── Data Loading ─────────────────────────────────────────────────────

def load_in_vitro() -> tuple[pd.DataFrame, dict]:
    """Load and prepare in vitro inhibition + dose response data."""
    iv = pd.read_parquet(_DATA / "in_vitro_inhibition_processed.parquet")
    dr = pd.read_parquet(_DATA / "dose_response_processed.parquet")
    iv["source"] = 0  # 0 = inhibition
    dr["source"] = 1  # 1 = dose_response
    df = pd.concat([iv, dr], ignore_index=True)
    df = df.dropna(subset=["dosage_nm", "HELM Annotation"])

    df["target"] = df["Inhibition_pct"].clip(0, 100) / 100.0

    # Categoricals
    tf_map = {t: i for i, t in enumerate(sorted(df["transfection_method"].dropna().unique()))}
    df["transfection_idx"] = df["transfection_method"].map(tf_map).fillna(0).astype(int)

    # Continuous
    df["log10_dose"] = np.log10(df["dosage_nm"].clip(lower=1))
    df["treatment_hrs"] = df["treatment_period_hrs"].fillna(df["treatment_period_hrs"].median())

    df["group"] = assign_groups(df)
    info = {"transfection_to_idx": tf_map, "n_transfection": len(tf_map)}
    return df, info


def _load_invivo(filename, species_col, species_val, admin_col,
                 biomarkers, cov_fn, invitro_df=None):
    """Load in vivo data: filter species, assign groups, explode biomarker arrays."""
    df = pd.read_parquet(_DATA / filename)
    df = df[df[species_col] == species_val].dropna(subset=["HELM Annotation"]).copy()
    groups = assign_groups(df, invitro_df)

    admin_map = {a: i for i, a in enumerate(sorted(df[admin_col].dropna().unique()))} if admin_col in df.columns else {}

    rows = []
    for idx, row in df.iterrows():
        shared = cov_fn(row, admin_map)
        shared["HELM Annotation"] = row["HELM Annotation"]
        shared["group"] = groups.get(idx)
        for col, task_id in biomarkers.items():
            for v in _parse_array(row.get(col)):
                rows.append({**shared, "task_id": task_id, "raw_value": float(v)})

    exploded = pd.DataFrame(rows)
    info = {"admin_to_idx": admin_map, "n_admin": len(admin_map)}
    return exploded, info


def load_hepatic_mouse(invitro_df=None) -> tuple[pd.DataFrame, dict]:
    """Load hepatotoxicity data (mouse), explode 6 biomarker arrays."""
    def covs(row, admin_map):
        return {
            "dose_mgkg": row.get("dosage_mg_per_kg", np.nan),
            "num_doses": row.get("num_doses", np.nan),
            "dosing_days": row.get("dosing_period_days", np.nan),
            "admin_idx": admin_map.get(row.get("adminstration_method", ""), 0),
        }
    df, info = _load_invivo(
        "hepatictoxicity_processed.parquet", "species", "mouse",
        "adminstration_method", {bm: TASK_ID[bm] for bm in HEPATIC_BIOMARKERS},
        covs, invitro_df,
    )
    if not df.empty:
        for col in ["dose_mgkg", "num_doses", "dosing_days"]:
            df[col] = df[col].fillna(df[col].median())
    return df, info


def load_neuro_mouse(invitro_df=None) -> tuple[pd.DataFrame, dict]:
    """Load neurotoxicity data (mouse), explode FOB arrays."""
    def covs(row, admin_map):
        return {
            "dose_ug": row.get("dosage_ug", np.nan),
            "latency_hrs_log": np.log1p(row.get("latency_time_hours", 0.0)),
            "admin_idx": admin_map.get(row.get("administration_method", ""), 0),
        }
    df, info = _load_invivo(
        "neurotoxicity_processed.parquet", "species", "Mouse",
        "administration_method", {"FOB_score": TASK_ID["FOB"]},
        covs, invitro_df,
    )
    if not df.empty:
        df["dose_ug"] = df["dose_ug"].fillna(df["dose_ug"].median())
    return df, info


# ── Splits ───────────────────────────────────────────────────────────

def build_splits(*dfs: pd.DataFrame, seed: int = 42,
                 train_frac: float = 0.8, val_frac: float = 0.1):
    """Split by target gene groups (80/10/10). Returns {split: set_of_groups}."""
    all_groups = sorted({
        g for df in dfs if "group" in df.columns
        for g in df["group"].dropna().unique()
    })
    rng = np.random.RandomState(seed)
    rng.shuffle(all_groups)

    n_train = int(len(all_groups) * train_frac)
    n_val = int(len(all_groups) * val_frac)
    splits = {
        "train": set(all_groups[:n_train]),
        "val": set(all_groups[n_train:n_train + n_val]),
        "test": set(all_groups[n_train + n_val:]),
    }
    print(f"  Splits: {' / '.join(f'{k}={len(v)}' for k, v in splits.items())} groups")
    return splits


def _assign_split(df: pd.DataFrame, splits: dict[str, set[str]]) -> pd.Series:
    """Map each row to a split via its group."""
    col = pd.Series(index=df.index, dtype=object)
    for name, groups in splits.items():
        col[df["group"].isin(groups)] = name
    return col


# ── Datasets ─────────────────────────────────────────────────────────

class InVitroDataset(Dataset):
    """In vitro inhibition dataset. HELM pre-encoded via HelmEncoder."""

    def __init__(self, df: pd.DataFrame, encoder: HelmEncoder, group_to_id: dict):
        self.n = len(df)
        self.target = torch.tensor(df["target"].values, dtype=torch.float32)
        self.continuous = torch.from_numpy(
            np.column_stack([df["log10_dose"].values, df["treatment_hrs"].values]).astype(np.float32),
        )
        self.transfection_idx = torch.tensor(df["transfection_idx"].values, dtype=torch.long)
        self.group_id = torch.tensor(
            df["group"].map(group_to_id).fillna(-1).values, dtype=torch.long,
        )
        self.source = torch.tensor(df["source"].values, dtype=torch.long)

        enc = encoder.gather(df["HELM Annotation"].values)
        self.base_idx = enc["base_idx"]
        self.sugar_idx = enc["sugar_idx"]
        self.backbone_idx = enc["backbone_idx"]
        self.mask = enc["mask"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in vars(self).items() if isinstance(v, torch.Tensor)}


class InVivoDataset(Dataset):
    """Combined hepatic + neuro in vivo dataset."""

    def __init__(self, hepatic_df, neuro_df, norm_stats: NormStats, encoder: HelmEncoder, group_to_id: dict):
        parts = []
        if not hepatic_df.empty:
            parts.append(hepatic_df)
        if not neuro_df.empty:
            parts.append(neuro_df)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        self.n = len(df)
        if not self.n:
            return

        # Normalize targets (vectorized)
        tasks = df["task_id"].values.astype(np.int64)
        raw = df["raw_value"].values.astype(np.float64)
        norm = raw.copy()
        for tid in [TASK_ID["ALT"], TASK_ID["AST"]]:
            m = tasks == tid
            if m.any():
                norm[m] = np.log1p(raw[m])
        m = tasks == TASK_ID["FOB"]
        if m.any():
            norm[m] = raw[m] / 7.0
        for tid in range(1, N_TASKS):
            m = tasks == tid
            if m.any() and tid in norm_stats.mean:
                norm[m] = (norm[m] - norm_stats.mean[tid]) / max(norm_stats.std[tid], 1e-8)

        self.target = torch.tensor(norm, dtype=torch.float32)
        self.task_id = torch.from_numpy(tasks)
        self.group_id = torch.tensor(
            df["group"].map(group_to_id).fillna(-1).values, dtype=torch.long,
        )

        # Covariates: [dose_mgkg, num_doses, dosing_days, dose_ug, latency_hrs_log]
        def col(name):
            return df[name].fillna(0).values if name in df.columns else np.zeros(self.n)

        self.continuous = torch.from_numpy(np.column_stack([
            col("dose_mgkg"), col("num_doses"), col("dosing_days"),
            col("dose_ug"), col("latency_hrs_log"),
        ]).astype(np.float32))
        self.admin_idx = torch.from_numpy(col("admin_idx").astype(np.int64))

        enc = encoder.gather(df["HELM Annotation"].values)
        self.base_idx = enc["base_idx"]
        self.sugar_idx = enc["sugar_idx"]
        self.backbone_idx = enc["backbone_idx"]
        self.mask = enc["mask"]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in vars(self).items() if isinstance(v, torch.Tensor)}


# ── Dataset Utilities ────────────────────────────────────────────────

def move_to(ds, device: torch.device):
    """Move all tensor attributes to device (call once; eliminates per-batch transfers)."""
    for k in list(vars(ds)):
        v = getattr(ds, k)
        if isinstance(v, torch.Tensor):
            setattr(ds, k, v.to(device))
    return ds


def iter_batches(ds, batch_size: int, shuffle: bool = True):
    """Yield batches by on-device tensor indexing (replaces DataLoader)."""
    fields = {k: v for k, v in vars(ds).items() if isinstance(v, torch.Tensor)}
    dev = next(iter(fields.values())).device
    idx = torch.randperm(ds.n, device=dev) if shuffle else torch.arange(ds.n, device=dev)
    for start in range(0, ds.n, batch_size):
        sel = idx[start:start + batch_size]
        yield {k: v[sel] for k, v in fields.items()}


# ── Norm Stats ───────────────────────────────────────────────────────

def compute_norm_stats(hepatic_df: pd.DataFrame, neuro_df: pd.DataFrame) -> NormStats:
    """Per-task mean/std from training data for standardization."""
    mean, std = {}, {}
    for bm in HEPATIC_BIOMARKERS:
        tid = TASK_ID[bm]
        vals = hepatic_df.loc[hepatic_df["task_id"] == tid, "raw_value"].values
        if len(vals) == 0:
            mean[tid], std[tid] = 0.0, 1.0
            continue
        if bm in LOG_TRANSFORM:
            vals = np.log1p(vals)
        mean[tid] = float(np.nanmean(vals))
        std[tid] = max(float(np.nanstd(vals)), 1e-8)

    fob = neuro_df.loc[neuro_df["task_id"] == TASK_ID["FOB"], "raw_value"].values if not neuro_df.empty else np.array([])
    if len(fob):
        scaled = fob / 7.0
        mean[TASK_ID["FOB"]] = float(np.nanmean(scaled))
        std[TASK_ID["FOB"]] = max(float(np.nanstd(scaled)), 1e-8)
    else:
        mean[TASK_ID["FOB"]], std[TASK_ID["FOB"]] = 0.0, 1.0

    return NormStats(mean=mean, std=std)


# ── Master Load Function ────────────────────────────────────────────

def load_all(seed: int = 42, max_len: int = 40):
    """Load all data, build HELM encoder, split, and create datasets.

    Returns dict with train/val/test datasets and metadata.
    """
    t0 = time.time()
    print("Loading data...")

    invitro_df, invitro_info = load_in_vitro()
    print(f"  In vitro: {len(invitro_df):,} rows ({time.time()-t0:.1f}s)")

    t1 = time.time()
    hepatic_df, hepatic_info = load_hepatic_mouse(invitro_df)
    hep_counts = "  ".join(
        f"{bm}={(hepatic_df['task_id']==TASK_ID[bm]).sum():,}"
        for bm in HEPATIC_BIOMARKERS
    ) if not hepatic_df.empty else "empty"
    print(f"  Hepatic: {len(hepatic_df):,} rows — {hep_counts} ({time.time()-t1:.1f}s)")

    t1 = time.time()
    neuro_df, neuro_info = load_neuro_mouse(invitro_df)
    print(f"  Neuro: {len(neuro_df):,} rows ({time.time()-t1:.1f}s)")

    # Parse + encode all HELM strings once (deduplicating across datasets)
    t1 = time.time()
    all_helms = list(invitro_df["HELM Annotation"])
    if not hepatic_df.empty:
        all_helms.extend(hepatic_df["HELM Annotation"])
    if not neuro_df.empty:
        all_helms.extend(neuro_df["HELM Annotation"])
    encoder = HelmEncoder(all_helms, max_len=max_len)
    print(f"  HELM encoding: {time.time()-t1:.1f}s")

    # Filter to valid HELMs
    invitro_df = invitro_df[invitro_df["HELM Annotation"].isin(encoder.valid)].copy()
    if not hepatic_df.empty:
        hepatic_df = hepatic_df[hepatic_df["HELM Annotation"].isin(encoder.valid)].copy()
    if not neuro_df.empty:
        neuro_df = neuro_df[neuro_df["HELM Annotation"].isin(encoder.valid)].copy()

    # Splits
    splits = build_splits(invitro_df, hepatic_df, neuro_df, seed=seed)
    invitro_df["split"] = _assign_split(invitro_df, splits)
    if not hepatic_df.empty:
        hepatic_df["split"] = _assign_split(hepatic_df, splits)
    if not neuro_df.empty:
        neuro_df["split"] = _assign_split(neuro_df, splits)

    # Drop unassigned
    invitro_df = invitro_df[invitro_df["split"].notna()]
    hepatic_df = hepatic_df[hepatic_df["split"].notna()] if not hepatic_df.empty else hepatic_df
    neuro_df = neuro_df[neuro_df["split"].notna()] if not neuro_df.empty else neuro_df

    # Norm stats from training set only
    hep_train = hepatic_df[hepatic_df["split"] == "train"] if not hepatic_df.empty else pd.DataFrame()
    neu_train = neuro_df[neuro_df["split"] == "train"] if not neuro_df.empty else pd.DataFrame()
    norm_stats = compute_norm_stats(hep_train, neu_train)

    # Build datasets
    t1 = time.time()

    # Global group→id mapping for median Spearman evaluation
    all_groups = sorted({
        g for df in [invitro_df, hepatic_df, neuro_df]
        for g in df["group"].dropna().unique()
    })
    group_to_id = {g: i for i, g in enumerate(all_groups)}

    def iv(split):
        return InVitroDataset(invitro_df[invitro_df["split"] == split].reset_index(drop=True), encoder, group_to_id)

    def vivo(split):
        h = hepatic_df[hepatic_df["split"] == split].reset_index(drop=True) if not hepatic_df.empty else pd.DataFrame()
        n = neuro_df[neuro_df["split"] == split].reset_index(drop=True) if not neuro_df.empty else pd.DataFrame()
        return InVivoDataset(h, n, norm_stats, encoder, group_to_id)

    result = {
        "train_invitro": iv("train"), "val_invitro": iv("val"), "test_invitro": iv("test"),
        "train_invivo": vivo("train"), "val_invivo": vivo("val"), "test_invivo": vivo("test"),
        "invitro_info": invitro_info, "hepatic_info": hepatic_info, "neuro_info": neuro_info,
        "norm_stats": norm_stats, "splits": splits,
    }
    print(f"  Datasets: {time.time()-t1:.1f}s")

    for split in ["train", "val", "test"]:
        print(f"    {split}: iv={len(result[f'{split}_invitro']):,}  vivo={len(result[f'{split}_invivo']):,}")
    print(f"  Total: {time.time()-t0:.1f}s")

    return result
