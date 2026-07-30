"""Held-out evaluation for an OligoAI fine-tune (runs on the RunPod pod).

Invoked by runpod/bootstrap.sh as:

    PYTHONPATH=/workspace/OligoAI python /workspace/oligoai_eval.py \
        --checkpoint /workspace/out/checkpoint.ckpt \
        --split-csv  /workspace/out/<run>.withsplit.csv \
        --out        /workspace/out/<run>.json

Loads the trained checkpoint, predicts inhibition for every row of the split
CSV (using the model's saved StandardScaler), and writes:

  * ``<out-stem>_predictions.parquet`` — the **test-split** rows with columns
    custom_id, helm_annotation, cell_line, target_RNA, aso_sequence_5_to_3,
    dosage, inhibition_percent (observed), prediction (model output). This is
    what analyses.logic.models.{oligoai_efs,geneholdout_enrichment} consume.
  * ``<out>`` (JSON) — overall + per-split regression metrics.

Inference logic (model load, ASODataset, batch unpacking, scaler inverse
transform) mirrors OligoAI's run_inference.py exactly, so predictions match the
training-time tokenisation/scaling.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.aso.dataset import ASODataset
from train_aso import ASOInhibitionPredictionWrapper

# Columns the downstream EF/enrichment scripts expect in the predictions parquet.
_PRED_COLS = ["custom_id", "helm_annotation", "cell_line", "target_RNA",
              "aso_sequence_5_to_3", "dosage", "inhibition_percent", "prediction"]


class ASODatasetWithLen(ASODataset):
    """ASODataset + __len__ for DataLoader (as in OligoAI run_inference.py)."""

    def __len__(self):
        return len(self.df)


def _predict(checkpoint: str, split_csv: str, batch_size: int = 64,
             num_workers: int = 8) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}; loading model from {checkpoint}")
    model = ASOInhibitionPredictionWrapper.load_from_checkpoint(checkpoint)
    model.eval(); model.to(device)
    print(f"[eval] model loaded (scaler={type(model.scaler).__name__})")

    df = pd.read_csv(split_csv)
    print(f"[eval] split distribution:\n{df['split'].value_counts()}")

    dataset = ASODatasetWithLen(data_path=split_csv, alphabet=Alphabet(),
                                pad_to_max_len=True)
    assert len(dataset) == len(df), f"{len(dataset)} dataset rows vs {len(df)} csv rows"
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=(device == "cuda"), shuffle=False)

    preds: list[float] = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            (aso, chem, backbone, context, _, dosage, transfection, _) = batch
            aso, chem, backbone, context = (aso.to(device), chem.to(device),
                                            backbone.to(device), context.to(device))
            dosage, transfection = dosage.to(device), transfection.to(device)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    scaled = model(aso, chem, backbone, context, dosage, transfection)
            else:
                scaled = model(aso, chem, backbone, context, dosage, transfection)
            preds.extend(model.scaler.inverse_transform(scaled).cpu().numpy())
            if (bi + 1) % 20 == 0:
                print(f"[eval] {(bi + 1) * batch_size} rows predicted ...")

    assert len(preds) == len(df), f"{len(preds)} preds vs {len(df)} rows"
    df["prediction"] = np.asarray(preds, dtype=float)
    return df


def _metrics(df: pd.DataFrame) -> dict:
    def reg(sub: pd.DataFrame) -> dict:
        if len(sub) < 2:
            return {"n": int(len(sub))}
        err = sub["prediction"].to_numpy(float) - sub["inhibition_percent"].to_numpy(float)
        r = float(pearsonr(sub["prediction"], sub["inhibition_percent"])[0])
        return {
            "n": int(len(sub)),
            "mae": round(float(np.mean(np.abs(err))), 4),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "r2": round(r ** 2, 4),
            "pearson": round(r, 4),
            "spearman": round(float(spearmanr(sub["prediction"], sub["inhibition_percent"])[0]), 4),
        }

    out = {"overall": reg(df)}
    for split in ("train", "val", "test"):
        s = df[df["split"] == split]
        if not s.empty:
            out[split] = reg(s)
    # Per-target test spearman (held-out genes) — useful sanity for the holdout.
    by_target = {}
    test = df[df["split"] == "test"]
    for tgt, g in test.groupby("target_RNA"):
        if len(g) > 1:
            by_target[str(tgt)] = round(float(spearmanr(g["prediction"], g["inhibition_percent"])[0]), 4)
    if by_target:
        out["test_spearman_by_target"] = by_target
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out", required=True, help="metrics JSON path; predictions "
                    "parquet is written as <stem>_predictions.parquet")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    df = _predict(args.checkpoint, args.split_csv, args.batch_size, args.num_workers)

    out_path = Path(args.out)
    metrics = _metrics(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote metrics -> {out_path}")
    print(f"[eval] test metrics: {metrics.get('test')}")

    # Test-split predictions parquet (the EF/enrichment input).
    test = df[df["split"] == "test"].copy()
    missing = [c for c in _PRED_COLS if c not in test.columns]
    if missing:
        raise SystemExit(f"split CSV missing columns for predictions parquet: {missing}")
    pred_path = out_path.with_name(out_path.stem + "_predictions.parquet")
    test[_PRED_COLS].to_parquet(pred_path, index=False)
    print(f"[eval] wrote {len(test)} test predictions -> {pred_path}")


if __name__ == "__main__":
    main()
