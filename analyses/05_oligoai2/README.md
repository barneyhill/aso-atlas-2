# OligoAI2

Multi-task regression model for predicting antisense oligonucleotide (ASO) activity and toxicity from HELM-encoded chemical structure.

## Architecture

- **Factored embedding** — additive embeddings over base, sugar, and backbone vocabularies
- **Transformer encoder** (2-layer, pre-norm) with rotary position embeddings (RoPE)
- **Perceiver bottleneck** — cross-attention from learned latent queries into encoder output, producing a fixed-size ASO representation
- **Covariate encoders** — separate MLPs for in vitro (dose, treatment time, transfection method) and in vivo (dosing regimen, administration route) covariates
- **FiLM-gated head** — task embedding modulates the ASO latent via learned gating; single shared head across all tasks

## Tasks

| Task | Source | Target |
|---|---|---|
| Inhibition | In vitro (inhibition + dose-response) | % knockdown (clipped 0–1) |
| ALT | Hepatotoxicity (mouse) | log1p-transformed ALT |
| AST | Hepatotoxicity (mouse) | log1p-transformed AST |
| FOB | Neurotoxicity (mouse) | FOB score / 7 |

In vivo targets are z-scored using training-set statistics. Loss is CCC (concordance correlation coefficient), with homoscedastic uncertainty weighting across tasks.

## Training

Three-phase training schedule:

1. **Pre-train** on ~249K in vitro samples (CCC loss, OneCycleLR, 30 epochs)
2. **In vivo heads** on ~27K in vivo samples (multi-task CCC, differential LR with slow encoder, early stopping, 100 epochs)
3. **Joint fine-tuning** — alternating in vitro / in vivo batches at low LR (20 epochs)

## Data

Reads processed parquet files from `../../data/oligostack/processed/`. Splits are 80/10/10 by target gene group (ensuring all samples for a given gene stay in the same split).

HELM strings are parsed once via `HelmEncoder`, deduplicated across datasets, and stored as pre-encoded tensors. All dataset tensors are moved to device upfront to eliminate per-batch transfers.

## Usage

```bash
# Full training run
uv run python -m analyses.05_oligoai2.train

# Quick smoke test (2 epochs per phase)
uv run python -m analyses.05_oligoai2.train --smoke-test

# Skip joint fine-tuning phase
uv run python -m analyses.05_oligoai2.train --skip-phase3
```

## Files

| File | Description |
|---|---|
| `data.py` | HELM tokenization, parquet loading, train/val/test splitting, PyTorch datasets |
| `model.py` | Transformer encoder, perceiver bottleneck, covariate encoders, FiLM head, CCC loss |
| `train.py` | 3-phase training loop, evaluation, CLI entry point |
