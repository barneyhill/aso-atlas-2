"""Build the OligoAI-format training CSV from our processed parquets.

Runs locally. Produces ``data/oligostack/processed/oligoai_train.csv.gz``
with the columns expected by OligoAI's ``rinalmo.data.downstream.aso.dataset.ASODataset``.

rna_context extraction ports ``get_rna_context()`` and the alignment loop from
the asogym2 create_Rinamlo_data notebook (cell 3):
pyensembl release 110 (human) + GRCh38 primary-assembly FASTA,
flank_size=50, unique-match required, strand-aware.

Prerequisites (one-time): run `just fetch-reference`, which downloads the
Ensembl release-110 GRCh38 primary-assembly FASTA to data/reference/ (verifying
its sha256) and installs the matching pyensembl annotation. The FASTA path
defaults to that canonical location; override with the OLIGOAI_GRCH38_FASTA env
var.
"""

from __future__ import annotations

import gzip
import importlib.util
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _ROOT / "data/oligostack/processed"
_OUT_PATH_DEFAULT = _DATA_DIR / "oligoai_train.csv.gz"


def _out_path_for_fold(fold_idx: int) -> Path:
    """Write per-fold CSVs to oligoai_train_fold{i}.csv.gz so 5-fold
    CV produces five distinct training corpora. Fold 0 is the legacy
    default and keeps the original filename for backward compatibility.
    """
    if fold_idx == 0 and os.environ.get("OLIGOAI_FOLD") is None:
        return _OUT_PATH_DEFAULT
    return _DATA_DIR / f"oligoai_train_fold{fold_idx}.csv.gz"

# Canonical location populated by `just fetch-reference` (Ensembl release-110
# GRCh38 primary assembly, sha256 651561b3…68fd). Override with OLIGOAI_GRCH38_FASTA.
_DEFAULT_FASTA = os.environ.get(
    "OLIGOAI_GRCH38_FASTA",
    str(_ROOT / "data/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"),
)

# OligoAI's hard-coded vocabularies (rinalmo/data/downstream/aso/dataset.py)
_ALLOWED_SUGARS = {"MOE", "DNA", "cEt"}          # our Helm.parse spelling
_SUGAR_TO_OLIGOAI = {"MOE": "MOE", "DNA": "DNA", "cEt": "cET"}  # OligoAI vocab spelling
_ALLOWED_TRANSFECTION = {"Electroporation", "Gymnosis", "Lipofection"}

# target_RNA symbols that aren't Ensembl gene names; map to the locus whose
# sequence the ASOs actually target (rna_context lookup only — the original
# target_RNA label is preserved in the output). UBE3A-ATS (the UBE3A antisense)
# maps to its host transcript SNHG14: 240/245 of its ASOs match the SNHG14 span.
_GENE_ALIASES = {"UBE3A-ATS": "SNHG14"}


# ── Helm import (bypass analyses/utils/__init__.py which pulls xgboost) ───────

def _load_helm():
    spec = importlib.util.spec_from_file_location(
        "_oligoai_helm", _ROOT / "analyses/utils/helm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_oligoai_helm"] = mod
    spec.loader.exec_module(mod)
    return mod.Helm


Helm = _load_helm()


# ── Reference genome + pyensembl ─────────────────────────────────────────────

def _load_reference_fasta(path: str) -> dict[str, str]:
    """Load a gzipped primary-assembly FASTA into {contig: upper-case DNA}."""
    from Bio import SeqIO  # noqa: PLC0415 — optional dep

    print(f"Loading reference FASTA into memory from {path} ...", flush=True)
    ref: dict[str, str] = {}
    with gzip.open(path, "rt") as fh:
        for record in SeqIO.parse(fh, "fasta"):
            ref[record.id] = str(record.seq).upper()
    print(f"  loaded {len(ref)} contigs", flush=True)
    return ref


def _reverse_complement(seq: str) -> str:
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(comp.get(b, b) for b in reversed(seq))


def _get_rna_context(
    gene,
    gene_dna_sequence: str,
    aso_length: int,
    match_start: int,
    flank_size: int = 50,
) -> str:
    """Ported from asogym2/create_Rinamlo_data.ipynb cell 3."""
    context_start = max(0, match_start - flank_size)
    context_end = min(len(gene_dna_sequence), match_start + aso_length + flank_size)
    dna_context = gene_dna_sequence[context_start:context_end]
    if gene.strand == "+":
        return dna_context.replace("T", "U")
    return _reverse_complement(dna_context).replace("T", "U")


# ── Row-level HELM → OligoAI columns ─────────────────────────────────────────

def _helm_to_oligoai_row(helm_str: str) -> tuple[str, list[str], list[str]] | None:
    """Return (aso_sequence_5_to_3 as DNA letters, sugar_mods, backbone_mods)
    or None if the HELM is invalid / contains unsupported chemistry."""
    parsed = Helm.parse(helm_str)
    if parsed is None:
        return None
    sugars = set(parsed.sugars)
    if not sugars or not sugars.issubset(_ALLOWED_SUGARS):
        return None
    seq = parsed.dna_sequence            # A/C/G/T
    sugar_mods = [_SUGAR_TO_OLIGOAI[s] for s in parsed.sugars]
    # Atlas convention: length-N list = [bb]*(N-1) + ['<PAD>']
    backbone_mods = list(parsed.backbones) + ["<PAD>"]
    assert len(sugar_mods) == parsed.length
    assert len(backbone_mods) == parsed.length
    return seq, sugar_mods, backbone_mods


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    fasta_path = os.environ.get("OLIGOAI_GRCH38_FASTA", _DEFAULT_FASTA)
    if not Path(fasta_path).exists():
        raise SystemExit(
            f"GRCh38 reference FASTA not found at {fasta_path}.\n"
            "Run `just fetch-reference` to download it (Ensembl release-110 primary "
            "assembly + pyensembl 110), or set OLIGOAI_GRCH38_FASTA to an existing copy."
        )

    try:
        from pyensembl import EnsemblRelease  # noqa: PLC0415 — optional dep
    except ModuleNotFoundError as e:
        raise SystemExit(
            "pyensembl not installed. Run: uv pip install pyensembl biopython && "
            "pyensembl install --release 110 --species human"
        ) from e

    # ── Load & concat ────────────────────────────────────────────────────
    iv = pd.read_parquet(_DATA_DIR / "in_vitro_inhibition_processed.parquet")
    dr = pd.read_parquet(_DATA_DIR / "dose_response_processed.parquet")
    df = pd.concat([iv, dr], ignore_index=True)
    n0 = len(df)
    print(f"Loaded {n0:,} rows (iv={len(iv):,}, dose-response={len(dr):,})")

    # ── Stage 1: human cells ─────────────────────────────────────────────
    df = df[df["cell_line_species"] == "human"].copy()
    print(f"  after human-only filter: {len(df):,}")

    # ── Stage 2: HELM parse + chemistry ──────────────────────────────────
    cache: dict[str, object] = {}
    def _encode(helm):
        if helm not in cache:
            cache[helm] = _helm_to_oligoai_row(helm)
        return cache[helm]

    parsed_series = df["HELM Annotation"].map(_encode)
    df = df[parsed_series.notna()].copy()
    parsed_series = parsed_series[parsed_series.notna()]
    df["aso_sequence_5_to_3"] = [p[0] for p in parsed_series]
    df["sugar_mods"] = [p[1] for p in parsed_series]
    df["backbone_mods"] = [p[2] for p in parsed_series]
    print(f"  after HELM/chemistry filter: {len(df):,}")

    # ── Stage 3: target_RNA present + transfection method valid ──────────
    df = df[df["target_RNA"].notna()].copy()
    df = df[df["transfection_method"].isin(_ALLOWED_TRANSFECTION)].copy()
    print(f"  after target/transfection filter: {len(df):,}")

    # ── Stage 4: pyensembl + reference alignment ─────────────────────────
    ens = EnsemblRelease(110, species="human")
    try:
        ens.db  # noqa: B018 — force load, clearer error message if not installed
    except Exception as exc:
        raise SystemExit(
            "pyensembl release 110 (human) not installed. Run:\n"
            "    pyensembl install --release 110 --species human"
        ) from exc

    reference = _load_reference_fasta(fasta_path)

    # Ported alignment loop from asogym2/create_Rinamlo_data.ipynb cell 3.
    # Iterate per gene to amortise the gene-sequence extraction.
    contexts: dict[int, str] = {}
    stats = Counter()
    unique_targets = df["target_RNA"].unique()
    print(f"  resolving {len(unique_targets)} unique target_RNA symbols via pyensembl ...")

    for gene_idx, gene_name in enumerate(unique_targets, start=1):
        sub = df[df["target_RNA"] == gene_name]
        try:
            gene = ens.genes_by_name(_GENE_ALIASES.get(gene_name, gene_name))[0]
        except (ValueError, IndexError):
            stats["gene_not_found"] += len(sub)
            continue

        contig = gene.contig
        if contig not in reference:
            stats["contig_missing"] += len(sub)
            continue

        gene_dna = reference[contig][gene.start - 1 : gene.end]
        if not gene_dna:
            stats["empty_gene_dna"] += len(sub)
            continue

        for row_idx, aso in sub["aso_sequence_5_to_3"].items():
            aso_up = aso.upper()
            probe = _reverse_complement(aso_up) if gene.strand == "+" else aso_up
            matches = [m.start() for m in re.finditer(re.escape(probe), gene_dna)]
            if len(matches) != 1:
                stats["non_unique_match"] += 1
                continue
            ctx = _get_rna_context(gene, gene_dna, len(aso_up), matches[0], flank_size=50)
            contexts[row_idx] = ctx

        if gene_idx % 25 == 0:
            print(
                f"    {gene_idx}/{len(unique_targets)} genes processed; "
                f"{len(contexts):,} rows contexted, "
                f"{stats['non_unique_match']:,} non-unique, "
                f"{stats['gene_not_found']:,} genes-not-found"
            )

    df["rna_context"] = df.index.map(contexts)
    df = df[df["rna_context"].notna()].copy()
    print(f"  after rna_context extraction: {len(df):,}")
    if stats:
        print("    drop reasons:", dict(stats))

    # ── Stage 5: dosage + inhibition cleanup ─────────────────────────────
    df["inhibition_percent"] = df["Inhibition_pct"].astype(float).clip(0, 100)
    # Median-impute dosage (computed on present values only, matching ASODataset)
    median_dose = df["dosage_nm"].median()
    df["dosage"] = df["dosage_nm"].fillna(median_dose).astype(float)

    # ── Stage 6: dedup (HELM, cell_line, target_RNA, dosage_nm) ──────────
    # Sort by USPTO ID so the same (key) always lands in the same patent split.
    df = df.sort_values(["HELM Annotation", "cell_line", "target_RNA",
                         "dosage_nm", "USPTO ID"], kind="stable")

    group_cols = ["HELM Annotation", "cell_line", "target_RNA", "dosage_nm"]
    agg = (
        df.groupby(group_cols, dropna=False)
          .agg(
              inhibition_percent=("inhibition_percent", "mean"),
              dosage=("dosage", "first"),
              transfection_method=("transfection_method", "first"),
              aso_sequence_5_to_3=("aso_sequence_5_to_3", "first"),
              sugar_mods=("sugar_mods", "first"),
              backbone_mods=("backbone_mods", "first"),
              rna_context=("rna_context", "first"),
              uspto_id=("USPTO ID", "first"),
              table_number=("Table Number", "first"),
              n_rows=("inhibition_percent", "size"),
          )
          .reset_index()
    )
    print(f"  after dedup: {len(agg):,} rows "
          f"(collapsed from {len(df):,}; mean fold = {len(df)/max(len(agg),1):.2f}×)")

    # ── Stage 7: build custom_id (matches ASODataset.extract_patent_id shape) ─
    def _make_custom_id(uspto: str, table_num) -> str:
        try:
            tn = f"{int(table_num):05d}"
        except (TypeError, ValueError):
            tn = str(table_num) if pd.notna(table_num) else "0"
        return f"our-data/inhibition_tables/{uspto}_table_{tn}.xml"

    agg["custom_id"] = [
        _make_custom_id(u, t) for u, t in zip(agg["uspto_id"], agg["table_number"])
    ]

    # ── Stage 8: write ───────────────────────────────────────────────────
    # Keep cell_line, target_RNA, HELM Annotation on the row so our post-training
    # eval can regroup dose-response curves; ASODataset only reads the columns it
    # needs and ignores extras.
    agg = agg.rename(columns={"HELM Annotation": "helm_annotation"})

    # Patent-level 5-fold GroupKFold with HELM-level dedup via cv_group
    # *reassignment* (not row-drop). Each HELM is mapped to its earliest
    # patent, and every row of that HELM uses cv_group = earliest_patent
    # for the GroupKFold split. This guarantees no sequence leakage across
    # folds while keeping all rows — the previous row-filter dropped ~11%
    # of measurements (rows where the HELM was published in a later patent).
    # The original `uspto_id` column is preserved unchanged for provenance.
    #
    # No val split: we use fixed hyperparameters (max_epochs=10, fixed
    # lr/dropout/batch in bootstrap.sh) and the eval picks the newest
    # checkpoint by mtime, so val isn't used for model selection.
    # OligoGym (the parity baseline) similarly has no val split.
    # Lightning's val_dataloader is allowed to be empty.
    # Gene-holdout mode (OLIGOAI_HOLDOUT_GENES="SCN2A,UBE3A-ATS"): force every
    # row of the named target genes into `test` and everything else into
    # `train`. Unlike the patent-grouped GroupKFold below, this guarantees a
    # *gene*-level holdout — no ASO of a held-out gene (from any patent) leaks
    # into training — which is what an honest "predict for an unseen gene"
    # evaluation requires. Used for the paper-3 q-feasibility enrichment.
    holdout_genes_env = os.environ.get("OLIGOAI_HOLDOUT_GENES")
    if holdout_genes_env:
        holdout_genes = {g.strip() for g in holdout_genes_env.split(",") if g.strip()}
        present = set(agg["target_RNA"].unique())
        missing = holdout_genes - present
        if missing:
            raise SystemExit(
                f"OLIGOAI_HOLDOUT_GENES names genes absent after filtering: "
                f"{sorted(missing)} (present example targets: {sorted(present)[:5]})"
            )
        held = agg["target_RNA"].isin(holdout_genes)
        agg["split"] = np.where(held, "test", "train")
        # Invariant: held-out genes are entirely in test, never in train.
        assert agg.loc[held, "split"].eq("test").all()
        assert not agg.loc[agg["split"].eq("train"), "target_RNA"].isin(holdout_genes).any()
        per_gene = agg[held].groupby("target_RNA").size().to_dict()
        print(f"  gene holdout {sorted(holdout_genes)}: "
              f"train={agg['split'].eq('train').sum()} "
              f"test={agg['split'].eq('test').sum()} "
              f"(held-out rows per gene: {per_gene}; no val split)")
    else:
        from sklearn.model_selection import GroupKFold
        agg = agg.sort_values(["helm_annotation", "uspto_id"], kind="stable")
        first_patent = (
            agg.drop_duplicates("helm_annotation", keep="first")
               [["helm_annotation", "uspto_id"]]
               .set_index("helm_annotation")["uspto_id"]
        )
        agg["cv_group"] = agg["helm_annotation"].map(first_patent)

        fold_idx = int(os.environ.get("OLIGOAI_FOLD", "0"))
        n_splits = 5
        groups = agg["cv_group"].values
        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(agg, agg["inhibition_percent"], groups))
        train_idx, test_idx = folds[fold_idx]
        agg["split"] = "train"
        agg.iloc[test_idx, agg.columns.get_loc("split")] = "test"
        print(f"  fold {fold_idx}/{n_splits}: "
              f"train={agg['split'].eq('train').sum()} "
              f"test={agg['split'].eq('test').sum()} "
              f"(no val split)")

    out = agg[[
        "aso_sequence_5_to_3",
        "sugar_mods",
        "backbone_mods",
        "rna_context",
        "inhibition_percent",
        "dosage",
        "transfection_method",
        "custom_id",
        "cell_line",
        "target_RNA",
        "helm_annotation",
        "split",
    ]].copy()

    # ASODataset uses ast.literal_eval on sugar_mods / backbone_mods, so write
    # them as Python-list repr strings (pandas does this by default for list
    # cells going through to_csv).
    out_path = (
        _DATA_DIR / "oligoai_train_geneholdout.csv.gz"
        if holdout_genes_env else _out_path_for_fold(fold_idx)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out):,} rows, {out_path.stat().st_size/1e6:.1f} MB)")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\nSummary:")
    print(f"  rows                 : {len(out):,}")
    if len(out) == 0:
        print("  (empty dataset — skipping per-column summary)")
        return
    print(f"  unique HELM          : {agg['helm_annotation'].nunique():,}")
    print(f"  unique target_RNA    : {agg['target_RNA'].nunique():,}")
    print(f"  unique patents       : {agg['uspto_id'].nunique():,}")
    print(f"  cell lines           : {agg['cell_line'].nunique():,}")
    print(
        f"  dose range (nM)      : "
        f"{out['dosage'].min():.3g} – {out['dosage'].max():.3g} "
        f"(median {out['dosage'].median():.3g})"
    )
    print(
        f"  rna_context length   : "
        f"min={out['rna_context'].str.len().min()}, "
        f"median={int(out['rna_context'].str.len().median())}, "
        f"max={out['rna_context'].str.len().max()}"
    )
    print(f"  transfection methods : {out['transfection_method'].value_counts().to_dict()}")
    print(f"  gene-split rows      : {out['split'].value_counts().to_dict()}")
    print(f"  gene-split genes     : "
          f"{out.groupby('split')['target_RNA'].nunique().to_dict()}")


if __name__ == "__main__":
    main()
