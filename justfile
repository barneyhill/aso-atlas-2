default: compile

# ── Reference genome (GRCh38 primary assembly, Ensembl release 110) ──────────
#    Needed by oligoai_build_csv.py to extract each ASO's rna_context window
#    (pyensembl 110 gives coordinates; this FASTA gives the sequence). Downloads
#    to data/reference/ (gitignored), verifies the sha256 against Ensembl's
#    published file, and installs the matching pyensembl annotation. Run once.
REF_FASTA := "data/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
REF_URL := "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz"
REF_SHA256 := "651561b3065aa6083b62290ec96344a90d07388883874070aa368b745fde68fd"
fetch-reference:
    mkdir -p data/reference
    test -f {{REF_FASTA}} || curl -L --fail -o {{REF_FASTA}} {{REF_URL}}
    echo "{{REF_SHA256}}  {{REF_FASTA}}" | shasum -a 256 -c -
    uv pip install pyensembl biopython
    uv run pyensembl install --release 110 --species human

# ── Logic (data processing) ──────────────────────────────
clean:
    uv run python -m analyses.logic.clean

hagerdorn:
    uv run python -m analyses.logic.models.hepatotox &
    uv run python -m analyses.logic.models.neurotox &
    wait

pipeline: hagerdorn
    uv run python -m analyses.logic.pipeline

analysis: clean
    just pipeline

# ── OligoAI enrichment factors (reads oligoai_predictions.parquet) ──
oligoai-efs:
    uv run python -m analyses.logic.models.oligoai_efs

# ── Species-transfer ablation (justifies 4-model OligoAI-tox) ──
species-transfer:
    uv run python -m analyses.logic.species_transfer

# ── Fold-level CV audit (confirms zero HELM leakage in splits) ──
cv-audit:
    uv run python -m analyses.logic.cv_audit

# ── OligoAI 5-fold split CSVs (P3). Each invocation writes one fold's
#    training CSV; drive training from RunPod against the matching
#    oligoai_train_fold{i}.csv.gz.
oligoai-splits:
    for i in 0 1 2 3 4; do OLIGOAI_FOLD=$i uv run python -m analyses.logic.models.oligoai_build_csv; done

# ── Launch all 5 OligoAI training folds on RunPod in parallel (heavy, ~$15-20). ──
#    Each fold runs on its own pod so 5 GPUs are held simultaneously. Logs land
#    in runpod/logs/<ts>.fold{N}.{train,bootstrap}.log; local artifacts use
#    fold-suffixed paths so there's no collision. Depends on `just oligoai-splits`
#    to have written the per-fold CSVs first.
oligoai-launch-folds:
    for i in 0 1 2 3 4; do uv run python runpod/launch.py --fold $i & done; wait

# ── Gene-holdout build + RunPod train (P3 q-feasibility). Holds the named genes
#    fully out of training, trains ONE model, and writes
#    data/results/oligoai_{GENEHOLDOUT_NAME}_predictions.parquet. ~1 GPU/~$1-2.
#    Parameterise per experiment, e.g. for the UBE3A-ATS single-gene holdout:
#      just GENEHOLDOUT_GENES="UBE3A-ATS" GENEHOLDOUT_NAME="UBE3A_ATS_holdout" oligoai-holdout-genes
GENEHOLDOUT_GENES := "SCN2A,UBE3A-ATS"
GENEHOLDOUT_NAME := "geneholdout"
oligoai-holdout-genes: fetch-reference
    OLIGOAI_HOLDOUT_GENES="{{GENEHOLDOUT_GENES}}" uv run python -m analyses.logic.models.oligoai_build_csv
    uv run python runpod/launch.py --holdout-name {{GENEHOLDOUT_NAME}}

# ── SCN2A/UBE3A-ATS held-out enrichment (reads oligoai_geneholdout_predictions.parquet) ──
geneholdout-ef:
    uv run python -m analyses.logic.models.geneholdout_enrichment

# ── Strategy leaderboard (reads oligogym + neurotox + pipeline JSONs) ──
ef-table: oligoai-efs
    uv run python -m analyses.logic.ef_table

# ── Primary lineage-rescued Gold content-recall audit ──
#    Connects each Gold readout to its frozen source/canonical raw row, follows
#    the actual production deduplication group, and verifies the released winner.
audit:
    uv run python analyses/validation/build_production_table_canonical_map.py
    uv run python analyses/validation/dedup_lineage.py
    uv run python analyses/validation/lineage_rescued_recall.py

# Superseded release-wide/value-based matcher, retained only for provenance.
legacy-audit:
    uv run python analyses/validation/gold_standard_audit.py

# ── Rebuttal statistical block (focused AC uncertainty + conditional enrichment) ──
#    Reads pipeline_results.json, ef_table.json, oligogym_benchmark.json and the
#    OligoAI fold prediction parquets. Supersedes s1_plateau / s1_ef_composition_bias
#    for anything quoted in the response.
stats:
    uv run python -m analyses.logic.models.benchmark_ids
    uv run python -m analyses.logic.models.threshold_sweep
    uv run python -m analyses.logic.models.s1_conditional_pipeline
    uv run python -m analyses.logic.models.ac_uncertainty

# ── Data-backed Markdown blocks in the rebuttal ────────────
rebuttal-tables:
    uv run python -m analyses.logic.models.threshold_sweep
    uv run python -m analyses.logic.models.s1_conditional_pipeline
    uv run python -m analyses.logic.models.ac_uncertainty
    uv run python -m analyses.render_rebuttal_tables
    uv run pytest tests/test_rebuttal_tables.py tests/test_release_gene_reconciliation.py tests/test_paper_release_scope.py -q

# ── Export (reads JSONs from data/results/) ──────────────
export: ef-table
    uv run python -m analyses.export_paper_numbers

# ── Plotting (reads processed data from data/results/) ──
plots:
    uv run python -m analyses.plotting.plot_fig_atlas
    uv run python -m analyses.plotting.plot_fig_pipeline
    uv run python -m analyses.plotting.plot_fig_cost
    uv run python -m analyses.plotting.plot_fig_concordance
    uv run python -m analyses.plotting.plot_fig_enrichment
    uv run python -m analyses.plotting.plot_fig_clinical
    uv run python -m analyses.plotting.plot_fig_species_transfer
    uv run python -m analyses.plotting.plot_fig_cost_sensitivity
    uv run python -m analyses.plotting.plot_fig_mouse_rat_alt

# ── Manuscript ───────────────────────────────────────────
compile:
    typst compile typst/main.typ

# Regenerate every release-scoped corpus count and Figure 1, then compile exports.
paper-release:
    uv run python -m analyses.export_paper_numbers
    uv run python -m analyses.plotting.plot_fig_atlas
    typst compile typst/main.typ typst/main.pdf
    uv run python typst/export_docx.py
    uv run pytest tests/test_paper_release_scope.py tests/test_release_gene_reconciliation.py -q

export-models:
    uv run python -m analyses.export_models

# ── OligoGym benchmark (heavy, run explicitly) ─────────────
oligogym:
    uv run python -m analyses.logic.models.oligogym_benchmark

# Reproduce the paper from the versioned processed snapshot and frozen benchmark
# outputs included in the code release. This does not require private API keys or
# the large raw LLM-extraction intermediates.
reproduce:
    just pipeline
    just export
    just plots
    just compile

# Full rebuild beginning with raw extraction outputs, when those local inputs are
# available. The LLM extraction itself is documented under patent_collate/.
build: analysis
    just export
    just plots
    just compile

test:
    uv run pytest tests/ -x -q

# ── HuggingFace dataset release ─────────────────────────────
# Rebuilds the sharded parquets, manifest, generated card and NeurIPS-complete
# Croissant metadata, then checks leakage, hashes and card/paper reconciliation.
aso-atlas-2-release:
    uv run python aso-atlas-2-release/prepare_hf_release.py
    uv run pytest tests/test_release_splits.py -q

upload: aso-atlas-2-release
    uv run python aso-atlas-2-release/upload_hf_release.py
