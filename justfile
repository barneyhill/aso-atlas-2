default: compile

# ── Logic (data processing) ──────────────────────────────
clean:
    uv run python -m analyses.logic.clean

hagerdorn:
    uv run python -m analyses.logic.models.hepatotox &
    uv run python -m analyses.logic.models.neurotox &
    wait

pipeline: clean hagerdorn
    uv run python -m analyses.logic.pipeline

analysis: clean hagerdorn pipeline

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

# ── Strategy leaderboard (reads oligogym + neurotox + pipeline JSONs) ──
ef-table: oligoai-efs
    uv run python -m analyses.logic.ef_table

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

export-models:
    uv run python -m analyses.export_models

# ── OligoGym benchmark (heavy, run explicitly) ─────────────
oligogym:
    uv run python -m analyses.logic.models.oligogym_benchmark

build: analysis export plots compile

test:
    uv run pytest tests/ -x -q

# ── HuggingFace dataset release ─────────────────────────────
aso-atlas-2-release:
    uv run python scripts/prepare_hf_release.py

upload: aso-atlas-2-release
    uv run python scripts/upload_hf_release.py
