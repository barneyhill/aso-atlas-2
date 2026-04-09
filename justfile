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

# ── Export (reads JSONs from data/results/) ──────────────
export:
    uv run python -m analyses.export_paper_numbers

# ── Plotting (reads processed data from data/results/) ──
plots:
    uv run python -m analyses.plotting.plot_figure1
    uv run python -m analyses.plotting.plot_figure2
    uv run python -m analyses.plotting.plot_figure3
    uv run python -m analyses.plotting.plot_figure4
    uv run python -m analyses.plotting.plot_figure5
    uv run python -m analyses.plotting.plot_figure6
    uv run python -m analyses.plotting.plot_figure7
    uv run python -m analyses.plotting.plot_supp_enrichment_sweep
    uv run python -m analyses.plotting.plot_supp_dinucleotide
    uv run python -m analyses.plotting.plot_supp_alt_ast
    uv run python -m analyses.plotting.plot_supp_biomarker_dist
    uv run python -m analyses.plotting.plot_oligogym_table

# ── Manuscript ───────────────────────────────────────────
compile:
    typst compile typst/main.typ

export-models:
    uv run python -m analyses.export_models

# ── OligoGym benchmark (heavy, run explicitly) ─────────────
oligogym:
    uv run python -m analyses.logic.models.oligogym_benchmark

oligogym-table:
    uv run python -m analyses.plotting.plot_oligogym_table

build: analysis export plots compile

test:
    uv run pytest tests/ -x -q
