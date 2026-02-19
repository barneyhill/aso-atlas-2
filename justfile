default: compile

# ── Logic (data processing) ──────────────────────────────
clean:
    uv run python -m analyses.logic.clean

pipeline: clean
    uv run python -m analyses.logic.pipeline

analysis: clean pipeline

# ── Evaluate & export ────────────────────────────────────
# Heavy: model inference on test sets (~2 min). Only re-run when checkpoint or data changes.
evaluate: analysis
    uv run python -m analyses.logic.evaluate_model

export: analysis
    uv run python -m analyses.export_paper_numbers

# Heavy: model forward pass + UMAP (~3 min). Only re-run when data or model changes.
embed: analysis
    uv run python -m analyses.logic.embed_umap

# ── Plotting ──────────────────────────────────────────────
plots: analysis
    uv run python -m analyses.plotting.plot_figure1
    uv run python -m analyses.plotting.plot_figure2
    uv run python -m analyses.plotting.plot_figure4
    uv run python -m analyses.plotting.plot_figure5

# Heavy: trains 3 ablation models (~15 min). One-off analysis, not in `build`.
ablation: analysis
    uv run python -m analyses.logic.run_ablation

# ── Manuscript ────────────────────────────────────────────
compile:
    typst compile typst/main.typ

build: analysis evaluate export embed plots compile

test:
    uv run pytest tests/ -x -q
