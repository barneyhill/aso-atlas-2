default: compile

# ── Logic (data processing) ──────────────────────────────
clean:
    uv run python -m analyses.logic.clean

hagerdorn:
    uv run python -m analyses.logic.models.hepatotox
    uv run python -m analyses.logic.models.neurotox

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

# ── Manuscript ───────────────────────────────────────────
compile:
    typst compile typst/main.typ

build: analysis export plots compile

test:
    uv run pytest tests/ -x -q
