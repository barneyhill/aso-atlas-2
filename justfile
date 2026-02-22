default: compile

# ── Logic (data processing) ──────────────────────────────
clean:
    uv run python -m analyses.logic.clean

pipeline: clean
    uv run python -m analyses.logic.pipeline

analysis: clean pipeline

# ── Hagerdorn models ─────────────────────────────────────
hagerdorn:
    uv run python -m analyses.04_hagerdorn.01_hepatotox
    uv run python -m analyses.04_hagerdorn.02_neurotox

# ── Export (reads CSVs/JSONs from analysis + hagerdorn) ──
export:
    uv run python -m analyses.export_paper_numbers

# ── Plotting (reads processed data from analysis + hagerdorn) ──
plots:
    uv run python -m analyses.plotting.plot_figure1
    uv run python -m analyses.plotting.plot_figure2
    uv run python -m analyses.plotting.plot_figure3
    uv run python -m analyses.plotting.plot_figure4
    uv run python -m analyses.plotting.plot_figure5

# ── Manuscript ───────────────────────────────────────────
compile:
    typst compile typst/main.typ

build: analysis hagerdorn export plots compile

test:
    uv run pytest tests/ -x -q
