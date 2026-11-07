"""Shared plotting style helpers.

Panel letters go through this module so that every figure in the manuscript
labels its panels the same way.
"""

from __future__ import annotations

# Every figure is placed at full column width, so a panel letter's printed size
# is its point size times (print width / figure width). A fixed point size
# therefore prints smaller on a wide figure than a narrow one. Scaling the
# point size in proportion to figure width cancels that, so a letter on the
# 18-inch pipeline figure and one on the 15-inch clinical figure land at the
# same size on the page.
PANEL_LABEL_SIZE = 16
PANEL_LABEL_REF_WIDTH_IN = 16.0

# Figure-fraction offsets: how far a letter sits left of its panel's left edge,
# and how far its baseline sits above the panel's top edge.
PANEL_LABEL_GAP = 0.02
PANEL_LABEL_RISE = 0.02


def panel_label(fig, letter: str, x: float, y: float) -> None:
    """Place a bold panel letter at figure coordinates ``(x, y)``.

    Anchored left on the baseline, so letters sharing a ``y`` sit on one line
    regardless of which glyphs they are.
    """
    scale = fig.get_size_inches()[0] / PANEL_LABEL_REF_WIDTH_IN
    fig.text(x, y, letter, fontsize=PANEL_LABEL_SIZE * scale,
             fontweight="bold", ha="left", va="baseline")


def panel_labels(fig, pairs, gap: float = PANEL_LABEL_GAP,
                 rise: float = PANEL_LABEL_RISE) -> None:
    """Label a row of panels from ``pairs`` of ``(axes, letter)``.

    Every letter in the row shares one baseline, taken from the tallest panel,
    and each sits a fixed distance left of its own panel. This replaces
    axes-relative placement, where a single offset such as ``x=-0.08`` resolves
    against each panel's own width and therefore lands at a different absolute
    position for a wide panel than a narrow one.
    """
    pairs = list(pairs)
    fig.canvas.draw()  # settle positions of aspect-constrained axes
    baseline = max(ax.get_position().y1 for ax, _ in pairs) + rise
    for ax, letter in pairs:
        panel_label(fig, letter, ax.get_position().x0 - gap, baseline)
