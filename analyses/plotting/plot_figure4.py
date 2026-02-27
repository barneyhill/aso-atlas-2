"""
Figure 4: Hagedorn model replication — hepatotoxicity and neurotoxicity.

Layout: 2 rows × 4 cols (Mouse left, Rat right), ROC + CM only.

              ── Mouse ──         ── Rat ──
Row 1 (Hep): [A: ROC] [B: CM]   [C: ROC] [D: CM]
Row 2 (Neu): [E: ROC] [F: CM]   [G: ROC] [H: CM]

Reads: data/results/{hepatotox,neurotox}.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
HEPATOTOX_JSON = _root / "data/results/hepatotox.json"
NEUROTOX_JSON = _root / "data/results/neurotox.json"
OUT_DIR = _root / "typst/plots/fig4"


# ── ROC panels ───────────────────────────────────────────────────

def draw_hep_roc(predictions, ax, pred_key="ALT", title="Mouse — ALT"):
    if pred_key in predictions:
        preds = np.array(predictions[pred_key]["predictions"])
        labels = np.array(predictions[pred_key]["labels"])
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = predictions[pred_key]["auc"]
        ax.plot(fpr, tpr, label=f"OligoAI-tox (AUC={auc_val:.3f})",
                color="#4878A8", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10, pad=6)


def draw_neuro_roc(predictions, ax, fob_key="FOB", title="Mouse — bFOB",
                   hagedorn_key=None):
    if fob_key in predictions:
        preds = np.array(predictions[fob_key]["predictions"])
        labels = np.array(predictions[fob_key]["labels"])
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = predictions[fob_key]["auc"]
        ax.plot(fpr, tpr, label=f"OligoAI-tox (AUC={auc_val:.3f})",
                color="#4878A8", linewidth=2)

    if hagedorn_key and hagedorn_key in predictions:
        preds_lin = np.array(predictions[hagedorn_key]["predictions"])
        labels_lin = np.array(predictions[hagedorn_key]["labels"])
        fpr_lin, tpr_lin, _ = roc_curve(labels_lin, preds_lin)
        auc_lin = predictions[hagedorn_key]["auc"]
        ax.plot(fpr_lin, tpr_lin, label=f"Hagedorn linear (AUC={auc_lin:.3f})",
                color="#888888", linewidth=1.5, linestyle="--")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10, pad=6)


# ── CM panels ────────────────────────────────────────────────────

def draw_hep_cm(predictions, ax, pred_key="ALT"):
    if pred_key not in predictions:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    cm_data = predictions[pred_key]["confusion"]
    _draw_cm(cm_data, ["<2\u00d7ULN", "\u22652\u00d7ULN"], predictions[pred_key]["accuracy"], ax)


def draw_neuro_cm(predictions, ax, pred_key="FOB"):
    if pred_key not in predictions:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    cm_data = predictions[pred_key]["confusion"]
    _draw_cm(cm_data, ["FOB\u22641", "FOB>1"], predictions[pred_key]["accuracy"], ax)


def _draw_cm(cm_data, tick_labels, accuracy, ax):
    cm = np.array([[cm_data["tn"], cm_data["fp"]],
                   [cm_data["fn"], cm_data["tp"]]])
    ax.imshow(cm, cmap="Blues", aspect="equal")
    for r in range(2):
        for c in range(2):
            color = "white" if cm[r, c] > cm.max() / 2 else "black"
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    pass  # no per-panel title


# ── Main ──────────────────────────────────────────────────────────

def main():
    for p in [HEPATOTOX_JSON, NEUROTOX_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `just hagerdorn` first.")

    with open(HEPATOTOX_JSON) as f:
        hep_data = json.load(f)
    hep_preds = hep_data["predictions"]

    with open(NEUROTOX_JSON) as f:
        neuro_data = json.load(f)
    neuro_preds = neuro_data["predictions"]

    # Rat data
    rat_hep_preds = hep_data.get("rat_predictions", {})
    rat_neuro_preds = neuro_data.get("rat_predictions", {})

    has_rat = bool(rat_hep_preds) or bool(rat_neuro_preds)

    if has_rat:
        # 2 rows × 4 cols: ROC + CM for Mouse and Rat
        fig = plt.figure(figsize=(16, 9), dpi=300)
        gs = fig.add_gridspec(
            2, 4, width_ratios=[1, 0.8, 1, 0.8],
            left=0.08, right=0.96, wspace=0.45, hspace=0.35,
        )
        axes = np.empty((2, 4), dtype=object)
        for r in range(2):
            for c in range(4):
                axes[r, c] = fig.add_subplot(gs[r, c])

        # Panel labels in figure coords — aligned grid
        fig.canvas.draw()  # force layout so get_position() is accurate
        col_xs = [axes[0, c].get_position().x0 - 0.03 for c in range(4)]
        row_ys = [axes[r, 0].get_position().y1 + 0.03 for r in range(2)]
        for i, letter in enumerate("ABCDEFGH"):
            row, col = divmod(i, 4)
            fig.text(col_xs[col], row_ys[row], letter,
                     fontsize=14, fontweight="bold", va="bottom", ha="left")

        fig.text(0.30, 0.97, "Mouse", ha="center", fontsize=14, fontweight="bold")
        fig.text(0.73, 0.97, "Rat", ha="center", fontsize=14, fontweight="bold")

        # Row labels on left margin
        fig.text(0.02, 0.72, "Hepatorenal\ntoxicity", ha="center", va="center",
                 fontsize=12, fontweight="bold", rotation=90)
        fig.text(0.02, 0.28, "Neurological\ntolerability", ha="center", va="center",
                 fontsize=12, fontweight="bold", rotation=90)

        # Top row: hepatotoxicity
        draw_hep_roc(hep_preds, axes[0, 0], pred_key="ALT", title="Mouse — ALT")
        draw_hep_cm(hep_preds, axes[0, 1], pred_key="ALT")
        draw_hep_roc(rat_hep_preds, axes[0, 2], pred_key="rat_ALT", title="Rat — ALT")
        draw_hep_cm(rat_hep_preds, axes[0, 3], pred_key="rat_ALT")

        # Bottom row: neurotoxicity (with Hagedorn linear baseline)
        draw_neuro_roc(neuro_preds, axes[1, 0], fob_key="FOB", title="Mouse — bFOB",
                       hagedorn_key="hagedorn_score")
        draw_neuro_cm(neuro_preds, axes[1, 1], pred_key="FOB")
        draw_neuro_roc(rat_neuro_preds, axes[1, 2], fob_key="rat_FOB", title="Rat — mFOB",
                       hagedorn_key="rat_hagedorn_score")
        draw_neuro_cm(rat_neuro_preds, axes[1, 3], pred_key="rat_FOB")
    else:
        # Fallback: 2×2 mouse only
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 9), dpi=300,
            gridspec_kw={"width_ratios": [1, 0.8], "wspace": 0.35, "hspace": 0.45},
        )

        for i, letter in enumerate("ABCD"):
            row, col = divmod(i, 2)
            axes[row, col].text(-0.08, 1.12, letter, transform=axes[row, col].transAxes,
                                fontsize=14, fontweight="bold")

        draw_hep_roc(hep_preds, axes[0, 0], title="Hepatotoxicity — ALT")
        draw_hep_cm(hep_preds, axes[0, 1])
        draw_neuro_roc(neuro_preds, axes[1, 0], title="Neurotoxicity — bFOB",
                       hagedorn_key="hagedorn_score")
        draw_neuro_cm(neuro_preds, axes[1, 1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig4.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # ── fig4-small: 2×2 confusion matrices (Mouse/Rat × Hep/Neuro) ──
    fig_s, axes_s = plt.subplots(
        2, 2, figsize=(8, 8), dpi=300,
        gridspec_kw={"wspace": 0.40, "hspace": 0.35, "left": 0.15},
    )

    fig_s.canvas.draw()
    col_xs = [axes_s[0, c].get_position().x0 - 0.03 for c in range(2)]
    row_ys = [axes_s[r, 0].get_position().y1 + 0.03 for r in range(2)]
    for i, letter in enumerate("ABCD"):
        row, col = divmod(i, 2)
        fig_s.text(col_xs[col], row_ys[row], letter,
                   fontsize=14, fontweight="bold", va="bottom", ha="left")

    fig_s.text(0.30, 0.97, "Mouse", ha="center", fontsize=14, fontweight="bold")
    fig_s.text(0.73, 0.97, "Rat", ha="center", fontsize=14, fontweight="bold")
    fig_s.text(0.02, 0.72, "Hepatorenal\ntoxicity", ha="center", va="center",
               fontsize=12, fontweight="bold", rotation=90)
    fig_s.text(0.02, 0.28, "Neurological\ntolerability", ha="center", va="center",
               fontsize=12, fontweight="bold", rotation=90)

    draw_hep_cm(hep_preds, axes_s[0, 0], pred_key="ALT")
    draw_hep_cm(rat_hep_preds, axes_s[0, 1], pred_key="rat_ALT")
    draw_neuro_cm(neuro_preds, axes_s[1, 0], pred_key="FOB")
    draw_neuro_cm(rat_neuro_preds, axes_s[1, 1], pred_key="rat_FOB")

    small_path = OUT_DIR / "fig4-small.svg"
    fig_s.savefig(small_path, format="svg", bbox_inches="tight")
    fig_s.savefig(small_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig_s)
    print(f"Saved {small_path}")


if __name__ == "__main__":
    main()
