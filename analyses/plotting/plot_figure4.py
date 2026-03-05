"""
Figure 4: Hagedorn model replication — hepatotoxicity and neurotoxicity.

Layout: 3 rows × 4 cols (Liver left, Neurological right).

                ── Liver ──          ── Neurological ──
Row 1 (Mouse):    [A: ROC] [B: CM]   [C: ROC] [D: CM]
Row 2 (Rat):      [E: ROC] [F: CM]   [G: ROC] [H: CM]
Row 3 (Combined): [I: ROC] [J: CM]   [K: ROC] [L: CM]

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

def draw_hep_roc(predictions, ax, pred_key="ALT", title="Mouse — ALT",
                 nodose_key=None):
    if pred_key in predictions:
        preds = np.array(predictions[pred_key]["predictions"])
        labels = np.array(predictions[pred_key]["labels"])
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = predictions[pred_key]["auc"]
        ax.plot(fpr, tpr, label=f"OligoAI-tox (AUC={auc_val:.3f})",
                color="#4878A8", linewidth=2, zorder=3)

    if nodose_key and nodose_key in predictions:
        preds_nd = np.array(predictions[nodose_key]["predictions"])
        labels_nd = np.array(predictions[nodose_key]["labels"])
        fpr_nd, tpr_nd, _ = roc_curve(labels_nd, preds_nd)
        auc_nd = predictions[nodose_key]["auc"]
        ax.plot(fpr_nd, tpr_nd, label=f"Sequence only (AUC={auc_nd:.3f})",
                color="#D97706", linewidth=1.5)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(
        range(len(labels)),
        key=lambda i: (
            0 if labels[i].startswith("OligoAI-tox") else
            1 if labels[i].startswith("Hagedorn") else
            2
        ),
    )
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10, pad=6)


def draw_neuro_roc(predictions, ax, fob_key="FOB", title="Mouse — bFOB",
                   hagedorn_key=None):
    hagedorn_handle = None
    oligo_handle = None
    hagedorn_label = None
    oligo_label = None

    if hagedorn_key and hagedorn_key in predictions:
        preds_lin = np.array(predictions[hagedorn_key]["predictions"])
        labels_lin = np.array(predictions[hagedorn_key]["labels"])
        fpr_lin, tpr_lin, _ = roc_curve(labels_lin, preds_lin)
        auc_lin = predictions[hagedorn_key]["auc"]
        hagedorn_label = f"Hagedorn et al. 2022 (AUC={auc_lin:.3f})"
        hagedorn_handle, = ax.plot(
            fpr_lin, tpr_lin, label=hagedorn_label,
            color="#D97706", linewidth=1.5, zorder=2,
        )

    if fob_key in predictions:
        preds = np.array(predictions[fob_key]["predictions"])
        labels = np.array(predictions[fob_key]["labels"])
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = predictions[fob_key]["auc"]
        oligo_label = f"OligoAI-tox (AUC={auc_val:.3f})"
        oligo_handle, = ax.plot(
            fpr, tpr, label=oligo_label,
            color="#4878A8", linewidth=2, zorder=3,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend_handles = []
    legend_labels = []
    if oligo_handle is not None:
        legend_handles.append(oligo_handle)
        legend_labels.append(oligo_label)
    if hagedorn_handle is not None:
        legend_handles.append(hagedorn_handle)
        legend_labels.append(hagedorn_label)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=8)
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
    _draw_cm(cm_data, ["ALT<2\u00d7ULN", "ALT\u22652\u00d7ULN"], predictions[pred_key]["accuracy"], ax)


def draw_neuro_cm(predictions, ax, pred_key="FOB"):
    if pred_key not in predictions:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    cm_data = predictions[pred_key]["confusion"]
    _draw_cm(cm_data, ["bFOB\u22641", "bFOB>1"], predictions[pred_key]["accuracy"], ax)


def _draw_cm(cm_data, tick_labels, accuracy, ax):
    cm = np.array([[cm_data["tn"], cm_data["fp"]],
                   [cm_data["fn"], cm_data["tp"]]])
    ax.imshow(cm, cmap="Blues", aspect="equal")
    for r in range(2):
        for c in range(2):
            color = "white" if cm[r, c] > cm.max() * 0.65 else "black"
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

    # Combined data
    combined_hep_preds = hep_data.get("combined_predictions", {})
    combined_neuro_preds = neuro_data.get("combined_predictions", {})

    has_combined = bool(combined_hep_preds) or bool(combined_neuro_preds)
    has_rat = bool(rat_hep_preds) or bool(rat_neuro_preds)

    if has_combined:
        # 3 rows × 4 cols: ROC + CM for Mouse, Rat, and Combined
        nrows = 3
        fig = plt.figure(figsize=(16, 13.5), dpi=300)
        gs = fig.add_gridspec(
            nrows, 4, width_ratios=[1, 0.8, 1, 0.8],
            left=0.08, right=0.96, wspace=0.45, hspace=0.35,
        )
        axes = np.empty((nrows, 4), dtype=object)
        for r in range(nrows):
            for c in range(4):
                axes[r, c] = fig.add_subplot(gs[r, c])

        # Panel labels in figure coords — aligned grid
        fig.canvas.draw()
        col_xs = [axes[0, c].get_position().x0 - 0.03 for c in range(4)]
        row_ys = [axes[r, 0].get_position().y1 + 0.03 for r in range(nrows)]
        for i, letter in enumerate("ABCDEFGHIJKL"):
            row, col = divmod(i, 4)
            fig.text(col_xs[col], row_ys[row], letter,
                     fontsize=14, fontweight="bold", va="bottom", ha="left")

        fig.text(0.30, 0.98, "Liver toxicity", ha="center", fontsize=14, fontweight="bold")
        fig.text(0.73, 0.98, "Neurological tolerability", ha="center", fontsize=14, fontweight="bold")

        # Row labels on left margin (evenly spaced in thirds)
        for label, row_idx in [("Mouse", 0), ("Rat", 1), ("Combined", 2)]:
            y_mid = (axes[row_idx, 0].get_position().y0 + axes[row_idx, 0].get_position().y1) / 2
            fig.text(0.02, y_mid, label, ha="center", va="center",
                     fontsize=12, fontweight="bold", rotation=90)

        # Row 1: Mouse
        draw_hep_roc(hep_preds, axes[0, 0], pred_key="ALT", title="Mouse — ALT")
        draw_hep_cm(hep_preds, axes[0, 1], pred_key="ALT")
        draw_neuro_roc(neuro_preds, axes[0, 2], fob_key="FOB", title="Mouse — bFOB",
                       hagedorn_key="hagedorn_score")
        draw_neuro_cm(neuro_preds, axes[0, 3], pred_key="FOB")

        # Row 2: Rat
        draw_hep_roc(rat_hep_preds, axes[1, 0], pred_key="rat_ALT", title="Rat — ALT")
        draw_hep_cm(rat_hep_preds, axes[1, 1], pred_key="rat_ALT")
        draw_neuro_roc(rat_neuro_preds, axes[1, 2], fob_key="rat_FOB", title="Rat — mFOB",
                       hagedorn_key="rat_hagedorn_score")
        draw_neuro_cm(rat_neuro_preds, axes[1, 3], pred_key="rat_FOB")

        # Row 3: Combined
        draw_hep_roc(combined_hep_preds, axes[2, 0], pred_key="combined_ALT",
                     title="Combined — ALT")
        draw_hep_cm(combined_hep_preds, axes[2, 1], pred_key="combined_ALT")
        draw_neuro_roc(combined_neuro_preds, axes[2, 2], fob_key="combined_FOB",
                       title="Combined — FOB", hagedorn_key="combined_hagedorn_score")
        draw_neuro_cm(combined_neuro_preds, axes[2, 3], pred_key="combined_FOB")

    elif has_rat:
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

        fig.canvas.draw()
        col_xs = [axes[0, c].get_position().x0 - 0.03 for c in range(4)]
        row_ys = [axes[r, 0].get_position().y1 + 0.03 for r in range(2)]
        for i, letter in enumerate("ABCDEFGH"):
            row, col = divmod(i, 4)
            fig.text(col_xs[col], row_ys[row], letter,
                     fontsize=14, fontweight="bold", va="bottom", ha="left")

        fig.text(0.30, 0.97, "Liver toxicity", ha="center", fontsize=14, fontweight="bold")
        fig.text(0.73, 0.97, "Neurological tolerability", ha="center", fontsize=14, fontweight="bold")

        fig.text(0.02, 0.72, "Mouse", ha="center", va="center",
                 fontsize=12, fontweight="bold", rotation=90)
        fig.text(0.02, 0.28, "Rat", ha="center", va="center",
                 fontsize=12, fontweight="bold", rotation=90)

        # Row 1: Mouse
        draw_hep_roc(hep_preds, axes[0, 0], pred_key="ALT", title="Mouse — ALT")
        draw_hep_cm(hep_preds, axes[0, 1], pred_key="ALT")
        draw_neuro_roc(neuro_preds, axes[0, 2], fob_key="FOB", title="Mouse — bFOB",
                       hagedorn_key="hagedorn_score")
        draw_neuro_cm(neuro_preds, axes[0, 3], pred_key="FOB")

        # Row 2: Rat
        draw_hep_roc(rat_hep_preds, axes[1, 0], pred_key="rat_ALT", title="Rat — ALT")
        draw_hep_cm(rat_hep_preds, axes[1, 1], pred_key="rat_ALT")
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

    # ── fig4-small: 3×2 confusion matrices (Mouse/Rat/Combined × Hep/Neuro) ──
    nrows_s = 3 if has_combined else 2
    fig_s, axes_s = plt.subplots(
        nrows_s, 2, figsize=(8, 4 * nrows_s), dpi=300,
        gridspec_kw={"wspace": 0.40, "hspace": 0.35, "left": 0.15},
    )

    fig_s.canvas.draw()
    col_xs = [axes_s[0, c].get_position().x0 - 0.03 for c in range(2)]
    row_ys = [axes_s[r, 0].get_position().y1 + 0.03 for r in range(nrows_s)]
    labels_s = "ABCDEF" if has_combined else "ABCD"
    for i, letter in enumerate(labels_s):
        row, col = divmod(i, 2)
        fig_s.text(col_xs[col], row_ys[row], letter,
                   fontsize=14, fontweight="bold", va="bottom", ha="left")

    fig_s.text(0.30, 0.97, "Liver\ntoxicity", ha="center", fontsize=14, fontweight="bold")
    fig_s.text(0.73, 0.97, "Neurological\ntolerability", ha="center", fontsize=14, fontweight="bold")

    row_labels = ["Mouse", "Rat", "Combined"] if has_combined else ["Mouse", "Rat"]
    for idx, label in enumerate(row_labels):
        y_mid = (axes_s[idx, 0].get_position().y0 + axes_s[idx, 0].get_position().y1) / 2
        fig_s.text(0.02, y_mid, label, ha="center", va="center",
                   fontsize=12, fontweight="bold", rotation=90)

    draw_hep_cm(hep_preds, axes_s[0, 0], pred_key="ALT")
    draw_neuro_cm(neuro_preds, axes_s[0, 1], pred_key="FOB")
    draw_hep_cm(rat_hep_preds, axes_s[1, 0], pred_key="rat_ALT")
    draw_neuro_cm(rat_neuro_preds, axes_s[1, 1], pred_key="rat_FOB")
    if has_combined:
        draw_hep_cm(combined_hep_preds, axes_s[2, 0], pred_key="combined_ALT")
        draw_neuro_cm(combined_neuro_preds, axes_s[2, 1], pred_key="combined_FOB")

    small_path = OUT_DIR / "fig4-small.svg"
    fig_s.savefig(small_path, format="svg", bbox_inches="tight")
    fig_s.savefig(small_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig_s)
    print(f"Saved {small_path}")

    # ── fig4-mice: 2×2 (mouse only, top: hepatorenal, bottom: neurological) ──
    fig_m = plt.figure(figsize=(9, 9), dpi=300)
    gs_m = fig_m.add_gridspec(
        2, 2, width_ratios=[1, 0.8],
        left=0.10, right=0.96, wspace=0.40, hspace=0.35,
    )
    axes_m = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes_m[r, c] = fig_m.add_subplot(gs_m[r, c], aspect="equal" if c == 0 else "auto")

    draw_hep_roc(hep_preds, axes_m[0, 0], pred_key="ALT", title="")
    draw_hep_cm(hep_preds, axes_m[0, 1], pred_key="ALT")
    draw_neuro_roc(neuro_preds, axes_m[1, 0], fob_key="FOB", title="",
                   hagedorn_key="hagedorn_score")
    draw_neuro_cm(neuro_preds, axes_m[1, 1], pred_key="FOB")

    # Row titles centered across both columns
    fig_m.canvas.draw()
    for row, label in [(0, "Liver toxicity"), (1, "Neurological tolerability")]:
        x_mid = (axes_m[row, 0].get_position().x0 + axes_m[row, 1].get_position().x1) / 2
        y_top = axes_m[row, 0].get_position().y1 + 0.02
        fig_m.text(x_mid, y_top, label, ha="center", va="bottom",
                   fontsize=13, fontweight="bold")

    mice_path = OUT_DIR / "fig4-mice.png"
    fig_m.savefig(mice_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig_m)
    print(f"Saved {mice_path}")


if __name__ == "__main__":
    main()
