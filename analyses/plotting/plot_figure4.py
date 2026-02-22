"""
Figure 4: Hagerdorn model replication — hepatotoxicity and neurotoxicity.

Top row (A–C): Hepatotoxicity (Hagerdorn 2013, ALT prediction)
  A — Grouped bar chart (accuracy, sensitivity, specificity) across 4 RF models.
  B — ROC curve for dinucleotide model.
  C — Confusion matrix for dinucleotide model.

Bottom row (D–F): Neurotoxicity (Hagerdorn 2022, FOB prediction)
  D — Grouped bar chart for 5 models (linear + 4 RF).
  E — ROC curves for linear and dinucleotide models.
  F — Confusion matrix for dinucleotide RF model.

Reads: data/results/{hepatotox,neurotox}.json
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

matplotlib.use("Agg")

_root = Path(__file__).resolve().parents[2]
HEPATOTOX_JSON = _root / "data/results/hepatotox.json"
NEUROTOX_JSON = _root / "data/results/neurotox.json"
OUT_DIR = _root / "typst/plots/fig4"


# ── Hepatotoxicity panels ────────────────────────────────────────

def draw_hep_metrics(results_df, ax):
    alt = results_df[results_df["biomarker"] == "ALT"].dropna(subset=["GK_accuracy"])
    if alt.empty:
        return

    metrics = ["GK_accuracy", "GK_sensitivity", "GK_specificity"]
    labels = ["Accuracy", "Sensitivity", "Specificity"]
    x = np.arange(len(alt))
    width = 0.22

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, alt[metric].values, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(alt["model"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Hepatotoxicity — ALT", fontsize=10, pad=6)


def draw_hep_roc(predictions, ax):
    if "ALT" in predictions:
        preds = np.array(predictions["ALT"]["predictions"])
        labels = np.array(predictions["ALT"]["labels"])
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = predictions["ALT"]["auc"]
        ax.plot(fpr, tpr, label=f"Dinucleotide (AUC={auc_val:.3f})",
                color="#4878A8", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)


def draw_hep_cm(predictions, ax):
    if "ALT" not in predictions:
        return
    cm_data = predictions["ALT"]["confusion"]
    _draw_cm(cm_data, ["Low", "High"], predictions["ALT"]["accuracy"], ax)


# ── Neurotoxicity panels ─────────────────────────────────────────

def draw_neuro_metrics(results_df, ax):
    subset = results_df.dropna(subset=["GK_accuracy"])
    if subset.empty:
        return

    metrics = ["GK_accuracy", "GK_sensitivity", "GK_specificity"]
    labels = ["Accuracy", "Sensitivity", "Specificity"]
    x = np.arange(len(subset))
    width = 0.22

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(x + offset, subset[metric].values, width, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(subset["model"], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Neurotoxicity — FOB", fontsize=10, pad=6)


def draw_neuro_roc(predictions, ax):
    colors = {"FOB": "#4878A8", "linear": "#ff7f0e"}
    labels_map = {"FOB": "Dinucleotide RF", "linear": "Linear (5 feat.)"}

    for key in ["FOB", "linear"]:
        if key in predictions:
            preds = np.array(predictions[key]["predictions"])
            labels = np.array(predictions[key]["labels"])
            fpr, tpr, _ = roc_curve(labels, preds)
            auc_val = predictions[key]["auc"]
            ax.plot(fpr, tpr, label=f"{labels_map[key]} (AUC={auc_val:.3f})",
                    color=colors[key], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)


def draw_neuro_cm(predictions, ax):
    if "FOB" not in predictions:
        return
    cm_data = predictions["FOB"]["confusion"]
    _draw_cm(cm_data, ["Non-toxic", "Neurotoxic"], predictions["FOB"]["accuracy"], ax)


# ── Shared ────────────────────────────────────────────────────────

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
    ax.set_title(f"Acc = {accuracy:.2f}", fontsize=9, pad=6)


# ── Main ──────────────────────────────────────────────────────────

def main():
    for p in [HEPATOTOX_JSON, NEUROTOX_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `just hagerdorn` first.")

    with open(HEPATOTOX_JSON) as f:
        hep_data = json.load(f)
    hep_results = pd.DataFrame(hep_data["models"])
    hep_preds = hep_data["predictions"]

    with open(NEUROTOX_JSON) as f:
        neuro_data = json.load(f)
    neuro_results = pd.DataFrame(neuro_data["models"])
    neuro_preds = neuro_data["predictions"]

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9), dpi=300,
        gridspec_kw={"width_ratios": [1.2, 1, 0.8], "wspace": 0.35, "hspace": 0.45},
    )

    # Panel labels
    for i, letter in enumerate("ABCDEF"):
        row, col = divmod(i, 3)
        axes[row, col].text(-0.08, 1.12, letter, transform=axes[row, col].transAxes,
                            fontsize=14, fontweight="bold")

    # Top row: hepatotoxicity
    draw_hep_metrics(hep_results, axes[0, 0])
    draw_hep_roc(hep_preds, axes[0, 1])
    draw_hep_cm(hep_preds, axes[0, 2])

    # Bottom row: neurotoxicity
    draw_neuro_metrics(neuro_results, axes[1, 0])
    draw_neuro_roc(neuro_preds, axes[1, 1])
    draw_neuro_cm(neuro_preds, axes[1, 2])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fig4.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
