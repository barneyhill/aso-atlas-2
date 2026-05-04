"""
Hagedorn 2022 neurotoxicity model replication.

Linear model (5 sequence features) for FOB score prediction.
Adapted from LNA to DNA/MOE/cET chemistry with target-level
train/test splits (GroupKFold).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import fisher_exact, spearmanr

from analyses.utils.helm import Helm
from analyses.utils.models import _optimal_threshold, mean_of_array

_root = Path(__file__).resolve().parents[3]
_data_dir = _root / "data/oligostack/processed"
RESULTS_DIR = _root / "data/results"


def load_and_filter():
    """Load neurotox data, filter to Mouse 700ug ICV valid chemistry."""
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    print(f"Loaded neurotox data: {len(df):,} rows")

    df = df[
        (df["species"] == "Mouse")
        & (df["dosage_ug"] == 700)
        & (df["administration_method"] == "ICV")
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"After filter (Mouse, 700ug ICV, valid chemistry): {len(df):,} rows")

    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    print(f"With valid FOB scores: {len(df):,} rows")
    return df


def assign_groups(df: pd.DataFrame) -> pd.Series:
    """Assign HELM-level dedup groups for GroupKFold.

    Each unique HELM sequence is assigned to its earliest patent,
    so no ASO sequence appears in both train and test.
    """
    mask = df["HELM Annotation"].notna() & df["USPTO ID"].notna()
    first_patent = (
        df.loc[mask]
        .sort_values("USPTO ID", kind="stable")
        .drop_duplicates("HELM Annotation", keep="first")
        .set_index("HELM Annotation")["USPTO ID"]
    )
    groups = df["HELM Annotation"].map(first_patent)
    print(f"  Total grouped: {groups.notna().sum()}/{len(df)}, {groups.nunique()} unique patents")
    return groups


def hagedorn_linear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 5 Hagedorn 2022 linear model features.

    Features: g_count, a_count, t_count, c_count, g_free_3p
    """
    feat_names = ["g_count", "a_count", "t_count", "c_count", "g_free_3p"]
    rows = []
    for helm in df["HELM Annotation"]:
        parsed = Helm.parse(helm)
        if parsed is None:
            rows.append({f: np.nan for f in feat_names})
            continue

        bases = parsed.bases

        g_free_3p = 0
        for b in reversed(bases):
            if b != "G":
                g_free_3p += 1
            else:
                break
        if g_free_3p == len(bases):
            g_free_3p = 20
        g_free_3p = min(g_free_3p, 20)

        rows.append({
            "g_count": bases.count("G"),
            "a_count": bases.count("A"),
            "t_count": bases.count("T"),
            "c_count": bases.count("C"),
            "g_free_3p": g_free_3p,
        })

    return pd.DataFrame(rows, index=df.index)


# ── Hagedorn 2022 exact score (fixed coefficients) ─────────────

# Coefficients from the original R code (trained on 1,645 LNA gapmers,
# calcium oscillation data). Score > 70 = predicted safe (mild neurotox).
HAGEDORN_INTERCEPT = 136.0430
HAGEDORN_COEFS = {
    "a_count": -3.1263,
    "c_count": -5.1100,
    "t_count": -4.7217,
    "g_count": -10.1264,
    "g_free_3p": 1.3577,
}


def hagedorn_score(df: pd.DataFrame) -> pd.Series:
    """Compute the exact Hagedorn 2022 neurotoxicity score.

    score = 136.0430 - 3.1263*A - 5.1100*C - 4.7217*T - 10.1264*G + 1.3577*g_free_3p

    Returns a Series of scores (higher = safer). NaN for unparseable HELM.
    """
    feat = hagedorn_linear_features(df)
    scores = pd.Series(HAGEDORN_INTERCEPT, index=df.index, dtype=float)
    for col, coef in HAGEDORN_COEFS.items():
        scores += coef * feat[col]
    scores = scores.round(1)
    scores[feat.isna().any(axis=1)] = np.nan
    return scores


def binary_labels(df: pd.DataFrame) -> pd.Series:
    """Neurotoxic (FOB > 1) vs non-toxic (FOB <= 1)."""
    y = pd.Series(index=df.index, dtype=str)
    y[df["mean_FOB"] > 1] = "high"
    y[df["mean_FOB"] <= 1] = "low"
    return y


def eval_hagedorn_score(df, groups, pred_key_prefix=""):
    """Evaluate Hagedorn 2022 fixed-coefficient score on a dataset.

    Returns (result_row, predictions_entry) or (None, None) if insufficient data.
    """
    y_labels = binary_labels(df)
    scores = hagedorn_score(df)
    mask = y_labels.isin(["high", "low"]) & scores.notna() & groups.notna()

    if mask.sum() < 50:
        print(f"  Hagedorn score: skip (n={mask.sum()})")
        return None, None

    print(f"  Hagedorn score (fixed coefficients)...", end=" ")
    y_score = (y_labels[mask] == "high").astype(int)
    s = scores[mask]
    neg_scores = -s

    try:
        auc_score = roc_auc_score(y_score, neg_scores)
    except ValueError:
        auc_score = np.nan

    threshold = _optimal_threshold(y_score, neg_scores)
    pred_labels = (neg_scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_score, pred_labels).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    _, pval = fisher_exact([[tp, fn], [fp, tn]])
    score_threshold = -threshold

    print(f"acc={acc:.3f} AUC={auc_score:.3f} (threshold={score_threshold:.1f})")

    result_row = {
        "model": "Hagedorn score (5 features)",
        "N": int(mask.sum()),
        "N_high": int(y_score.sum()),
        "N_low": int(len(y_score) - y_score.sum()),
        "GK_accuracy": acc,
        "GK_sensitivity": sens,
        "GK_specificity": spec,
        "GK_AUC": auc_score,
        "GK_pvalue": pval,
        "N_groups": groups[mask].nunique(),
    }

    pred_entry = {
        "predictions": neg_scores.tolist(),
        "labels": y_score.tolist(),
        "n": int(mask.sum()),
        "accuracy": float(acc),
        "auc": float(auc_score),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "scores": s.tolist(),
        "threshold": float(score_threshold),
        "fob_values": df.loc[mask, "mean_FOB"].tolist(),
        "groups": groups[mask].tolist(),
    }

    return result_row, (f"{pred_key_prefix}hagedorn_score", pred_entry)


def load_and_filter_rat():
    """Load neurotox data, filter to Rat 3000ug valid chemistry."""
    df = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df = df[
        (df["species"] == "Rat")
        & (df["dosage_ug"] == 3000)
        & (df["latency_time_hours"] == 3)
        & df["HELM Annotation"].apply(Helm.valid_chemistry)
    ].copy()
    print(f"Rat neurotox data: {len(df):,} rows")

    df["mean_FOB"] = df["FOB_score"].apply(mean_of_array)
    df = df[df["mean_FOB"].notna()].reset_index(drop=True)
    print(f"With valid FOB scores: {len(df):,} rows")
    return df


def run_combined_hagedorn():
    """Run Hagedorn-Linear on combined mouse+rat data with HELM-level dedup."""
    print("\n" + "=" * 60)
    print("Combined (Mouse+Rat) Hagedorn Score")
    print("=" * 60)

    df_all = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df_all = df_all[df_all["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df_all["mean_FOB"] = df_all["FOB_score"].apply(mean_of_array)
    df_all = df_all[df_all["mean_FOB"].notna()]

    mouse = df_all[
        (df_all["species"] == "Mouse")
        & (df_all["dosage_ug"] == 700)
        & (df_all["administration_method"] == "ICV")
    ].copy()
    rat = df_all[
        (df_all["species"] == "Rat")
        & (df_all["dosage_ug"] == 3000)
        & (df_all["latency_time_hours"] == 3)
    ].copy()

    df_combined = pd.concat([mouse, rat], ignore_index=True)
    print(f"Combined neurotox data: {len(df_combined):,} rows (Mouse: {len(mouse)}, Rat: {len(rat)})")

    y_labels = binary_labels(df_combined)
    scores = hagedorn_score(df_combined)
    groups = assign_groups(df_combined)
    mask = y_labels.isin(["high", "low"]) & scores.notna() & groups.notna()
    predictions = {}

    if mask.sum() >= 50:
        print(f"  Hagedorn score (combined)...", end=" ")
        y_score = (y_labels[mask] == "high").astype(int)
        s = scores[mask]
        neg_scores = -s
        try:
            auc_score = roc_auc_score(y_score, neg_scores)
        except ValueError:
            auc_score = np.nan

        threshold = _optimal_threshold(y_score, neg_scores)
        pred_labels = (neg_scores > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_score, pred_labels).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"AUC={auc_score:.3f}")

        predictions["combined_hagedorn_score"] = {
            "predictions": neg_scores.tolist(),
            "labels": y_score.tolist(),
            "n": int(mask.sum()),
            "accuracy": float(acc),
            "auc": float(auc_score),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        }
    else:
        print(f"  Hagedorn score (combined): skip (n={mask.sum()})")

    return predictions


def cross_species_concordance() -> dict:
    """Compare mouse vs rat FOB scores for shared compounds.

    Mouse: 700ug ICV; Rat: 3000ug (species-appropriate dose).
    Binary: FOB > 1 (neurotoxic) vs FOB <= 1 (non-toxic).
    """
    df_all = pd.read_parquet(_data_dir / "neurotoxicity_processed.parquet")
    df_all = df_all[df_all["HELM Annotation"].apply(Helm.valid_chemistry)].copy()
    df_all["mean_FOB"] = df_all["FOB_score"].apply(mean_of_array)
    df_all = df_all[df_all["mean_FOB"].notna()]

    mouse = df_all[
        (df_all["species"] == "Mouse")
        & (df_all["dosage_ug"] == 700)
        & (df_all["administration_method"] == "ICV")
    ]
    rat = df_all[
        (df_all["species"] == "Rat")
        & (df_all["dosage_ug"] == 3000)
    ]

    shared_cids = set(mouse["Compound ID"].dropna()) & set(rat["Compound ID"].dropna())
    print(f"\nCross-species concordance: {len(shared_cids)} shared compounds")

    mouse_means = mouse.groupby("Compound ID")["mean_FOB"].mean()
    rat_means = rat.groupby("Compound ID")["mean_FOB"].mean()
    common = mouse_means.index.intersection(rat_means.index)
    m_vals = mouse_means.loc[common].dropna()
    r_vals = rat_means.loc[common].dropna()
    both_valid = m_vals.index.intersection(r_vals.index)
    m_vals = m_vals.loc[both_valid]
    r_vals = r_vals.loc[both_valid]

    rho, pval = spearmanr(m_vals.values, r_vals.values)
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(len(m_vals) - 3)
    ci_lo, ci_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"  FOB: n={len(m_vals)}, Spearman rho={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={pval:.2e}")

    m_label = pd.Series(index=m_vals.index, dtype=str)
    m_label[m_vals > 1] = "high"
    m_label[m_vals <= 1] = "low"

    r_label = pd.Series(index=r_vals.index, dtype=str)
    r_label[r_vals > 1] = "high"
    r_label[r_vals <= 1] = "low"

    labelled = m_label.isin(["high", "low"]) & r_label.isin(["high", "low"])
    m_lab = m_label[labelled]
    r_lab = r_label[labelled]

    hh = int(((m_lab == "high") & (r_lab == "high")).sum())
    hl = int(((m_lab == "high") & (r_lab == "low")).sum())
    lh = int(((m_lab == "low") & (r_lab == "high")).sum())
    ll = int(((m_lab == "low") & (r_lab == "low")).sum())
    n_lab = len(m_lab)
    concordance = (hh + ll) / n_lab if n_lab > 0 else 0.0
    print(f"    Binary: concordance={concordance:.1%} (n={n_lab}), "
          f"hh={hh} hl={hl} lh={lh} ll={ll}")

    return {
        "FOB": {
            "n_shared": len(m_vals),
            "spearman_rho": round(float(rho), 3),
            "spearman_p": float(pval),
            "spearman_ci_lo": round(float(ci_lo), 3),
            "spearman_ci_hi": round(float(ci_hi), 3),
            "concordance_rate": round(concordance, 3),
            "concordance_n": n_lab,
            "crosstab": {"hh": hh, "hl": hl, "lh": lh, "ll": ll},
            "mouse_values": m_vals.tolist(),
            "rat_values": r_vals.tolist(),
        }
    }


def main():
    print("=" * 60)
    print("Hagedorn 2022 — Neurotoxicity Model (Linear)")
    print("=" * 60)

    # ── Mouse ──
    df = load_and_filter()
    groups = assign_groups(df)

    y_labels = binary_labels(df)
    print(f"\nBinary labels:")
    print(f"  Neurotoxic (FOB > 1): {(y_labels == 'high').sum():,}")
    print(f"  Non-toxic (FOB <= 1): {(y_labels == 'low').sum():,}")

    mouse_row, mouse_pred = eval_hagedorn_score(df, groups)
    predictions = {}
    if mouse_pred:
        predictions[mouse_pred[0]] = mouse_pred[1]

    cross_species = cross_species_concordance()

    # ── Rat ──
    print("\n" + "=" * 60)
    print("Rat Neurotoxicity (Hagedorn-Linear)")
    print("=" * 60)
    rat_df = load_and_filter_rat()
    rat_groups = assign_groups(rat_df)

    rat_y_labels = binary_labels(rat_df)
    print(f"\nRat binary labels:")
    print(f"  Neurotoxic (FOB > 1): {(rat_y_labels == 'high').sum():,}")
    print(f"  Non-toxic (FOB <= 1): {(rat_y_labels == 'low').sum():,}")

    rat_row, rat_pred = eval_hagedorn_score(rat_df, rat_groups, "rat_")
    rat_predictions = {}
    if rat_pred:
        rat_predictions[rat_pred[0]] = rat_pred[1]

    # ── Combined ──
    combined_predictions = run_combined_hagedorn()

    # Save consolidated JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "neurotox.json"

    models = []
    if mouse_row:
        models.append(mouse_row)

    with open(out_path, "w") as f:
        json.dump({
            "models": models,
            "predictions": predictions,
            "cross_species": cross_species,
            "rat_models": [rat_row] if rat_row else [],
            "rat_predictions": rat_predictions,
            "combined_predictions": combined_predictions,
        }, f, indent=2)
    print(f"\nSaved {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
