#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "scipy>=1.10",
#     "matplotlib>=3.7",
#     "seaborn>=0.12",
#     "statsmodels>=0.14",
#     "pyarrow>=14.0",
# ]
# ///
"""
Hypothesis 004: IC50 Potency vs ALT Toxicity Relationship

Investigates whether optimizing for low hepatotoxicity (ALT) inadvertently
selects for low potency (high IC50) ASOs.

Filters:
- IC50: Hepatocyte cell lines (HepG2, Hep3B, HepaRG), electroporation only
- ALT: Mouse, subcutaneous only

Run with: uv run analyses/hypotheses/004_ic50_potency_toxicity/analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm

# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DOSE_RESPONSE_PATH = PROJECT_ROOT / "data/oligostack/processed/dose_response_processed.parquet"
HEPATIC_PATH = PROJECT_ROOT / "data/oligostack/processed/hepatictoxicity_processed.parquet"
FIGURES_DIR = SCRIPT_DIR / "figures"

# =============================================================================
# THRESHOLDS
# =============================================================================

IC50_POTENT_THRESHOLD = 500  # nM - compounds below this are "potent"
ALT_TOXIC_THRESHOLD = 100  # IU/L - compounds above this are "toxic"
IC50_R2_THRESHOLD = 0.7  # Minimum R² for valid IC50 fit

# Hepatocyte cell lines for IC50
HEPATOCYTE_CELL_LINES = [
    "HepG2", "Hep3B", "HepaRG",
    "Primary human hepatocytes", "human primary hepatocytes",
    "primary mouse hepatocytes", "mouse primary hepatocytes",
    "primary hepatocytes"
]


# =============================================================================
# IC50 CALCULATION
# =============================================================================

def hill_equation(dose, bottom, top, ic50, hill):
    """4-parameter logistic (Hill) equation for dose-response curves."""
    return bottom + (top - bottom) / (1 + (ic50 / dose) ** hill)


def fit_ic50(doses: np.ndarray, responses: np.ndarray) -> dict:
    """Fit IC50 from dose-response data using 4-parameter logistic model."""
    mask = ~(np.isnan(doses) | np.isnan(responses))
    doses = np.array(doses)[mask]
    responses = np.array(responses)[mask]

    if len(doses) < 4:
        return {"ic50_nm": np.nan, "r_squared": np.nan, "fit_success": False}

    # Initial parameter guesses
    bottom_init = np.min(responses)
    top_init = np.max(responses)
    ic50_init = np.median(doses)
    hill_init = 1.0

    min_dose = np.min(doses[doses > 0]) if np.any(doses > 0) else 1e-6
    max_dose = np.max(doses)

    try:
        popt, _ = curve_fit(
            hill_equation,
            doses,
            responses,
            p0=[bottom_init, top_init, ic50_init, hill_init],
            bounds=(
                [0, 0, min_dose / 100, 0.1],
                [100, 100, max_dose * 100, 10]
            ),
            maxfev=5000
        )
        bottom, top, ic50, hill = popt

        # Calculate R²
        y_pred = hill_equation(doses, *popt)
        ss_res = np.sum((responses - y_pred) ** 2)
        ss_tot = np.sum((responses - np.mean(responses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "ic50_nm": ic50,
            "r_squared": r_squared,
            "fit_success": True,
            "hill": hill,
            "bottom": bottom,
            "top": top
        }
    except Exception:
        return {"ic50_nm": np.nan, "r_squared": np.nan, "fit_success": False}


def calculate_ic50_for_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate IC50 for each dose-response curve in the dataset."""
    results = []

    # Group by curve identifiers
    grouped = df.groupby(["Table Number", "Compound ID", "cell_line", "transfection_method"])

    for (table, compound, cell_line, transfection), group in grouped:
        if len(group) < 4:
            continue

        doses = group["dosage_nm"].values
        responses = group["Inhibition_pct"].values

        fit_result = fit_ic50(doses, responses)
        fit_result["Compound ID"] = compound
        fit_result["cell_line"] = cell_line
        fit_result["transfection_method"] = transfection
        fit_result["n_points"] = len(group)
        results.append(fit_result)

    return pd.DataFrame(results)


# =============================================================================
# ALT PROCESSING
# =============================================================================

def flatten_alt(alt_value):
    """Convert ALT value (possibly array) to single float (mean)."""
    if alt_value is None:
        return np.nan
    if isinstance(alt_value, (int, float)):
        return float(alt_value)
    if isinstance(alt_value, np.ndarray):
        if len(alt_value) == 0:
            return np.nan
        return float(np.mean(alt_value))
    try:
        return float(alt_value)
    except (ValueError, TypeError):
        return np.nan


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HYPOTHESIS 004: IC50 POTENCY vs ALT TOXICITY")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load and filter dose-response data
    # -------------------------------------------------------------------------
    print("\n[1] Loading dose-response data...")
    df_dr = pd.read_parquet(DOSE_RESPONSE_PATH)
    print(f"    Total dose-response measurements: {len(df_dr):,}")

    # Filter to hepatocyte cell lines
    df_dr_hep = df_dr[df_dr["cell_line"].isin(HEPATOCYTE_CELL_LINES)].copy()
    print(f"    After hepatocyte filter: {len(df_dr_hep):,}")

    # Filter to electroporation
    df_dr_elec = df_dr_hep[df_dr_hep["transfection_method"] == "Electroporation"].copy()
    print(f"    After electroporation filter: {len(df_dr_elec):,}")
    print(f"    Cell lines: {df_dr_elec['cell_line'].unique().tolist()}")

    # -------------------------------------------------------------------------
    # Calculate IC50 values
    # -------------------------------------------------------------------------
    print("\n[2] Calculating IC50 values...")
    ic50_df = calculate_ic50_for_curves(df_dr_elec)
    print(f"    Total curves: {len(ic50_df):,}")
    print(f"    Successful fits: {ic50_df['fit_success'].sum():,}")

    # Filter to high-quality fits
    ic50_valid = ic50_df[
        (ic50_df["fit_success"]) &
        (ic50_df["r_squared"] > IC50_R2_THRESHOLD)
    ].copy()
    print(f"    High-quality fits (R² > {IC50_R2_THRESHOLD}): {len(ic50_valid):,}")
    print(f"    Unique compounds: {ic50_valid['Compound ID'].nunique():,}")

    # Aggregate IC50 per compound (median across curves)
    ic50_agg = ic50_valid.groupby("Compound ID")["ic50_nm"].median().reset_index()
    print(f"    Compounds with IC50: {len(ic50_agg):,}")

    # -------------------------------------------------------------------------
    # Load and filter ALT data
    # -------------------------------------------------------------------------
    print("\n[3] Loading ALT data...")
    df_hep = pd.read_parquet(HEPATIC_PATH)
    print(f"    Total hepatic records: {len(df_hep):,}")

    # Filter to mouse, subcutaneous
    df_hep = df_hep[
        (df_hep["species"] == "mouse") &
        (df_hep["adminstration_method"] == "subcutaneous")
    ].copy()
    print(f"    Mouse, subcutaneous: {len(df_hep):,}")

    # Calculate ALT mean
    df_hep["ALT_flat"] = df_hep["ALT"].apply(flatten_alt)
    df_hep = df_hep[df_hep["ALT_flat"].notna() & (df_hep["ALT_flat"] > 0)].copy()
    print(f"    With valid ALT: {len(df_hep):,}")

    # Calculate mg/kg/day
    df_hep["mg_kg_day"] = (
        df_hep["dosage_mg_per_kg"] * df_hep["num_doses"]
    ) / df_hep["dosing_period_days"]
    df_hep = df_hep[df_hep["mg_kg_day"].notna() & (df_hep["mg_kg_day"] > 0)].copy()

    # Aggregate ALT per compound
    alt_agg = df_hep.groupby("Compound ID").agg({
        "ALT_flat": "mean",
        "mg_kg_day": "mean"
    }).reset_index()
    alt_agg.rename(columns={"ALT_flat": "ALT"}, inplace=True)
    print(f"    Unique compounds with ALT: {len(alt_agg):,}")

    # -------------------------------------------------------------------------
    # Merge IC50 and ALT
    # -------------------------------------------------------------------------
    print("\n[4] Merging IC50 and ALT data...")
    merged = ic50_agg.merge(alt_agg, on="Compound ID")
    print(f"    Compounds with both IC50 and ALT: {len(merged):,}")

    if len(merged) < 10:
        print("\n    ERROR: Not enough overlapping compounds for analysis!")
        return

    # Log transforms
    merged["log_IC50"] = np.log10(merged["ic50_nm"])
    merged["log_ALT"] = np.log10(merged["ALT"])
    merged["log_mg_kg_day"] = np.log10(merged["mg_kg_day"].clip(lower=1e-6))

    # Drop rows with any NaN/inf in log values
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log_IC50", "log_ALT", "log_mg_kg_day"]
    )
    print(f"    After dropping NaN/inf: {len(merged)}")

    # -------------------------------------------------------------------------
    # Statistical Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STATISTICAL RESULTS")
    print("=" * 70)

    # Spearman correlation
    rho, p_spearman = stats.spearmanr(merged["ic50_nm"], merged["ALT"])
    print(f"\n1. Spearman Correlation (IC50 vs ALT):")
    print(f"   ρ = {rho:.4f}")
    print(f"   p = {p_spearman:.2e}")
    print(f"   n = {len(merged)}")

    if rho > 0:
        print("   Interpretation: Higher IC50 (less potent) → Higher ALT (more toxic)")
    else:
        print("   Interpretation: Higher IC50 (less potent) → Lower ALT (less toxic)")

    # Linear regression: log(ALT) ~ log(IC50)
    print(f"\n2. Linear Regression: log(ALT) ~ log(IC50)")
    X = sm.add_constant(merged["log_IC50"])
    model = sm.OLS(merged["log_ALT"], X).fit()
    print(f"   Slope = {model.params['log_IC50']:.4f} (95% CI: {model.conf_int().loc['log_IC50', 0]:.4f}, {model.conf_int().loc['log_IC50', 1]:.4f})")
    print(f"   R² = {model.rsquared:.4f}")
    print(f"   p = {model.pvalues['log_IC50']:.2e}")

    # Multiple regression: log(ALT) ~ log(IC50) + log(mg_kg_day)
    print(f"\n3. Multiple Regression: log(ALT) ~ log(IC50) + log(dose)")
    X_multi = sm.add_constant(merged[["log_IC50", "log_mg_kg_day"]])
    model_multi = sm.OLS(merged["log_ALT"], X_multi).fit()
    print(f"   IC50 coefficient = {model_multi.params['log_IC50']:.4f}, p = {model_multi.pvalues['log_IC50']:.2e}")
    print(f"   Dose coefficient = {model_multi.params['log_mg_kg_day']:.4f}, p = {model_multi.pvalues['log_mg_kg_day']:.2e}")
    print(f"   R² = {model_multi.rsquared:.4f}")

    # -------------------------------------------------------------------------
    # Quadrant Analysis
    # -------------------------------------------------------------------------
    print(f"\n4. Quadrant Analysis (IC50 < {IC50_POTENT_THRESHOLD} nM, ALT < {ALT_TOXIC_THRESHOLD} IU/L):")

    merged["potent"] = merged["ic50_nm"] < IC50_POTENT_THRESHOLD
    merged["safe"] = merged["ALT"] < ALT_TOXIC_THRESHOLD

    q_potent_safe = ((merged["potent"]) & (merged["safe"])).sum()
    q_potent_toxic = ((merged["potent"]) & (~merged["safe"])).sum()
    q_weak_safe = ((~merged["potent"]) & (merged["safe"])).sum()
    q_weak_toxic = ((~merged["potent"]) & (~merged["safe"])).sum()

    print(f"\n   {'':20} | {'Safe (ALT<100)':>15} | {'Toxic (ALT≥100)':>15} | Total")
    print(f"   {'-'*20}-+-{'-'*15}-+-{'-'*15}-+-------")
    print(f"   {'Potent (IC50<500)':<20} | {q_potent_safe:>15} | {q_potent_toxic:>15} | {q_potent_safe + q_potent_toxic:>5}")
    print(f"   {'Weak (IC50≥500)':<20} | {q_weak_safe:>15} | {q_weak_toxic:>15} | {q_weak_safe + q_weak_toxic:>5}")
    print(f"   {'-'*20}-+-{'-'*15}-+-{'-'*15}-+-------")
    print(f"   {'Total':<20} | {q_potent_safe + q_weak_safe:>15} | {q_potent_toxic + q_weak_toxic:>15} | {len(merged):>5}")

    # Chi-square test for independence
    contingency = [[q_potent_safe, q_potent_toxic], [q_weak_safe, q_weak_toxic]]
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n   Chi-square test for independence:")
    print(f"   χ² = {chi2:.2f}, p = {p_chi2:.4f}, dof = {dof}")

    if p_chi2 < 0.05:
        # Calculate odds ratio
        odds_ratio = (q_potent_safe * q_weak_toxic) / (q_potent_toxic * q_weak_safe) if (q_potent_toxic * q_weak_safe) > 0 else np.inf
        print(f"   Odds ratio (potent & safe vs weak & safe): {odds_ratio:.2f}")

    # Mann-Whitney U: ALT in potent vs weak
    print(f"\n5. Mann-Whitney U Test: ALT in Potent vs Weak ASOs")
    alt_potent = merged[merged["potent"]]["ALT"]
    alt_weak = merged[~merged["potent"]]["ALT"]
    u_stat, p_mw = stats.mannwhitneyu(alt_potent, alt_weak, alternative="two-sided")
    print(f"   Potent: n={len(alt_potent)}, median ALT={alt_potent.median():.1f}")
    print(f"   Weak: n={len(alt_weak)}, median ALT={alt_weak.median():.1f}")
    print(f"   U = {u_stat:.0f}, p = {p_mw:.4f}")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Generating visualizations...")
    print("=" * 70)

    # 1. Scatter plot: IC50 vs ALT
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(merged["ic50_nm"], merged["ALT"], alpha=0.5, s=30, c="steelblue", edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Add regression line
    x_range = np.logspace(np.log10(merged["ic50_nm"].min()), np.log10(merged["ic50_nm"].max()), 100)
    y_pred = 10 ** (model.params["const"] + model.params["log_IC50"] * np.log10(x_range))
    ax.plot(x_range, y_pred, "r-", linewidth=2, label=f"Regression (R²={model.rsquared:.3f})")

    # Add threshold lines
    ax.axvline(x=IC50_POTENT_THRESHOLD, color="green", linestyle="--", alpha=0.7, label=f"IC50={IC50_POTENT_THRESHOLD} nM")
    ax.axhline(y=ALT_TOXIC_THRESHOLD, color="orange", linestyle="--", alpha=0.7, label=f"ALT={ALT_TOXIC_THRESHOLD} IU/L")

    ax.set_xlabel("IC50 (nM)", fontsize=12)
    ax.set_ylabel("ALT (IU/L)", fontsize=12)
    ax.set_title(f"IC50 vs ALT (Hepatocyte cells, Electroporation, n={len(merged)})\nSpearman ρ={rho:.3f}, p={p_spearman:.2e}", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ic50_vs_alt_scatter.png", dpi=150)
    plt.savefig(FIGURES_DIR / "ic50_vs_alt_scatter.svg")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'ic50_vs_alt_scatter.png'}")

    # 2. Quadrant plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = []
    for _, row in merged.iterrows():
        if row["potent"] and row["safe"]:
            colors.append("green")
        elif row["potent"] and not row["safe"]:
            colors.append("orange")
        elif not row["potent"] and row["safe"]:
            colors.append("blue")
        else:
            colors.append("red")

    ax.scatter(merged["ic50_nm"], merged["ALT"], c=colors, alpha=0.6, s=40, edgecolors="none")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.axvline(x=IC50_POTENT_THRESHOLD, color="black", linestyle="--", linewidth=2)
    ax.axhline(y=ALT_TOXIC_THRESHOLD, color="black", linestyle="--", linewidth=2)

    # Add quadrant labels
    ax.text(10, 30, f"Potent & Safe\nn={q_potent_safe}", fontsize=12, ha="left", va="bottom", color="green", fontweight="bold")
    ax.text(10, 300, f"Potent & Toxic\nn={q_potent_toxic}", fontsize=12, ha="left", va="top", color="orange", fontweight="bold")
    ax.text(5000, 30, f"Weak & Safe\nn={q_weak_safe}", fontsize=12, ha="right", va="bottom", color="blue", fontweight="bold")
    ax.text(5000, 300, f"Weak & Toxic\nn={q_weak_toxic}", fontsize=12, ha="right", va="top", color="red", fontweight="bold")

    ax.set_xlabel("IC50 (nM)", fontsize=12)
    ax.set_ylabel("ALT (IU/L)", fontsize=12)
    ax.set_title("Therapeutic Window: Potency vs Toxicity\n(Green = optimal)", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ic50_alt_quadrants.png", dpi=150)
    plt.savefig(FIGURES_DIR / "ic50_alt_quadrants.svg")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'ic50_alt_quadrants.png'}")

    # 3. Boxplot: ALT by IC50 quartiles
    fig, ax = plt.subplots(figsize=(10, 6))
    merged["IC50_quartile"] = pd.qcut(merged["ic50_nm"], q=4, labels=["Q1\n(most potent)", "Q2", "Q3", "Q4\n(least potent)"])

    quartile_data = [merged[merged["IC50_quartile"] == q]["ALT"].values for q in merged["IC50_quartile"].cat.categories]
    bp = ax.boxplot(quartile_data, tick_labels=[str(q) for q in merged["IC50_quartile"].cat.categories], patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.axhline(y=ALT_TOXIC_THRESHOLD, color="orange", linestyle="--", alpha=0.7, label=f"ALT={ALT_TOXIC_THRESHOLD}")
    ax.set_ylabel("ALT (IU/L)", fontsize=12)
    ax.set_xlabel("IC50 Quartile", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("ALT by IC50 Quartiles", fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "alt_by_ic50_quartiles.png", dpi=150)
    plt.savefig(FIGURES_DIR / "alt_by_ic50_quartiles.svg")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'alt_by_ic50_quartiles.png'}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if rho > 0 and p_spearman < 0.05:
        print(f"""
FINDING: Weak POSITIVE correlation between IC50 and ALT (ρ={rho:.3f}, p={p_spearman:.2e})

This means: Higher IC50 (LESS potent) correlates with higher ALT (MORE toxic).

IMPLICATION: Optimizing for LOW toxicity does NOT select for low potency.
In fact, more potent ASOs tend to be LESS hepatotoxic.

Therapeutic window analysis:
- {q_potent_safe} compounds ({100*q_potent_safe/len(merged):.1f}%) are both potent AND safe
- {q_potent_toxic} compounds ({100*q_potent_toxic/len(merged):.1f}%) are potent but toxic
- {q_weak_safe} compounds ({100*q_weak_safe/len(merged):.1f}%) are weak but safe
- {q_weak_toxic} compounds ({100*q_weak_toxic/len(merged):.1f}%) are both weak AND toxic
""")
    elif rho < 0 and p_spearman < 0.05:
        print(f"""
FINDING: NEGATIVE correlation between IC50 and ALT (ρ={rho:.3f}, p={p_spearman:.2e})

This means: Higher IC50 (LESS potent) correlates with lower ALT (LESS toxic).

WARNING: Optimizing for LOW toxicity MAY select for low potency ASOs.
This is a potential concern for therapeutic development.
""")
    else:
        print(f"""
FINDING: No significant correlation between IC50 and ALT (ρ={rho:.3f}, p={p_spearman:.2e})

This means: Potency and toxicity appear to be INDEPENDENT.
Optimizing for low toxicity should not systematically affect potency.
""")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
