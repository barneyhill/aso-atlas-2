#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "rdkit>=2023.9.1",
#     "scipy>=1.11",
#     "statsmodels>=0.14",
#     "pyarrow>=14.0",
# ]
# ///
"""
RDKit Descriptor Analysis for ASO Hepatotoxicity

Maps HELM-encoded ASO sequences to nucleoside SMILES, calculates RDKit
molecular descriptors, sums across sequences, and tests associations
with ALT (liver enzyme biomarker).

Run with: uv run analyses/hypotheses/003_rdkit_descriptors/analysis.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats
from statsmodels.stats.multitest import multipletests


DATA_DIR = Path(__file__).parent.parent.parent / "data"

# =============================================================================
# NUCLEOSIDE SMILES MAPPINGS
# =============================================================================

# DNA nucleosides (2'-deoxyribose sugar)
DNA_NUCLEOSIDES = {
    "dA": "Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1",  # 2'-deoxyadenosine
    "dG": "Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1",  # 2'-deoxyguanosine
    "dC": "Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1",  # 2'-deoxycytidine
    "dT": "Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]c1=O",  # thymidine
    "d5meC": "Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)nc1N",  # 5-methyl-2'-deoxycytidine
}

# MOE nucleosides (2'-O-methoxyethyl ribose sugar)
# The 2'-O-methoxyethyl modification: 2'-OH replaced with -O-CH2-CH2-O-CH3
MOE_NUCLEOSIDES = {
    "moeA": "COCCO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(N)ncnc12",  # MOE-adenosine
    "moeG": "COCCO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(=O)[nH]c(N)nc12",  # MOE-guanosine
    "moeC": "COCCO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1ccc(N)nc1=O",  # MOE-cytidine
    "moeT": "COCCO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(=O)[nH]c1=O",  # MOE-thymidine
    "moe5meC": "COCCO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(N)nc1=O",  # MOE-5-methylcytidine
}

# cET nucleosides (constrained ethyl / 2'-4' constrained ethyl BNA)
# Bridge between 2' and 4' positions with ethyl constraint
CET_NUCLEOSIDES = {
    "cetA": "C[C@H]1O[C@H]2[C@@H](CO)O[C@H]([C@@H]12)n1cnc2c(N)ncnc12",  # cEt-adenosine
    "cetG": "C[C@H]1O[C@H]2[C@@H](CO)O[C@H]([C@@H]12)n1cnc2c(=O)[nH]c(N)nc12",  # cEt-guanosine
    "cetC": "C[C@H]1O[C@H]2[C@@H](CO)O[C@H]([C@@H]12)n1ccc(N)nc1=O",  # cEt-cytidine
    "cetT": "C[C@H]1O[C@H]2[C@@H](CO)O[C@H]([C@@H]12)n1cc(C)c(=O)[nH]c1=O",  # cEt-thymidine
    "cet5meC": "C[C@H]1O[C@H]2[C@@H](CO)O[C@H]([C@@H]12)n1cc(C)c(N)nc1=O",  # cEt-5-methylcytidine
}

# 2'-OMe nucleosides (2'-O-methyl ribose sugar)
# The 2'-O-methyl modification: 2'-OH replaced with -O-CH3
OME_NUCLEOSIDES = {
    "omeA": "CO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(N)ncnc12",  # 2'-OMe-adenosine
    "omeG": "CO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(=O)[nH]c(N)nc12",  # 2'-OMe-guanosine
    "omeC": "CO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1ccc(N)nc1=O",  # 2'-OMe-cytidine
    "omeT": "CO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(=O)[nH]c1=O",  # 2'-OMe-thymidine
    "omeU": "CO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1ccc(=O)[nH]c1=O",  # 2'-OMe-uridine
    "ome5meC": "CO[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(N)nc1=O",  # 2'-OMe-5-methylcytidine
}

# LNA nucleosides (Locked Nucleic Acid - 2'-4' methylene bridge)
LNA_NUCLEOSIDES = {
    "lnaA": "OC[C@H]1O[C@H]([C@H]2O[C@H]1CO2)n1cnc2c(N)ncnc12",  # LNA-adenosine
    "lnaG": "OC[C@H]1O[C@H]([C@H]2O[C@H]1CO2)n1cnc2c(=O)[nH]c(N)nc12",  # LNA-guanosine
    "lnaC": "OC[C@H]1O[C@H]([C@H]2O[C@H]1CO2)n1ccc(N)nc1=O",  # LNA-cytidine
    "lnaT": "OC[C@H]1O[C@H]([C@H]2O[C@H]1CO2)n1cc(C)c(=O)[nH]c1=O",  # LNA-thymidine
    "lnaU": "OC[C@H]1O[C@H]([C@H]2O[C@H]1CO2)n1ccc(=O)[nH]c1=O",  # LNA-uridine
    "lna5meC": "OC[C@H]1O[C@H]([C@H]2O[C@H]1CO2)n1cc(C)c(N)nc1=O",  # LNA-5-methylcytidine
}

# 2'-Fluoro nucleosides (2'-F)
FLUORO_NUCLEOSIDES = {
    "fA": "F[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(N)ncnc12",  # 2'-F-adenosine
    "fG": "F[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(=O)[nH]c(N)nc12",  # 2'-F-guanosine
    "fC": "F[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1ccc(N)nc1=O",  # 2'-F-cytidine
    "fT": "F[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(=O)[nH]c1=O",  # 2'-F-thymidine
    "fU": "F[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1ccc(=O)[nH]c1=O",  # 2'-F-uridine
    "f5meC": "F[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(N)nc1=O",  # 2'-F-5-methylcytidine
}

# Standard RNA nucleosides (ribose sugar)
RNA_NUCLEOSIDES = {
    "rA": "O[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(N)ncnc12",  # adenosine
    "rG": "O[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cnc2c(=O)[nH]c(N)nc12",  # guanosine
    "rC": "O[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1ccc(N)nc1=O",  # cytidine
    "rU": "O[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1ccc(=O)[nH]c1=O",  # uridine
    "rT": "O[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(=O)[nH]c1=O",  # ribothymidine
    "r5meC": "O[C@@H]1[C@H](O[C@H](CO)[C@@H]1O)n1cc(C)c(N)nc1=O",  # 5-methylcytidine
}

ALL_NUCLEOSIDES = {
    **DNA_NUCLEOSIDES,
    **MOE_NUCLEOSIDES,
    **CET_NUCLEOSIDES,
    **OME_NUCLEOSIDES,
    **LNA_NUCLEOSIDES,
    **FLUORO_NUCLEOSIDES,
    **RNA_NUCLEOSIDES,
}

# =============================================================================
# RDKIT DESCRIPTORS
# =============================================================================

DESCRIPTOR_FUNCS = {
    # === Original descriptors ===
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "MolWt": Descriptors.MolWt,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "FractionCSP3": Descriptors.FractionCSP3,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "MolMR": Descriptors.MolMR,
    "LabuteASA": Descriptors.LabuteASA,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "BertzCT": Descriptors.BertzCT,
    # === Charge distribution ===
    "MaxPartialCharge": Descriptors.MaxPartialCharge,
    "MinPartialCharge": Descriptors.MinPartialCharge,
    "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
    "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
    # === Electrotopological state ===
    "MaxAbsEStateIndex": Descriptors.MaxAbsEStateIndex,
    "MinAbsEStateIndex": Descriptors.MinAbsEStateIndex,
    "MinEStateIndex": Descriptors.MinEStateIndex,
    "MaxEStateIndex": Descriptors.MaxEStateIndex,
    # === Ring descriptors ===
    "NumSaturatedRings": Descriptors.NumSaturatedRings,
    "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
    "RingCount": Descriptors.RingCount,
    # === Connectivity indices ===
    "Chi0": Descriptors.Chi0,
    "Chi1": Descriptors.Chi1,
    "Chi0n": Descriptors.Chi0n,
    "Chi1n": Descriptors.Chi1n,
    "Chi0v": Descriptors.Chi0v,
    "Chi1v": Descriptors.Chi1v,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
    "HallKierAlpha": Descriptors.HallKierAlpha,
    # === Hydrogen bonding detail ===
    "NHOHCount": Descriptors.NHOHCount,
    "NOCount": Descriptors.NOCount,
    # === VSA descriptors (surface area by property) ===
    "PEOE_VSA1": Descriptors.PEOE_VSA1,  # Most negative partial charge
    "PEOE_VSA2": Descriptors.PEOE_VSA2,
    "PEOE_VSA3": Descriptors.PEOE_VSA3,
    "PEOE_VSA6": Descriptors.PEOE_VSA6,
    "PEOE_VSA7": Descriptors.PEOE_VSA7,
    "PEOE_VSA8": Descriptors.PEOE_VSA8,  # Most positive partial charge
    "SlogP_VSA1": Descriptors.SlogP_VSA1,  # Most hydrophilic
    "SlogP_VSA2": Descriptors.SlogP_VSA2,
    "SlogP_VSA3": Descriptors.SlogP_VSA3,
    "SlogP_VSA5": Descriptors.SlogP_VSA5,
    "SlogP_VSA6": Descriptors.SlogP_VSA6,  # Most hydrophobic
    "SMR_VSA1": Descriptors.SMR_VSA1,
    "SMR_VSA3": Descriptors.SMR_VSA3,
    "SMR_VSA5": Descriptors.SMR_VSA5,
    "SMR_VSA7": Descriptors.SMR_VSA7,
    "EState_VSA1": Descriptors.EState_VSA1,
    "EState_VSA2": Descriptors.EState_VSA2,
    "EState_VSA3": Descriptors.EState_VSA3,
    "EState_VSA4": Descriptors.EState_VSA4,
    "EState_VSA5": Descriptors.EState_VSA5,
    "EState_VSA6": Descriptors.EState_VSA6,
}


# =============================================================================
# HELM PARSING
# =============================================================================


def parse_helm_to_nucleoside_keys(helm: str) -> list[str] | None:
    """
    Parse HELM string to list of nucleoside dictionary keys.

    Args:
        helm: HELM annotation string, e.g.,
            RNA1{{[moe](G)[sp].[moe]([5meC])[sp].d(A)[sp]...}}$$$$

    Returns:
        List of nucleoside keys, e.g., ['moeG', 'moe5meC', 'dA', ...]
        Returns None if HELM contains unknown bases (e.g., 'N') or is unparseable.
    """
    # Extract sequence part between {{ and }}
    match = re.search(r"\{\{(.+?)\}\}", helm)
    if not match:
        return None

    seq = match.group(1)
    nucleotides = seq.split(".")

    # Known bases
    known_bases = {"A", "G", "C", "T", "U", "5meC"}

    keys = []
    for nuc in nucleotides:
        nuc = nuc.strip()
        if not nuc:
            continue

        # Determine sugar type
        if nuc.startswith("[moe]"):
            sugar = "moe"
        elif nuc.startswith("[cet]") or nuc.startswith("[cEt]"):
            sugar = "cet"
        elif nuc.startswith("[m]") or nuc.startswith("m("):  # 2'-O-methyl
            sugar = "ome"
        elif nuc.startswith("[lna]"):  # Locked nucleic acid
            sugar = "lna"
        elif nuc.startswith("[fR]"):  # 2'-fluoro
            sugar = "f"
        elif nuc.startswith("r("):  # Standard RNA ribose
            sugar = "r"
        elif nuc.startswith("d("):  # DNA
            sugar = "d"
        else:
            # Unknown sugar type (e.g., [?], conjugates)
            continue

        # Extract base from parentheses
        base_match = re.search(r"\(([^)]+)\)", nuc)
        if not base_match:
            continue
        base = base_match.group(1)

        # Normalize base - handle [5meC] or 5meC
        base = base.strip("[]")
        if base.lower() == "5mec":
            base = "5meC"

        # Check for unknown base (e.g., 'N' for any nucleotide)
        if base not in known_bases:
            return None  # Reject entire HELM if any base is unknown

        # Build key
        key = f"{sugar}{base}"
        keys.append(key)

    return keys if keys else None


# =============================================================================
# DESCRIPTOR CALCULATION
# =============================================================================


def calc_nucleoside_descriptors(smiles: str) -> dict[str, float]:
    """Calculate all descriptors for a single nucleoside SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: np.nan for name in DESCRIPTOR_FUNCS}

    return {name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()}


def precompute_nucleoside_descriptors() -> dict[str, dict[str, float]]:
    """Precompute descriptors for all nucleoside types."""
    descriptors = {}
    for key, smiles in ALL_NUCLEOSIDES.items():
        descriptors[key] = calc_nucleoside_descriptors(smiles)
    return descriptors


def sum_descriptors_for_aso(
    nucleoside_keys: list[str],
    nucleoside_descriptors: dict[str, dict[str, float]],
) -> dict[str, float]:
    """
    Sum descriptors across all nucleosides in an ASO.

    Args:
        nucleoside_keys: List of nucleoside keys (e.g., ['moeG', 'moeG', 'dA', ...])
        nucleoside_descriptors: Precomputed descriptors for each nucleoside type

    Returns:
        Dictionary of summed descriptors
    """
    summed = {name: 0.0 for name in DESCRIPTOR_FUNCS}

    for key in nucleoside_keys:
        if key in nucleoside_descriptors:
            for name, value in nucleoside_descriptors[key].items():
                if not np.isnan(value):
                    summed[name] += value

    return summed


# =============================================================================
# DATA LOADING
# =============================================================================


def flatten_biomarker(val) -> float:
    """Convert biomarker array to mean value."""
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val) if not np.isnan(val) else np.nan
    if isinstance(val, np.ndarray):
        if len(val) == 0:
            return np.nan
        valid = val[~np.isnan(val)]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def load_and_process_data(
    nucleoside_descriptors: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Load HELM sequences, calculate descriptors, and merge with ALT data."""
    # 1. Load hepatotoxicity parquet and get all unique HELM annotations
    parquet_file = DATA_DIR / "oligostack/processed/hepatictoxicity_processed.parquet"
    df_bio = pd.read_parquet(parquet_file)

    # Get unique HELM annotations with ALT data
    helm_list = df_bio["HELM Annotation"].dropna().unique().tolist()
    print(f"Loaded {len(helm_list)} unique HELM sequences from hepatotoxicity data")

    # 2. Calculate summed descriptors for each ASO
    aso_descriptors = []
    parse_failures = 0
    short_sequences = 0
    min_length = 10  # Minimum ASO length to include

    for helm in helm_list:
        nuc_keys = parse_helm_to_nucleoside_keys(helm)
        if nuc_keys is None:
            parse_failures += 1
            continue

        if len(nuc_keys) < min_length:
            short_sequences += 1
            continue

        summed = sum_descriptors_for_aso(nuc_keys, nucleoside_descriptors)
        summed["HELM"] = helm
        summed["n_nucleotides"] = len(nuc_keys)
        aso_descriptors.append(summed)

    df_desc = pd.DataFrame(aso_descriptors)
    print(f"Calculated descriptors for {len(df_desc)} ASOs")
    print(f"  ({parse_failures} HELM parse failures, {short_sequences} short sequences filtered)")

    # Flatten ALT
    df_bio["ALT_flat"] = df_bio["ALT"].apply(flatten_biomarker)

    # Aggregate by HELM Annotation
    df_bio_agg = (
        df_bio.groupby("HELM Annotation").agg({"ALT_flat": "mean"}).reset_index()
    )
    df_bio_agg.rename(columns={"HELM Annotation": "HELM"}, inplace=True)
    print(f"Unique HELM with ALT data: {len(df_bio_agg)}")

    # 4. Merge
    merged = pd.merge(df_desc, df_bio_agg, on="HELM", how="inner")
    print(f"Merged dataset: {len(merged)} ASOs with both descriptors and ALT")

    return merged


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlations between summed descriptors and ALT."""
    descriptor_cols = [
        c for c in df.columns if c in DESCRIPTOR_FUNCS and c not in ["HELM", "ALT_flat"]
    ]

    results = []
    for desc in descriptor_cols:
        valid = df[[desc, "ALT_flat"]].dropna()
        n = len(valid)

        if n < 10:
            continue

        # Check for constant input (would cause correlation to be undefined)
        if valid[desc].nunique() <= 1:
            continue

        rho, p = stats.spearmanr(valid[desc], valid["ALT_flat"])

        results.append(
            {
                "descriptor": desc,
                "n": n,
                "spearman_rho": rho,
                "p_value": p,
            }
        )

    results_df = pd.DataFrame(results)

    # FDR correction (only on valid p-values)
    if len(results_df) > 0:
        valid_p = results_df["p_value"].notna()
        if valid_p.sum() > 0:
            _, p_fdr, _, _ = multipletests(
                results_df.loc[valid_p, "p_value"], method="fdr_bh"
            )
            results_df.loc[valid_p, "p_fdr"] = p_fdr
        else:
            results_df["p_fdr"] = np.nan
        results_df["significant"] = results_df["p_fdr"] < 0.05

    return results_df.sort_values("p_value")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("RDKIT DESCRIPTOR - ALT CORRELATION ANALYSIS")
    print("=" * 70)

    # Validate SMILES
    print("\nValidating nucleoside SMILES...")
    invalid = []
    for key, smiles in ALL_NUCLEOSIDES.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(key)
            print(f"  WARNING: Invalid SMILES for {key}")
    if invalid:
        print(f"\n{len(invalid)} invalid SMILES found. Please check mappings.")
    else:
        print(f"  All {len(ALL_NUCLEOSIDES)} nucleoside SMILES are valid")

    # Precompute descriptors
    print("\nPrecomputing nucleoside descriptors...")
    nucleoside_descriptors = precompute_nucleoside_descriptors()
    print(f"  Computed {len(DESCRIPTOR_FUNCS)} descriptors for {len(nucleoside_descriptors)} nucleoside types")

    # Show example descriptor values
    print("\nExample descriptor values (dA vs moeA):")
    print(f"  {'Descriptor':<20} {'dA':>12} {'moeA':>12} {'diff':>12}")
    print("  " + "-" * 58)
    for desc in ["MolLogP", "TPSA", "MolWt", "NumHDonors"]:
        dA_val = nucleoside_descriptors["dA"][desc]
        moeA_val = nucleoside_descriptors["moeA"][desc]
        print(f"  {desc:<20} {dA_val:>12.2f} {moeA_val:>12.2f} {moeA_val - dA_val:>+12.2f}")

    # Load and process data
    print("\nLoading data...")
    merged = load_and_process_data(nucleoside_descriptors)

    if len(merged) < 10:
        print("\nERROR: Too few matched samples to compute correlations!")
        return

    # Compute correlations
    print("\nComputing Spearman correlations...")
    results = compute_correlations(merged)
    print(f"Total correlations tested: {len(results)}")

    # Report results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    n_sig = results["significant"].sum() if "significant" in results.columns else 0
    print(f"\nSignificant correlations (FDR < 0.05): {n_sig}")

    if n_sig > 0:
        sig = results[results["significant"]].sort_values("p_fdr")
        print("\n{:<25} {:>5} {:>10} {:>12}".format("Descriptor", "N", "rho", "p_FDR"))
        print("-" * 55)
        for _, row in sig.iterrows():
            print(
                "{:<25} {:>5} {:>10.3f} {:>12.2e}".format(
                    row["descriptor"], row["n"], row["spearman_rho"], row["p_fdr"]
                )
            )

    # Show all correlations
    print("\n" + "=" * 70)
    print("ALL CORRELATIONS (sorted by p-value)")
    print("=" * 70)
    print("\n{:<25} {:>5} {:>10} {:>12} {:>12}".format("Descriptor", "N", "rho", "p_raw", "p_FDR"))
    print("-" * 68)
    for _, row in results.iterrows():
        sig_marker = "*" if row.get("significant", False) else ""
        print(
            "{:<25} {:>5} {:>10.3f} {:>12.2e} {:>12.2e} {}".format(
                row["descriptor"],
                row["n"],
                row["spearman_rho"],
                row["p_value"],
                row["p_fdr"],
                sig_marker,
            )
        )

    # Save results
    output_file = Path(__file__).parent / "rdkit_descriptor_correlations.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Also save the merged data for further exploration
    data_output = Path(__file__).parent / "rdkit_descriptor_data.csv"
    merged.to_csv(data_output, index=False)
    print(f"Merged data saved to: {data_output}")


if __name__ == "__main__":
    main()
