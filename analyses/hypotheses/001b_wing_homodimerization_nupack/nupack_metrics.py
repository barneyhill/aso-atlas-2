"""
NUPACK-based thermodynamic calculations for ASO self-complementarity.

Uses NUPACK4 to compute free energies of homodimer formation,
providing thermodynamically rigorous metrics for the wing homodimerization
hypothesis (vs simple contiguous base-pair counting in 001).

Key metrics:
- homodimer_dG: Free energy of two identical ASO strands forming a complex
- monomer_dG: Free energy of intramolecular folding (hairpin/structure)
- ddG_dimerization: Propensity to dimerize = homodimer_dG - 2*monomer_dG
"""

import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

# Add NUPACK to path
NUPACK_PATH = Path("/Users/barneyh/dphil/nonspecificactivity_ionis_paper/nupack-4.0.2.0")
sys.path.insert(0, str(NUPACK_PATH / "source" / "python"))

from nupack import Model, Strand, Tube, SetSpec, tube_analysis


@dataclass
class NupackMetrics:
    """Container for all NUPACK-computed thermodynamic metrics."""
    # Full ASO metrics
    monomer_dG: float           # Intramolecular folding free energy (kcal/mol)
    homodimer_dG: float         # Homodimer complex free energy (kcal/mol)
    ddG_dimerization: float     # homodimer_dG - 2*monomer_dG (dimerization propensity)

    # Wing-specific metrics
    wing5_monomer_dG: float     # 5' wing intramolecular folding
    wing5_homodimer_dG: float   # 5' wing homodimer (matches 6YCS crystal structure)
    wing3_monomer_dG: float     # 3' wing intramolecular folding
    wing3_homodimer_dG: float   # 3' wing homodimer
    wing5_wing3_dG: float       # Heterodimer: one ASO's 5' wing + another's 3' wing

    # MFE structure info (for inspection)
    homodimer_mfe_structure: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return asdict(self)


def create_model(celsius: float = 37, sodium: float = 0.15,
                 magnesium: float = 0.001, ensemble: str = 'stacking') -> Model:
    """
    Create NUPACK model with physiological conditions.

    Args:
        celsius: Temperature in Celsius (default 37, body temperature)
        sodium: Sodium concentration in M (default 0.15 = 150 mM, physiological)
        magnesium: Magnesium concentration in M (default 0.001 = 1 mM)
        ensemble: Stacking ensemble type

    Returns:
        NUPACK Model object
    """
    return Model(
        material='dna',
        celsius=celsius,
        sodium=sodium,
        magnesium=magnesium,
        ensemble=ensemble
    )


def compute_monomer_dG(seq: str, model: Model) -> float:
    """
    Compute free energy of intramolecular folding (single strand).

    Args:
        seq: DNA sequence (5'->3')
        model: NUPACK Model object

    Returns:
        Free energy in kcal/mol (more negative = more stable structure)
    """
    if not seq or len(seq) < 2:
        return 0.0

    strand = Strand(seq, name='aso')
    tube = Tube(
        strands={strand: 1e-6},  # 1 uM, arbitrary for single-strand
        complexes=SetSpec(max_size=1),
        name='monomer_tube'
    )
    result = tube_analysis(tubes=[tube], model=model)

    # Get free energy from result - access via result[complex].free_energy
    for cplx in result.complexes:
        if len(cplx.strands) == 1:
            return float(result[cplx].free_energy)

    return 0.0


def compute_homodimer_dG(seq: str, model: Model) -> tuple[float, Optional[str]]:
    """
    Compute free energy of homodimer formation (two identical strands).

    Args:
        seq: DNA sequence (5'->3')
        model: NUPACK Model object

    Returns:
        Tuple of (free_energy in kcal/mol, MFE structure string)
    """
    if not seq or len(seq) < 2:
        return 0.0, None

    strand = Strand(seq, name='aso')
    tube = Tube(
        strands={strand: 1e-6},  # 1 uM per strand
        complexes=SetSpec(max_size=2),  # Allow monomer and dimer
        name='dimer_tube'
    )
    result = tube_analysis(tubes=[tube], model=model)

    # Get homodimer free energy - access via result[complex].free_energy
    dimer_dG = 0.0
    mfe_structure = None
    for cplx in result.complexes:
        # Homodimer has 2 identical strands
        if len(cplx.strands) == 2:
            cplx_result = result[cplx]
            dimer_dG = float(cplx_result.free_energy)
            # Get MFE structure
            try:
                mfe_structure = str(cplx_result.mfe[0].structure)
            except Exception:
                pass
            break

    return dimer_dG, mfe_structure


def compute_heterodimer_dG(seq1: str, seq2: str, model: Model) -> float:
    """
    Compute free energy of heterodimer formation (two different sequences).

    Used for wing5-wing3 interaction between different ASO molecules.

    Args:
        seq1: First DNA sequence
        seq2: Second DNA sequence
        model: NUPACK Model object

    Returns:
        Free energy in kcal/mol
    """
    if not seq1 or not seq2 or len(seq1) < 2 or len(seq2) < 2:
        return 0.0

    strand1 = Strand(seq1, name='wing5')
    strand2 = Strand(seq2, name='wing3')
    tube = Tube(
        strands={strand1: 1e-6, strand2: 1e-6},
        complexes=SetSpec(max_size=2),
        name='heterodimer_tube'
    )
    result = tube_analysis(tubes=[tube], model=model)

    # Get heterodimer free energy
    for cplx in result.complexes:
        # Heterodimer has 2 different strands
        if len(cplx.strands) == 2:
            strand_names = {s.name for s in cplx.strands}
            if 'wing5' in strand_names and 'wing3' in strand_names:
                return float(result[cplx].free_energy)

    return 0.0


def compute_all_metrics(full_seq: str, wing5: str, wing3: str,
                        model: Model) -> NupackMetrics:
    """
    Compute all NUPACK thermodynamic metrics for an ASO.

    Args:
        full_seq: Full ASO sequence
        wing5: 5' wing sequence
        wing3: 3' wing sequence
        model: NUPACK Model object

    Returns:
        NupackMetrics dataclass with all computed values
    """
    # Full ASO metrics
    monomer_dG = compute_monomer_dG(full_seq, model)
    homodimer_dG, mfe_structure = compute_homodimer_dG(full_seq, model)
    ddG = homodimer_dG - 2 * monomer_dG

    # 5' wing metrics (NaN if no wing)
    w5_mono = compute_monomer_dG(wing5, model) if wing5 else np.nan
    w5_homo, _ = compute_homodimer_dG(wing5, model) if wing5 else (np.nan, None)

    # 3' wing metrics (NaN if no wing)
    w3_mono = compute_monomer_dG(wing3, model) if wing3 else np.nan
    w3_homo, _ = compute_homodimer_dG(wing3, model) if wing3 else (np.nan, None)

    # Wing5-Wing3 heterodimer (NaN if either wing missing)
    w5w3_dG = compute_heterodimer_dG(wing5, wing3, model) if (wing5 and wing3) else np.nan

    return NupackMetrics(
        monomer_dG=monomer_dG,
        homodimer_dG=homodimer_dG,
        ddG_dimerization=ddG,
        wing5_monomer_dG=w5_mono,
        wing5_homodimer_dG=w5_homo,
        wing3_monomer_dG=w3_mono,
        wing3_homodimer_dG=w3_homo,
        wing5_wing3_dG=w5w3_dG,
        homodimer_mfe_structure=mfe_structure
    )


def test_nupack_sanity():
    """
    Run sanity checks to verify NUPACK is working correctly.

    Returns:
        True if all tests pass, raises AssertionError otherwise
    """
    model = create_model()

    # Test 1: Self-complementary sequence should have very negative dimer dG
    self_comp = "GCGCGCGC"
    dG_self, _ = compute_homodimer_dG(self_comp, model)
    print(f"Self-complementary (GCGCGCGC) homodimer dG: {dG_self:.2f} kcal/mol")
    assert dG_self < -5, f"Expected stable dimer for self-comp sequence, got {dG_self}"

    # Test 2: Poly-A should have less negative dG (weak/no structure)
    poly_a = "AAAAAAAA"
    dG_poly_a, _ = compute_homodimer_dG(poly_a, model)
    print(f"Poly-A (AAAAAAAA) homodimer dG: {dG_poly_a:.2f} kcal/mol")
    # Poly-A can still form some structure, but less than self-comp
    assert dG_poly_a > dG_self, f"Expected poly-A less stable than self-comp"

    # Test 3: Check ddG calculation
    mono_dG = compute_monomer_dG(self_comp, model)
    homo_dG, _ = compute_homodimer_dG(self_comp, model)
    ddG = homo_dG - 2 * mono_dG
    print(f"Self-comp monomer dG: {mono_dG:.2f}, homodimer dG: {homo_dG:.2f}, ddG: {ddG:.2f}")

    # Test 4: Typical ASO-like sequence
    aso_like = "TGCTGATTAGTGTCGAT"  # Random 17-mer
    metrics = compute_all_metrics(aso_like, aso_like[:5], aso_like[-5:], model)
    print(f"ASO-like sequence metrics: homodimer_dG={metrics.homodimer_dG:.2f}, ddG={metrics.ddG_dimerization:.2f}")

    print("\nAll sanity checks passed!")
    return True


if __name__ == "__main__":
    test_nupack_sanity()
