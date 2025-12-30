"""Map parsed HELM nucleotides to modXNA fragment codes.

This module converts HELM-parsed nucleotide sequences to modXNA residue
definitions suitable for molecular dynamics simulations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .helm_parser import Nucleotide


def load_catalog(catalog_path: Path = None) -> dict:
    """Load fragment catalog JSON.

    Args:
        catalog_path: Path to fragment_catalog.json. If None, uses default
            location at ../data/fragment_catalog.json relative to this module.

    Returns:
        Dictionary containing fragment catalog data.

    Raises:
        FileNotFoundError: If the catalog file doesn't exist.
    """
    if catalog_path is None:
        # From src/modxna/ go up to modxna/ then into data/
        catalog_path = Path(__file__).parent.parent.parent / "data" / "fragment_catalog.json"
    with open(catalog_path) as f:
        return json.load(f)


@dataclass
class ModXNAResidue:
    """Represents a modXNA residue for molecular dynamics."""
    resname: str      # 3-letter code (e.g., "5MG", "CEG")
    backbone: str     # modXNA backbone code
    sugar: str        # modXNA sugar code
    base: str         # modXNA base code
    is_5prime: bool
    is_3prime: bool

    def __hash__(self):
        return hash((self.backbone, self.sugar, self.base, self.is_5prime, self.is_3prime))

    def __eq__(self, other):
        if not isinstance(other, ModXNAResidue):
            return False
        return (self.backbone == other.backbone and
                self.sugar == other.sugar and
                self.base == other.base and
                self.is_5prime == other.is_5prime and
                self.is_3prime == other.is_3prime)


# Default mappings when no catalog is available
DEFAULT_SUGAR_MAP = {
    "": "RIB",           # Default ribose for RNA
    "d": "DEO",          # Deoxyribose
    "[moe]": "MOE",      # 2'-O-methoxyethyl
    "[MOE]": "MOE",
    "[cet]": "CET",      # Constrained ethyl (cEt)
    "[CET]": "CET",
    "[lna]": "LNA",      # Locked nucleic acid
    "[LNA]": "LNA",
    "[fana]": "FAN",     # 2'-fluoro-arabino
    "[FANA]": "FAN",
    "[2f]": "2FL",       # 2'-fluoro
    "[2F]": "2FL",
    "[fR]": "2FL",       # 2'-fluoro (alternate notation)
    "[m]": "OME",        # 2'-O-methyl
    "[M]": "OME",
}

DEFAULT_BACKBONE_MAP = {
    "": "PO4",           # Standard phosphate (terminal)
    "[sp]": "PS",        # Phosphorothioate
    "[SP]": "PS",
    "[po]": "PO4",       # Phosphodiester
    "[PO]": "PO4",
}

DEFAULT_BASE_MAP = {
    # RNA bases
    "A": "ADE",
    "G": "GUA",
    "C": "CYT",
    "U": "URA",
    # DNA bases
    "T": "THY",
    # Modified bases
    "5meC": "5MC",       # 5-methylcytosine
    "5MeC": "5MC",
}


def map_to_modxna(nucleotides: list[Nucleotide], catalog: dict = None) -> list[ModXNAResidue]:
    """Convert parsed HELM to modXNA residues.

    Args:
        nucleotides: List of Nucleotide objects from HELM parsing
        catalog: Optional fragment catalog dictionary. If None, uses defaults.

    Returns:
        List of ModXNAResidue objects representing the modXNA sequence.
    """
    if catalog is None:
        catalog = {}

    # Extract mappings from catalog or use defaults
    sugar_map = catalog.get('sugar', DEFAULT_SUGAR_MAP)
    backbone_map = catalog.get('backbone', DEFAULT_BACKBONE_MAP)
    base_map = catalog.get('base', DEFAULT_BASE_MAP)
    terminal_map = catalog.get('terminal_backbone', {})

    residues = []
    for nuc in nucleotides:
        # Determine RNA vs DNA context from sugar
        is_dna = nuc.sugar == 'd' or nuc.sugar.lower() in ['d', '[d]', 'dna']

        # Map sugar
        sugar = _lookup_sugar(nuc.sugar, sugar_map)

        # Map backbone (5'-terminal gets special 5PO code)
        if nuc.is_5prime:
            backbone = terminal_map.get('5prime', '5PO')
        elif nuc.is_3prime:
            # 3' terminal uses regular backbone with --3cap flag in modXNA
            backbone = _lookup_backbone(nuc.backbone, backbone_map) if nuc.backbone else "RPO"
        else:
            backbone = _lookup_backbone(nuc.backbone, backbone_map)

        # Map base (with RNA/DNA context)
        base = _lookup_base(nuc.base, base_map, is_dna=is_dna)

        # Generate 3-letter residue name
        resname = generate_resname(backbone, sugar, base, nuc.is_5prime, nuc.is_3prime)

        residues.append(ModXNAResidue(
            resname=resname,
            backbone=backbone,
            sugar=sugar,
            base=base,
            is_5prime=nuc.is_5prime,
            is_3prime=nuc.is_3prime
        ))

    return residues


def _lookup_sugar(helm_sugar: str, sugar_map: dict) -> str:
    """Look up modXNA sugar code from HELM sugar notation."""
    if helm_sugar in sugar_map:
        return sugar_map[helm_sugar]
    # Try case-insensitive lookup
    for key, value in sugar_map.items():
        if key.lower() == helm_sugar.lower():
            return value
    # Return as-is if not found (might already be modXNA code)
    return helm_sugar.strip('[]').upper()[:3] if helm_sugar else "RIB"


def _lookup_backbone(helm_backbone: str, backbone_map: dict) -> str:
    """Look up modXNA backbone code from HELM backbone notation."""
    if helm_backbone in backbone_map:
        return backbone_map[helm_backbone]
    # Try case-insensitive lookup
    for key, value in backbone_map.items():
        if key.lower() == helm_backbone.lower():
            return value
    # Return as-is if not found
    return helm_backbone.strip('[]').upper()[:3] if helm_backbone else "PO4"


def _lookup_base(helm_base: str, base_map: dict, is_dna: bool = False) -> str:
    """Look up modXNA base code from HELM base notation.

    Args:
        helm_base: HELM base notation (A, G, C, T, U, 5meC, etc.)
        base_map: Dictionary mapping HELM bases to modXNA codes
        is_dna: Whether the context is DNA (affects A, G, C lookup)

    Returns:
        modXNA 3-letter base code
    """
    if helm_base in base_map:
        value = base_map[helm_base]
        # Handle dict with rna/dna keys
        if isinstance(value, dict):
            context = 'dna' if is_dna else 'rna'
            return value.get(context, value.get('rna', value.get('dna', helm_base)))
        return value
    # Handle case variations
    for key, val in base_map.items():
        if key.lower() == helm_base.lower():
            if isinstance(val, dict):
                context = 'dna' if is_dna else 'rna'
                return val.get(context, val.get('rna', val.get('dna', helm_base)))
            return val
    # Return truncated version if not found
    return helm_base.upper()[:3]


def generate_resname(backbone: str, sugar: str, base: str,
                     is_5prime: bool, is_3prime: bool) -> str:
    """Generate unique 3-letter residue name.

    The naming convention follows modXNA patterns:
    - 5' terminal: starts with '5' (e.g., "5CG", "5MG")
    - 3' terminal: starts with '3' (e.g., "3EC", "3MG")
    - Central: uses modification identifier (e.g., "CEG", "MOG", "MEC")

    Args:
        backbone: modXNA backbone code
        sugar: modXNA sugar code
        base: modXNA base code (e.g., RAA, RGG, DAA, M5C)
        is_5prime: Whether this is a 5' terminal residue
        is_3prime: Whether this is a 3' terminal residue

    Returns:
        3-letter residue name string

    Examples:
        - 5' MOE guanine: "5MG"
        - Central MOE cytosine: "MEC"
        - 3' MOE cytosine: "3EC"
        - Central PS backbone: "CEG" (central guanine)
    """
    # Get base character from modXNA code
    # Codes like RAA->A, RGG->G, DAA->A, DGG->G, M5C->C, DTT->T, RUU->U
    base_char = _get_base_char(base)

    # Get sugar/modification character
    sugar_char = _get_modification_char(sugar)

    if is_5prime:
        # 5' terminal: "5" + sugar_char + base_char
        return f"5{sugar_char}{base_char}"
    elif is_3prime:
        # 3' terminal: "3" + sugar_char + base_char
        return f"3{sugar_char}{base_char}"
    else:
        # Central: sugar_char + base_char + "C" (for central) or similar
        # Or: modification code + base
        if sugar in ("MOE", "MO"):
            return f"MO{base_char}"
        elif sugar in ("CET", "CE"):
            return f"CE{base_char}"
        elif sugar in ("OME", "OM"):
            return f"OM{base_char}"
        elif sugar in ("LNA", "LN"):
            return f"LN{base_char}"
        elif sugar in ("2FL", "2F"):
            return f"2F{base_char}"
        elif sugar == "DEO":
            return f"D{base_char}C"  # Deoxy central
        else:
            return f"C{sugar_char}{base_char}"  # Central + sugar + base


def _get_modification_char(sugar: str) -> str:
    """Get single character representing sugar modification.

    Args:
        sugar: modXNA sugar code

    Returns:
        Single character for residue name
    """
    mod_chars = {
        "RIB": "R",
        "RC3": "R",
        "DEO": "D",
        "DC2": "D",
        "MOE": "M",
        "CET": "C",
        "OME": "O",
        "LNA": "L",
        "2FL": "F",
        "AF2": "F",
        "FAN": "F",
    }
    return mod_chars.get(sugar, sugar[0] if sugar else "R")


def _get_base_char(base: str) -> str:
    """Get single character representing the nucleobase.

    Args:
        base: modXNA base code (e.g., RAA, RGG, DAA, M5C)

    Returns:
        Single character for the base (A, G, C, T, U)
    """
    # Map modXNA base codes to single characters
    base_chars = {
        "RAA": "A", "DAA": "A",  # Adenine
        "RGG": "G", "DGG": "G",  # Guanine
        "RCC": "C", "DCC": "C",  # Cytosine
        "RUU": "U",              # Uracil (RNA)
        "DTT": "T",              # Thymine (DNA)
        "M5C": "C",              # 5-methylcytosine
    }
    if base in base_chars:
        return base_chars[base]
    # Fallback: try to extract from code pattern (e.g., XYZ -> Y for single letter bases)
    if len(base) >= 2:
        return base[1] if base[1] in 'AGCTU' else base[0]
    return base[0] if base else 'X'


def get_unique_residues(residues: list[ModXNAResidue]) -> dict[str, ModXNAResidue]:
    """Group residues by unique modification combination.

    This identifies all unique residue types needed for force field
    parameterization.

    Args:
        residues: List of ModXNAResidue objects

    Returns:
        Dictionary keyed by resname with ModXNAResidue as values.
        Only unique combinations of (backbone, sugar, base, position) are kept.
    """
    unique = {}
    for res in residues:
        key = (res.backbone, res.sugar, res.base, res.is_5prime, res.is_3prime)
        if res.resname not in unique:
            unique[res.resname] = res
    return unique


def get_residue_summary(residues: list[ModXNAResidue]) -> dict:
    """Generate summary of residue types for parameterization.

    Args:
        residues: List of ModXNAResidue objects

    Returns:
        Dictionary with summary information for each unique residue type.
    """
    unique = get_unique_residues(residues)
    summary = {}

    for resname, res in unique.items():
        position = "5-terminal" if res.is_5prime else "3-terminal" if res.is_3prime else "central"
        summary[resname] = {
            'backbone': res.backbone,
            'sugar': res.sugar,
            'base': res.base,
            'position': position,
            'count': sum(1 for r in residues if r.resname == resname)
        }

    return summary


def residues_to_dict(residues: list[ModXNAResidue]) -> list[dict]:
    """Convert residues to list of dictionaries for JSON serialization.

    Args:
        residues: List of ModXNAResidue objects

    Returns:
        List of dictionaries representing residues.
    """
    return [
        {
            'resname': r.resname,
            'backbone': r.backbone,
            'sugar': r.sugar,
            'base': r.base,
            'is_5prime': r.is_5prime,
            'is_3prime': r.is_3prime
        }
        for r in residues
    ]
