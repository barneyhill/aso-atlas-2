"""Parse HELM format oligonucleotide strings.

HELM format example: `RNA1{[moe](C)[sp].[moe](T)[sp].d(A)[sp].d(G)}$$$$`

Structure:
- Starts with `RNA1{` or `DNA1{`
- Each nucleotide: `[sugar](base)[backbone].`
- Sugar is optional, defaults to ribose for RNA, deoxy for DNA
- Backbone linkage follows base in brackets
- Nucleotides separated by `.`
- Ends with `}$$$$`
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Nucleotide:
    """Represents a single nucleotide parsed from HELM format."""
    position: int
    sugar: str  # e.g., "[moe]", "d", "" for default
    base: str   # e.g., "A", "G", "C", "T", "U", "5meC"
    backbone: str  # e.g., "[sp]", ".", "" for terminal
    is_5prime: bool
    is_3prime: bool


def parse_helm(helm_string: str) -> list[Nucleotide]:
    """Parse HELM string into list of Nucleotide objects.

    Args:
        helm_string: HELM format string, e.g.,
            "RNA1{[moe](C)[sp].[moe](T)[sp].d(A)[sp].d(G)}$$$$"

    Returns:
        List of Nucleotide objects representing the parsed sequence.

    Raises:
        ValueError: If the HELM string format is invalid.
    """
    # Validate and extract the polymer type and content
    match = re.match(r'^(RNA|DNA)1\{(.+)\}\$\$\$\$$', helm_string.strip())
    if not match:
        raise ValueError(
            f"Invalid HELM format. Expected 'RNA1{{...}}$$$$' or 'DNA1{{...}}$$$$', "
            f"got: {helm_string[:50]}..."
        )

    polymer_type = match.group(1)  # "RNA" or "DNA"
    content = match.group(2)

    # Split by '.' to get individual nucleotides
    # Need to handle dots inside brackets carefully
    nucleotide_strings = _split_nucleotides(content)

    if not nucleotide_strings:
        raise ValueError("No nucleotides found in HELM string")

    nucleotides = []
    total_count = len(nucleotide_strings)

    for i, nuc_str in enumerate(nucleotide_strings):
        position = i + 1  # 1-indexed
        is_5prime = (i == 0)
        is_3prime = (i == total_count - 1)

        sugar, base, backbone = _parse_nucleotide(nuc_str, polymer_type, is_3prime)

        nucleotides.append(Nucleotide(
            position=position,
            sugar=sugar,
            base=base,
            backbone=backbone,
            is_5prime=is_5prime,
            is_3prime=is_3prime
        ))

    return nucleotides


def _split_nucleotides(content: str) -> list[str]:
    """Split HELM content by '.' separator, respecting brackets.

    Args:
        content: The content inside RNA1{...} or DNA1{...}

    Returns:
        List of nucleotide strings.
    """
    nucleotides = []
    current = []
    bracket_depth = 0

    for char in content:
        if char == '[':
            bracket_depth += 1
            current.append(char)
        elif char == ']':
            bracket_depth -= 1
            current.append(char)
        elif char == '.' and bracket_depth == 0:
            if current:
                nucleotides.append(''.join(current))
                current = []
        else:
            current.append(char)

    # Don't forget the last nucleotide (no trailing '.')
    if current:
        nucleotides.append(''.join(current))

    return nucleotides


def _parse_nucleotide(nuc_str: str, polymer_type: str, is_3prime: bool) -> tuple[str, str, str]:
    """Parse a single nucleotide string into components.

    Args:
        nuc_str: Single nucleotide string, e.g., "[moe](C)[sp]" or "d(A)" or "(G)"
        polymer_type: "RNA" or "DNA" for default sugar assignment
        is_3prime: Whether this is the 3' terminal nucleotide

    Returns:
        Tuple of (sugar, base, backbone)
    """
    # Pattern to match: optional_sugar(base)optional_backbone
    # Sugar can be: [xxx] or single letter like 'd' or empty
    # Base is always in parentheses: (X) or (5meC) etc.
    # Backbone can be: [xxx] or empty (terminal)

    # Regex pattern explanation:
    # ^(\[[^\]]+\]|[a-zA-Z])?  - Optional sugar: either [something] or a single letter
    # \(([^)]+)\)              - Base in parentheses (required)
    # (\[[^\]]+\])?$           - Optional backbone: [something]

    pattern = r'^(\[[^\]]+\]|[a-zA-Z])?'  # Optional sugar
    pattern += r'\(([^)]+)\)'              # Base (required)
    pattern += r'(\[[^\]]+\])?$'           # Optional backbone

    match = re.match(pattern, nuc_str)
    if not match:
        raise ValueError(f"Invalid nucleotide format: {nuc_str}")

    sugar_raw = match.group(1) or ""
    base = match.group(2)
    backbone_raw = match.group(3) or ""

    # Normalize base (handle 5meC/5MeC variations)
    base = _normalize_base(base)

    # Determine sugar
    if sugar_raw:
        sugar = sugar_raw
    else:
        # Default sugar based on polymer type
        sugar = "" if polymer_type == "RNA" else "d"

    # Determine backbone
    if is_3prime:
        # Terminal nucleotide has no backbone linkage
        backbone = ""
    else:
        backbone = backbone_raw

    return sugar, base, backbone


def _normalize_base(base: str) -> str:
    """Normalize base modification names.

    Args:
        base: Base string from HELM, e.g., "A", "5meC", "5MeC"

    Returns:
        Normalized base string.
    """
    # Normalize 5-methylcytosine variations
    if base.lower() in ('5mec', '5-mec', '5-methylc'):
        return '5meC'
    return base


def get_sequence(nucleotides: list[Nucleotide]) -> str:
    """Extract simple sequence string from nucleotides.

    Args:
        nucleotides: List of Nucleotide objects

    Returns:
        Simple sequence string (e.g., "CTAG")
    """
    base_map = {
        '5meC': 'C',  # Methylated cytosine still pairs as C
    }
    return ''.join(base_map.get(n.base, n.base[0] if len(n.base) > 1 else n.base)
                   for n in nucleotides)


def helm_to_dict(helm_string: str) -> dict:
    """Parse HELM string and return as dictionary for JSON serialization.

    Args:
        helm_string: HELM format string

    Returns:
        Dictionary with 'sequence' and 'nucleotides' keys
    """
    nucleotides = parse_helm(helm_string)
    return {
        'sequence': get_sequence(nucleotides),
        'nucleotides': [
            {
                'position': n.position,
                'sugar': n.sugar,
                'base': n.base,
                'backbone': n.backbone,
                'is_5prime': n.is_5prime,
                'is_3prime': n.is_3prime
            }
            for n in nucleotides
        ]
    }
