"""
HELM parsing utilities for ASO gapmer analysis.

Single entry-point: the ``Helm`` dataclass.  Use ``Helm.parse(helm_string)``
to obtain a frozen, validated object with properties for wing structure,
5-10-5 MOE classification, PS count, and DNA-equivalent sequence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------

_SUGAR_MAP: dict[str, str] = {
    "moe": "MOE",
    "lna": "LNA",
    "cet": "cEt",
    "fr": "fR",       # NB: parser lowercases tokens, so key must be 'fr'
    "m": "OMe",
    "d": "DNA",
}

_BASE_NORM: dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "T": "T",
    "5meC": "C",
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_nucleotide(nuc: str) -> tuple[str | None, str | None, str | None]:
    """Parse a single nucleotide token.

    Patterns:
    - [moe](G)[sp] -> sugar=MOE, base=G
    - d([5meC])[sp] -> sugar=DNA, base=C, mod=5meC
    - (A)[sp] -> sugar=RNA, base=A
    - [lna](T) -> sugar=LNA, base=T

    Returns:
        (sugar_type, base, base_modification) or (None, None, None) if unparseable
    """
    nuc_clean = nuc.replace("[sp]", "").strip()

    sugar = "RNA"
    base = None
    base_mod = None

    # Pattern: [sugar_type](base) or [sugar_type]([base_mod])
    sugar_match = re.match(r"\[(\w+)\]\(([^)]+)\)", nuc_clean)

    if sugar_match:
        sugar_token = sugar_match.group(1).lower()
        base_content = sugar_match.group(2)

        sugar = _SUGAR_MAP.get(sugar_token, sugar_token.upper())

        if base_content.startswith("[") and base_content.endswith("]"):
            base_mod = base_content[1:-1]
            base = _BASE_NORM.get(base_mod, base_mod[0] if base_mod else None)
        else:
            base = base_content.upper()

    # Pattern: d(base) or d([base_mod]) — DNA
    elif nuc_clean.startswith("d("):
        sugar = "DNA"
        base_content = nuc_clean[2:-1] if nuc_clean.endswith(")") else nuc_clean[2:]

        if base_content.startswith("[") and base_content.endswith("]"):
            base_mod = base_content[1:-1]
            base = _BASE_NORM.get(base_mod, base_mod[0] if base_mod else None)
        else:
            base = base_content.upper()

    # Pattern: (base) — unmodified RNA
    elif nuc_clean.startswith("(") and nuc_clean.endswith(")"):
        sugar = "RNA"
        base_content = nuc_clean[1:-1]

        if base_content.startswith("[") and base_content.endswith("]"):
            base_mod = base_content[1:-1]
            base = _BASE_NORM.get(base_mod, base_mod[0] if base_mod else None)
        else:
            base = base_content.upper()

    # Normalize U/T based on sugar type
    if base == "T" and sugar == "RNA":
        base = "U"
    elif base == "U" and sugar == "DNA":
        base = "T"

    return sugar, base, base_mod


# ---------------------------------------------------------------------------
# Public Helm class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Helm:
    """Parsed HELM annotation with full structural information."""

    sequence: str
    length: int
    sugars: tuple[str, ...]
    bases: tuple[str, ...]
    base_mods: tuple[str | None, ...]
    backbones: tuple[str, ...]

    def __post_init__(self) -> None:
        assert len(self.sugars) == self.length
        assert len(self.bases) == self.length
        assert len(self.base_mods) == self.length
        assert len(self.backbones) == self.length - 1

    # ---- constructors -----------------------------------------------------

    @classmethod
    def parse(cls, helm: str) -> Helm | None:
        """Parse a HELM annotation string.

        Returns a ``Helm`` instance, or ``None`` if the string is not
        parseable (including ``None``, empty string, ``NaN``).
        """
        if not helm or not isinstance(helm, str):
            return None

        match = re.search(r"\{\{(.+?)\}\}", helm)
        if not match:
            return None

        seq_part = match.group(1)
        nucleotides = seq_part.split(".")

        sugars: list[str] = []
        bases: list[str] = []
        base_mods: list[str | None] = []
        linkage_types: list[str] = []

        for nuc in nucleotides:
            nuc = nuc.strip()
            if not nuc:
                continue

            has_ps = "[sp]" in nuc
            sugar, base, base_mod = _parse_nucleotide(nuc)

            if sugar is None or base is None:
                continue

            sugars.append(sugar)
            bases.append(base)
            base_mods.append(base_mod)
            linkage_types.append("PS" if has_ps else "PO")

        if not sugars:
            return None

        backbones = linkage_types[: len(sugars) - 1] if len(sugars) > 1 else []
        sequence = "".join(bases)

        return cls(
            sequence=sequence,
            length=len(sugars),
            sugars=tuple(sugars),
            bases=tuple(bases),
            base_mods=tuple(base_mods),
            backbones=tuple(backbones),
        )

    # ---- properties -------------------------------------------------------

    @property
    def ps_count(self) -> int:
        """Count of phosphorothioate linkages."""
        return sum(1 for b in self.backbones if b == "PS")

    @property
    def wings(self) -> tuple[int, int, int]:
        """Gapmer wing structure as (5' wing, DNA gap, 3' wing).

        Wings are counted as contiguous MOE nucleotides from each end;
        the gap is the number of DNA nucleotides in between.
        """
        sugar_types: list[str] = []
        for s in self.sugars:
            if s == "MOE":
                sugar_types.append("MOE")
            elif s == "DNA":
                sugar_types.append("DNA")
            else:
                sugar_types.append("OTHER")

        five_prime = 0
        for s in sugar_types:
            if s == "MOE":
                five_prime += 1
            else:
                break

        three_prime = 0
        for s in reversed(sugar_types):
            if s == "MOE":
                three_prime += 1
            else:
                break

        if five_prime + three_prime >= len(sugar_types):
            dna_gap = 0
        else:
            end = len(sugar_types) - three_prime if three_prime > 0 else len(sugar_types)
            middle = sugar_types[five_prime:end]
            dna_gap = sum(1 for s in middle if s == "DNA")

        return (five_prime, dna_gap, three_prime)

    @property
    def is_5_10_5_moe(self) -> bool:
        """True if this is a 5-10-5 MOE gapmer."""
        return self.wings == (5, 10, 5)

    @property
    def dna_sequence(self) -> str:
        """DNA-equivalent sequence (U → T)."""
        return self.sequence.replace("U", "T")

    # ---- static helpers ---------------------------------------------------

    @staticmethod
    def valid_chemistry(helm: str) -> bool:
        """Check if HELM contains only DNA/MOE/cEt sugars (quick pre-filter)."""
        if pd.isna(helm) or helm == "None":
            return False
        for ex in ("[lna]", "[LNA]", "[fR]", "[FR]", "[am]", "[AM]", "[?]"):
            if ex in helm:
                return False
        return not re.search(r"\[m\]\(", helm)
