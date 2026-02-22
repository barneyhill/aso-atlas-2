"""
HELM parsing utilities for ASO gapmer analysis.

Functions for parsing HELM notation to extract wing structure information.
"""

import re


def parse_helm_wings(helm: str) -> tuple[int, int, int] | None:
    """
    Parse HELM annotation to extract gapmer wing structure.

    Returns (5' wing MOE count, DNA gap count, 3' wing MOE count) or None if not parseable.

    5-10-5 MOE gapmer structure:
    - 5 MOE nucleotides at 5' end
    - 10 DNA (deoxy) nucleotides in middle
    - 5 MOE nucleotides at 3' end
    """
    if not helm or not isinstance(helm, str):
        return None

    # Extract the sequence part between {{ and }}
    match = re.search(r'\{\{(.+?)\}\}', helm)
    if not match:
        return None

    seq = match.group(1)

    # Split by '.' to get individual nucleotides
    # Each nucleotide looks like: [moe](A)[sp] or d(C)[sp]
    nucleotides = seq.split('.')

    # Classify each nucleotide as MOE or DNA
    sugar_types = []
    for nuc in nucleotides:
        nuc = nuc.strip()
        if not nuc:
            continue
        if nuc.startswith('[moe]'):
            sugar_types.append('MOE')
        elif nuc.startswith('d('):
            sugar_types.append('DNA')
        elif nuc.startswith('[cet]') or nuc.startswith('[lna]') or nuc.startswith('[fR]') or nuc.startswith('[m]'):
            # Other modifications - not a standard MOE gapmer
            sugar_types.append('OTHER')
        else:
            sugar_types.append('UNKNOWN')

    if not sugar_types:
        return None

    # Count 5' wing (MOE from start)
    five_prime_moe = 0
    for s in sugar_types:
        if s == 'MOE':
            five_prime_moe += 1
        else:
            break

    # Count 3' wing (MOE from end)
    three_prime_moe = 0
    for s in reversed(sugar_types):
        if s == 'MOE':
            three_prime_moe += 1
        else:
            break

    # Count DNA gap in middle
    if five_prime_moe + three_prime_moe >= len(sugar_types):
        # No gap or all MOE
        dna_gap = 0
    else:
        middle = sugar_types[five_prime_moe:len(sugar_types) - three_prime_moe if three_prime_moe > 0 else len(sugar_types)]
        dna_gap = sum(1 for s in middle if s == 'DNA')

    return (five_prime_moe, dna_gap, three_prime_moe)


def is_5_10_5_moe(helm: str) -> bool:
    """Check if HELM annotation represents a 5-10-5 MOE gapmer."""
    result = parse_helm_wings(helm)
    if result is None:
        return False
    five_prime, gap, three_prime = result
    return five_prime == 5 and gap == 10 and three_prime == 5
