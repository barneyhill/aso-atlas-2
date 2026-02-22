"""Tests for the Helm class in analyses.utils.helm."""

import math

import pytest

from analyses.utils.helm import Helm


# ---------------------------------------------------------------------------
# Sample HELM strings
# ---------------------------------------------------------------------------

# 5-10-5 MOE gapmer, all-PS, 20-mer
MOE_5_10_5 = (
    "RNA1{{"
    "[moe](G)[sp].[moe](A)[sp].[moe](C)[sp].[moe](C)[sp].[moe](T)[sp]."
    "d(G)[sp].d(T)[sp].d(G)[sp].d(A)[sp].d(A)[sp].d(G)[sp].d(T)[sp].d(T)[sp].d(A)[sp].d(C)[sp]."
    "[moe](C)[sp].[moe](A)[sp].[moe](T)[sp].[moe](G)[sp].[moe](A)"
    "}}$$$$"
)

# 5-10-5 MOE gapmer with 5meC modifications
MOE_5_10_5_5MEC = (
    "RNA1{{"
    "[moe]([5meC])[sp].[moe](A)[sp].[moe](G)[sp].[moe](T)[sp].[moe]([5meC])[sp]."
    "d(A)[sp].d(G)[sp].d(T)[sp].d([5meC])[sp].d(A)[sp].d(G)[sp].d(T)[sp].d(C)[sp].d(A)[sp].d(G)[sp]."
    "[moe](T)[sp].[moe]([5meC])[sp].[moe](A)[sp].[moe](G)[sp].[moe](T)"
    "}}$$$$"
)

# cEt gapmer: 3-10-3
CET_3_10_3 = (
    "RNA1{{"
    "[cet](G)[sp].[cet](A)[sp].[cet](C)[sp]."
    "d(T)[sp].d(G)[sp].d(A)[sp].d(C)[sp].d(T)[sp].d(G)[sp].d(A)[sp].d(C)[sp].d(T)[sp].d(G)[sp]."
    "[cet](A)[sp].[cet](C)[sp].[cet](T)"
    "}}$$$$"
)

# Mixed PS/PO backbone (last linkage is PO)
MIXED_BACKBONE = (
    "RNA1{{"
    "[moe](A)[sp].[moe](G)[sp].d(C)[sp].d(T).d(A)[sp].[moe](G)[sp].[moe](C)"
    "}}$$$$"
)


# ---------------------------------------------------------------------------
# Parsing basics
# ---------------------------------------------------------------------------

class TestParse:
    def test_5_10_5_moe(self):
        h = Helm.parse(MOE_5_10_5)
        assert h is not None
        assert h.length == 20
        assert h.sequence == "GACCTGTGAAGTTACCATGA"

    def test_sugars_correct(self):
        h = Helm.parse(MOE_5_10_5)
        assert h.sugars[:5] == ("MOE",) * 5
        assert h.sugars[5:15] == ("DNA",) * 10
        assert h.sugars[15:] == ("MOE",) * 5

    def test_backbones_all_ps(self):
        h = Helm.parse(MOE_5_10_5)
        assert all(b == "PS" for b in h.backbones)
        assert len(h.backbones) == 19

    def test_5mec_handled(self):
        h = Helm.parse(MOE_5_10_5_5MEC)
        assert h is not None
        # 5meC normalises to C in the sequence
        assert h.bases[0] == "C"
        assert h.base_mods[0] == "5meC"
        # Plain bases have None mod
        assert h.base_mods[1] is None

    def test_cet_gapmer(self):
        h = Helm.parse(CET_3_10_3)
        assert h is not None
        assert h.length == 16
        assert h.sugars[:3] == ("cEt",) * 3
        assert h.sugars[3:13] == ("DNA",) * 10
        assert h.sugars[13:] == ("cEt",) * 3

    def test_mixed_backbone(self):
        h = Helm.parse(MIXED_BACKBONE)
        assert h is not None
        # d(T) has no [sp] → PO linkage
        assert h.backbones == ("PS", "PS", "PS", "PO", "PS", "PS")

    def test_frozen(self):
        h = Helm.parse(MOE_5_10_5)
        with pytest.raises(AttributeError):
            h.length = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases → None
# ---------------------------------------------------------------------------

class TestParseEdgeCases:
    def test_none(self):
        assert Helm.parse(None) is None  # type: ignore[arg-type]

    def test_empty_string(self):
        assert Helm.parse("") is None

    def test_nan(self):
        assert Helm.parse(float("nan")) is None  # type: ignore[arg-type]

    def test_garbage(self):
        assert Helm.parse("not a helm string") is None

    def test_no_braces(self):
        assert Helm.parse("RNA1{stuff}$$$$") is None

    def test_integer(self):
        assert Helm.parse(42) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_ps_count_all_ps(self):
        h = Helm.parse(MOE_5_10_5)
        assert h.ps_count == 19

    def test_ps_count_mixed(self):
        h = Helm.parse(MIXED_BACKBONE)
        assert h.ps_count == 5

    def test_wings_5_10_5(self):
        h = Helm.parse(MOE_5_10_5)
        assert h.wings == (5, 10, 5)

    def test_wings_cet(self):
        h = Helm.parse(CET_3_10_3)
        # cEt wings count as OTHER → 0 MOE wings
        assert h.wings == (0, 10, 0)

    def test_wings_mixed(self):
        h = Helm.parse(MIXED_BACKBONE)
        assert h.wings == (2, 3, 2)

    def test_is_5_10_5_moe_true(self):
        h = Helm.parse(MOE_5_10_5)
        assert h.is_5_10_5_moe is True

    def test_is_5_10_5_moe_false(self):
        h = Helm.parse(CET_3_10_3)
        assert h.is_5_10_5_moe is False

    def test_dna_sequence(self):
        h = Helm.parse(MOE_5_10_5)
        assert h.dna_sequence == h.sequence  # No U in this one

    def test_dna_sequence_u_to_t(self):
        # RNA sugar: T → U in sequence, then dna_sequence converts back
        rna_helm = "RNA1{{(A)[sp].(U)[sp].(G)}}$$$$"
        h = Helm.parse(rna_helm)
        assert h is not None
        assert h.sequence == "AUG"
        assert h.dna_sequence == "ATG"

    def test_length(self):
        h = Helm.parse(MOE_5_10_5)
        assert h.length == 20

    def test_sequence(self):
        h = Helm.parse(MOE_5_10_5)
        assert len(h.sequence) == h.length


# ---------------------------------------------------------------------------
# valid_chemistry static method
# ---------------------------------------------------------------------------

class TestValidChemistry:
    def test_moe_accepted(self):
        assert Helm.valid_chemistry(MOE_5_10_5) is True

    def test_cet_accepted(self):
        assert Helm.valid_chemistry(CET_3_10_3) is True

    def test_lna_rejected(self):
        helm = "RNA1{{[lna](G)[sp].d(A)}}$$$$"
        assert Helm.valid_chemistry(helm) is False

    def test_lna_upper_rejected(self):
        helm = "RNA1{{[LNA](G)[sp].d(A)}}$$$$"
        assert Helm.valid_chemistry(helm) is False

    def test_fr_rejected(self):
        helm = "RNA1{{[fR](G)[sp].d(A)}}$$$$"
        assert Helm.valid_chemistry(helm) is False

    def test_ome_rejected(self):
        helm = "RNA1{{[m](G)[sp].d(A)}}$$$$"
        assert Helm.valid_chemistry(helm) is False

    def test_uncertain_rejected(self):
        helm = "RNA1{{[?](G)[sp].d(A)}}$$$$"
        assert Helm.valid_chemistry(helm) is False

    def test_none_rejected(self):
        assert Helm.valid_chemistry(None) is False  # type: ignore[arg-type]

    def test_nan_rejected(self):
        assert Helm.valid_chemistry(float("nan")) is False

    def test_string_none_rejected(self):
        assert Helm.valid_chemistry("None") is False
