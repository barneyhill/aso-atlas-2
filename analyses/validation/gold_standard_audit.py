"""Deprecated release-wide diagnostic for ASO Atlas 2.0.

This value-based, all-release matcher is retained only to reproduce the earlier
analysis and must not be presented as primary precision or recall.  It permits
unverified family/source substitutions and does not enforce a unified
one-to-one production source scope.  Primary scoring belongs in
``canonical_source_audit.py`` after a Gold-to-production-canonical table map is
frozen before Atlas comparison.

The submitted paper reported a 100-row spot check. Reviewers (Tw5Q W2/Q2) and the
AC (priority #2) asked for something categorically different: an audit that is
stratified, that separates extraction from normalisation accuracy, and that
estimates *coverage* (recall), which a row-level precision check cannot see.

This script computes that audit against an exhaustively human-verified corpus:
every table of four patents was independently re-extracted and manually verified
table by table (325 tables, ~21.9k rows), giving a ground truth in which BOTH
directions are defined:

    precision  = of the Atlas rows filed under these patents, what fraction is
                 corroborated by the verified re-extraction?
    recall     = of the verified measurements printed in these patents, what
                 fraction is present anywhere in the released Atlas?

Two historical choices dominate these diagnostic numbers:

1.  Recall was evaluated against the WHOLE release, not the same
    (patent, table). Patent families republish tables verbatim: 2,715 of the
    C9ORF72 measurements are filed in Atlas under the continuation
    US20230112920A1. This motivates source-aware validation but does not justify
    unrestricted release-wide matching.
2.  Matching on (compound, value, dose) across the whole release admits chance
    collisions, because a compound can carry dozens of records. We therefore
    measure the chance floor directly by permuting values across compounds and
    re-running the identical matcher, and report it alongside every rate.

Usage:
    uv run python analyses/validation/gold_standard_audit.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "aso-atlas-2-release"
SCRAPE = Path.home() / "dphil" / "tox-patent-scrape"
GOLD_RUNS = SCRAPE / "verify" / "data" / "runs"
XML_DIR = SCRAPE / "data" / "step2"
OUT_JSON = ROOT / "data" / "results" / "gold_standard_audit.json"

PATENTS = {
    "US20160251655A1": "C9ORF72",
    "US20240026353A1": "SCN2A",
    "US20210038631A1": "IRF4",
    "US20190160090A1": "AGT",
}

# gold class -> release parquet(s). in_vitro_inhibition and dose_response are one
# schema; the release splits them on doses-per-compound at export, so a gold
# inhibition row may legitimately land in either.
CLASS_DATASETS = {
    "inhibition": ["in_vitro_inhibition", "dose_response"],
    "hepatotox": ["hepatotoxicity"],
    "neurotox": ["neurotoxicity"],
}

# hepatotoxicity is counted per biomarker readout, which is the unit the paper
# reports (one row contributes up to 7 measurements). gold column -> Atlas column.
BIOMARKERS = {
    "ALT_(IU/L)": "ALT",
    "AST_(IU/L)": "AST",
    "BIL_(mg/dL)": "TBIL",
    "BUN_(mg/dL)": "BUN",
    "ALB_(g/dL)": "ALB",
    "CREA_(mg/dL)": "CREA",
    "urine_protein/creatinine_ratio": "PC_ratio",
}

# Secondary fields, split by what an error in them would mean. An extraction
# field is a number or string printed in the table; a normalisation field is one
# the pipeline derived, mapped, or harmonised, and is where vocabulary drift
# lives. Reporting them together (as the 100-row audit did) conflates "misread
# the page" with "spelled it differently".
FIELD_KIND = {
    "inhibition": {
        "dosage_nm": "extraction",
        "cells_per_well": "extraction",
        "treatment_period_hrs": "extraction",
        "cell_line": "normalisation",
        "cell_line_species": "normalisation",
        "target_RNA": "normalisation",
        "transfection_method": "normalisation",
    },
    "hepatotox": {
        "dosage_mg_per_kg": "extraction",
        "num_doses": "extraction",
        "dosing_period_days": "extraction",
        "species": "normalisation",
        "adminstration_method": "normalisation",
    },
    "neurotox": {
        "dosage_ug": "extraction",
        "latency_time_hours": "extraction",
        "species": "normalisation",
        "administration_method": "normalisation",
        "tolerability_score_type": "normalisation",
    },
}

# gold column -> Atlas column, for the secondary-field comparison
FIELD_MAP = {
    "inhibition": {
        "dosage_nm": "dosage_nm",
        "cells_per_well": "cells_per_well",
        "treatment_period_hrs": "treatment_period_hrs",
        "cell_line": "cell_line",
        "cell_line_species": "cell_line_species",
        "target_RNA": "target_RNA",
        "transfection_method": "transfection_method",
    },
    "hepatotox": {
        "dosage_(mg/kg)": "dosage_mg_per_kg",
        "num_doses": "num_doses",
        "dosing_period_days": "dosing_period_days",
        "species": "species",
        "administration_method": "adminstration_method",
    },
    "neurotox": {
        "dosage_(ug)": "dosage_ug",
        "latency_time_hours": "latency_time_hours",
        "species": "species",
        "administration_method": "administration_method",
        "tolerability_score_type": "tolerability_score_type",
    },
}

# gold primary-measurement column -> (Atlas channel name, gold dose column). The
# channel name is the Atlas one on both sides: gold calls the FOB vector
# "tolerability_score" and the release calls it "FOB_score", and keying the match
# on the gold name silently zeroes neurotoxicity recall.
PRIMARY = {
    "inhibition": ("Inhibition_pct", "Inhibition_pct", "dosage_nm"),
    "hepatotox": (None, None, "dosage_(mg/kg)"),  # exploded over BIOMARKERS
    "neurotox": ("tolerability_score", "FOB_score", "dosage_(ug)"),
}

# Controlled-vocabulary synonyms. These are presentation differences, not data
# errors: the patent writes "Intracerebroventricular", the release writes "ICV".
# We report agreement both before and after applying them, so the reader can see
# exactly how much of the disagreement is vocabulary.
VOCAB = {
    "icv": "icv", "intracerebroventricular": "icv",
    "it": "it", "intrathecal": "it",
    "sc": "sc", "subcutaneous": "sc", "subcutaneously": "sc",
    "iv": "iv", "intravenous": "iv",
    "ip": "ip", "intraperitoneal": "ip",
    "po": "po", "oral": "po", "gavage": "po",
    "electroporation": "electroporation", "electroporated": "electroporation",
    "transfection": "transfection", "cationic lipid": "cationic lipid",
    "lipofectamine": "cationic lipid", "lipofectin": "cationic lipid",
    "free uptake": "gymnosis", "gymnosis": "gymnosis", "gymnotic": "gymnosis",
    "human": "human", "mouse": "mouse", "mice": "mouse", "rat": "rat",
    "rats": "rat", "monkey": "monkey", "rhesus": "monkey",
    "behavioral": "behavioural", "behavioural": "behavioural",
}


# ── value comparison (ported from the verifier so both agree exactly) ────────

def as_vector(v):
    """Parse a per-animal score set into a list of floats, else None.

    Three encodings of the same vector are in play across the two systems: the
    verified CSV text "7, 5, 7", a JSON list [7.0, 5.0, 7.0], and a numpy repr
    "[7. 5. 7.]". Comparing them as strings is what made neurotoxicity
    concordance read 4/252 for data that agrees exactly.
    """
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    s = str(v if v is not None else "").strip()
    bracketed = s.startswith("[") and s.endswith("]")
    if bracketed:
        s = s[1:-1].strip()
    if not s:
        return None
    if not bracketed:
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
            return None  # "1,000" is one number, not two
        if not re.fullmatch(r"[\d.\s,;/+-]+", s) or not re.search(r"[,;/]", s):
            return None
    parts = [p for p in re.split(r"[\s,;/]+", s) if p]
    try:
        out = [float(p) for p in parts]
    except ValueError:
        return None
    return out or None


def as_scalar(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if not s or s.upper() in {"NA", "N/A", "NONE", "NULL", "-"}:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        m = re.search(r"-?\d*\.?\d+", s)
        return float(m.group()) if m else None


def same_value(a, b, tol=1e-6):
    va, vb = as_vector(a), as_vector(b)
    if va is not None and vb is not None:
        return len(va) == len(vb) and all(abs(x - y) < tol for x, y in zip(va, vb))
    if va is not None or vb is not None:
        v = va if va is not None else vb
        other = as_scalar(b if va is not None else a)
        return len(v) == 1 and other is not None and abs(v[0] - other) < tol
    na, nb = as_scalar(a), as_scalar(b)
    if na is not None and nb is not None:
        return abs(na - nb) < tol or abs(na - nb) / max(abs(na), abs(nb), 1.0) < 1e-6
    sa = str(a if a is not None else "").strip().lower()
    sb = str(b if b is not None else "").strip().lower()
    if not sa or sa in {"na", "n/a", "none"}:
        return not sb or sb in {"na", "n/a", "none", "null"}
    if not sb or sb == "null":
        return False
    return sa == sb


def norm_text(v):
    s = str(v if v is not None else "").strip().lower()
    s = re.sub(r"[\s_\-/]+", " ", s).strip()
    return s


def vocab(v):
    s = norm_text(v)
    if s in VOCAB:
        return VOCAB[s]
    # "BALB/c mice" -> mouse, "Sprague Dawley rats" -> rat. Plurals must be
    # listed explicitly: \brat\b does not match "rats".
    for k in ("mice", "mouse", "rats", "rat", "rhesus", "monkey", "human"):
        if re.search(rf"\b{k}\b", s):
            return VOCAB[k]
    return s


def wilson(k, n, z=1.96):
    """Wilson score interval. Normal-approximation CIs are useless at the 100%
    and small-n cells this audit is full of."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def cid_of(v):
    m = re.search(r"\d{4,7}", str(v if v is not None else ""))
    return m.group() if m else None


# ── HELM construction (ported from the verifier) ─────────────────────────────

SUGAR_HELM = {"d": "d", "e": "[moe]", "k": "[cet]", "l": "[lna]"}
SUGAR_TOKEN = {"d": "d", "[d]": "d", "[moe]": "e", "[cet]": "k", "[lna]": "l"}


def clean_seq(v):
    s = str(v if v is not None else "").strip()
    if not s or re.fullmatch(r"(na|n/a|none|-|—)", s, re.I):
        return ""
    return re.sub(r"[^ACGTU]", "", s.upper())


def expand_sugar(motif, n):
    if not motif:
        return None
    m = str(motif).strip()
    if re.fullmatch(r"[dekl]+", m, re.I):
        return m.lower()
    g = re.fullmatch(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*(MOE|cET|LNA)?", m, re.I)
    if g:
        w = {"moe": "e", "cet": "k", "lna": "l"}.get((g.group(4) or "moe").lower(), "e")
        return w * int(g.group(1)) + "d" * int(g.group(2)) + w * int(g.group(3))
    return None


def expand_backbone(bb, n_links):
    if not bb:
        return None
    m = str(bb).strip()
    if re.fullmatch(r"[so]+", m, re.I):
        return m.lower()
    if re.fullmatch(r"PS", m, re.I) or re.search(r"phosphorothioate", m, re.I):
        return "s" * n_links if n_links else None
    if re.fullmatch(r"PO", m, re.I) or re.search(r"phosphodiester", m, re.I):
        return "o" * n_links if n_links else None
    return None


def build_helm(sequence, sugar_std, backbone_std, methyl_c):
    seq = clean_seq(sequence)
    if not seq or not sugar_std or len(sugar_std) != len(seq):
        return None
    if backbone_std and len(backbone_std) != len(seq) - 1:
        return None
    parts = []
    for i, b in enumerate(seq):
        sug = SUGAR_HELM.get(sugar_std[i])
        if not sug:
            return None
        base = "[5meC]" if (b == "C" and methyl_c) else b
        link = ""
        if backbone_std and i < len(seq) - 1:
            link = "[sp]" if backbone_std[i] == "s" else ""
        parts.append(f"{sug}({base}){link}")
    return "RNA1{{" + ".".join(parts) + "}}$$$$"


def parse_helm(helm):
    """HELM -> (bases, sugars, linkages) in the same per-position alphabet the
    patent motifs expand to, so the two are directly comparable."""
    if not helm:
        return None
    m = re.search(r"RNA1\{\{(.*)\}\}", str(helm))
    if not m:
        return None
    units = m.group(1).split(".")
    bases, sugars, links = [], [], []
    for i, u in enumerate(units):
        g = re.match(r"^(\[[a-zA-Z0-9]+\]|d)\(([^)]+)\)(\[sp\])?$", u.strip())
        if not g:
            return None
        sug = SUGAR_TOKEN.get(g.group(1))
        if not sug:
            return None
        sugars.append(sug)
        bases.append(g.group(2))
        if i < len(units) - 1:
            links.append("s" if g.group(3) else "o")
    return {
        "bases": bases,
        "seq": "".join("C" if b == "[5meC]" else b for b in bases),
        "sugars": "".join(sugars),
        "links": "".join(links),
        "methyl": "[5meC]" in bases,
    }


def strip_5mec(helm):
    return str(helm or "").replace("[5meC]", "C").replace("[d](", "d(")


# ── load the human-verified gold corpus ──────────────────────────────────────

CONTROL_NAME = re.compile(r"pbs|saline|vehicle|untreated|control|utc|ltc", re.I)
N_CONTROL_ROWS = 0          # counted while loading, reported in the audit


def load_gold():
    """Return (records, tables). One record per measurement, at the same grain
    the paper counts: one inhibition value, one FOB observation, or one
    biomarker readout."""
    global N_CONTROL_ROWS
    N_CONTROL_ROWS = 0
    records, tables, chem = [], {}, {}
    for f in sorted(GOLD_RUNS.glob("*.json")):
        pid, tnum = f.stem.split("_t")
        if pid not in PATENTS:
            continue
        d = json.loads(f.read_text())
        tables[(pid, int(tnum))] = {
            "status": d.get("status"),
            "klass": d.get("klass"),
            "n_rows": len(d.get("rows") or []),
        }
        if d.get("status") != "verified":
            continue
        klass = d.get("klass")
        rows = d.get("rows") or []

        # sequence tables are the compound->chemistry lookup, audited separately
        if klass == "sequence":
            for r in rows:
                cid = cid_of(r.get("Compound ID"))
                if cid and clean_seq(r.get("sequence")):
                    chem.setdefault(cid, dict(r, _src="sequence"))
            continue
        if klass not in CLASS_DATASETS:
            continue

        # inhibition tables also print chemistry; use as fallback lookup only
        for r in rows:
            cid = cid_of(r.get("Compound ID"))
            if cid and clean_seq(r.get("sequence")) and cid not in chem:
                chem[cid] = dict(r, _src="inhibition")

        prim_col, channel, dose_col = PRIMARY[klass]
        for i, r in enumerate(rows):
            cid_field = "Compound ID" if klass == "inhibition" else "treatment_or_ISIS_number"
            cid = cid_of(r.get(cid_field))
            if not cid:
                # Vehicle and PBS control arms carry no compound number. The
                # release drops them deliberately, so excluding them here too
                # keeps that decision out of the coverage figure entirely.
                if CONTROL_NAME.search(str(r.get(cid_field) or "")):
                    N_CONTROL_ROWS += 1
                continue
            dose = as_scalar(r.get(dose_col))
            base = dict(pid=pid, table=int(tnum), klass=klass, cid=cid,
                        dose=dose, row=r, row_idx=i)
            if klass == "hepatotox":
                for gcol, acol in BIOMARKERS.items():
                    vec = as_vector(r.get(gcol))
                    vals = vec if vec is not None else (
                        [as_scalar(r.get(gcol))] if as_scalar(r.get(gcol)) is not None else [])
                    for v in vals:
                        records.append(dict(base, channel=acol, value=v))
            else:
                v = r.get(prim_col)
                if as_vector(v) is None and as_scalar(v) is None:
                    continue
                records.append(dict(base, channel=channel, value=v))
    return records, tables, chem


# ── load the released Atlas ──────────────────────────────────────────────────

def read_release(name):
    """Read either the former flat parquet or the current split shards."""
    flat = RELEASE / f"{name}.parquet"
    if flat.exists():
        return pd.read_parquet(flat)
    shards = [RELEASE / name / f"{split}.parquet"
              for split in ("train", "validation", "test")]
    missing = [path for path in shards if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing release shards: {missing}")
    return pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)


def load_atlas():
    """Return (all_records_by_cid, records_filed_under_our_patents)."""
    by_cid = defaultdict(list)
    ours = []
    specs = {
        "in_vitro_inhibition": ("Inhibition_pct", "dosage_nm"),
        "dose_response": ("Inhibition_pct", "dosage_nm"),
        "neurotoxicity": ("FOB_score", "dosage_ug"),
    }
    for name, (vcol, dcol) in specs.items():
        df = read_release(name)
        for r in df.to_dict("records"):
            cid = cid_of(r["Compound ID"])
            if cid is None:
                continue
            rec = {"ds": name, "pid": r["USPTO ID"], "table": int(r["Table Number"]),
                   "cid": cid, "channel": vcol, "value": r[vcol],
                   "dose": as_scalar(r[dcol]), "raw": r}
            by_cid[cid].append(rec)
            if r["USPTO ID"] in PATENTS:
                ours.append(rec)

    df = read_release("hepatotoxicity")
    for r in df.to_dict("records"):
        cid = cid_of(r["Compound ID"])
        if cid is None:
            continue
        dose = as_scalar(r["dosage_mg_per_kg"])
        for acol in BIOMARKERS.values():
            vec = as_vector(r.get(acol))
            if vec is None:
                s = as_scalar(r.get(acol))
                vec = [s] if s is not None else []
            for v in vec:
                rec = {"ds": "hepatotoxicity", "pid": r["USPTO ID"],
                       "table": int(r["Table Number"]), "cid": cid, "channel": acol,
                       "value": v, "dose": dose, "raw": r}
                by_cid[cid].append(rec)
                if r["USPTO ID"] in PATENTS:
                    ours.append(rec)
    return by_cid, ours


# ── matching ─────────────────────────────────────────────────────────────────

def candidates(by_cid, rec):
    want = set(CLASS_DATASETS[rec["klass"]])
    return [h for h in by_cid.get(rec["cid"], [])
            if h["ds"] in want and h["channel"] == rec["channel"]]


def context_of(rec):
    """The assay context that distinguishes two measurements of one compound at
    one dose: which cell line, which species, which readout."""
    row = rec["row"]
    if rec["klass"] == "inhibition":
        return norm_text(row.get("cell_line"))
    return vocab(row.get("species"))


def atlas_context(h):
    if h["ds"] in ("in_vitro_inhibition", "dose_response"):
        return norm_text(h["raw"].get("cell_line"))
    return vocab(h["raw"].get("species"))


def find_match(cands, value, dose, pid=None, ctx=None):
    """First candidate agreeing on (value, dose), preferring one that also
    agrees on assay context.

    Without the context preference a compound with the same percent inhibition
    in two cell lines matches whichever row comes first, so a HepG2 measurement
    can be paired with an LLC-MK2 row. That does not change whether the
    measurement is present, but it corrupts every secondary-field comparison
    downstream: 162 of the apparent cell_line_species disagreements were this
    artefact rather than a discrepancy in either dataset.
    """
    fallback = None
    for h in cands:
        if pid is not None and h["pid"] != pid:
            continue
        if not same_value(value, h["value"]):
            continue
        if dose is not None and h["dose"] is not None and abs(dose - h["dose"]) > 1e-6:
            continue
        if ctx is None or atlas_context(h) == ctx:
            return h
        if fallback is None:
            fallback = h
    return fallback


# ── table-level strata, read from the patent XML ─────────────────────────────

def table_strata(gold_records, gold_tables):
    """Assign each verified measurement table a format and unit-conversion
    stratum. Format is read from the extracted shape (how many doses and how
    many measurement contexts per table); unit conversion is read from the
    patent's own caption, not from our code, so the stratum is not circular.

    USPTO full-text XML separates caption words with the em-space entity
    &#x2003; rather than a space, so the caption must be entity-decoded before
    any word-boundary regex will fire on it."""
    header = {}
    for pid in PATENTS:
        xml_files = list(XML_DIR.glob(f"{pid}-*.XML"))
        if not xml_files:
            continue
        xml = xml_files[0].read_text(errors="ignore")
        for m in re.finditer(r'<tables[^>]*num="([^"]*)"[^>]*>([\s\S]*?)</tables>', xml):
            num, body = int(m.group(1)), m.group(2)
            text = unescape(re.sub(r"<[^>]+>", " ", body))
            text = text.replace(" ", " ").replace(" ", " ")
            header[(pid, num)] = re.sub(r"\s+", " ", text)[:1500].lower()

    per_table = defaultdict(lambda: {"doses": set(), "ctx": set(), "n": 0})
    for r in gold_records:
        k = (r["pid"], r["table"])
        per_table[k]["n"] += 1
        if r["dose"] is not None:
            per_table[k]["doses"].add(round(r["dose"], 6))
        row = r["row"]
        if r["klass"] == "inhibition":
            # a "measurement context" is one assay column of the printed grid:
            # same compound, same dose, different cell line / readout / timepoint
            per_table[k]["ctx"].add((norm_text(row.get("cell_line")),
                                     norm_text(row.get("target_RNA")),
                                     norm_text(row.get("treatment_period_hrs"))))
        elif r["klass"] == "neurotox":
            per_table[k]["ctx"].add((norm_text(row.get("latency_time_hours")),
                                     norm_text(row.get("tolerability_score_type"))))
        else:
            per_table[k]["ctx"].add((r["channel"],))

    strata = {}
    for k, v in per_table.items():
        h = header.get(k, "")
        pct_control = bool(re.search(r"%\s*(control|utc|ltc)|percent\s+control|"
                                     r"\bof\s+control\b|untreated control", h))
        dose_unit = bool(re.search(r"\d\s*(µm|μm|um\b|mm\b)|micromolar|millimolar", h))
        strata[k] = {
            "format": "dose-series" if len(v["doses"]) >= 3 else "single-dose",
            "multi_column": len(v["ctx"]) > 1,
            "n_contexts": len(v["ctx"]),
            "pct_control_conversion": pct_control,
            "dose_unit_conversion": dose_unit,
            "n": v["n"],
        }
    return strata


# ── replaying the released pipeline's own QC rules ───────────────────────────
# Verbatim from analyses/logic/clean.py and Appendix A.2. A measurement the
# release does not contain is only an extraction failure if none of these
# accounts for it, so each is checked against the verified row before any
# omission is blamed on the LLM.

QC_VALUE_RANGE = {          # channel -> (min, max), documented range filters
    "Inhibition_pct": (-1000, 100),
    "ALT": (10, 50000), "AST": (10, 50000), "ALB": (1, 100),
    "FOB_score": (0, 7),
}
HEPATIC_ROUTES = {"subcutaneous", "intraperitoneal"}

# documented deduplication keys, per released table
DEDUP_KEYS = {
    "in_vitro_inhibition": ("cid", "value"),
    "dose_response": ("cid", "dose", "value"),
    "neurotox": ("cid", "species", "route", "score_type", "dose"),
    "hepatotox": ("cid", "species", "num_doses", "period", "dose", "route", "channel"),
}


def collapsed_by_dedup(r, cands, strata, rel_tables):
    """Would the documented deduplication have collapsed this verified
    measurement into a row the release does contain?

    The test has to run against the RELEASE rows, not against other verified
    rows, because the dedup keys are narrower than the measurement itself:

      in_vitro_inhibition  drop_duplicates(['Compound ID', 'Inhibition_pct'])
      neurotoxicity        drop_duplicates([HELM, species, route, score type, dose])

    Neither key contains everything that distinguishes two real measurements.
    A compound assayed at 700 nM and at 500 nM with the same percent inhibition
    survives in_vitro dedup exactly once, so the 700 nM reading is absent from
    the release by design rather than by extraction failure. Scoring it as a
    miss blames the LLM for a deliberate collapse.
    """
    if not cands:
        return None
    # Deduplication can only remove a row the pipeline actually read. If the
    # release holds NOTHING from this source table, the table was never ingested
    # and the omission is a coverage failure, not a deliberate collapse.
    #
    # Without this guard the rule fires on chance collisions: a compound may hold
    # dozens of records across the release, so "some record of this compound has
    # the same inhibition value" is easy to satisfy by luck. It was mis-attributing
    # 484 of 496 omissions, which flattered filtering recall and hid genuine loss.
    # C9ORF72 t43/t44/t45 were the bulk of it, and they are not replicates of the
    # ingested t42: they carry different compounds and different values.
    if rel_tables.get((r["pid"], r["table"]), 0) == 0:
        return None
    if r["klass"] == "inhibition":
        if strata[(r["pid"], r["table"])]["format"] != "single-dose":
            return None            # dose IS in the dose_response key
        if any(same_value(r["value"], h["value"]) for h in cands):
            return ("collapsed by deduplication: in_vitro key is compound + "
                    "inhibition value, excluding dose")
        return None
    if r["klass"] == "neurotox":
        # latency_time_hours is not in the key, so two timepoints at one dose
        # collapse to one row
        if any(h["dose"] is not None and r["dose"] is not None
               and abs(h["dose"] - r["dose"]) < 1e-6 for h in cands):
            return ("collapsed by deduplication: neurotoxicity key excludes "
                    "readout timepoint")
        return None
    return None                     # hepatic groups on dose, so no collapse


def source_table_absent(r, dead_tables):
    """No measurement of this source table appears anywhere in the release.

    The test must be "did anything from this table match, under any patent",
    not "does the release hold rows under this (patent, table)". Continuations
    refile the same table under a new application number and a new table number,
    so the latter test calls 173 tables missing when only a handful truly are.

    Worth its own category because the remedy differs: one un-ingested table is a
    single failure, not hundreds of independent extraction errors, and merging
    the two makes a few lost tables look like diffuse inaccuracy."""
    if (r["pid"], r["table"], r["klass"]) in dead_tables:
        return "no measurement of this source table appears in the release"
    return None


def qc_rule_removing(r, gold_chem, helm_by_cid):
    """Name the documented QC rule that removes this verified measurement, or
    None if the release should have kept it."""
    # 1. HELM-level filters, applied uniformly across all four categories
    chem = gold_chem.get(r["cid"])
    if chem:
        seq = clean_seq(chem.get("sequence"))
        sug = expand_sugar(chem.get("sugar_motif"), len(seq) or None)
        if seq and len(seq) <= 10:
            return "HELM filter: sequence of 10 nt or fewer"
        if seq and len(set(seq)) == 1:
            return "HELM filter: homopolymer sequence"
        if sug:
            if "d" not in sug:
                return "HELM filter: no DNA gap (steric blocker)"
            if set(sug) == {"d"}:
                return "HELM filter: naked DNA, no sugar modification"
    if r["cid"] not in helm_by_cid:
        return "compound carries no HELM annotation in the release"

    # 2. documented value-range filters
    rng = QC_VALUE_RANGE.get(r["channel"])
    if rng:
        vec = as_vector(r["value"])
        vals = vec if vec is not None else [as_scalar(r["value"])]
        vals = [v for v in vals if v is not None]
        if vals and not all(rng[0] <= v <= rng[1] for v in vals):
            return f"value outside the documented QC range for {r['channel']}"

    # 3. category-specific admission rules
    row = r["row"]
    if r["klass"] == "hepatotox":
        if vocab(row.get("administration_method")) not in {"sc", "ip"}:
            return "hepatic filter: administration route not subcutaneous or intraperitoneal"
        if r["dose"] is not None and r["dose"] > 10000:
            return "hepatic filter: dose above 10,000 mg/kg"
    if r["klass"] == "inhibition" and r["dose"] is not None and r["dose"] <= 0:
        return "dose-response filter: non-positive dosage"
    return None


def chemistry_class(helm):
    h = parse_helm(helm)
    if not h:
        return "unparsed"
    s = h["sugars"]
    if re.fullmatch(r"e{5}d{10}e{5}", s):
        return "5-10-5 MOE gapmer"
    if "k" in s and "e" in s:
        return "mixed MOE/cET"
    if "k" in s:
        return "cET gapmer"
    if "l" in s:
        return "LNA-containing"
    if re.fullmatch(r"e+d+e+", s):
        return "MOE gapmer (other wing)"
    return "other"


# ── main analysis ────────────────────────────────────────────────────────────

def rate(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "pct": round(p, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)}


def main(seed=2026, n_perm=20):
    rng = np.random.default_rng(seed)
    gold, gold_tables, gold_chem = load_gold()
    n_control_rows = N_CONTROL_ROWS
    rel_tables = Counter()
    for _n in ["in_vitro_inhibition", "dose_response", "hepatotoxicity", "neurotoxicity"]:
        _df = read_release(_n)
        for _p, _t in zip(_df["USPTO ID"], _df["Table Number"]):
            rel_tables[(_p, int(_t))] += 1
    by_cid, atlas_ours = load_atlas()
    strata = table_strata(gold, gold_tables)

    # ---- recall: is each verified measurement present in the release? -------
    for r in gold:
        cands = candidates(by_cid, r)
        r["_n_cand"] = len(cands)
        ctx = context_of(r)
        hit = find_match(cands, r["value"], r["dose"], ctx=ctx)
        r["_hit"] = hit
        # Recall is judged release-wide (patent families republish tables), but a
        # metadata comparison may not be: two patents can carry the same compound
        # at the same value under entirely different studies, and comparing dosing
        # schedules across them manufactures disagreements. The "17 vs 17.5"
        # num_doses discrepancy attributed to IRF4 t147 was really this patent's
        # row being compared against US20220241320A1 t147, a sister filing with a
        # different protocol. Secondary fields therefore use the same-patent match
        # only, and simply abstain when there is none.
        same_pid = find_match(cands, r["value"], r["dose"], pid=r["pid"], ctx=ctx)
        r["_hit_same_patent"] = same_pid
        r["_same_patent"] = same_pid is not None
        r["_same_table"] = bool(hit and hit["pid"] == r["pid"] and hit["table"] == r["table"])
        r["_state"] = ("match" if hit else ("absent" if not cands else "missing"))

    # chance floor: permute values across compounds within (class, channel) and
    # re-run the identical matcher. Anything at or near this rate is noise.
    null_rates = []
    groups = defaultdict(list)
    for i, r in enumerate(gold):
        groups[(r["klass"], r["channel"])].append(i)
    for _ in range(n_perm):
        hits = 0
        for idxs in groups.values():
            vals = [gold[i]["value"] for i in idxs]
            perm = rng.permutation(len(vals))
            for j, i in enumerate(idxs):
                r = gold[i]
                if find_match(candidates(by_cid, r), vals[perm[j]], r["dose"]):
                    hits += 1
        null_rates.append(100 * hits / len(gold))
    null = {"mean_pct": round(float(np.mean(null_rates)), 2),
            "sd_pct": round(float(np.std(null_rates)), 2), "n_permutations": n_perm}

    # ---- precision: is each released row corroborated by the gold corpus? ---
    gold_by_patent = defaultdict(list)
    for r in gold:
        gold_by_patent[(r["pid"], r["klass"], r["channel"], r["cid"])].append(r)
    ds_to_class = {d: c for c, ds in CLASS_DATASETS.items() for d in ds}
    gold_cids_any = {r["cid"] for r in gold}
    # Precision is only defined over released rows whose source table the corpus
    # actually verified. Tables still under review are NOT ground truth, and
    # scoring them looks up the compound's values from OTHER tables of the same
    # patent, which manufactures false discordances: every released row of
    # C9ORF72 t19 was reported as uncorroborated purely because t19 is still
    # `review`, even though the verified re-extraction of it agrees exactly.
    gold_tables_seen = {(r["pid"], r["table"]) for r in gold}
    atlas_unverified = [a for a in atlas_ours
                        if (a["pid"], a["table"]) not in gold_tables_seen]
    atlas_ours = [a for a in atlas_ours
                  if (a["pid"], a["table"]) in gold_tables_seen]
    prec_hit = prec_miss = prec_nocompound = 0
    prec_by_ds = defaultdict(lambda: [0, 0])
    prec_examples = []
    prec_scored = []            # (released row, corroborated?) for stratification
    uncovered = Counter()
    for a in atlas_ours:
        klass = ds_to_class[a["ds"]]
        key = (a["pid"], klass, a["channel"], a["cid"])
        cands = gold_by_patent.get(key, [])
        if not cands:
            prec_nocompound += 1
            # why is there no gold counterpart? distinguishing these matters: a
            # compound we never saw is a different problem from one we saw in a
            # table the release filed under a different endpoint.
            if a["cid"] not in gold_cids_any:
                uncovered["compound not present in the verified corpus"] += 1
            elif (a["pid"], a["table"]) not in gold_tables_seen:
                uncovered["table filed under this patent but absent from it"] += 1
            else:
                uncovered["compound verified elsewhere, not at this endpoint"] += 1
            continue
        ok = any(same_value(g["value"], a["value"]) and
                 (g["dose"] is None or a["dose"] is None or
                  abs(g["dose"] - a["dose"]) < 1e-6) for g in cands)
        prec_by_ds[a["ds"]][1] += 1
        prec_scored.append((a, bool(ok)))
        if ok:
            prec_hit += 1
            prec_by_ds[a["ds"]][0] += 1
        else:
            prec_miss += 1
            if len(prec_examples) < 25:
                prec_examples.append({
                    "patent": a["pid"], "table": a["table"], "compound": a["cid"],
                    "channel": a["channel"], "atlas_value": str(a["value"]),
                    "atlas_dose": a["dose"],
                    "gold_values": sorted({str(g["value"]) for g in cands})[:6],
                })

    # ---- secondary fields, split extraction vs normalisation ---------------
    def stated(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return False
        return str(v).strip().upper() not in {"", "NA", "N/A", "NONE", "NULL"}

    fields = defaultdict(lambda: {"raw": [0, 0], "vocab": [0, 0], "modes": Counter(),
                                  "examples": [], "gold_only": 0, "atlas_only": 0,
                                  "neither": 0})
    for r in gold:
        hit = r["_hit_same_patent"]
        if not hit:
            continue
        for gcol, acol in FIELD_MAP[r["klass"]].items():
            gv = r["row"].get(gcol)
            av = hit["raw"].get(acol)
            key = (r["klass"], acol)
            # A field only one side records is a coverage gap, not a disagreement.
            # Counting it either way would be wrong, so it is tallied separately.
            if not stated(gv) or not stated(av):
                if stated(gv):
                    fields[key]["gold_only"] += 1
                elif stated(av):
                    fields[key]["atlas_only"] += 1
                else:
                    fields[key]["neither"] += 1
                continue
            ok = same_value(gv, av)
            fields[key]["raw"][1] += 1
            fields[key]["raw"][0] += int(ok)
            ok_v = ok or vocab(gv) == vocab(av)
            fields[key]["vocab"][1] += 1
            fields[key]["vocab"][0] += int(ok_v)
            if not ok_v:
                fields[key]["modes"][f"{norm_text(gv)[:32]} != {norm_text(av)[:32]}"] += 1
                if len(fields[key]["examples"]) < 3:
                    fields[key]["examples"].append(
                        {"patent": r["pid"], "table": r["table"], "compound": r["cid"],
                         "gold": str(gv)[:60], "atlas": str(av)[:60]})

    field_report = []
    for (klass, acol), v in sorted(fields.items()):
        field_report.append({
            "endpoint": klass, "field": acol,
            "kind": FIELD_KIND[klass].get(acol, "extraction"),
            "raw": rate(*v["raw"]), "after_vocabulary": rate(*v["vocab"]),
            "coverage": {"both_state_it": v["raw"][1], "only_patent_states_it": v["gold_only"],
                         "only_atlas_states_it": v["atlas_only"], "neither": v["neither"]},
            "top_mismatch_modes": v["modes"].most_common(3),
            "examples": v["examples"],
        })

    # ---- chemistry / HELM normalisation ------------------------------------
    helm_by_cid = {}
    for cid, recs in by_cid.items():
        for h in recs:
            hv = h["raw"].get("HELM Annotation")
            if isinstance(hv, str) and hv:
                helm_by_cid.setdefault(cid, set()).add(hv)

    chem_rows = []
    for cid, r in gold_chem.items():
        atlas_helms = helm_by_cid.get(cid)
        if not atlas_helms:
            continue
        seq = clean_seq(r.get("sequence"))
        sugar = expand_sugar(r.get("sugar_motif"), len(seq) or None)
        back = expand_backbone(r.get("backbone_motif"), len(sugar) - 1 if sugar else
                               (len(seq) - 1 if seq else None))
        methyl_stated = bool(re.fullmatch(r"(y|yes|true|1)", str(r.get("cytosine_methylated") or ""), re.I))
        ours = build_helm(seq, sugar, back, methyl_stated)
        best = None
        for ah in atlas_helms:
            p = parse_helm(ah)
            if not p:
                continue
            score = (p["seq"] == seq, p["sugars"] == sugar, p["links"] == back)
            if best is None or sum(score) > sum(best[1]):
                best = (ah, score, p)
        if best is None:
            continue
        ah, (seq_ok, sug_ok, link_ok), p = best
        # `best` is the Atlas HELM that agrees with us most, which on its own
        # would be circular: given several annotations we would be scoring the
        # one we like. Recording the strict variant too (every annotation for
        # this compound must agree) makes the selection auditable rather than
        # something a reader has to take on trust. The gap turns out to be
        # negligible because only 3 compounds carry more than one HELM, but that
        # is a fact to report, not to assume.
        parsed = [q for q in (parse_helm(x) for x in sorted(atlas_helms)) if q]
        chem_rows.append({
            "seq_ok_all": all(q["seq"] == seq for q in parsed),
            "sugar_ok_all": all(q["sugars"] == sugar for q in parsed) if sugar else None,
            "link_ok_all": all(q["links"] == back for q in parsed) if back else None,
            "n_atlas_helms": len(parsed),
            "cid": cid, "n_atlas_seqs": len({parse_helm(x)["seq"] for x in atlas_helms
                                             if parse_helm(x)}),
            "seq_ok": seq_ok, "sugar_ok": sug_ok, "link_ok": link_ok,
            # A motif the patent table never printed is unknown, not wrong. Scoring
            # those as failures is what made an earlier pass read sugar 44% /
            # linkage 18% for chemistry that agrees whenever it is stated at all.
            "sugar_stated": sugar is not None,
            "link_stated": back is not None,
            "helm_exact": bool(ours) and ours == ah,
            "helm_no5mec": bool(ours) and strip_5mec(ours) == strip_5mec(ah),
            "methyl_stated": methyl_stated,
            "atlas_has_5mec": p["methyl"],
            "chem_class": chemistry_class(ah),
            "buildable": ours is not None,
            "src": r.get("_src"),
        })
    cdf = pd.DataFrame(chem_rows)
    chem_report = {}
    if len(cdf):
        sug, lnk = cdf[cdf.sugar_stated], cdf[cdf.link_stated]
        # HELM can only be compared where the patent states BOTH sugar and
        # linkage. With linkage unstated build_helm emits an all-phosphodiester
        # backbone, which never matches a PS gapmer: scoring those as failures
        # made exact HELM read 21% for chemistry that is otherwise 99.8% right.
        bld = cdf[cdf.buildable & cdf.sugar_stated & cdf.link_stated]
        chem_report = {
            "n_compounds": int(len(cdf)),
            "sequence": rate(int(cdf.seq_ok_all.sum()), len(cdf)),
            "sugar_motif_where_stated": rate(int(sug.sugar_ok_all.sum()), len(sug)),
            "linkage_motif_where_stated": rate(int(lnk.link_ok_all.sum()), len(lnk)),
            "selection_sensitivity": {
                "note": ("headline figures above require EVERY Atlas annotation for "
                         "the compound to agree. The permissive variant, scoring the "
                         "best-agreeing annotation, is shown for comparison; the two "
                         "differ only because 3 compounds carry more than one HELM."),
                "compounds_with_multiple_helms": int((cdf.n_atlas_helms > 1).sum()),
                "sequence_best_agreeing": rate(int(cdf.seq_ok.sum()), len(cdf)),
                "sugar_best_agreeing": rate(int(sug.sugar_ok.sum()), len(sug)),
                "linkage_best_agreeing": rate(int(lnk.link_ok.sum()), len(lnk)),
            },
            "helm_denominator_note": (
                "helm_exact and helm_ignoring_5mec are computed only where the "
                "patent states BOTH the sugar and the linkage motif, so a full HELM "
                "can be built at all. That is a minority of compounds; the coverage "
                "block below gives how often each motif is stated."),
            "helm_exact": rate(int(bld.helm_exact.sum()), len(bld)),
            "helm_ignoring_5mec": rate(int(bld.helm_no5mec.sum()), len(bld)),
            "unique_sequence_in_atlas": rate(int((cdf.n_atlas_seqs == 1).sum()), len(cdf)),
            "coverage": {
                "sugar_motif_stated_in_patent": rate(int(cdf.sugar_stated.sum()), len(cdf)),
                "linkage_motif_stated_in_patent": rate(int(cdf.link_stated.sum()), len(cdf)),
                "cytosine_methylation_stated_in_table": rate(int(cdf.methyl_stated.sum()), len(cdf)),
                "atlas_annotates_5meC": rate(int(cdf.atlas_has_5mec.sum()), len(cdf)),
            },
            "note": ("The dominant HELM discrepancy is 5-methyl-cytosine, which the "
                     "patents state once in prose ('each cytosine residue is a 5-methyl "
                     "cytosine') rather than per compound in the table. helm_exact vs "
                     "helm_ignoring_5mec isolates that single normalisation error mode."),
        }

    # ---- stratified recall --------------------------------------------------
    def stratify(keyfn, label):
        acc = defaultdict(lambda: [0, 0])
        for r in gold:
            k = keyfn(r)
            if k is None:
                continue
            acc[k][1] += 1
            acc[k][0] += int(r["_state"] == "match")
        return {"by": label,
                "levels": {str(k): rate(*v) for k, v in sorted(acc.items(), key=lambda x: -x[1][1])}}

    endpoint_of = {"inhibition": "in vitro inhibition", "hepatotox": "hepatotoxicity",
                   "neurotox": "neurotoxicity"}
    chem_of = {}
    for cid in {r["cid"] for r in gold}:
        hs = helm_by_cid.get(cid)
        chem_of[cid] = chemistry_class(sorted(hs)[0]) if hs else "not in Atlas"

    # A compound the release carries no HELM for cannot match anything, so its
    # recall is 0 by construction. Leaving it in the chemistry stratum reads as
    # "this chemistry class has 0% recall", which is not what it means: the class
    # is unknown precisely because the compound is absent. It is reported as its
    # own coverage quantity instead.
    def chem_stratum(r):
        c = chem_of[r["cid"]]
        return None if c == "not in Atlas" else c

    unannotated = [r for r in gold if chem_of[r["cid"]] == "not in Atlas"]

    strat = [
        stratify(lambda r: endpoint_of[r["klass"]], "endpoint"),
        stratify(lambda r: r["channel"] if r["klass"] == "hepatotox" else None,
                 "hepatotoxicity biomarker channel"),
        stratify(lambda r: strata[(r["pid"], r["table"])]["format"], "table format"),
        stratify(lambda r: ("multi-column melt" if strata[(r["pid"], r["table"])]["multi_column"]
                            else "single measurement column"), "table layout"),
        stratify(lambda r: ("% control -> % inhibition"
                            if strata[(r["pid"], r["table"])]["pct_control_conversion"]
                            else "no conversion") if r["klass"] == "inhibition" else None,
                 "unit conversion (potency scale)"),
        stratify(lambda r: ("uM/mM -> nM" if strata[(r["pid"], r["table"])]["dose_unit_conversion"]
                            else "dose stated in nM") if r["klass"] == "inhibition" else None,
                 "unit conversion (dose)"),
        stratify(chem_stratum, "chemistry class"),
        stratify(lambda r: f'{r["pid"]} ({PATENTS[r["pid"]]})', "patent"),
    ]

    # Endpoint and table layout are not independent axes in this corpus: almost
    # every hepatotoxicity table is a multi-column melt. Reporting both without
    # saying so presents one population twice and reads as two findings
    # corroborating each other.
    xtab = defaultdict(lambda: [0, 0])
    for r in gold:
        k = (endpoint_of[r["klass"]],
             "multi-column melt" if strata[(r["pid"], r["table"])]["multi_column"]
             else "single measurement column")
        xtab[k][1] += 1
        xtab[k][0] += int(r["_state"] == "match")
    confounding = {
        "note": ("endpoint and table layout overlap heavily, so the two strata "
                 "above are not independent evidence; this cross-tab shows how "
                 "much of each layout level is one endpoint."),
        "cells": {f"{e} / {l}": rate(*v) for (e, l), v in sorted(xtab.items())},
    }

    # Precision, stratified on the same axes as recall. Q2 asks for coverage "in
    # addition to row-level correctness", which is only answerable if both are
    # cut the same way; precision by dataset alone cannot say whether the release
    # is less accurate on, say, melted tables.
    ds_endpoint = {"in_vitro_inhibition": "in vitro inhibition",
                   "dose_response": "in vitro inhibition",
                   "hepatotoxicity": "hepatotoxicity",
                   "neurotoxicity": "neurotoxicity"}

    def prec_stratify(keyfn, label):
        acc = defaultdict(lambda: [0, 0])
        for a, ok in prec_scored:
            k = keyfn(a)
            if k is None:
                continue
            acc[k][1] += 1
            acc[k][0] += int(ok)
        return {"by": label,
                "levels": {str(k): rate(*v)
                           for k, v in sorted(acc.items(), key=lambda x: -x[1][1])}}

    def a_layout(a):
        s = strata.get((a["pid"], a["table"]))
        if not s:
            return None
        return "multi-column melt" if s["multi_column"] else "single measurement column"

    def a_chem(a):
        hs = helm_by_cid.get(a["cid"])
        return chemistry_class(sorted(hs)[0]) if hs else None

    def a_conv(a):
        s = strata.get((a["pid"], a["table"]))
        if not s or ds_endpoint[a["ds"]] != "in vitro inhibition":
            return None
        return "% control -> % inhibition" if s["pct_control_conversion"] else "no conversion"

    prec_strat = [
        prec_stratify(lambda a: ds_endpoint[a["ds"]], "endpoint"),
        prec_stratify(lambda a: strata.get((a["pid"], a["table"]), {}).get("format"),
                      "table format"),
        prec_stratify(a_layout, "table layout"),
        prec_stratify(a_conv, "unit conversion (potency scale)"),
        prec_stratify(a_chem, "chemistry class"),
        prec_stratify(lambda a: f'{a["pid"]} ({PATENTS[a["pid"]]})', "patent"),
    ]

    # ---- coverage: what the release omitted --------------------------------
    tbl_recall = defaultdict(lambda: [0, 0])
    for r in gold:
        k = (r["pid"], r["table"], r["klass"])
        tbl_recall[k][1] += 1
        tbl_recall[k][0] += int(r["_state"] == "match")
    missed_tables = [{"patent": p, "table": t, "endpoint": endpoint_of[k],
                      "gold_measurements": v[1]}
                     for (p, t, k), v in sorted(tbl_recall.items()) if v[0] == 0]
    partial_tables = [{"patent": p, "table": t, "endpoint": endpoint_of[k],
                       "recalled": v[0], "gold_measurements": v[1]}
                      for (p, t, k), v in sorted(tbl_recall.items(), key=lambda x: x[1][0] / x[1][1])
                      if 0 < v[0] < v[1]][:15]

    # Why each omission happened. "Dose arm dropped" is the diagnostic one: the
    # release captured the compound at this endpoint, but only at some of the
    # doses the table prints. It is invisible to any row-level precision audit
    # and it concentrates in the multi-dose in vivo tolerability tables that the
    # cost model's late stages depend on.
    omission_modes = Counter()
    omission_by_endpoint = defaultdict(Counter)
    dose_arm_examples = []
    for r in gold:
        if r["_state"] == "match":
            continue
        cands = candidates(by_cid, r)
        if not cands:
            mode = "compound has no record at this endpoint in the release"
        elif r["dose"] is not None and not any(
                h["dose"] is not None and abs(h["dose"] - r["dose"]) < 1e-6 for h in cands):
            mode = "dose arm dropped (compound captured at other doses only)"
            if len(dose_arm_examples) < 12:
                dose_arm_examples.append({
                    "patent": r["pid"], "table": r["table"], "compound": r["cid"],
                    "endpoint": endpoint_of[r["klass"]], "channel": r["channel"],
                    "patent_dose": r["dose"], "patent_value": str(r["value"]),
                    "doses_in_release": sorted({h["dose"] for h in cands
                                                if h["dose"] is not None})[:8],
                })
        else:
            mode = "dose arm captured, this value absent"
        omission_modes[mode] += 1
        omission_by_endpoint[endpoint_of[r["klass"]]][mode] += 1

    # ---- filtering recall, separated from extraction recall ----------------
    # The AC asks for these as distinct quantities, and they are: the pipeline
    # documents QC rules that remove rows ON PURPOSE (Appendix A.2), so a
    # measurement absent from the release is only an extraction failure if no
    # documented rule accounts for it. We replay each rule against the verified
    # row and attribute the omission to the first rule that fires.
    # tables from which nothing at all reached the release, under any patent
    _tbl = defaultdict(lambda: [0, 0])
    for r in gold:
        k = (r["pid"], r["table"], r["klass"])
        _tbl[k][1] += 1
        _tbl[k][0] += int(r["_state"] == "match")
    dead_tables = {k for k, v in _tbl.items() if v[0] == 0}

    filt = Counter()
    filt_by_endpoint = defaultdict(Counter)
    filt_examples = defaultdict(list)
    for r in gold:
        if r["_state"] == "match":
            filt["retained and present"] += 1
            filt_by_endpoint[endpoint_of[r["klass"]]]["retained and present"] += 1
            continue
        rule = qc_rule_removing(r, gold_chem, helm_by_cid)
        if rule is None:
            rule = collapsed_by_dedup(r, candidates(by_cid, r), strata, rel_tables)
        if rule is None:
            rule = source_table_absent(r, dead_tables)
        if rule is None:
            rule = "unexplained by any documented filter"
        filt[rule] += 1
        filt_by_endpoint[endpoint_of[r["klass"]]][rule] += 1
        if len(filt_examples[rule]) < 3:
            filt_examples[rule].append(
                {"patent": r["pid"], "table": r["table"], "compound": r["cid"],
                 "endpoint": endpoint_of[r["klass"]], "channel": r["channel"],
                 "value": str(r["value"]), "dose": r["dose"]})

    # A never-ingested table is NOT a documented filter, so it must not be
    # credited to filtering recall. It is a coverage failure with a known cause.
    NOT_A_FILTER = ("retained and present",
                    "unexplained by any documented filter",
                    "no measurement of this source table appears in the release")
    removed_by_design = sum(v for k, v in filt.items() if k not in NOT_A_FILTER)
    unexplained = filt["unexplained by any documented filter"]
    n_gold = len(gold)
    filtering_report = {
        "definition": ("filtering recall = share of verified measurements that the "
                       "documented QC rules retain; extraction recall = share of "
                       "those retained measurements the pipeline actually captured. "
                       "Their product is the end-to-end recall."),
        "filtering_recall": rate(n_gold - removed_by_design, n_gold),
        "extraction_recall_of_retained": rate(
            sum(r["_state"] == "match" for r in gold), n_gold - removed_by_design),
        "end_to_end_recall": rate(sum(r["_state"] == "match" for r in gold), n_gold),
        "unexplained_omissions": rate(unexplained, n_gold),
        "by_rule": dict(filt.most_common()),
        "by_rule_and_endpoint": {k: dict(v.most_common())
                                 for k, v in filt_by_endpoint.items()},
        "examples": {k: v for k, v in filt_examples.items()},
    }

    # ---- unit-conversion accuracy ------------------------------------------
    # Distinct from the unit-conversion recall stratum: this asks whether the
    # conversion the pipeline performed produced the right NUMBER, by testing
    # the two ways it can go wrong. Orientation: patents 2 and 3 print % control,
    # so the release must store 100 - x; if it stored x the value is inverted.
    # Scale: a dose or value off by a power of ten is an unconverted unit.
    orient = Counter()
    orient_examples = []
    for r in gold:
        if r["klass"] != "inhibition":
            continue
        if not strata[(r["pid"], r["table"])]["pct_control_conversion"]:
            continue
        cands = candidates(by_cid, r)
        if not cands:
            continue
        gv = as_scalar(r["value"])
        if gv is None:
            continue
        dose_ok = lambda h: (r["dose"] is None or h["dose"] is None
                             or abs(h["dose"] - r["dose"]) < 1e-6)
        gold_cell = norm_text(r["row"].get("cell_line"))

        def context_ok(h):
            atlas_cell = norm_text(h["raw"].get("cell_line"))
            return dose_ok(h) and (
                not gold_cell or not atlas_cell or gold_cell == atlas_cell
            )

        if any(context_ok(h) and same_value(gv, h["value"]) for h in cands):
            orient["converted correctly (% inhibition)"] += 1
        elif any(context_ok(h) and same_value(100 - gv, h["value"]) for h in cands):
            orient["stored as % control, not converted"] += 1
            if len(orient_examples) < 8:
                orient_examples.append(
                    {"patent": r["pid"], "table": r["table"], "compound": r["cid"],
                     "patent_pct_inhibition": gv, "value_in_release": 100 - gv})
        else:
            orient["no counterpart to compare"] += 1

    scale = Counter()
    scale_examples = []
    for r in gold:
        if r["_state"] == "match":
            continue
        cands = candidates(by_cid, r)
        gv = as_scalar(r["value"])
        if not cands or gv is None or gv == 0:
            continue
        hit = None
        for k in (-3, -2, -1, 1, 2, 3):
            f = 10.0 ** k
            for h in cands:
                hv = as_scalar(h["value"])
                if hv is not None and hv != 0 and abs(hv - gv * f) < 1e-6 * max(abs(hv), 1):
                    hit = ("value", k, h)
                    break
            if hit:
                break
        if hit is None and r["dose"] is not None and r["dose"] != 0:
            for k in (-3, -2, -1, 1, 2, 3):
                f = 10.0 ** k
                for h in cands:
                    if (h["dose"] is not None and same_value(gv, h["value"])
                            and abs(h["dose"] - r["dose"] * f) < 1e-6 * max(abs(h["dose"]), 1)):
                        hit = ("dose", k, h)
                        break
                if hit:
                    break
        if hit:
            what, k, h = hit
            scale[f"{what} differs by 10^{k}"] += 1
            if len(scale_examples) < 8:
                scale_examples.append(
                    {"patent": r["pid"], "table": r["table"], "compound": r["cid"],
                     "field": what, "factor": 10.0 ** k, "patent_value": gv,
                     "patent_dose": r["dose"], "release_value": str(h["value"]),
                     "release_dose": h["dose"]})

    # How many tables actually require each conversion. Without this the dose
    # stratum reports a single level and looks like padding; the honest statement
    # is that the conversion is absent from this corpus, with the count of tables
    # inspected to back it.
    inhib_tables = {(r["pid"], r["table"]) for r in gold if r["klass"] == "inhibition"}
    n_dose_conv = sum(1 for k in inhib_tables if strata[k]["dose_unit_conversion"])
    n_pct_conv = sum(1 for k in inhib_tables if strata[k]["pct_control_conversion"])

    n_orient = sum(orient.values()) - orient["no counterpart to compare"]
    unit_report = {
        "conversions_present_in_the_audited_corpus": {
            "in_vitro_tables_inspected": len(inhib_tables),
            "tables_printing_%_control_(need 100-x)": n_pct_conv,
            "tables_printing_dose_in_uM_or_mM_(need x1000)": n_dose_conv,
            "note": ("Every in-vitro table in these four patents states its "
                     "concentrations in nM, so the dose-unit conversion has no "
                     "exercised level here and its recall stratum has a single "
                     "cell. That is a property of the corpus, not an omission "
                     "from the audit. The potency-scale conversion is exercised "
                     "and is reported below."),
        },
        "potency_scale_orientation": {
            "definition": ("tables printing % control must be stored as "
                           "100 - x; a value stored uninverted is a conversion failure"),
            "correct": rate(orient["converted correctly (% inhibition)"], n_orient),
            "counts": dict(orient),
            "examples": orient_examples,
        },
        "power_of_ten_errors": {
            "definition": ("among omitted measurements, how many are present in the "
                           "release at exactly 10^k times the printed value or dose"),
            "n_detected": int(sum(scale.values())),
            "of_omissions": int(sum(omission_modes.values())),
            "by_pattern": dict(scale),
            "examples": scale_examples,
        },
    }

    # ---- self-check: is the verified corpus itself complete? ---------------
    # An audit whose ground truth is incomplete manufactures false precision
    # errors, so the corpus is checked against the release in the SAME way the
    # release is checked against it. Per table, compare the set of dose arms
    # each side holds. A dose the release has and the corpus lacks is our gap;
    # the reverse is the release's. Both are reported.
    gold_doses = defaultdict(set)
    for r in gold:
        if r["dose"] is not None:
            gold_doses[(r["pid"], r["table"], r["klass"])].add(round(r["dose"], 1))
    rel_doses = defaultdict(set)
    for a in atlas_ours:
        if a["dose"] is not None:
            rel_doses[(a["pid"], a["table"], ds_to_class[a["ds"]])].add(round(a["dose"], 1))

    corpus_gaps, release_gaps, transposed = [], [], []
    for k, gd in sorted(gold_doses.items()):
        rd = rel_doses.get(k, set())
        if not rd:
            continue
        pid, tnum, klass = k
        # Special case worth naming rather than counting as a corpus gap: if the
        # release's "doses" for a table coincide with the corpus's measurement
        # VALUES, the release read the value column as the dose column.
        gvals = {round(as_scalar(x["value"]), 1) for x in gold
                 if (x["pid"], x["table"], x["klass"]) == k
                 and as_scalar(x["value"]) is not None}
        # Require a real dose ladder and a genuine mismatch before calling it a
        # transposition: with one dose the overlap test is trivially satisfied.
        if (rd - gd) and len(rd) >= 4 and gvals and len(rd & gvals) >= 0.8 * len(rd):
            transposed.append({"patent": pid, "table": tnum,
                               "endpoint": endpoint_of[klass],
                               "release_doses": sorted(rd)[:10],
                               "corpus_doses": sorted(gd)[:10],
                               "diagnosis": ("release dose column holds the "
                                             "measurement values")})
            continue
        if rd - gd:
            corpus_gaps.append({"patent": pid, "table": tnum, "endpoint": endpoint_of[klass],
                                "doses_only_in_release": sorted(rd - gd)[:12],
                                "doses_in_corpus": sorted(gd)[:12]})
        if gd - rd:
            release_gaps.append({"patent": pid, "table": tnum, "endpoint": endpoint_of[klass],
                                 "doses_only_in_corpus": sorted(gd - rd)[:12],
                                 "doses_in_release": sorted(rd)[:12]})
    self_check = {
        "method": ("per table, compare the dose arms each side holds. Applied in "
                   "both directions so a gap in the verified corpus is not "
                   "reported as a defect in the release."),
        "tables_compared": len([k for k in gold_doses if rel_doses.get(k)]),
        "tables_where_corpus_is_incomplete": corpus_gaps,
        "tables_where_release_is_incomplete": release_gaps,
        "n_corpus_incomplete": len(corpus_gaps),
        "n_release_incomplete": len(release_gaps),
        "tables_with_transposed_dose_column": transposed,
    }

    # ---- error taxonomy -----------------------------------------------------
    # Every disagreement the audit found, named, counted, and labelled by whether
    # it can reach a modelled endpoint. The distinction matters: a wrong ALT value
    # changes a pass/fail label and therefore the benchmark; a cells-per-well
    # digit error changes nothing any model in the paper consumes.
    MODEL_FIELDS = {"Inhibition_pct", "FOB_score", "ALT", "AST", "TBIL", "BUN",
                    "ALB", "CREA", "PC_ratio", "dosage_nm", "dosage_ug",
                    "dosage_mg_per_kg", "HELM Annotation"}
    taxonomy = []
    for x in field_report:
        v = x["after_vocabulary"]
        if v["n"] == 0 or v["k"] == v["n"]:
            continue
        taxonomy.append({
            "mode": f'{x["endpoint"]}.{x["field"]} disagrees',
            "kind": x["kind"], "n": v["n"] - v["k"], "of": v["n"],
            "reaches_a_modelled_endpoint": x["field"] in MODEL_FIELDS,
            "example": (x["examples"] or [None])[0],
            "dominant_pattern": (x["top_mismatch_modes"] or [[None, 0]])[0][0],
        })
    taxonomy.append({
        "mode": "measurement printed in the patent but absent from the release",
        "kind": "coverage", "n": int(sum(omission_modes.values())), "of": len(gold),
        "reaches_a_modelled_endpoint": True,
        "example": None,
        "dominant_pattern": max(omission_modes, key=omission_modes.get) if omission_modes else None,
    })
    taxonomy.append({
        "mode": "released measurement not corroborated by the verified re-extraction",
        "kind": "extraction", "n": prec_miss, "of": prec_hit + prec_miss,
        "reaches_a_modelled_endpoint": True,
        "example": (prec_examples or [None])[0], "dominant_pattern": None,
    })
    if chem_report:
        ex = chem_report["helm_exact"]
        taxonomy.append({
            "mode": "HELM differs only by 5-methyl-cytosine annotation",
            "kind": "normalisation", "n": ex["n"] - ex["k"], "of": ex["n"],
            "reaches_a_modelled_endpoint": True,
            "example": None,
            "dominant_pattern": ("patents state methylation once in prose, not per "
                                 "compound in the table"),
        })
    taxonomy.sort(key=lambda t: -t["n"])

    # ---- audit scope, stated explicitly -------------------------------------
    # Ionis files continuations that republish the same studies under new
    # application numbers, so the compounds in these four patents also appear in
    # the release under sister filings we did not annotate. Recall is measured
    # release-wide and therefore already credits that republication, but
    # PRECISION is only defined over rows filed under the four audited IDs. The
    # rest is out of scope and is quantified here rather than left implicit.
    four_rows = sum(len(read_release(n_).query("`USPTO ID` in @PATENTS")) for n_ in
                    ["in_vitro_inhibition", "dose_response", "hepatotoxicity",
                     "neurotoxicity"])
    fam = Counter()
    gold_cids = {r["cid"] for r in gold}
    for name in ["in_vitro_inhibition", "dose_response", "hepatotoxicity",
                 "neurotoxicity"]:
        df = read_release(name)
        for pid_, cid_ in zip(df["USPTO ID"], df["Compound ID"]):
            if pid_ not in PATENTS and cid_of(cid_) in gold_cids:
                fam[pid_] += 1
    scope = {
        "audited_patents": list(PATENTS),
        "released_rows_under_the_audited_patents": int(four_rows),
        "released_rows_under_family_members_sharing_these_compounds": int(sum(fam.values())),
        "family_members": dict(fam.most_common(12)),
        "note": ("Recall is matched release-wide, so a measurement republished by a "
                 "continuation still counts as covered. Precision is scored only on "
                 "rows filed under the four audited patents; rows under family "
                 "members are outside the annotated ground truth and are not "
                 "scored either way."),
    }

    # ---- assemble -----------------------------------------------------------
    n = len(gold)
    n_verified_tables = sum(1 for v in gold_tables.values() if v["status"] == "verified")
    report = {
        "validation_role": (
            "deprecated release-wide diagnostic; not primary precision or recall"
        ),
        "primary_validation": (
            "analyses/validation/canonical_source_audit.py with a frozen "
            "Gold-to-production-canonical source-table mapping"
        ),
        "corpus": {
            "patents": [{"id": p, "gene": g,
                         "verified_tables": sum(1 for (pp, _), v in gold_tables.items()
                                                if pp == p and v["status"] == "verified"),
                         "gold_measurements": sum(1 for r in gold if r["pid"] == p)}
                        for p, g in PATENTS.items()],
            "verified_tables_total": n_verified_tables,
            "tables_still_under_review": sum(1 for v in gold_tables.values()
                                             if v["status"] != "verified"),
            "gold_measurements_total": n,
            "gold_compounds": len({r["cid"] for r in gold}),
            "gold_chemistry_compounds": len(gold_chem),
            "measurements_by_endpoint": dict(Counter(endpoint_of[r["klass"]] for r in gold)),
            "unit_definition": ("one measurement = one assay readout: one inhibition value, "
                                "one FOB observation, or one biomarker readout "
                                "(a hepatotoxicity row contributes up to 7)"),
        },
        "audit_scope": scope,
        "controls_excluded": {
            "note": ("Vehicle and control arms (PBS, saline, untreated) are dropped "
                     "from the release on purpose. They are excluded from BOTH "
                     "sides here, so that choice is never scored as a coverage "
                     "failure: load_gold() keeps a row only if its treatment field "
                     "carries a 4-7 digit compound number, which no control arm "
                     "does. Counted here so the exclusion is visible rather than "
                     "implicit."),
            "control_rows_in_the_verified_tables": n_control_rows,
            "control_rows_reaching_the_audit": 0,
            "gold_measurements_at_zero_dose": sum(1 for r in gold if r["dose"] == 0),
        },
        "recall": {
            "anywhere_in_release": rate(sum(r["_state"] == "match" for r in gold), n),
            "same_patent_only": rate(sum(r["_same_patent"] for r in gold), n),
            "same_patent_and_table": rate(sum(r["_same_table"] for r in gold), n),
            "chance_floor_permuted_values": null,
            "note": ("Historical release-wide, value-based diagnostic only. Patent-family "
                     "republication motivates a predeclared production-canonical source "
                     "scope; it does not make unrestricted anywhere-in-release matching "
                     "a valid primary recall estimate."),
        },
        "precision": {
            "corroborated": rate(prec_hit, prec_hit + prec_miss),
            "released_rows_audited": prec_hit + prec_miss,
            "released_rows_for_compounds_absent_from_gold": prec_nocompound,
            "released_rows_excluded_table_still_under_review": len(atlas_unverified),
            "unauditable_row_reasons": dict(uncovered),
            "by_dataset": {k: rate(v[0], v[1]) for k, v in sorted(prec_by_ds.items())},
            "stratified": prec_strat,
            "discordance_examples": prec_examples,
        },
        "endpoint_layout_confounding": confounding,
        "compounds_with_no_chemistry_in_the_release": {
            "measurements": len(unannotated),
            "compounds": len({r["cid"] for r in unannotated}),
            "note": ("These carry no HELM in the release, so they cannot be "
                     "matched and cannot be assigned a chemistry class. They are "
                     "excluded from the chemistry stratum (where they would read "
                     "as a 0%-recall class) and counted here instead."),
            "by_endpoint": dict(Counter(endpoint_of[r["klass"]] for r in unannotated)),
        },
        "corpus_self_check": self_check,
        "filtering_vs_extraction_recall": filtering_report,
        "unit_conversion_accuracy": unit_report,
        "stratified_recall": strat,
        "error_taxonomy": taxonomy,
        "secondary_fields": field_report,
        "chemistry_normalisation": chem_report,
        "coverage": {
            "omission_modes": dict(omission_modes),
            "omission_modes_by_endpoint": {k: dict(v) for k, v in omission_by_endpoint.items()},
            "dropped_dose_arm_examples": dose_arm_examples,
            "tables_with_zero_coverage": missed_tables,
            "n_tables_with_zero_coverage": len(missed_tables),
            "measurements_in_zero_coverage_tables": sum(t["gold_measurements"] for t in missed_tables),
            "lowest_coverage_partial_tables": partial_tables,
        },
        "provenance": {"seed": seed, "release_dir": str(RELEASE),
                       "gold_dir": str(GOLD_RUNS)},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    return report


def summarise(rep):
    c, rc, pr = rep["corpus"], rep["recall"], rep["precision"]
    f = lambda d: f'{d["pct"]:.1f}% [{d["ci_lo"]:.1f}-{d["ci_hi"]:.1f}] ({d["k"]}/{d["n"]})'
    print(f'\ncorpus: {c["verified_tables_total"]} verified tables, '
          f'{c["gold_measurements_total"]:,} measurements, {c["gold_compounds"]:,} compounds')
    for p in c["patents"]:
        print(f'  {p["id"]} ({p["gene"]}): {p["verified_tables"]} tables, '
              f'{p["gold_measurements"]:,} measurements')
    print(f'\nRECALL   anywhere-in-release {f(rc["anywhere_in_release"])}')
    print(f'         same patent only    {f(rc["same_patent_only"])}')
    print(f'         same patent+table   {f(rc["same_patent_and_table"])}')
    print(f'         chance floor        {rc["chance_floor_permuted_values"]["mean_pct"]:.1f}%'
          f' +/- {rc["chance_floor_permuted_values"]["sd_pct"]:.1f}')
    print(f'\nPRECISION corroborated     {f(pr["corroborated"])}'
          f'  (+{pr["released_rows_for_compounds_absent_from_gold"]} rows for compounds not in gold)')
    for s in rep["stratified_recall"]:
        print(f'\n-- recall by {s["by"]}')
        for k, v in s["levels"].items():
            print(f'   {k:38s} {f(v)}')
    print("\n== row-level correctness (precision), same strata ==")
    for s in rep["precision"]["stratified"]:
        print(f'-- precision by {s["by"]}')
        for k, v in s["levels"].items():
            print(f'   {k:38s} {f(v)}')
    ua = rep["compounds_with_no_chemistry_in_the_release"]
    print(f'\n-- unmatchable: {ua["measurements"]} measurements / {ua["compounds"]} '
          f'compounds carry no HELM in the release {ua["by_endpoint"]}')
    print('-- endpoint x layout confounding')
    for k, v in rep["endpoint_layout_confounding"]["cells"].items():
        print(f'   {k:52s} {f(v)}')
    cp = rep["unit_conversion_accuracy"]["conversions_present_in_the_audited_corpus"]
    print(f'\n-- conversions exercised: {cp["in_vitro_tables_inspected"]} in-vitro tables; '
          f'{cp["tables_printing_%_control_(need 100-x)"]} need 100-x; '
          f'{cp["tables_printing_dose_in_uM_or_mM_(need x1000)"]} need a dose-unit change')
    print("\n-- secondary fields (extraction vs normalisation)")
    for x in rep["secondary_fields"]:
        print(f'   {x["endpoint"]:11s} {x["field"]:24s} {x["kind"]:13s} '
              f'raw {f(x["raw"])}  vocab {f(x["after_vocabulary"])}')
    if rep["chemistry_normalisation"]:
        print("\n-- chemistry normalisation")
        for k, v in rep["chemistry_normalisation"].items():
            if isinstance(v, dict) and "pct" in v:
                print(f'   {k:28s} {f(v)}')
    sc = rep["corpus_self_check"]
    print(f'\n-- corpus self-check ({sc["tables_compared"]} tables compared both ways)')
    print(f'   tables where OUR corpus is incomplete : {sc["n_corpus_incomplete"]}')
    for t in sc["tables_where_corpus_is_incomplete"]:
        print(f'      {t["patent"]} t{t["table"]} doses only in release: {t["doses_only_in_release"]}')
    for t in sc["tables_with_transposed_dose_column"]:
        print(f'   TRANSPOSED  {t["patent"]} t{t["table"]}: {t["diagnosis"]}')
        print(f'      release doses {t["release_doses"]} vs printed {t["corpus_doses"]}')
    print(f'   tables where the RELEASE is incomplete: {sc["n_release_incomplete"]}')
    for t in sc["tables_where_release_is_incomplete"][:6]:
        print(f'      {t["patent"]} t{t["table"]} doses only in corpus: {t["doses_only_in_corpus"]}')
    fr = rep["filtering_vs_extraction_recall"]
    print(f'\n-- recall decomposition')
    print(f'   filtering recall (QC rules retain)      {f(fr["filtering_recall"])}')
    print(f'   extraction recall (of retained)         {f(fr["extraction_recall_of_retained"])}')
    print(f'   end-to-end                              {f(fr["end_to_end_recall"])}')
    print(f'   unexplained by any documented filter    {f(fr["unexplained_omissions"])}')
    for k, v in fr["by_rule"].items():
        print(f'      {v:6d}  {k}')
    uc = rep["unit_conversion_accuracy"]
    print(f'\n-- unit-conversion accuracy')
    print(f'   % control -> % inhibition orientation   {f(uc["potency_scale_orientation"]["correct"])}')
    print(f'   power-of-ten errors among omissions     '
          f'{uc["power_of_ten_errors"]["n_detected"]}/{uc["power_of_ten_errors"]["of_omissions"]}'
          f'  {uc["power_of_ten_errors"]["by_pattern"]}')
    print("\n-- error taxonomy (largest first)")
    for t in rep["error_taxonomy"]:
        flag = "affects a modelled endpoint" if t["reaches_a_modelled_endpoint"] else "metadata only"
        print(f'   {t["n"]:5d}/{t["of"]:<6d} {t["kind"]:14s} {t["mode"][:58]:58s} {flag}')
        if t["dominant_pattern"]:
            print(f'                          -> {t["dominant_pattern"]}')
    cov = rep["coverage"]
    print(f'\n-- coverage: {cov["omission_modes"]}, '
          f'{cov["n_tables_with_zero_coverage"]} tables with zero coverage '
          f'({cov["measurements_in_zero_coverage_tables"]} measurements)')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--permutations", type=int, default=20)
    a = ap.parse_args()
    summarise(main(a.seed, a.permutations))
    print(f"\nwrote {OUT_JSON}")
