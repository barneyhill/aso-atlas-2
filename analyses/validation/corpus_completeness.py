"""Deterministic completeness check of the human-verified gold corpus.

The gold-standard audit (`gold_standard_audit.py`) measures the released Atlas
against a human-verified re-extraction of four patents. That is only meaningful
if the verified corpus is itself complete and faithful: a measurement the corpus
missed becomes an invisible false negative, and a value the corpus invented
becomes a false discordance charged to the release.

This script establishes that ground truth independently of any model. It re-reads
the USPTO full-text XML for each of the four patents, rebuilds every table as a
literal cell grid, and accounts for every printed cell against the extraction:

    coverage  = of the measurement cells the patent prints, what fraction does
                the corpus carry?
    fidelity  = of the values the corpus carries, what fraction traces back to a
                cell the patent actually prints?

Neither direction involves the LLM, the verifier UI, or the release. A cell is
consumed only if some extracted row for the same table and the same treatment
carries that number, either verbatim or as the documented `100 - x` conversion
that %-control tables require.

Column-level aggregation is what makes the output readable. Whole columns are
legitimately unconsumed: SEQ ID NO, compound-name text, standard deviations the
schema does not model. Those show up as 0% consumed and are classified as
non-measurement columns. A column that is *partially* consumed is the diagnostic
signal, because it means the same column was read for some rows and not others,
which is exactly the multi-column melt failure that produced the t17/t19 defect.

Usage:
    uv run python analyses/validation/corpus_completeness.py
    uv run python analyses/validation/corpus_completeness.py --table US20210038631A1:111
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRAPE = Path.home() / "dphil" / "tox-patent-scrape"
GOLD_RUNS = SCRAPE / "verify" / "data" / "runs"
XML_DIR = SCRAPE / "data" / "step2"
OUT_JSON = ROOT / "data" / "results" / "corpus_completeness.json"

PATENTS = {
    "US20160251655A1": "C9ORF72",
    "US20240026353A1": "SCN2A",
    "US20210038631A1": "IRF4",
    "US20190160090A1": "AGT",
}

# Fields holding a printed measurement, per class. Everything else in a row is
# context the pipeline derives or copies from the caption, and is not expected to
# appear as a cell of the row it annotates.
VALUE_FIELDS = {
    "inhibition": ["Inhibition_pct"],
    "neurotox": ["tolerability_score"],
    "hepatotox": ["ALT_(IU/L)", "AST_(IU/L)", "BIL_(mg/dL)", "BUN_(mg/dL)",
                  "ALB_(g/dL)", "CREA_(mg/dL)", "urine_protein/creatinine_ratio"],
    "sequence": [],
}
ID_FIELD = {
    "inhibition": "Compound ID",
    "neurotox": "treatment_or_ISIS_number",
    "hepatotox": "treatment_or_ISIS_number",
    "sequence": "Compound ID",
}

# Control treatments print under many names and carry no compound number, so they
# cannot be matched by the digit rule.
CONTROL = re.compile(r"^(pbs|utc|ltc|saline|vehicle|untreated|control|no\s*asos?|"
                     r"mock|water|naive)\b", re.I)

# Concentrations as the patent prints them in a column header.
DOSE_TOKEN = re.compile(r"([\d,]+\.?\d*)\s*(nm|um|μm|µm|mm|pm)\b", re.I)
DOSE_TO_NM = {"pm": 1e-3, "nm": 1.0, "um": 1e3, "μm": 1e3, "µm": 1e3, "mm": 1e6}

# Cells that are legitimately not a number: an unmeasurable well, a censored
# value, a footnote marker.
BLANK = re.compile(r"^(n\.?\s*d\.?|n\.?\s*a\.?|nt|-{1,3}|—|\*+|<\s*\d|>\s*\d|)$", re.I)


def ascii_signs(s):
    """USPTO XML writes negative numbers with the typographic minus U+2212 and
    ranges with en/em dashes. A regex anchored on ASCII '-' silently rejects
    every negative cell, which reads as the corpus having invented the value.
    Negative percent inhibition is common (RNA above untreated control), so this
    is not a rare path."""
    return (str(s).replace("−", "-").replace("–", "-")
            .replace("—", "-").replace("‐", "-").replace("­", ""))


def cell_number(s):
    """A cell as a single float, or None if it is not one number."""
    t = ascii_signs(s).strip().replace(",", "")
    if not t or BLANK.match(t):
        return None
    m = re.fullmatch(r"[+-]?\d*\.?\d+", t)
    return float(m.group()) if m else None


def cell_vector(s):
    """A cell as a per-animal score vector, or None. '7, 5, 7' is four numbers;
    '1,000' is one, which is why the thousands-separator form is excluded first."""
    t = ascii_signs(s).strip()
    if not t or re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", t):
        return None
    if not re.search(r"[,;/]", t):
        return None
    parts = [p for p in re.split(r"[\s,;/]+", t) if p]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def norm_val(v):
    """Canonical key for comparing a printed cell to an extracted value."""
    vec = cell_vector(v)
    if vec is not None:
        return tuple(round(x, 6) for x in vec)
    n = cell_number(v)
    return None if n is None else (round(n, 6),)


def parse_grid(xml, num):
    """Every <row> of the table as a list of cell strings.

    USPTO XML writes an empty cell as the self-closing <entry/>, so the capture
    alternation must include it or the columns shift left and every downstream
    column index is wrong. Word separators inside a cell are the em-space entity
    &#x2003;, which is not whitespace to a regex until it is decoded."""
    m = re.search(r'<tables[^>]*num="0*%d"[^>]*>([\s\S]*?)</tables>' % num, xml)
    if not m:
        return None
    rows = []
    for rm in re.finditer(r"<row>([\s\S]*?)</row>", m.group(1)):
        cells = []
        for c in re.findall(r"<entry[^>]*>([\s\S]*?)</entry>|<entry[^>]*/>", rm.group(1)):
            t = unescape(re.sub(r"<[^>]+>", "", c or ""))
            t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
            cells.append(re.sub(r"\s+", " ", t).strip())
        rows.append(cells)
    return rows


def load_xml():
    out = {}
    for pid in PATENTS:
        f = sorted(XML_DIR.glob(f"{pid}-*.XML"))
        if f:
            out[pid] = f[0].read_text(errors="ignore")
    return out


def ids_of(rows, klass):
    """Treatment identifiers the extraction claims for this table."""
    field = ID_FIELD[klass]
    ids = set()
    for r in rows:
        v = str(r.get(field) or "").strip()
        if not v:
            continue
        m = re.search(r"\d{4,7}", v)
        ids.add(m.group() if m else v.lower())
    return ids


def row_identifier(cells, known_ids):
    """Index and key of the cell naming this grid row's treatment, or None.

    Matching against the identifiers the extraction claims (rather than any
    digit run) is deliberate: patents print SEQ ID NO, well counts and dose
    headers as bare integers, and treating those as compound IDs invents data
    rows that were never measurements."""
    for i, c in enumerate(cells):
        m = re.search(r"\d{4,7}", c)
        if m and m.group() in known_ids:
            return i, m.group()
    for i, c in enumerate(cells):
        if CONTROL.match(c) and c.strip().lower() in known_ids:
            return i, c.strip().lower()
    return None


def consumed_keys(rows, klass, ident):
    """Multiset of printed forms the extraction accounts for, for one treatment.

    Both orientations are admitted for inhibition because %-control tables are
    stored as 100 - x by design; requiring the verbatim cell would score every
    correctly converted table as a total miss.

    Returned per concentration, not as one pool. Percent-control and percent-
    inhibition are mirror images, so a compound measured at four doses carries
    both x and 100-x for each of them, and the pooled multiset can absorb a
    mistyped readout using the mirror of a different dose's value. Keying the
    pool on the dose closes that hole: a cell printed under the 40 nM header may
    only be redeemed by a value the extraction filed at 40 nM.
    """
    field = ID_FIELD[klass]
    by_dose = defaultdict(Counter)
    for r in rows:
        v = str(r.get(field) or "").strip()
        m = re.search(r"\d{4,7}", v)
        key = m.group() if m else v.lower()
        if key != ident:
            continue
        d = str(r.get("dosage_nm") or "").strip()
        dose = float(d) if re.fullmatch(r"[\d.]+", d) else None
        for f in VALUE_FIELDS[klass]:
            k = norm_val(r.get(f))
            if k is None:
                continue
            by_dose[dose][k] += 1
            if klass == "inhibition" and len(k) == 1:
                by_dose[dose][(round(100 - k[0], 6),)] += 1
    return by_dose


def header_doses(grid, id_col):
    """Column index -> concentration in nM, read from the table's header rows.

    Only columns whose header states a concentration get an entry; everything
    else falls back to dose-agnostic matching."""
    out = {}
    for cells in grid:
        for j, c in enumerate(cells):
            if j == id_col or j in out:
                continue
            m = DOSE_TOKEN.search(c)
            if m and not cell_number(c):
                out[j] = round(float(m.group(1).replace(",", ""))
                               * DOSE_TO_NM[m.group(2).lower()], 4)
    return out


def check_table(pid, tnum, klass, rows, xml):
    grid = parse_grid(xml, tnum)
    if grid is None:
        return {"error": "table not found in the patent XML"}
    known = ids_of(rows, klass)
    if not known:
        return {"error": "extraction claims no treatment identifier"}

    consumed_pool = {i: consumed_keys(rows, klass, i) for i in known}

    # Pass 1: locate the data rows and collect every candidate cell, without
    # consuming anything yet.
    data = []                       # (identifier, id_col, [(col, key, cell, row)])
    id_cols = Counter()
    raw = []
    for cells in grid:
        found = row_identifier(cells, known)
        if found is None:
            continue
        idx, ident = found
        id_cols[idx] += 1
        raw.append((ident, idx, cells))
    if not raw:
        return {"error": "no grid row names a treatment the extraction claims"}
    id_col = id_cols.most_common(1)[0][0]

    # USPTO drops a cell from a row when the publisher merges or omits one, so a
    # short row shifts every later cell one column left. Aligning short rows on
    # their trailing edge puts the readout back under the readout column; without
    # it a single ragged row reports the table's real measurement column as
    # partially accounted and its SEQ ID column as a lost measurement.
    modal_width = Counter(len(c) for _, _, c in raw).most_common(1)[0][0]
    for ident, idx, cells in raw:
        shift = modal_width - len(cells)
        cand = []
        for j, c in enumerate(cells):
            if j == idx:
                continue
            k = norm_val(c)
            if k is None:
                continue
            cand.append((j + shift if (shift and j > idx) else j, k, c, cells))
        data.append((ident, idx, cand))

    # Pass 2: rank columns by affinity, the share of their cells that appear
    # anywhere in the matching treatment's extracted values. Consumption order
    # matters because the multiset is finite: a SEQ ID column that happens to
    # print 100 will otherwise steal the credit for a real 100% inhibition cell
    # in a later column, and the genuine measurement column then reads as
    # partially accounted. Ranking by affinity lets real measurement columns
    # claim their values first.
    # Where the column header names a concentration, a cell may only be redeemed
    # by a value the extraction filed at that same concentration. Elsewhere (a
    # biomarker column, an unlabelled readout) any dose is admissible.
    col_dose = header_doses(grid, id_col)

    def pool_for(ident, j):
        p = consumed_pool[ident]
        d = col_dose.get(j)
        return p[d] if d is not None and d in p else None

    def avail(ident, j, k, used):
        p = pool_for(ident, j)
        if p is not None:
            return used[(ident, j)][k] < p.get(k, 0)
        return used[(ident, None)][k] < sum(c.get(k, 0) for c in consumed_pool[ident].values())

    aff = defaultdict(lambda: [0, 0])
    for ident, _, cand in data:
        for j, k, _c, _r in cand:
            aff[j][1] += 1
            p = pool_for(ident, j)
            hit = (p.get(k, 0) if p is not None
                   else sum(c.get(k, 0) for c in consumed_pool[ident].values()))
            aff[j][0] += int(hit > 0)
    order = sorted(aff, key=lambda j: (-aff[j][0] / aff[j][1], j))

    per_col = defaultdict(lambda: {"cells": 0, "consumed": 0, "unconsumed_examples": []})
    used = defaultdict(Counter)
    for j in order:
        for ident, _, cand in data:
            for jj, k, c, row in cand:
                if jj != j:
                    continue
                per_col[j]["cells"] += 1
                if avail(ident, j, k, used):
                    used[(ident, j if pool_for(ident, j) is not None else None)][k] += 1
                    per_col[j]["consumed"] += 1
                elif len(per_col[j]["unconsumed_examples"]) < 4:
                    per_col[j]["unconsumed_examples"].append(
                        {"treatment": ident, "cell": c, "row": row[:9],
                         "column_dose_nm": col_dose.get(j)})

    ids_seen = {ident for ident, _, _ in data}
    data_rows = len(data)

    # What counts as a measurement column cannot be decided by how much of it the
    # extraction consumed, because a column that was missed ENTIRELY then looks
    # exactly like a metadata column, gets excluded from its own denominator, and
    # coverage reads 100% for a table with a whole dose arm absent. That is how a
    # dropped 10,000 nM column in SCN2A t97 passed as clean.
    #
    # So the patent's own header decides it: a column headed with a concentration
    # is a measurement column whether or not anything consumed it. Consumption
    # rate is only the fallback for columns whose header names no dose.
    cols, partial = {}, []
    coincidental = 0
    for j, v in sorted(per_col.items()):
        frac = v["consumed"] / v["cells"] if v["cells"] else 0.0
        is_meas = (j in col_dose) or frac >= 0.5
        kind = ("measurement column, fully accounted" if frac == 1.0 else
                "MEASUREMENT COLUMN, PARTIALLY ACCOUNTED" if is_meas else
                "not a modelled measurement column" if frac == 0.0 else
                "not a measurement column (coincidental value matches)")
        cols[j] = {"cells": v["cells"], "consumed": v["consumed"],
                   "pct": round(100 * frac, 1), "is_measurement_column": is_meas,
                   "kind": kind, "unconsumed_examples": v["unconsumed_examples"]}
        if is_meas and frac < 1.0:
            partial.append(j)
        if not is_meas:
            coincidental += v["consumed"]

    meas_cells = sum(c["cells"] for c in cols.values() if c["is_measurement_column"])
    meas_consumed = sum(c["consumed"] for c in cols.values() if c["is_measurement_column"])

    # A grid whose data rows disagree on cell count has shifted columns, so the
    # per-column verdicts above are not reliable for it and it is named as such.
    widths = Counter(len(cand[0][3]) for _, _, cand in data if cand)
    ragged = len(widths) > 1

    # A grid row naming a compound the extraction never claims is a dropped row,
    # which no value-level check can see. Two constraints keep this from firing on
    # everything: the identifier must sit in the column the real data rows use
    # (patents print genomic start and stop sites as bare 5-6 digit integers, and
    # without this every coordinate reads as a lost compound), and the row must
    # carry a number in a column already established as a measurement column
    # (rows whose readout is 'n.d.' have nothing to lose).
    meas_idx = {j for j, c in cols.items() if c["is_measurement_column"]}
    unmatched_grid_ids = []
    for cells in grid:
        if row_identifier(cells, known) is not None or len(cells) <= id_col:
            continue
        m = re.fullmatch(r"\s*(\d{5,7})\s*", cells[id_col])
        if not m or m.group(1) in known:
            continue
        if any(j in meas_idx and (cell_number(x) is not None or cell_vector(x) is not None)
               for j, x in enumerate(cells)):
            unmatched_grid_ids.append({"id": m.group(1), "row": cells[:9]})

    # fidelity: values the corpus holds that no printed cell in the table supports
    grid_keys = Counter()
    for cells in grid:
        for c in cells:
            k = norm_val(c)
            if k is not None:
                grid_keys[k] += 1
    invented = []
    for r in rows:
        for f in VALUE_FIELDS[klass]:
            k = norm_val(r.get(f))
            if k is None:
                continue
            alt = (round(100 - k[0], 6),) if (klass == "inhibition" and len(k) == 1) else None
            if not grid_keys.get(k) and not (alt and grid_keys.get(alt)):
                if len(invented) < 6:
                    invented.append({"treatment": str(r.get(ID_FIELD[klass]))[:20],
                                     "field": f, "value": str(r.get(f))[:40]})

    # Dose fidelity. A dose-series table names its concentrations in the column
    # headers, so every concentration the extraction reports has to be one of
    # them. An extracted dose the patent never prints means a column was read as
    # a concentration arm when it is something else: this is how the IC50 column
    # of SCN2A t52 became a fabricated 2000 nM arm carrying 100 - IC50 as percent
    # inhibition. Coverage cannot see it, because the IC50 cells really are
    # printed; only the header disagrees.
    unprinted_doses = None
    if klass == "inhibition":
        got = set()
        for r in rows:
            v = str(r.get("dosage_nm") or "").strip()
            if re.fullmatch(r"[\d.]+", v):
                got.add(float(v))
        printed = set()
        for cells in grid:
            for c in cells:
                for m in DOSE_TOKEN.finditer(c):
                    printed.add(round(float(m.group(1).replace(",", ""))
                                      * DOSE_TO_NM[m.group(2).lower()], 4))
        if len(got) >= 2 and printed:
            extra = sorted(x for x in got
                           if not any(abs(x - p) <= max(0.01, 1e-3 * p) for p in printed))
            if extra:
                unprinted_doses = {"extracted_but_not_printed": extra,
                                   "printed_in_headers": sorted(printed)[:12],
                                   "rows_affected": sum(
                                       1 for r in rows
                                       if str(r.get("dosage_nm") or "").strip()
                                       and re.fullmatch(r"[\d.]+", str(r.get("dosage_nm")).strip())
                                       and float(str(r.get("dosage_nm")).strip()) in set(extra))}

    # Identifier fidelity. The compound number is the join key to the release, so
    # reading it from the wrong column silently detaches every measurement in the
    # table from everything else while leaving all the values correct. A column of
    # 5-7 digit integers sitting to the LEFT of the one the extraction used is the
    # signature: patents print ISIS NO first, then genomic start and stop sites.
    wrong_id_col = None
    for j in range(id_col):
        n = sum(1 for _i, _idx, cand in data
                for jj, _k, c, _r in cand
                if jj == j and re.fullmatch(r"\s*\d{5,7}\s*", c))
        if n >= 0.8 * data_rows:
            distinct = len({c for _i, _idx, cand in data for jj, _k, c, _r in cand
                            if jj == j})
            if distinct >= 0.8 * data_rows:
                wrong_id_col = {
                    "identifier_column_used": id_col,
                    "unused_identifier_column": j,
                    "rows": data_rows,
                    "diagnosis": ("the extraction keyed on a column to the right of "
                                  "an unused column of compound-shaped integers"),
                }
                break

    return {
        "patent": pid, "table": tnum, "class": klass,
        "extracted_rows": len(rows),
        "grid_data_rows": data_rows,
        "treatments_extracted": len(known),
        "treatments_found_in_grid": len(ids_seen),
        "treatments_not_found_in_grid": sorted(known - ids_seen)[:8],
        "measurement_cells": meas_cells,
        "measurement_cells_accounted": meas_consumed,
        "coverage_pct": round(100 * meas_consumed / meas_cells, 2) if meas_cells else None,
        "partially_accounted_columns": partial,
        "coincidental_matches_in_metadata_columns": coincidental,
        "ragged_grid": ragged,
        "identifier_column_suspect": wrong_id_col,
        "doses_not_printed_in_headers": unprinted_doses,
        "columns": cols,
        "grid_rows_naming_an_unextracted_compound": unmatched_grid_ids[:8],
        "n_grid_rows_naming_an_unextracted_compound": len(unmatched_grid_ids),
        "values_with_no_printed_cell": invented,
        "n_values_with_no_printed_cell": len(invented),
    }


def main(only=None):
    xmls = load_xml()
    results = []
    for f in sorted(GOLD_RUNS.glob("*.json")):
        pid, tnum = f.stem.split("_t")
        if pid not in PATENTS or pid not in xmls:
            continue
        tnum = int(tnum)
        if only and (pid, tnum) != only:
            continue
        d = json.loads(f.read_text())
        klass, rows = d.get("klass"), d.get("rows") or []
        # Only signed-off measurement tables belong to the Gold corpus.  Run
        # files can retain stale provisional classes after the classification
        # ledger moves a table to ``skip`` (the twelve IRF4 proliferation
        # tables are the current example); treating those review files as
        # inhibition would enlarge the audit universe incorrectly.
        if d.get("status") != "verified":
            continue
        if klass not in VALUE_FIELDS or klass == "sequence" or not rows:
            continue
        r = check_table(pid, tnum, klass, rows, xmls[pid])
        r["status"] = d.get("status")
        results.append(r)

    ok = [r for r in results if "error" not in r]
    flagged = [r for r in ok
               if r["partially_accounted_columns"]
               or r["n_grid_rows_naming_an_unextracted_compound"]
               or r["n_values_with_no_printed_cell"]
               or r["treatments_not_found_in_grid"]
               or r["identifier_column_suspect"]
               or r["doses_not_printed_in_headers"]]
    tot_cells = sum(r["measurement_cells"] for r in ok)
    tot_acc = sum(r["measurement_cells_accounted"] for r in ok)

    report = {
        "method": ("every table of the four audited patents rebuilt from USPTO "
                   "full-text XML as a literal cell grid, then each printed cell "
                   "accounted for against the verified extraction. No model, no "
                   "verifier UI, no release involved."),
        "tables_checked": len(ok),
        "tables_not_found_in_xml": [r for r in results if "error" in r],
        "measurement_cells_printed": tot_cells,
        "measurement_cells_accounted": tot_acc,
        "coverage_pct": round(100 * tot_acc / tot_cells, 3) if tot_cells else None,
        "tables_clean": len(ok) - len(flagged),
        "tables_flagged": len(flagged),
        "flagged": sorted(flagged, key=lambda r: (r["coverage_pct"] or 0)),
        "by_status": dict(Counter(r["status"] for r in ok)),
        "all_tables": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    return report


def summarise(rep):
    print(f'\n{rep["tables_checked"]} tables rebuilt from patent XML  '
          f'({rep["by_status"]})')
    print(f'printed measurement cells accounted for: '
          f'{rep["measurement_cells_accounted"]:,}/{rep["measurement_cells_printed"]:,} '
          f'= {rep["coverage_pct"]}%')
    print(f'clean tables {rep["tables_clean"]}   flagged {rep["tables_flagged"]}')
    if rep["tables_not_found_in_xml"]:
        print(f'  NOT FOUND IN XML: {len(rep["tables_not_found_in_xml"])}')
    for r in rep["flagged"]:
        bits = []
        if r["identifier_column_suspect"]:
            bits.append("WRONG IDENTIFIER COLUMN")
        if r["doses_not_printed_in_headers"]:
            bits.append(f'UNPRINTED DOSE '
                        f'{r["doses_not_printed_in_headers"]["extracted_but_not_printed"]}')
        if r["partially_accounted_columns"]:
            bits.append(f'partial cols {r["partially_accounted_columns"]}')
        if r["ragged_grid"]:
            bits.append("ragged grid")
        if r["n_grid_rows_naming_an_unextracted_compound"]:
            bits.append(f'{r["n_grid_rows_naming_an_unextracted_compound"]} dropped rows')
        if r["n_values_with_no_printed_cell"]:
            bits.append(f'{r["n_values_with_no_printed_cell"]} unsourced values')
        if r["treatments_not_found_in_grid"]:
            bits.append(f'{len(r["treatments_not_found_in_grid"])} treatments not in grid')
        print(f'  {r["patent"]} t{r["table"]:<5d} {r["class"]:11s} {str(r["status"]):9s} '
              f'cov {str(r["coverage_pct"]):>6s}%  ' + "; ".join(bits))


def self_test():
    """Negative control: inject each defect class into a clean table and confirm
    the check fires.

    A completeness check that reports 100% is worthless unless it can be shown to
    fail on data that is actually broken, and every defect below is one the four
    patents really produced at some point: a dropped table row, a mistyped
    readout, the identifier taken from the wrong column, and a column read as a
    concentration arm the patent never prints."""
    xmls = load_xml()
    pid, tnum, klass = "US20210038631A1", 111, "inhibition"
    base = json.loads((GOLD_RUNS / f"{pid}_t{tnum:05d}.json").read_text())["rows"]

    clean = check_table(pid, tnum, klass, base, xmls[pid])
    cases = [("baseline, untouched", base, None)]

    drop = [r for r in base if str(r.get("Compound ID")) != "935762"]
    cases.append(("a compound's rows deleted", drop, "treatments_not_found_in_grid"))

    typo = [dict(r) for r in base]
    typo[0]["Inhibition_pct"] = str(float(typo[0]["Inhibition_pct"]) + 3)
    cases.append(("one readout mistyped", typo, "coverage_pct"))

    shifted = [dict(r, **{"Compound ID": "99" + str(r["Compound ID"])[2:]}) for r in base]
    cases.append(("identifiers replaced wholesale", shifted, "any"))

    fake = [dict(r) for r in base] + [dict(base[0], dosage_nm="7777")]
    cases.append(("a dose the patent never prints", fake, "doses_not_printed_in_headers"))

    print(f"negative control on {pid} t{tnum} "
          f"({clean['measurement_cells']} measurement cells)\n")
    ok = True
    for label, rows, _expect in cases:
        r = check_table(pid, tnum, klass, rows, xmls[pid])
        if "error" in r:
            caught = True
            detail = r["error"]
        else:
            flags = []
            if (r["coverage_pct"] or 100) < 100:
                flags.append(f'coverage {r["coverage_pct"]}%')
            if r["treatments_not_found_in_grid"]:
                flags.append("treatment missing from grid")
            if r["n_grid_rows_naming_an_unextracted_compound"]:
                flags.append(f'{r["n_grid_rows_naming_an_unextracted_compound"]} dropped rows')
            if r["n_values_with_no_printed_cell"]:
                flags.append(f'{r["n_values_with_no_printed_cell"]} unsourced values')
            if r["identifier_column_suspect"]:
                flags.append("wrong identifier column")
            if r["doses_not_printed_in_headers"]:
                flags.append("unprinted dose")
            caught = bool(flags)
            detail = "; ".join(flags) or "no defect reported"
        expected = label != "baseline, untouched"
        good = caught == expected
        ok &= good
        print(f'  {"PASS" if good else "FAIL"}  {label:34s} -> {detail}')
    print(f'\nnegative control {"passed" if ok else "FAILED"}')
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", help="PID:NUM to check a single table verbosely")
    ap.add_argument("--self-test", action="store_true",
                    help="inject known defects and confirm each is detected")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if self_test() else 1)
    if a.table:
        pid, num = a.table.split(":")
        rep = main((pid, int(num)))
        print(json.dumps(rep["all_tables"], indent=2, default=str))
    else:
        summarise(main())
        print(f"\nwrote {OUT_JSON}")
