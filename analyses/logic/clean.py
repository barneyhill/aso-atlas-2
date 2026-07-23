"""
Clean raw oligostack CSVs and produce processed parquets.

Reads in_vitro_inhibition, dose_response (nM + uM), neurotoxicity, and
hepatotoxicity from data/oligostack/raw/ and writes cleaned, deduplicated
parquets to data/oligostack/processed/.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.utils.helm import Helm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parents[2]
DATA_RAW = _root / "data/oligostack/raw"
DATA_PROCESSED = _root / "data/oligostack/processed"


def _filter_helm(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop rows with uncertain, too-short, uniformly-modified, or homopolymer HELM annotations."""
    n_before = len(df)

    # 1. Uncertain chemistry: HELM contains '?'
    has_q = df["HELM Annotation"].str.contains(r"\?", na=False)

    # 2. Parse valid HELMs for length, sugar, and base checks
    helm_cache: dict = {}
    def _parse(h):
        if h not in helm_cache:
            helm_cache[h] = Helm.parse(h) if pd.notna(h) else None
        return helm_cache[h]

    parsed = df["HELM Annotation"].map(_parse)
    too_short = parsed.map(lambda p: p is not None and p.length <= 10)
    no_dna_gap = parsed.map(lambda p: p is not None and "DNA" not in p.sugars)
    monomer = parsed.map(lambda p: p is not None and len(set(p.bases)) == 1)
    naked_dna = parsed.map(lambda p: p is not None and all(s == "DNA" for s in p.sugars))

    drop = has_q | too_short | no_dna_gap | monomer | naked_dna
    result = df[~drop].copy()
    n_dropped = n_before - len(result)
    print(f"  {label}: dropped {n_dropped:,} rows "
          f"(?={has_q.sum():,}, short={too_short.sum():,}, "
          f"no-gap={no_dna_gap.sum():,}, monomer={monomer.sum():,}, "
          f"naked-DNA={naked_dna.sum():,})")
    return result


# ---------------------------------------------------------------------------
# CCLE cell line enrichment helpers
# ---------------------------------------------------------------------------
_CCLE_FILE = _root / "data/CCLE_celllines.csv"
_CELL_LINE_MANUAL_MAPPINGS_FILE = DATA_PROCESSED / "cell_line_manual_mappings.json"

_PRIMARY_CELL_PATTERNS = [
    r"primary.*hepatocyte", r"hepatocyte.*primary", r"mouse hepatocyte",
    r"hPHH", r"PHH", r"cryopreserved.*hepatocyte", r"HepatoPac",
    r"iPSC", r"IPS.*cell", r"IPS.*neuron", r"iCell", r"ReproNeuro",
    r"HUVEC", r"primary.*cell", r"primary.*neuron", r"primary.*culture",
    r"human primary", r"mouse primary", r"rat primary",
    r"transgenic.*hepatocyte", r"^CD4.*T.*cell", r"T-reg", r"^T-cell",
    r"hSKMC", r"hSKMc", r"HSMM", r"human.*tendon",
    r"differentiated.*adipocyte", r"^GM\d{4,5}", r"^GMO\d{4,5}",
    r"^F\d{2}-\d{3}", r"^SCA\d+-\d+", r"DM1 fibroblast", r"Steinert DM1",
    r"^HBE[s]?$", r"^3T3", r"NIH 3T3", r"^4T1$", r"^AML12$", r"^B16",
    r"^EMT-6$", r"^HEPA 1-6$", r"^MH-S$", r"^P388D1$", r"^RAW 264",
    r"^TCMK", r"^b\.END$", r"BACHD", r"^COS-7$", r"^LLC-MK", r"^NRK$",
    r"^Vero", r"^4MBr", r"^HEK.*\(", r"HEK-SORT1", r"GLP1R HEK",
    r"^HepAD38$", r"^HepG2\.2\.15$", r"^HepaRG$",
    r"Angelman.*neuron", r"Ube3a.*neuron", r"Ube3a.*YFP",
    r"human neuron", r"^ThioMac$",
]


def _normalize_cell_line_name(name: str) -> str:
    if name is None:
        return ""
    result = str(name).upper()
    for char in ['-', ' ', '.', ':', '/', '(', ')', ',', '_']:
        result = result.replace(char, '')
    return result


def _classify_cell_line(name: str) -> str | None:
    if name is None:
        return None
    for pattern in _PRIMARY_CELL_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return "primary"
    return None


def _load_ccle_lookup() -> dict[str, dict]:
    df = pd.read_csv(_CCLE_FILE)
    lookup = {}
    for _, row in df.iterrows():
        stripped = row.get("StrippedCellLineName")
        if pd.notna(stripped):
            key = _normalize_cell_line_name(stripped)
            lookup[key] = {
                "ccle_cell_line_name": row.get("CellLineName"),
                "ccle_model_id": row.get("ModelID"),
                "ccle_oncotree_lineage": row.get("OncotreeLineage"),
                "ccle_oncotree_disease": row.get("OncotreePrimaryDisease"),
            }
    return lookup


def _lookup_cell_line(name, ccle_lookup, manual_mappings):
    if name is None:
        return None, "unmapped"
    if name in manual_mappings:
        mapped = manual_mappings[name]
        if mapped:
            norm = _normalize_cell_line_name(mapped)
            if norm in ccle_lookup:
                return ccle_lookup[norm], "manual"
        return None, "manual"
    norm = _normalize_cell_line_name(name)
    if norm in ccle_lookup:
        return ccle_lookup[norm], "ccle_exact"
    prefix = [(k, v) for k, v in ccle_lookup.items() if k.startswith(norm) and len(norm) >= 3]
    if len(prefix) == 1:
        return prefix[0][1], "ccle_prefix"
    return None, "unmapped"


def _enrich_ccle(df: pd.DataFrame, ccle_lookup: dict, manual_mappings: dict) -> pd.DataFrame:
    """Add CCLE columns to a DataFrame based on its cell_line column."""
    cell_lines = df["cell_line"].unique()
    mappings: dict[str, dict] = {}
    for cl in cell_lines:
        if cl is None or (isinstance(cl, float) and pd.isna(cl)):
            mappings[cl] = {
                "ccle_cell_line_name": None, "ccle_model_id": None,
                "ccle_oncotree_lineage": None, "ccle_oncotree_disease": None,
                "cell_line_mapping_source": "unmapped",
            }
            continue
        classification = _classify_cell_line(cl)
        if classification == "primary":
            mappings[cl] = {
                "ccle_cell_line_name": None, "ccle_model_id": None,
                "ccle_oncotree_lineage": None, "ccle_oncotree_disease": None,
                "cell_line_mapping_source": "primary",
            }
            continue
        ccle_data, source = _lookup_cell_line(cl, ccle_lookup, manual_mappings)
        if ccle_data:
            mappings[cl] = {**ccle_data, "cell_line_mapping_source": source}
        else:
            mappings[cl] = {
                "ccle_cell_line_name": None, "ccle_model_id": None,
                "ccle_oncotree_lineage": None, "ccle_oncotree_disease": None,
                "cell_line_mapping_source": source,
            }
    df = df.copy()
    for col in ["ccle_cell_line_name", "ccle_model_id", "ccle_oncotree_lineage",
                 "ccle_oncotree_disease", "cell_line_mapping_source"]:
        df[col] = df["cell_line"].map(lambda x, c=col: mappings.get(x, {}).get(c))
    return df


def main():
    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    in_vitro_inhibition_df = pd.read_csv(DATA_RAW / "in_vitro_inhibition_collation_results.csv")

    dose_response_nm_df = pd.read_csv(DATA_RAW / "dose_response_nm_collation_results.csv")
    dose_response_um_df = pd.read_csv(DATA_RAW / "dose_response_um_collation_results.csv")
    dose_response_um_df['dosage'] = dose_response_um_df['dosage'] * 1000
    dose_response_df = pd.concat([dose_response_nm_df, dose_response_um_df])
    dose_response_df.rename(columns={'dosage': 'dosage_nm'}, inplace=True)
    del dose_response_nm_df, dose_response_um_df

    neurotoxicity_df = pd.read_csv(DATA_RAW / "neurotox_collation_results.csv")
    hepatictoxicity_df = pd.read_csv(DATA_RAW / "hepatictox_collation_results.csv")

    # ------------------------------------------------------------------
    # 1b. HELM quality filters (uncertain, short, all-MOE)
    # ------------------------------------------------------------------
    print("HELM quality filters:")
    in_vitro_inhibition_df = _filter_helm(in_vitro_inhibition_df, "in_vitro")
    dose_response_df = _filter_helm(dose_response_df, "dose_response")
    neurotoxicity_df = _filter_helm(neurotoxicity_df, "neurotoxicity")
    hepatictoxicity_df = _filter_helm(hepatictoxicity_df, "hepatic")

    # ------------------------------------------------------------------
    # 1c. Gene standardization (target_RNA)
    # ------------------------------------------------------------------
    with open(DATA_PROCESSED / "manual_mappings.json") as f:
        gene_mappings = json.load(f)
    for df_label, df_ref in [
        ("in_vitro", in_vitro_inhibition_df),
        ("dose_response", dose_response_df),
    ]:
        if "target_RNA" not in df_ref.columns:
            continue
        before = df_ref["target_RNA"].nunique()
        df_ref["target_RNA"] = df_ref["target_RNA"].replace(gene_mappings)
        after = df_ref["target_RNA"].nunique()
        remapped = before - after
        if remapped > 0:
            print(f"  {df_label}: remapped {remapped} target_RNA aliases")
    print(f"Gene standardization: applied {len(gene_mappings)} mappings")

    # ------------------------------------------------------------------
    # 2. Clean in vitro inhibition
    # ------------------------------------------------------------------
    clean_in_vitro_inhibition_df = in_vitro_inhibition_df[in_vitro_inhibition_df['Inhibition_pct'].between(-1000, 100)]
    clean_in_vitro_inhibition_df['cell_line_species'] = (
        clean_in_vitro_inhibition_df['cell_line_species']
        .replace({'Cynomolgus': 'monkey', 'cynomolgus': 'monkey', 'rhesus': 'monkey'})
    )

    def standardise_cell_line(cell_line):
        """Standardise cell line spellings to canonical names."""
        mapping = {
            'HepG2.2.15': 'HepG2.2.15',
            'HepG2': 'HepG2',
            'A431': 'A431',
            'A-431': 'A431',
            'A-549': 'A549',
            'A549': 'A549',
            'A459': 'A549',
            'Hep3B': 'Hep3B',
            'HepB3': 'Hep3B',
            'SH-SY5Y': 'SH-SY5Y',
            'SH-SY-5Y': 'SH-SY5Y',
            'SK-MEL-28': 'SK-MEL-28',
            'T-24': 'T24',
            'T24': 'T24',
            'HuVEC': 'HUVEC',
            'HUVEC': 'HUVEC',
            'LNCaP': 'LNCaP',
            'U251': 'U251',
            'iCell cardiomyocytes': 'iCell cardiomyocytes',
            'iCell cardiomyocytes2': 'iCell cardiomyocytes',
            'VCaP': 'VCaP',
            'iCell GABANeurons': 'iCell GABANeurons',
            'Huh7': 'Huh7',
            'HepaRG': 'HepaRG',
            'THP-1': 'THP-1',
            'MM.1R': 'MM.1R',
            'SNU-449': 'SNU-449',
            '54-2': '54-2',
            'LLC-MK2': 'LLC-MK2',
            'K-562': 'K-562',
            'hSKMc': 'hSKMc',
            'hSKMC': 'hSKMc',
            'hSKM': 'hSKMc',
            'HepAD38': 'HepAD38',
            'primary hepatocytes': 'primary hepatocytes',
            'primary human hepatocytes': 'primary human hepatocytes',
            'primary transgenic mouse hepatocytes': 'primary transgenic mouse hepatocytes',
            'primary hamster hepatocytes': 'primary hamster hepatocytes',
            'primary rabbit hepatocytes': 'primary rabbit hepatocytes',
            'primary mouse hepatocytes': 'primary mouse hepatocytes',
            'mouse primary hepatocytes': 'mouse primary hepatocytes',
            'Mouse primary hepatocyte': 'mouse primary hepatocytes',
            'mouse primary hepatocyte': 'mouse primary hepatocytes',
            'mouse primary hepatocyte cells': 'mouse primary hepatocytes',
            'Rat primary hepatocytes': 'rat primary hepatocytes',
            'rat primary hepatocytes': 'rat primary hepatocytes',
            'A172': 'A172',
            'A10': 'A10',
            'HEPA 1-6': 'HEPA 1-6',
            'COS-7': 'COS-7',
            'HeLa': 'HeLa',
            'Hela': 'HeLa',
            '3T3-L1': '3T3-L1',
            '4T1': '4T1',
            'ReproNeuro Neurons': 'ReproNeuro Neurons',
            'Human IPS cell derived ReproNeuro Neurons': 'ReproNeuro Neurons',
            'Primary Cynomolgus hepatocytes': 'Primary Cynomolgus hepatocytes',
            'b.END': 'b.END',
            'bEND': 'b.END',
            'B16-F10': 'B16-F10',
            'MCF7': 'MCF7',
            'PC3': 'PC3',
            'P388D1': 'P388D1',
            'EMT-6': 'EMT-6',
            'SW872': 'SW872',
            'Jurkat': 'Jurkat',
            'SKOV3': 'SKOV3',
            'Vero C1008': 'Vero C1008',
            'NRK': 'NRK',
            'HK-2': 'HK-2',
            'RAW 264.7': 'RAW 264.7',
            'U266': 'U266',
            'G-361': 'G-361',
            'TCMK-1': 'TCMK-1',
            'Ts 24': 'T24',
            'transgenic primary mouse hepatocytes': 'primary transgenic mouse hepatocytes',
            'transgenic mouse primary hepatocytes': 'primary transgenic mouse hepatocytes',
            'Transgenic mouse primary hepatocytes': 'primary transgenic mouse hepatocytes',
            'Primary hepatocytes': 'primary hepatocytes',
            'Primary rat hepatocytes': 'rat primary hepatocytes',
            'Primary hepatocytes from human apo(a) transgenic mice': 'primary hepatocytes',
            'differentiated human adipocytes': 'differentiated human adipocytes',
            'AML12': 'AML12',
        }
        name = str(cell_line).replace('\u2013', '-').replace('_', '-').replace('.', '.').strip()
        lower = name.lower().replace('-', '').replace(' ', '').replace('.', '')
        reverse = {k.lower().replace('-', '').replace(' ', '').replace('.', ''): v for k, v in mapping.items()}
        return reverse.get(lower, cell_line)

    clean_in_vitro_inhibition_df['cell_line'] = clean_in_vitro_inhibition_df['cell_line'].map(standardise_cell_line)

    # ------------------------------------------------------------------
    # 3. Deduplicate in vitro inhibition
    # ------------------------------------------------------------------
    dedup_clean_in_vitro_inhibition_df = (
        clean_in_vitro_inhibition_df
        .sort_values('USPTO ID', ascending=False)
        .drop_duplicates(subset=['Compound ID', 'Inhibition_pct'], keep='first')
        .reset_index(drop=True)
    )
    dedup_clean_in_vitro_inhibition_df.to_parquet(DATA_PROCESSED / 'in_vitro_inhibition_processed.parquet')
    print(f"Saved in_vitro_inhibition_processed.parquet ({len(dedup_clean_in_vitro_inhibition_df):,} rows)")

    # ------------------------------------------------------------------
    # 4. Clean dose response
    # ------------------------------------------------------------------
    clean_dose_response_df = dose_response_df[dose_response_df['Inhibition_pct'].between(-1000, 100)]
    clean_dose_response_df['cell_line_species'] = (
        clean_dose_response_df['cell_line_species']
        .replace({'rhesus': 'monkey', 'cynomolgus': 'monkey', 'cyno': 'monkey', 'cyano': 'monkey', 'cynomolgous': 'monkey'})
    )
    clean_dose_response_df = clean_dose_response_df[clean_dose_response_df['dosage_nm'] > 0]

    canonical_cell_lines = {
        'A431': 'A431', 'A-431': 'A431',
        'HepG2': 'HepG2', 'HEPG2': 'HepG2', 'Hep G2': 'HepG2', 'HEP-G2': 'HepG2', 'HepG2s': 'HepG2',
        'Hep3B': 'Hep3B', 'HepB3': 'Hep3B', 'Hep-3B': 'Hep3B',
        'SH-SY5Y': 'SH-SY5Y', 'SHSY5Y': 'SH-SY5Y', 'SH-SY-5Y': 'SH-SY5Y',
        'HUVEC': 'HUVEC', 'HUVECs': 'HUVEC',
        'HepaRG': 'HepaRG', 'Heparg': 'HepaRG',
        'THP-1': 'THP-1', 'THP1': 'THP-1',
        'SK-MEL-28': 'SK-MEL-28', 'SKMEL28': 'SK-MEL-28',
        'A549': 'A549',
        'Huh7': 'Huh7', 'HUH-7': 'Huh7',
        'iCell cardiomyocytes2': 'iCell cardiomyocytes2',
        'iCell cardiomyocytes': 'iCell cardiomyocytes',
        'primary hepatocytes': 'primary hepatocytes', 'Primary hepatocytes': 'primary hepatocytes',
        'SUP-M2': 'SUP-M2', 'SUPM2': 'SUP-M2',
        'MM.1R': 'MM.1R', 'MM1R': 'MM.1R',
        'U251': 'U251', 'U-251': 'U251', 'U251-MG': 'U251-MG',
        'iCell GABANeurons': 'iCell GABANeurons', 'iCell GABA Neurons': 'iCell GABANeurons',
        'HepG2.2.15': 'HepG2.2.15', 'HEPG2.2.15': 'HepG2.2.15',
        'LLC-MK2': 'LLC-MK2', 'LLCMK2': 'LLC-MK2',
        'VCaP': 'VCaP',
        'GM04281': 'GM04281',
        'transgenic mouse primary hepatocytes': 'transgenic mouse primary hepatocytes',
        'primary mouse hepatocytes': 'primary mouse hepatocytes',
        'SNU-449': 'SNU-449', 'SNU449': 'SNU-449',
        'HEK293': 'HEK293', 'HEK-293': 'HEK293',
        'GM02171': 'GM02171',
        'Primary human hepatocytes': 'Primary human hepatocytes',
        'primary human hepatocytes': 'Primary human hepatocytes',
        '54-2': '54-2',
        'cynomolgus primary hepatocytes': 'cynomolgus primary hepatocytes',
        'hSKMC': 'hSKMC',
        'GM02173B': 'GM02173B',
        'b.END': 'b.END', 'bEND': 'b.END',
        'mouse primary hepatocytes': 'mouse primary hepatocytes',
        '4T1': '4T1',
        'MDA-MB-436': 'MDA-MB-436', 'MDA MB 436': 'MDA-MB-436',
        'CD4 T-cells': 'CD4 T-cells',
        'CD4+ T-cells': 'CD4+ T-cells', 'Human CD4+ T-cells': 'CD4+ T-cells',
        'Angelman IPS-derived neurons': 'Angelman IPS-derived neurons',
        'primary rat hepatocytes': 'primary rat hepatocytes',
        'LNCaP': 'LNCaP',
        'T-reg': 'T-reg',
        'KARPAS-229': 'KARPAS-229',
        'SK-BR-3': 'SK-BR-3', 'SKBR3': 'SK-BR-3',
        'SCA2-04': 'SCA2-04',
        'K-562': 'K-562', 'K562': 'K-562',
        'HeLa': 'HeLa', 'hela': 'HeLa',
        'C4-2B': 'C4-2B', 'C4-2': 'C4-2',
        'KMS11': 'KMS11',
        'HepatoPac': 'HepatoPac',
        'HSMM': 'HSMM',
        'NCI-H460': 'NCI-H460', 'NCIH460': 'NCI-H460',
        'HBE': 'HBE',
        'T-24': 'T-24',
        'A10': 'A10',
        'cynomolgus monkey primary hepatocytes': 'cynomolgus monkey primary hepatocytes',
        'cyno monkey primary hepatocytes': 'cynomolgus monkey primary hepatocytes',
        'DM1 fibroblast': 'DM1 fibroblast',
        'Steinert DM1': 'Steinert DM1',
        'Rhesus monkey primary hepatocytes': 'Rhesus monkey primary hepatocytes',
        'human tendon cells': 'human tendon cells',
        'GM04022': 'GM04022',
        'RAW 264.7': 'RAW 264.7', 'RAW264.7': 'RAW 264.7',
        'C4-2B MR': 'C4-2B MR',
        'MDA-MB-231': 'MDA-MB-231', 'MDA MB 231': 'MDA-MB-231',
        'BACHD mouse hepatocytes': 'BACHD mouse hepatocytes', 'BACHD': 'BACHD',
        'MH-S': 'MH-S',
        'SCC25': 'SCC25', 'SCC-25': 'SCC25',
        'SW872': 'SW872',
        'B16-F10': 'B16-F10', 'B16F10': 'B16-F10',
        'F09-152': 'F09-152', 'F09-229': 'F09-229',
        'Ube3a-YFP primary neurons': 'Ube3a-YFP primary neurons',
        'Ube3a-YFP primary neuronal cultures': 'Ube3a-YFP primary neuronal cultures',
        'Primary neuronal cultures': 'Primary neuronal cultures',
        'GM04478': 'GM04478',
        'PiZ primary hepatocytes': 'PiZ primary hepatocytes',
        'Ube3a-YFP': 'Ube3a-YFP',
        'A172': 'A172',
        'H929': 'H929',
        'IPS-cell derived neurons': 'IPS-cell derived neurons',
        'U266': 'U266',
        'NIH 3T3': 'NIH 3T3', '3T3-L1': '3T3-L1',
        '4MBr-5': '4MBr-5',
        'Vero C1008': 'Vero C1008',
        'CaCo-2': 'CaCo-2', 'Caco2': 'CaCo-2',
        'U87MG': 'U87MG',
        'MCF7': 'MCF7',
        'U2932': 'U2932',
        'primary rabbit hepatocytes': 'primary rabbit hepatocytes',
        'primary hamster hepatocytes': 'primary hamster hepatocytes',
        'SCC-4': 'SCC-4',
        'Granta519': 'Granta519',
        'Mino': 'Mino',
        'primary monkey hepatocytes': 'primary monkey hepatocytes',
        'Z138': 'Z138',
        'mouse hepatocytes': 'mouse hepatocytes',
        'CWR22-RV1': 'CWR22-RV1',
        'PC9': 'PC9',
        'CAL33': 'CAL33',
        'MAVER1': 'MAVER1',
        'BICR56': 'BICR56',
        'JVM2': 'JVM2',
        'CAL27': 'CAL27',
        'COS-7': 'COS-7',
        'Detroit-562': 'Detroit-562',
        'UPCI:SCC090': 'UPCI:SCC090',
        'NCI-H747': 'NCI-H747',
        'BICR-22': 'BICR-22',
        'NCI-H292': 'NCI-H292',
        'SCC-9': 'SCC-9',
        'Colo201': 'Colo201',
        'SNU-899': 'SNU-899',
        'SNU-1214': 'SNU-1214',
        'SCC-15': 'SCC-15',
        'SNU-1076': 'SNU-1076',
        'SNU-1066': 'SNU-1066',
        'HEK-SORT1': 'HEK-SORT1',
        'GLP1R HEK': 'GLP1R HEK',
        'primary murine hepatocytes': 'primary murine hepatocytes',
        'ThioMac': 'ThioMac',
        'BT474M1': 'BT474M1',
    }

    def clean_cell_line(name):
        if pd.isnull(name):
            return name
        name_orig = str(name).strip()
        key = name_orig.replace('-', '').replace(' ', '').replace('.', '').replace('+', '').lower()
        for alt, canon in canonical_cell_lines.items():
            alt_key = str(alt).replace('-', '').replace(' ', '').replace('.', '').replace('+', '').lower()
            if key == alt_key:
                return canon
        return name_orig

    clean_dose_response_df['cell_line'] = clean_dose_response_df['cell_line'].map(clean_cell_line)

    # ------------------------------------------------------------------
    # 5. Deduplicate dose response
    # ------------------------------------------------------------------
    dedup_clean_dose_response_df = (
        clean_dose_response_df
        .sort_values('USPTO ID', ascending=False)
        .drop_duplicates(subset=['Compound ID', 'dosage_nm', 'Inhibition_pct'], keep='first')
        .reset_index(drop=True)
    )
    dedup_clean_dose_response_df.to_parquet(DATA_PROCESSED / 'dose_response_processed.parquet')
    print(f"Saved dose_response_processed.parquet ({len(dedup_clean_dose_response_df):,} rows)")

    # ------------------------------------------------------------------
    # 5b. CCLE cell line enrichment
    # ------------------------------------------------------------------
    ccle_lookup = _load_ccle_lookup()
    with open(_CELL_LINE_MANUAL_MAPPINGS_FILE) as f:
        ccle_manual = json.load(f)
    print(f"CCLE enrichment: loaded {len(ccle_lookup)} CCLE lines, {len(ccle_manual)} manual mappings")

    dedup_clean_in_vitro_inhibition_df = _enrich_ccle(
        dedup_clean_in_vitro_inhibition_df, ccle_lookup, ccle_manual
    )
    dedup_clean_in_vitro_inhibition_df.to_parquet(DATA_PROCESSED / 'in_vitro_inhibition_processed.parquet')

    dedup_clean_dose_response_df = _enrich_ccle(
        dedup_clean_dose_response_df, ccle_lookup, ccle_manual
    )
    dedup_clean_dose_response_df.to_parquet(DATA_PROCESSED / 'dose_response_processed.parquet')

    sources = dedup_clean_in_vitro_inhibition_df["cell_line_mapping_source"].value_counts()
    print(f"  in_vitro CCLE sources: {dict(sources)}")
    sources = dedup_clean_dose_response_df["cell_line_mapping_source"].value_counts()
    print(f"  dose_response CCLE sources: {dict(sources)}")

    # ------------------------------------------------------------------
    # 6. Clean neurotoxicity
    # ------------------------------------------------------------------
    def process_fob_score_to_list(val):
        def valid_score(x):
            try:
                v = float(x)
                return 0 <= v <= 7
            except Exception:
                return False

        if isinstance(val, str):
            val = val.strip()
            try:
                float_list = [
                    float(x.strip()) for x in val.split(',')
                    if x.strip() != '' and valid_score(x.strip())
                ]
                return float_list if float_list else np.nan
            except Exception:
                return np.nan
        elif isinstance(val, (int, float)):
            return [float(val)] if valid_score(val) else np.nan
        else:
            return np.nan

    clean_neurotoxicity_df = neurotoxicity_df
    clean_neurotoxicity_df['FOB_score'] = neurotoxicity_df['FOB_score'].apply(process_fob_score_to_list)

    species_mapping = {
        'C57/B16 mice': 'C57BL/6 mice',
        'C57/Bl6 mice': 'C57BL/6 mice',
        'C57bl6 mice': 'C57BL/6 mice',
        'C57/B16': 'C57BL/6 mice',
        'wild-type female C57/B16 mice': 'C57BL/6 mice',
        'Wild-type female C57/Bl6 mice': 'C57BL/6 mice',
        'Wild-type female C57/B16 mice': 'C57BL/6 mice',
        'C57BL/6 female mice': 'C57BL/6 mice',
        'C57B16/J female mice': 'C57BL/6 mice',
        'Sprague Dawley rats': 'Sprague-Dawley rats',
        'Sprague Dawley': 'Sprague-Dawley rats',
        'BALB/C mice': 'BALB/c mice',
        'BALB/c wild type mice': 'BALB/c mice',
        'balb/c mice': 'BALB/c mice',
    }
    clean_neurotoxicity_df['species_strain'] = clean_neurotoxicity_df['species_strain'].replace(species_mapping)

    # ------------------------------------------------------------------
    # 6b. Assign target_RNA to neurotoxicity via compound lookup + patent majority vote
    # ------------------------------------------------------------------
    # Build Compound ID → target_RNA lookup from in_vitro + dose_response (already cleaned)
    gene_lookup = pd.concat([
        dedup_clean_in_vitro_inhibition_df[['Compound ID', 'target_RNA']],
        dedup_clean_dose_response_df[['Compound ID', 'target_RNA']],
    ]).dropna(subset=['target_RNA']).drop_duplicates(subset=['Compound ID'], keep='first')
    compound_to_gene = dict(zip(gene_lookup['Compound ID'], gene_lookup['target_RNA']))

    # Direct match
    clean_neurotoxicity_df['target_RNA'] = clean_neurotoxicity_df['Compound ID'].map(compound_to_gene)

    # Patent-level majority vote for unmatched compounds
    matched = clean_neurotoxicity_df['target_RNA'].notna()
    if not matched.all():
        patent_gene_counts = (
            clean_neurotoxicity_df[matched]
            .groupby('USPTO ID')['target_RNA']
            .agg(lambda x: x.value_counts().index[0])
        )
        patent_to_gene = patent_gene_counts.to_dict()
        unmatched_mask = clean_neurotoxicity_df['target_RNA'].isna()
        clean_neurotoxicity_df.loc[unmatched_mask, 'target_RNA'] = (
            clean_neurotoxicity_df.loc[unmatched_mask, 'USPTO ID'].map(patent_to_gene)
        )

    n_with_gene = clean_neurotoxicity_df['target_RNA'].notna().sum()
    print(f"  Neurotox target_RNA assigned: {n_with_gene}/{len(clean_neurotoxicity_df)} "
          f"({100*n_with_gene/len(clean_neurotoxicity_df):.0f}%)")

    # ------------------------------------------------------------------
    # 7. Deduplicate neurotoxicity
    # ------------------------------------------------------------------
    dedup_cols = [
        "HELM Annotation",
        "species",
        "administration_method",
        "tolerability_score_type",
        "dosage_ug",
    ]
    clean_neurotoxicity_df_dedup = (
        clean_neurotoxicity_df
        .drop_duplicates(subset=dedup_cols, keep='first')
        .reset_index(drop=True)
    )
    clean_neurotoxicity_df_dedup.to_parquet(DATA_PROCESSED / 'neurotoxicity_processed.parquet')
    print(f"Saved neurotoxicity_processed.parquet ({len(clean_neurotoxicity_df_dedup):,} rows)")

    # ------------------------------------------------------------------
    # 8. Clean hepatic toxicity
    # ------------------------------------------------------------------
    def split_dosage(dosage):
        if isinstance(dosage, str):
            match = re.match(r"\s*([\d\.]+)\s*(.*)", dosage)
            if match:
                try:
                    value = float(match.group(1))
                except Exception:
                    value = None
                unit = match.group(2).strip() if match.group(2) else None
                return value, unit
        return None, None

    clean_hepatictoxicity_df = hepatictoxicity_df
    clean_hepatictoxicity_df[['dosage_value', 'dosage_unit']] = clean_hepatictoxicity_df['dosage'].apply(
        lambda x: pd.Series(split_dosage(x))
    )

    # Convert 'mg/kg/wk' to 'mg/kg'
    mask = clean_hepatictoxicity_df['dosage_unit'] == 'mg/kg/wk'
    valid = (
        mask &
        clean_hepatictoxicity_df['dosing_period_days'].notnull() &
        clean_hepatictoxicity_df['num_doses'].notnull() &
        (clean_hepatictoxicity_df['num_doses'] > 0)
    )

    def convert_mgkgwk_to_mgkg(row):
        try:
            return (row['dosage_value'] * row['dosing_period_days'] / 7) / row['num_doses']
        except Exception:
            return row['dosage_value']

    clean_hepatictoxicity_df.loc[valid, 'dosage_value'] = clean_hepatictoxicity_df[valid].apply(convert_mgkgwk_to_mgkg, axis=1)
    clean_hepatictoxicity_df.loc[mask, 'dosage_unit'] = clean_hepatictoxicity_df.loc[mask, 'dosage_unit'].replace('mg/kg/wk', 'mg/kg')

    clean_hepatictoxicity_df = clean_hepatictoxicity_df[clean_hepatictoxicity_df['dosage_unit'] == 'mg/kg']
    clean_hepatictoxicity_df = clean_hepatictoxicity_df.rename(columns={'dosage_value': 'dosage_mg_per_kg'})
    clean_hepatictoxicity_df = clean_hepatictoxicity_df.drop(columns=['dosage_unit', 'dosage'])

    clean_hepatictoxicity_df = clean_hepatictoxicity_df[clean_hepatictoxicity_df['dosage_mg_per_kg'] <= 1e4]

    clean_hepatictoxicity_df['adminstration_method'] = clean_hepatictoxicity_df['adminstration_method'].replace(
        {'intraperitoneally': 'intraperitoneal'}
    )
    clean_hepatictoxicity_df = clean_hepatictoxicity_df[
        clean_hepatictoxicity_df['adminstration_method'].isin(['intraperitoneal', 'subcutaneous'])
    ]
    clean_hepatictoxicity_df = clean_hepatictoxicity_df[
        clean_hepatictoxicity_df['measurement_source'].isin(['plasma', 'urine'])
    ]

    clean_hepatictoxicity_df['species'] = clean_hepatictoxicity_df['species'].replace({
        'cynomolgus monkey': 'monkey',
        'beagle dog': 'dog',
    })

    biomarker_ranges = {
        "ALB": (1, 100),
        "ALT": (10, 50000),
        "AST": (10, 50000),
    }

    def filter_biomarker_series(biomarker_series, vmin, vmax):
        def filter_list(vals):
            if isinstance(vals, list):
                return [v for v in vals if (pd.isnull(v) or (vmin <= v <= vmax))]
            elif pd.isnull(vals):
                return vals
            else:
                return vals if (vmin <= vals <= vmax) else None
        return biomarker_series.apply(filter_list)

    for biomarker, (vmin, vmax) in biomarker_ranges.items():
        if biomarker in clean_hepatictoxicity_df.columns:
            clean_hepatictoxicity_df[biomarker] = filter_biomarker_series(clean_hepatictoxicity_df[biomarker], vmin, vmax)

    # ------------------------------------------------------------------
    # 9. Collapse hepatic rows
    # ------------------------------------------------------------------
    group_cols = ['Compound ID', 'species', 'num_doses', 'dosing_period_days', 'dosage_mg_per_kg', 'measurement_source', 'adminstration_method']
    biomarker_cols = ['ALB', 'ALT', 'AST', 'BUN', 'CREA', 'TBIL', 'PC_ratio']

    other_cols = [c for c in clean_hepatictoxicity_df.columns if c not in group_cols + biomarker_cols]

    def first_non_nan(s):
        non_null = s.dropna()
        return non_null.iloc[0] if len(non_null) > 0 else None

    agg_dict = {col: lambda x: list(x.dropna()) for col in biomarker_cols}
    agg_dict.update({col: first_non_nan for col in other_cols})

    collapsed_hepatictoxicity_df = (
        clean_hepatictoxicity_df
        .groupby(group_cols, dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 10. Standardize species_strain for hepatic
    # ------------------------------------------------------------------
    species_strain_rename = {
        "CD1 mice": "CD-1 mice",
        "Sprague-Dawley rats": "Sprague-Dawley rats",
        "CD-1 mice": "CD-1 mice",
        "BALB/c mice": "BALB/c mice",
        "BALB/C mice": "BALB/c mice",
        "Tg mice": "Tg mice",
        "Sprague-Dawley": "Sprague-Dawley rats",
        "Sprague Dawley rats": "Sprague-Dawley rats",
        "Balb/c mice": "BALB/c mice",
        "CD1": "CD-1 mice",
        "Sprague-Dawley rat": "Sprague-Dawley rats",
        "C57BL/6 mice": "C57BL/6 mice",
        "PCSK9 transgenic mice": "PCSK9 transgenic mice",
        "male CD-1 mice": "CD-1 mice",
        "Balb/c": "BALB/c mice",
        "male CD1 mice": "CD-1 mice",
        "Male Sprague-Dawley rats": "Sprague-Dawley rats",
        "BALB/c": "BALB/c mice",
        "BACHD mice": "BACHD mice",
        "TTR transgenic mice": "TTR transgenic mice",
        "cynomolgus monkey": "cynomolgus monkey",
        "CD/IGS rats": "CD/IGS rats",
        "hTTR transgenic mice": "hTTR transgenic mice",
        "transgenic mice": "transgenic mice",
        "huAGT mice": "huAGT mice",
        "NCr nude mice": "NCr nude mice",
        "C57bl6 mice": "C57BL/6 mice",
        "CD/IGS rat": "CD/IGS rats",
        "Balb/c male mice": "BALB/c mice",
        "CD-1 male mice": "CD-1 mice",
        "CD1 mice (Charles River, Mass.)": "CD-1 mice",
        "cynomolgus male monkeys": "cynomolgus monkey",
        "BALB/C": "BALB/c mice",
        "female Balb/c mice": "BALB/c mice",
        "hTTR transgenic female mice": "hTTR transgenic mice",
        "huAGT transgenic mice": "huAGT mice",
        "female transgenic huAGT mice": "huAGT mice",
        "Sprague Dawley rat": "Sprague-Dawley rats",
        "NOD-SCID mice": "NOD-SCID mice",
        "male rats": "Sprague-Dawley rats",
        "Sprague Dawley": "Sprague-Dawley rats",
        "male cynomolgus monkeys": "cynomolgus monkey",
        "male cynomolgus monkey": "cynomolgus monkey",
        "huTMPRSS6 Tg mice": "huTMPRSS6 Tg mice",
        "CD-1": "CD-1 mice",
        "Sprague-Dawley rats (Charles River)": "Sprague-Dawley rats",
        "Male Sprague-Dawley rat": "Sprague-Dawley rats",
        "hu-PBMC-NSG (NOD.Cg-Prkdcscid Il2rgtm1Wj1/SzJ)": "hu-PBMC-NSG (NOD.Cg-Prkdcscid Il2rgtm1Wj1/SzJ)",
        "Male Sprague-Dawley rats, 9-10 weeks old": "Sprague-Dawley rats",
        "Male CD1 mice": "CD-1 mice",
        "Sprague-Dawley rats, nine- to ten-week old": "Sprague-Dawley rats",
        "C57B1/6 mice": "C57BL/6 mice",
        "Male Sprague Dawley rats": "Sprague-Dawley rats",
        "24-25 week old male beagle dogs": "beagle dogs",
        "C57BL/6": "C57BL/6 mice",
        "Male cynomolgus monkeys": "cynomolgus monkey",
        "Female Tg mice": "Tg mice",
        "C57BL/6 wildtype mice": "C57BL/6 mice",
        "male BALB/C mice": "BALB/c mice",
        "ob/ob mice": "ob/ob mice",
        "CrTac:NCr-Foxn1nu mice": "CrTac:NCr-Foxn1nu mice",
        "C57BL/6NTac-TgN(APOB100)": "C57BL/6NTac-TgN(APOB100)",
        "LDLr-/- mice": "LDLr-/- mice",
        "wild type mice": "wild type mice",
    }
    collapsed_hepatictoxicity_df['species_strain'] = collapsed_hepatictoxicity_df['species_strain'].replace(species_strain_rename)

    collapsed_hepatictoxicity_df.to_parquet(DATA_PROCESSED / 'hepatictoxicity_processed.parquet')
    print(f"Saved hepatictoxicity_processed.parquet ({len(collapsed_hepatictoxicity_df):,} rows)")


if __name__ == "__main__":
    main()
