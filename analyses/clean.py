import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    ## import
    """)
    return


@app.cell
def _():
    import pandas as pd

    base_path = "../data/oligostack/raw/"

    # in vitro inhibition
    in_vitro_inhibition_df = pd.read_csv(f"{base_path}/in_vitro_inhibition_collation_results.csv")

    # dose response (nM)
    dose_response_nm_df = pd.read_csv(f"{base_path}/dose_response_nm_collation_results.csv")

    # dose response (uM) + convert to nM

    dose_response_um_df = pd.read_csv(f"{base_path}/dose_response_um_collation_results.csv")
    dose_response_um_df['dosage'] = dose_response_um_df['dosage'] * 1000

    dose_response_df = pd.concat([dose_response_nm_df, dose_response_um_df])
    dose_response_df.rename(columns={'dosage': 'dosage_nm'}, inplace=True)

    del dose_response_nm_df, dose_response_um_df

    # neurotoxicity
    neurotoxicity_df = pd.read_csv(f"{base_path}/neurotox_collation_results.csv")

    # hepatictoxicity

    hepatictoxicity_df = pd.read_csv(f"{base_path}/hepatictox_collation_results.csv")
    return (
        dose_response_df,
        hepatictoxicity_df,
        in_vitro_inhibition_df,
        neurotoxicity_df,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Clean in vitro inhibition
    """)
    return


@app.cell
def _(in_vitro_inhibition_df):
    clean_in_vitro_inhibition_df = in_vitro_inhibition_df[in_vitro_inhibition_df['Inhibition_pct'].between(-1000, 100)]
    clean_in_vitro_inhibition_df['cell_line_species'] = (
        clean_in_vitro_inhibition_df['cell_line_species']
        .replace({'Cynomolgus': 'monkey', 'cynomolgus': 'monkey', 'rhesus': 'monkey'})
    )

    def standardise_cell_line(cell_line):
        """
        Standardise cell line spellings to canonical names based on provided mapping.
        """
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
            'Primary Cynomolgus hepatocytes': 'Primary Cynomolgus hepatocytes',
            'differentiated human adipocytes': 'differentiated human adipocytes',
            'MM.1R': 'MM.1R',
            'AML12': 'AML12',
            'HepAD38': 'HepAD38',
            'HepB3': 'Hep3B',
        }
        # Normalise names: strip spaces, case-insensitive, hyphens/periods
        name = str(cell_line).replace('–', '-').replace('_', '-').replace('.', '.').strip()
        lower = name.lower().replace('-', '').replace(' ', '').replace('.', '')
        reverse = {k.lower().replace('-', '').replace(' ', '').replace('.', ''): v for k, v in mapping.items()}
        return reverse.get(lower, cell_line)  # fallback to original if not found

    # Apply to the column and get standardised value counts
    clean_in_vitro_inhibition_df['cell_line'] = clean_in_vitro_inhibition_df['cell_line'].map(standardise_cell_line)
    return (clean_in_vitro_inhibition_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Deduplicate in vitro inhibition
    """)
    return


@app.cell
def _(clean_in_vitro_inhibition_df):
    # Sort by USPTO ID descending (latest first), then drop duplicates keeping first (latest)
    dedup_clean_in_vitro_inhibition_df = (
        clean_in_vitro_inhibition_df
        .sort_values('USPTO ID', ascending=False)
        .drop_duplicates(subset=['Compound ID', 'Inhibition_pct'], keep='first')
        .reset_index(drop=True)
    )
    return (dedup_clean_in_vitro_inhibition_df,)


@app.cell
def _(dedup_clean_in_vitro_inhibition_df):
    dedup_clean_in_vitro_inhibition_df.to_parquet('../data/oligostack/processed/in_vitro_inhibition_processed.parquet')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Clean dose response
    """)
    return


@app.cell
def _(dose_response_df, pd):
    ## Clean dose response

    clean_dose_response_df = dose_response_df[dose_response_df['Inhibition_pct'].between(-1000, 100)]
    clean_dose_response_df['cell_line_species'] = (
        clean_dose_response_df['cell_line_species']
        .replace({'rhesus': 'monkey', 'cynomolgus': 'monkey', 'cyno': 'monkey', 'cyano': 'monkey', 'cynomolgous': 'monkey'})
    )

    clean_dose_response_df = clean_dose_response_df[clean_dose_response_df['dosage_nm'] > 0]

    # Clean up/standardize alternative cell_line names to canonical names

    canonical_cell_lines = {
        # Top hits with most common alternative spellings/plurals etc.
        'A431': 'A431',
        'A-431': 'A431',

        'HepG2': 'HepG2',
        'HEPG2': 'HepG2',
        'Hep G2': 'HepG2',
        'HEP-G2': 'HepG2',
        'HepG2s': 'HepG2',

        'Hep3B': 'Hep3B',
        'HepB3': 'Hep3B',
        'Hep-3B': 'Hep3B',

        'SH-SY5Y': 'SH-SY5Y',
        'SHSY5Y': 'SH-SY5Y',
        'SH-SY-5Y': 'SH-SY5Y',

        'HUVEC': 'HUVEC',
        'HUVECs': 'HUVEC',

        'HepaRG': 'HepaRG',
        'Heparg': 'HepaRG',

        'THP-1': 'THP-1',
        'THP1': 'THP-1',

        'SK-MEL-28': 'SK-MEL-28',
        'SKMEL28': 'SK-MEL-28',

        'A549': 'A549',

        'Huh7': 'Huh7',
        'HUH-7': 'Huh7',

        'iCell cardiomyocytes2': 'iCell cardiomyocytes2',
        'iCell cardiomyocytes': 'iCell cardiomyocytes',

        'primary hepatocytes': 'primary hepatocytes',
        'Primary hepatocytes': 'primary hepatocytes',

        'SUP-M2': 'SUP-M2',
        'SUPM2': 'SUP-M2',

        'MM.1R': 'MM.1R',
        'MM1R': 'MM.1R',

        'U251': 'U251',
        'U-251': 'U251',
        'U251-MG': 'U251-MG',

        'iCell GABANeurons': 'iCell GABANeurons',
        'iCell GABA Neurons': 'iCell GABANeurons',

        'HepG2.2.15': 'HepG2.2.15',
        'HEPG2.2.15': 'HepG2.2.15',

        'LLC-MK2': 'LLC-MK2',
        'LLCMK2': 'LLC-MK2',

        'VCaP': 'VCaP',

        'GM04281': 'GM04281',

        'transgenic mouse primary hepatocytes': 'transgenic mouse primary hepatocytes',

        'primary mouse hepatocytes': 'primary mouse hepatocytes',

        'SNU-449': 'SNU-449',
        'SNU449': 'SNU-449',

        'HEK293': 'HEK293',
        'HEK-293': 'HEK293',

        'GM02171': 'GM02171',

        'Primary human hepatocytes': 'Primary human hepatocytes',
        'primary human hepatocytes': 'Primary human hepatocytes',

        '54-2': '54-2',

        'cynomolgus primary hepatocytes': 'cynomolgus primary hepatocytes',

        'hSKMC': 'hSKMC',

        'GM02173B': 'GM02173B',

        'b.END': 'b.END',
        'bEND': 'b.END',

        'mouse primary hepatocytes': 'mouse primary hepatocytes',

        '4T1': '4T1',

        'MDA-MB-436': 'MDA-MB-436',
        'MDA MB 436': 'MDA-MB-436',

        'CD4 T-cells': 'CD4 T-cells',
        'CD4+ T-cells': 'CD4+ T-cells',
        'Human CD4+ T-cells': 'CD4+ T-cells',

        'Angelman IPS-derived neurons': 'Angelman IPS-derived neurons',

        'primary rat hepatocytes': 'primary rat hepatocytes',

        'LNCaP': 'LNCaP',

        'T-reg': 'T-reg',

        'KARPAS-229': 'KARPAS-229',

        'SK-BR-3': 'SK-BR-3',
        'SKBR3': 'SK-BR-3',

        'SCA2-04': 'SCA2-04',

        'K-562': 'K-562',
        'K562': 'K-562',

        'HeLa': 'HeLa',
        'hela': 'HeLa',

        'C4-2B': 'C4-2B',
        'C4-2': 'C4-2',

        'KMS11': 'KMS11',

        'HepatoPac': 'HepatoPac',

        'HSMM': 'HSMM',

        'NCI-H460': 'NCI-H460',
        'NCIH460': 'NCI-H460',

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

        'RAW 264.7': 'RAW 264.7',
        'RAW264.7': 'RAW 264.7',

        'C4-2B MR': 'C4-2B MR',

        'MDA-MB-231': 'MDA-MB-231',
        'MDA MB 231': 'MDA-MB-231',

        'BACHD mouse hepatocytes': 'BACHD mouse hepatocytes',
        'BACHD': 'BACHD',

        'MH-S': 'MH-S',

        'SCC25': 'SCC25',
        'SCC-25': 'SCC25',

        'SW872': 'SW872',

        'B16-F10': 'B16-F10',
        'B16F10': 'B16-F10',

        'F09-152': 'F09-152',
        'F09-229': 'F09-229',

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

        'NIH 3T3': 'NIH 3T3',
        '3T3-L1': '3T3-L1',

        '4MBr-5': '4MBr-5',

        'Vero C1008': 'Vero C1008',

        'CaCo-2': 'CaCo-2',
        'Caco2': 'CaCo-2',

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
        # Strip and capital/space-insensitive match to mapping (also handle '-'/spaces)
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
    return (clean_dose_response_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Deduplicate dose response
    """)
    return


@app.cell
def _(clean_dose_response_df):
    # Sort by USPTO ID descending (latest first), then drop duplicates keeping first (latest)
    dedup_clean_dose_response_df = (
        clean_dose_response_df
        .sort_values('USPTO ID', ascending=False)
        .drop_duplicates(subset=['Compound ID', 'dosage_nm', 'Inhibition_pct'], keep='first')
        .reset_index(drop=True)
    )
    return (dedup_clean_dose_response_df,)


@app.cell
def _(dedup_clean_dose_response_df):
    dedup_clean_dose_response_df.to_parquet('../data/oligostack/processed/dose_response_processed.parquet')
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Clean neurotox
    """)
    return


@app.cell
def _(neurotoxicity_df):
    # Process 'FOB_score': convert everything into list of floats wherever possible, 
    # but ensure missing/unparsable values are np.nan (not [None], not [np.nan], not None).

    import numpy as np

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

    # Standardise strain names using mapping
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
    return clean_neurotoxicity_df, np


@app.cell
def _(mo):
    mo.md(r"""
    ## Dedup neuro tox
    """)
    return


@app.cell
def _(clean_neurotoxicity_df):
    # Deduplicate on HELM, species, admin, test type, dosage - keep first instance
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
    return (clean_neurotoxicity_df_dedup,)


@app.cell
def _(clean_neurotoxicity_df_dedup):
    clean_neurotoxicity_df_dedup.to_parquet('../data/oligostack/processed/neurotoxicity_processed.parquet')
    return


@app.cell
def _(hepatictoxicity_df, pd):
    # Split 'dosage' field into 'dosage_value' and 'dosage_unit' columns
    import re

    def split_dosage(dosage):
        """
        Splits a dosage string into value and unit.
        Example: '10 mg/kg' -> (10.0, 'mg/kg')
        Returns (value, unit) or (None, None) if not parsable.
        """
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
    clean_hepatictoxicity_df['dosage_unit'].value_counts()

    # Convert 'mg/kg/wk' to 'mg/kg' by dividing the total mg/kg over all doses during the period
    # The logic: for each row with 'mg/kg/wk', compute mg/kg = (dosage_value * dosing_period_days / 7) / num_doses,
    # or equivalently: (dosage_value / 7) * dosing_period_days / num_doses
    # This assumes dosage_value stores the total mg/kg per week, and we want mg/kg per dose

    mask = clean_hepatictoxicity_df['dosage_unit'] == 'mg/kg/wk'
    # Only update rows where necessary values exist and num_doses > 0
    valid = (
        mask &
        clean_hepatictoxicity_df['dosing_period_days'].notnull() &
        clean_hepatictoxicity_df['num_doses'].notnull() &
        (clean_hepatictoxicity_df['num_doses'] > 0)
    )

    def convert_mgkgwk_to_mgkg(row):
        # Calculate mg/kg per dose
        try:
            return (row['dosage_value'] * row['dosing_period_days'] / 7) / row['num_doses']
        except Exception:
            return row['dosage_value']

    clean_hepatictoxicity_df.loc[valid, 'dosage_value'] = clean_hepatictoxicity_df[valid].apply(convert_mgkgwk_to_mgkg, axis=1)
    clean_hepatictoxicity_df.loc[mask, 'dosage_unit'] = clean_hepatictoxicity_df.loc[mask, 'dosage_unit'].replace('mg/kg/wk', 'mg/kg')

    clean_hepatictoxicity_df = clean_hepatictoxicity_df[clean_hepatictoxicity_df['dosage_unit'] == 'mg/kg']
    clean_hepatictoxicity_df = clean_hepatictoxicity_df.rename(columns={'dosage_value': 'dosage_mg_per_kg'})
    clean_hepatictoxicity_df = clean_hepatictoxicity_df.drop(columns=['dosage_unit', 'dosage'])

    # Remove rows with dosage > 10^4 mg/kg
    clean_hepatictoxicity_df = clean_hepatictoxicity_df[clean_hepatictoxicity_df['dosage_mg_per_kg'] <= 1e4]

    # Map 'intraperitoneally' -> 'intraperitoneal'
    clean_hepatictoxicity_df['adminstration_method'] = clean_hepatictoxicity_df['adminstration_method'].replace(
        {'intraperitoneally': 'intraperitoneal'}
    )
    # Filter to only 'intraperitoneal' or 'subcutaneous'
    clean_hepatictoxicity_df = clean_hepatictoxicity_df[
        clean_hepatictoxicity_df['adminstration_method'].isin(['intraperitoneal', 'subcutaneous'])
    ]

    # filter to only 'plasma or urine':

    clean_hepatictoxicity_df = clean_hepatictoxicity_df[
        clean_hepatictoxicity_df['measurement_source'].isin(['plasma', 'urine'])
    ]

    # Standardize species names
    clean_hepatictoxicity_df['species'] = clean_hepatictoxicity_df['species'].replace({
        'cynomolgus monkey': 'monkey',
        'beagle dog': 'dog',
    })

    # TODO - Define valid ranges for all biomarkers
    biomarker_ranges = {
        "ALB": (1, 100),
        "ALT": (10, 50000),
        "AST": (10, 50000),
        #"BUN": (0, 50000),
        #"CREA": (0, 50000),
        #"TBIL": (0, 50000),
        #"PC_ratio": (0, 5)
    }

    def filter_biomarker_series(biomarker_series, vmin, vmax):
        # Handles lists, drops values <vmin or >vmax, keeps NaNs
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
    return (clean_hepatictoxicity_df,)


@app.cell
def _(clean_hepatictoxicity_df):
    # Collapse rows with shared grouping columns, creating a list of biomarkers
    group_cols = ['Compound ID', 'species', 'num_doses', 'dosing_period_days', 'dosage_mg_per_kg', 'measurement_source', 'adminstration_method']
    biomarker_cols = ['ALB', 'ALT', 'AST', 'BUN', 'CREA', 'TBIL', 'PC_ratio']

    # For other columns (e.g. USPTO ID), take the first non-NaN value
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
    return (collapsed_hepatictoxicity_df,)


@app.cell
def _(collapsed_hepatictoxicity_df):
    # Standardize species_strain values by simple renaming
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
        "male rats": "Sprague-Dawley rats",  # Best-effort
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
    return


@app.cell
def _(collapsed_hepatictoxicity_df):
    collapsed_hepatictoxicity_df.to_parquet('../data/oligostack/processed/hepatictoxicity_processed.parquet')
    return


if __name__ == "__main__":
    app.run()
