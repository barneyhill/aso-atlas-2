"""The release, its card and the paper must not drift apart again.

Every count on the dataset card comes from release_manifest.json, and the paper's
total comes from paper_numbers.json. These tests pin the two together and check
the complete corpus and HELM-resolved modelling subset remain explicit.
"""

import hashlib
import json
import runpy
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

RELEASE = Path("aso-atlas-2-release")
MANIFEST_PATH = RELEASE / "release_manifest.json"
CROISSANT_PATH = RELEASE / "croissant.json"
PAPER_NUMBERS = Path("typst/data/paper_numbers.json")

CONFIGS = ["in_vitro_inhibition", "dose_response", "hepatotoxicity", "neurotoxicity"]
pytestmark = pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason="release not built; run aso-atlas-2-release/prepare_hf_release.py",
)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def croissant():
    return json.loads(CROISSANT_PATH.read_text())


@pytest.fixture(scope="module")
def release():
    frames = []
    for config in CONFIGS:
        df = pq.read_table(
            RELEASE / f"{config}.parquet",
            columns=["USPTO ID", "Compound ID", "HELM Annotation", "model_eligible"],
        ).to_pandas()
        frames.append(df.assign(config=config))
    return pd.concat(frames, ignore_index=True)


def test_every_config_exists():
    for config in CONFIGS:
        assert (RELEASE / f"{config}.parquet").exists()


def test_model_eligibility_exactly_tracks_resolved_helm(release):
    assert release["model_eligible"].equals(release["HELM Annotation"].notna())
    assert (~release["model_eligible"]).any()


def test_manifest_row_counts_match_files(manifest, release):
    for config in CONFIGS:
        entry = manifest["configs"][config]
        actual = release[release["config"] == config]
        assert entry["rows"] == len(actual), config
        assert entry["model_eligible_rows"] == int(actual["model_eligible"].sum())
        assert entry["unresolved_helm_rows"] == int((~actual["model_eligible"]).sum())
    assert "released_rows" not in manifest["totals"]
    assert "model_eligible_rows" not in manifest["totals"]
    assert "unresolved_helm_rows" not in manifest["totals"]


def test_card_is_generated_from_manifest(manifest):
    """Card numbers are interpolated, so the totals must appear verbatim."""
    card = (RELEASE / "README.md").read_text()
    t = manifest["totals"]
    for value in (t["released_assay_readouts"], t["compounds"], t["patents"]):
        assert f"{value:,}" in card, value
    assert manifest["version"] in card
    assert "<<" not in card


def test_submitted_source_id_linkage_is_canonical(manifest, release):
    linkage = manifest["cross_category_linkage"]
    category_sets = {
        config: set(rows["Compound ID"].dropna().unique())
        for config, rows in release.groupby("config")
    }
    all_source_ids = set().union(*category_sets.values())
    n_multi = sum(
        sum(source_id in ids for ids in category_sets.values()) >= 2
        for source_id in all_source_ids
    )

    assert linkage["identity_key"] == "Compound ID"
    assert linkage["total_source_ids"] == len(all_source_ids) == 168_537
    assert linkage["source_ids_in_multiple_categories"] == n_multi == 15_339
    assert linkage["pct_in_multiple_categories"] == 9.1

    paper = json.loads(PAPER_NUMBERS.read_text())
    paper_linkage = paper["category_overlap"]
    assert paper_linkage["identity_key"] == "Compound ID"
    assert paper_linkage["n_total_source_ids"] == 168_537
    assert paper_linkage["n_in_multi"] == 15_339
    assert paper_linkage["pct_in_multi"] == 9.1

    card = (RELEASE / "README.md").read_text()
    assert manifest["linkage_definition"] in card


def test_card_declares_full_partition_per_config(manifest):
    card = (RELEASE / "README.md").read_text()
    front_matter = card.split("---")[1]
    for config in CONFIGS:
        assert f"path: {config}.parquet" in front_matter
        assert "split: full" in front_matter


def test_card_does_not_claim_a_missing_split(manifest):
    """The exact wording reviewer Tw5Q W5 quoted must not come back."""
    card = (RELEASE / "README.md").read_text().lower()
    assert "no canonical train/test split" not in card


def test_neurips_croissant_has_core_and_minimal_rai(croissant, manifest):
    """NeurIPS 2026 E&D requires core Croissant plus these RAI fields."""
    for key in (
        "@context", "@type", "name", "description", "conformsTo", "url",
        "license", "creator", "datePublished", "distribution", "recordSet",
    ):
        assert croissant.get(key), key

    assert croissant["@type"] == "sc:Dataset"
    assert "http://mlcommons.org/croissant/1.1" in croissant["conformsTo"]
    assert "http://mlcommons.org/croissant/RAI/1.0" in croissant["conformsTo"]
    assert croissant["version"] == manifest["version"]
    assert manifest["release_tag"] in croissant["url"]

    for key in (
        "rai:dataLimitations", "rai:dataBiases",
        "rai:personalSensitiveInformation", "rai:dataUseCases",
        "rai:dataSocialImpact", "rai:hasSyntheticData",
        "prov:wasDerivedFrom", "prov:wasGeneratedBy",
    ):
        assert key in croissant, key
    assert croissant["rai:hasSyntheticData"] is False


def test_croissant_resources_are_complete_pinned_and_hashed(croissant, manifest):
    expected = {f"{config}.parquet" for config in CONFIGS}
    expected |= {
        "folds/in_vitro_inhibition_folds.csv.gz",
        "folds/potency_folds.csv.gz",
        "folds/mouse_hepatic_folds.csv.gz",
        "folds/rat_hepatic_folds.csv.gz",
        "folds/mouse_neuro_folds.csv.gz",
        "folds/rat_neuro_folds.csv.gz",
        "folds/oligoai_folds.csv.gz",
    }
    resources = {entry["@id"]: entry for entry in croissant["distribution"]}
    assert set(resources) == expected

    for relative, entry in resources.items():
        path = RELEASE / relative
        assert path.exists(), relative
        assert f"/{manifest['release_tag']}/{relative}" in entry["contentUrl"]
        assert entry["contentSize"] == f"{path.stat().st_size} B"
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    assert len(croissant["recordSet"]) == len(CONFIGS)
    assert all(record["field"] for record in croissant["recordSet"])


def test_publish_allowlist_is_exact_and_excludes_build_inputs():
    upload = runpy.run_path(str(RELEASE / "upload_hf_release.py"))
    publish_files = set(upload["PUBLISH_FILES"])
    assert "croissant.json" in publish_files
    assert "README.md" in publish_files
    assert "release_manifest.json" in publish_files
    assert not any(path.endswith(".py") for path in publish_files)
    assert not any("validation" in path.lower() and not path.endswith("validation.parquet")
                   for path in publish_files)
    assert "scn2a_all_data.csv" not in publish_files
    stale = upload["stale_remote_files"]({
        ".gitattributes", "in_vitro_inhibition.parquet", "stray.csv", *publish_files,
    })
    assert stale == ["stray.csv"]


def test_neurips_large_dataset_sample_rule_stays_inactive(croissant):
    """If the bundle exceeds 4 GB, NeurIPS requires an inspectable small sample."""
    total_bytes = sum(int(item["contentSize"].split()[0])
                      for item in croissant["distribution"])
    assert total_bytes <= 4 * 1024**3


def test_card_documents_neurips_rai_topics():
    card = (RELEASE / "README.md").read_text()
    for phrase in (
        "## Intended uses", "not validated for clinical decision-making",
        "### Source terms and attribution", "### Social impact",
        "Croissant 1.1 core + RAI", "DepMap Public 25Q3",
    ):
        assert phrase in card


@pytest.mark.skipif(not PAPER_NUMBERS.exists(), reason="paper numbers not exported")
def test_release_retains_all_null_helm_rows(manifest):
    """The release matches the paper and explicitly marks unresolved chemistry."""
    paper = json.loads(PAPER_NUMBERS.read_text())
    xref = manifest["paper_cross_reference"]
    assert xref["paper_assay_readouts"] == paper["genes"]["n_measurements_all"]

    processed = Path("data/oligostack/processed")
    if not processed.exists():
        pytest.skip("processed data not available")

    from analyses.export_paper_numbers import BIOMARKER_COLS
    from analyses.utils.compounds import has_measurement

    unresolved = 0
    for name, filename, kind in [
        ("in_vitro_inhibition", "in_vitro_inhibition_processed.parquet", "rows"),
        ("dose_response", "dose_response_processed.parquet", "rows"),
        ("hepatotoxicity", "hepatictoxicity_processed.parquet", "biomarkers"),
        ("neurotoxicity", "neurotoxicity_processed.parquet", "fob"),
    ]:
        df = pq.read_table(processed / filename).to_pandas()
        null_helm = df[df["HELM Annotation"].isna()]
        if kind == "rows":
            unresolved += len(null_helm)
        elif kind == "biomarkers":
            unresolved += int(
                null_helm[BIOMARKER_COLS]
                .apply(lambda r: sum(has_measurement(r[c]) for c in BIOMARKER_COLS), axis=1)
                .sum()
            )
        else:
            unresolved += int(null_helm["FOB_score"].apply(has_measurement).sum())

    assert xref["delta"] == 0
    assert manifest["totals"]["unresolved_helm_readouts"] == unresolved
    assert manifest["totals"]["released_assay_readouts"] == xref["paper_assay_readouts"]
