"""Upload ASO Atlas 2.0 aso-atlas-2-release/ directory to HuggingFace Hub.

Tags the current remote state before pushing, so the snapshot a reviewer may have
already downloaded stays reachable and diffable, then publishes the reconciled
endpoint parquets and exact paper fold assignments.

    uv run python aso-atlas-2-release/upload_hf_release.py
"""

import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "barneyhill/aso-atlas-2"
RELEASE = Path("aso-atlas-2-release")

# Tag applied to the pre-existing remote state before it is overwritten.
PRIOR_REVISION_TAG = "neurips-submission"

CONFIGS = ["in_vitro_inhibition", "dose_response", "hepatotoxicity", "neurotoxicity"]
PAPER_FOLD_FILES = [
    "in_vitro_inhibition_folds.csv.gz",
    "potency_folds.csv.gz",
    "mouse_hepatic_folds.csv.gz",
    "rat_hepatic_folds.csv.gz",
    "mouse_neuro_folds.csv.gz",
    "rat_neuro_folds.csv.gz",
    "oligoai_folds.csv.gz",
]
PUBLISH_FILES = [
    "README.md",
    "release_manifest.json",
    "croissant.json",
    *[f"{config}.parquet" for config in CONFIGS],
    "folds/README.md",
    *[f"folds/{name}" for name in PAPER_FOLD_FILES],
]


def stale_remote_files(remote_files: set[str]) -> list[str]:
    return sorted(remote_files - set(PUBLISH_FILES) - {".gitattributes"})


def main() -> None:
    api = HfApi()

    manifest_path = RELEASE / "release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing: {manifest_path} — run prepare_hf_release.py first")
    manifest = json.loads(manifest_path.read_text())

    required = [RELEASE / path for path in PUBLISH_FILES]
    for f in required:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f} — run prepare_hf_release.py first")

    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    # Preserve whatever is currently published before overwriting it.
    try:
        api.create_tag(
            REPO_ID,
            tag=PRIOR_REVISION_TAG,
            repo_type="dataset",
            tag_message="Unversioned snapshot as released with the NeurIPS submission.",
        )
        print(f"Tagged current remote state as '{PRIOR_REVISION_TAG}'")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print(f"Tag '{PRIOR_REVISION_TAG}' already exists — leaving it untouched")
        else:
            raise

    version = manifest["version"]
    release_tag = manifest["release_tag"]
    print(f"Uploading {release_tag} to {REPO_ID}...")
    # An allowlist controls new uploads but does not remove unrelated files that
    # already exist remotely. The legacy revision is safely tagged above, so
    # remove every stale artifact from main and leave only HF's git metadata plus
    # the agreed publication bundle.
    remote_files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    stale_files = stale_remote_files(remote_files)
    if stale_files:
        print("Removing stale remote files:")
        for path in stale_files:
            print(f"  {path}")

    api.upload_folder(
        folder_path=str(RELEASE),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=(
            f"ASO Atlas 2.0 {release_tag}: reconciled counts and exact paper folds"
        ),
        delete_patterns=stale_files,
        # Publish an explicit allowlist. The staging directory also contains build
        # scripts, templates, caches and exploratory files that must never leak into
        # the dataset repository merely because they were created locally.
        allow_patterns=PUBLISH_FILES,
    )

    api.create_tag(
        REPO_ID,
        tag=release_tag,
        repo_type="dataset",
        tag_message=f"ASO Atlas 2.0 {release_tag}",
        exist_ok=True,
    )
    print(f"Tagged {release_tag}")
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
