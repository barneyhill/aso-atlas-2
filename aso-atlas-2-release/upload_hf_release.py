"""Upload ASO Atlas 2.0 aso-atlas-2-release/ directory to HuggingFace Hub."""

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "barneyhill/aso-atlas-2"
RELEASE = Path("aso-atlas-2-release")


def main() -> None:
    api = HfApi()

    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    files = list(RELEASE.glob("*.parquet")) + [RELEASE / "README.md"]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Missing: {f} — run prepare_hf_release.py first")

    print(f"Uploading {len(files)} files to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(RELEASE),
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"Done: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
