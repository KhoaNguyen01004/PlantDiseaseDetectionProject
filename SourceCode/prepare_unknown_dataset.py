import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("data/leafsnap")
OUTPUT_DIR = Path("data/unknown")
DEFAULT_KAGGLE_DATASET = "xhlulu/leafsnap-dataset"


def collect_field_images(source_dir: Path):
    field_root = source_dir / "images" / "field"
    if not field_root.exists():
        raise FileNotFoundError(
            f"Expected Leafsnap field images under: {field_root}"
        )

    image_files = []
    for extension in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(sorted(field_root.rglob(extension)))

    return [path for path in image_files if path.is_file()]


def find_leafsnap_root(root_dir: Path):
    if (root_dir / "images" / "field").exists():
        return root_dir

    for child in root_dir.iterdir():
        if child.is_dir() and (child / "images" / "field").exists():
            return child

    raise FileNotFoundError(
        f"Could not locate Leafsnap root containing images/field under: {root_dir}"
    )


def download_leafsnap_dataset(download_dir: Path, dataset: str = DEFAULT_KAGGLE_DATASET):
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed or not available in PATH. "
            "Install the Kaggle CLI and authenticate with your Kaggle account."
        )

    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Leafsnap dataset '{dataset}' into {download_dir}...")

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(download_dir), "--force"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed: {result.stderr.strip()}"
        )

    archive_files = list(download_dir.glob("*.zip"))
    if not archive_files:
        raise FileNotFoundError(
            f"No downloaded zip archive found in {download_dir}"
        )

    for archive in archive_files:
        print(f"Extracting {archive.name}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(download_dir)

    return find_leafsnap_root(download_dir)


def prepare_unknown_dataset(source_dir: Path, output_dir: Path, discard_old: bool = True):
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    images = collect_field_images(source_dir)
    if not images:
        raise FileNotFoundError(
            f"No field images found in Leafsnap dataset at {source_dir}/images/field"
        )

    if discard_old and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in images:
        destination = output_dir / f"unknown_{count:06d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare unknown dataset from Leafsnap field images."
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Path to the Leafsnap dataset root directory"
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Path where the unknown dataset will be saved"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Automatically download the Leafsnap Kaggle dataset if source is missing"
    )
    parser.add_argument(
        "--kaggle-dataset",
        default=DEFAULT_KAGGLE_DATASET,
        help="Kaggle dataset identifier to download"
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Temporary directory to download and extract the Leafsnap dataset"
    )
    parser.add_argument(
        "--keep-old",
        action="store_true",
        help="Keep existing unknown images instead of discarding them"
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep downloaded Leafsnap archive and extracted files after preparing unknown images"
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    temp_download_dir = None

    if not source_dir.exists() and args.download:
        if args.download_dir:
            download_dir = Path(args.download_dir)
        else:
            temp_download_dir = Path(tempfile.mkdtemp(prefix="leafsnap_"))
            download_dir = temp_download_dir

        try:
            source_dir = download_leafsnap_dataset(download_dir, args.kaggle_dataset)
        except Exception:
            if temp_download_dir is not None and temp_download_dir.exists() and not args.keep_download:
                shutil.rmtree(temp_download_dir, ignore_errors=True)
            raise
    elif not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}.\n"
            "Use --download to automatically fetch the Kaggle Leafsnap dataset."
        )

    print("=" * 60)
    print("Preparing UNKNOWN dataset from Leafsnap field images")
    print("=" * 60)

    count = prepare_unknown_dataset(
        source_dir,
        output_dir,
        discard_old=not args.keep_old
    )

    if temp_download_dir is not None and not args.keep_download:
        shutil.rmtree(temp_download_dir, ignore_errors=True)

    print("\nDone")
    print(f"Total unknown images copied: {count}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
