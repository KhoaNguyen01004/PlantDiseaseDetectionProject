#!/usr/bin/env python3

import os
import zipfile
import argparse
from pathlib import Path

import kagglehub
from tqdm import tqdm


DATASET = "abdallahalidev/plantvillage-dataset"


def extract_all_zips(root_dir):
    """
    Recursively extract all zip files
    """

    zip_files = []

    for path, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(path, file))

    if not zip_files:
        print("No zip files found.")
        return

    print(f"\nFound {len(zip_files)} zip file(s).\n")

    for zip_path in tqdm(zip_files, desc="Extracting ZIPs"):

        extract_dir = os.path.dirname(zip_path)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            os.remove(zip_path)

        except Exception as e:
            print(f"\nFailed to extract: {zip_path}")
            print(e)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to store dataset",
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PlantVillage Dataset Downloader (Kaggle)")
    print("=" * 60)

    print("\nDownloading dataset from Kaggle...\n")

    dataset_path = kagglehub.dataset_download(DATASET)

    print(f"\nDownloaded to cache:")
    print(dataset_path)

    print("\nCopying files to project data folder...\n")

    # copy dataset structure
    import shutil

    target_dir = os.path.join(data_dir, "plantvillage")

    if os.path.exists(target_dir):
        print(f"{target_dir} already exists.")
    else:
        shutil.copytree(dataset_path, target_dir)

    print("\nExtracting zip files if needed...\n")

    extract_all_zips(target_dir)

    print("\n" + "=" * 60)
    print("Dataset ready!")
    print("=" * 60)

    print("\nDataset location:")
    print(target_dir)

    print("\nNext step:")
    print("python src/train.py")


if __name__ == "__main__":
    main()