#!/usr/bin/env python3
"Download and prepare PlantVillage dataset (~54k images, 38 classes)."
import os
import urllib.request
import zipfile
from pathlib import Path
import argparse

URLS = {
    'train': 'https://github.com/spMohanty/PlantVillage-Dataset/raw/master/color/New Plant Diseases Dataset(Augmented)/train.zip',
    'val': 'https://github.com/spMohanty/PlantVillage-Dataset/raw/master/color/New Plant Diseases Dataset(Augmented)/valid.zip'
}

def download_and_extract(url, dest_dir):
    zip_path = f'{dest_dir}.zip'
    print(f'Downloading {url}...')
    urllib.request.urlretrieve(url, zip_path)
    
    print(f'Extracting to {dest_dir}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    os.remove(zip_path)
    print(f'Extracted to ./New Plant Diseases Dataset(Augmented)/{dest_dir.split("/")[-1]}')
    print('Move to data/train and data/val manually or symlink.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Target dir')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    for split, url in URLS.items():
        dest = os.path.join(args.data_dir, split)
        if os.path.exists(dest):
            print(f'{dest} exists, skipping.')
            continue
        download_and_extract(url, dest)
    
    print('\nDataset ready! Expected ~43k train, ~11k val images.')
    print('Run: python src/train.py')

if __name__ == '__main__':
    main()
