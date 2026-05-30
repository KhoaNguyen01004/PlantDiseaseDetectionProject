#!/usr/bin/env python3
"""Prepare real-world dataset structure for Plant Disease Detection.

Creates directory structure for future smartphone photo collection:
- data/plantvillage/ (existing clean data)
- data/real_world_train/ (future smartphone photos for training)
- data/real_world_test/ (future smartphone photos for testing)
"""
import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dataset_structure(base_dir: str = "data"):
    """Create the complete dataset structure for real-world data collection.
    
    Directory structure:
    data/
    ├── plantvillage/           # Existing clean PlantVillage data
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── real_world_train/       # Future smartphone photos (training)
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    ├── real_world_test/        # Future smartphone photos (testing)
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── unknown/                # Images that don't belong to any class
    """
    directories = [
        os.path.join(base_dir, "plantvillage"),
        os.path.join(base_dir, "real_world_train"),
        os.path.join(base_dir, "real_world_test"),
        os.path.join(base_dir, "unknown"),
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    logger.info("Dataset structure ready for real-world data collection.")
    return directories


def load_class_names_from_plantvillage(plantvillage_dir: str) -> List[str]:
    """Load class names from existing PlantVillage dataset.
    
    Args:
        plantvillage_dir: Path to PlantVillage directory.
    
    Returns:
        List of class names.
    """
    if not os.path.exists(plantvillage_dir):
        logger.warning(f"PlantVillage directory not found: {plantvillage_dir}")
        return []
    
    class_names = []
    for item in os.listdir(plantvillage_dir):
        item_path = os.path.join(plantvillage_dir, item)
        if os.path.isdir(item_path) and item != "__pycache__":
            class_names.append(item)
    
    logger.info(f"Found {len(class_names)} classes in PlantVillage dataset.")
    return sorted(class_names)


def create_class_directories(base_dir: str, class_names: List[str]):
    """Create class subdirectories in real_world_train and real_world_test.
    
    Args:
        base_dir: Base data directory.
        class_names: List of class names to create.
    """
    for split in ["real_world_train", "real_world_test"]:
        split_dir = os.path.join(base_dir, split)
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            logger.debug(f"Created: {class_dir}")
    
    logger.info(f"Created class directories for {len(class_names)} classes in real_world splits.")


def validate_image_count(directory: str, min_count: int = 0) -> bool:
    """Validate that a directory has at least min_count images.
    
    Args:
        directory: Directory to check.
        min_count: Minimum number of images required.
    
    Returns:
        True if directory has at least min_count images.
    """
    if not os.path.exists(directory):
        return False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    count = 0
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in image_extensions:
            count += 1
    
    return count >= min_count


def get_dataset_statistics(base_dir: str) -> dict:
    """Get statistics about the dataset.
    
    Args:
        base_dir: Base data directory.
    
    Returns:
        Dictionary with dataset statistics.
    """
    stats = {
        "plantvillage": {"classes": 0, "images": 0},
        "real_world_train": {"classes": 0, "images": 0},
        "real_world_test": {"classes": 0, "images": 0},
        "unknown": {"images": 0},
    }
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for split in ["plantvillage", "real_world_train", "real_world_test"]:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            classes = 0
            images = 0
            for item in os.listdir(split_dir):
                item_path = os.path.join(split_dir, item)
                if os.path.isdir(item_path):
                    classes += 1
                    for file in os.listdir(item_path):
                        if Path(file).suffix.lower() in image_extensions:
                            images += 1
                elif Path(item).suffix.lower() in image_extensions:
                    images += 1
            stats[split]["classes"] = classes
            stats[split]["images"] = images
    
    # Count unknown images
    unknown_dir = os.path.join(base_dir, "unknown")
    if os.path.exists(unknown_dir):
        images = 0
        for file in os.listdir(unknown_dir):
            if Path(file).suffix.lower() in image_extensions:
                images += 1
        stats["unknown"]["images"] = images
    
    return stats


def print_dataset_summary(base_dir: str):
    """Print a summary of the dataset.
    
    Args:
        base_dir: Base data directory.
    """
    stats = get_dataset_statistics(base_dir)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"{'Split':<20} {'Classes':<10} {'Images':<10}")
    print("-"*60)
    
    for split, data in stats.items():
        classes = data.get("classes", "N/A")
        images = data.get("images", 0)
        print(f"{split:<20} {classes:<10} {images:<10}")
    
    total_images = sum(data.get("images", 0) for data in stats.values())
    print("-"*60)
    print(f"{'TOTAL':<20} {'':<10} {total_images:<10}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare real-world dataset structure for Plant Disease Detection'
    )
    parser.add_argument(
        '--base-dir',
        default='data',
        help='Base directory for dataset (default: data)'
    )
    parser.add_argument(
        '--create-classes',
        action='store_true',
        help='Create class directories based on existing PlantVillage classes'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print dataset summary'
    )
    args = parser.parse_args()
    
    # Create base structure
    logger.info("Creating dataset structure...")
    create_dataset_structure(args.base_dir)
    
    # Create class directories if requested
    if args.create_classes:
        plantvillage_dir = os.path.join(args.base_dir, "plantvillage")
        class_names = load_class_names_from_plantvillage(plantvillage_dir)
        if class_names:
            create_class_directories(args.base_dir, class_names)
    
    # Print summary
    if args.summary:
        print_dataset_summary(args.base_dir)
    
    logger.info("Real-world dataset preparation complete!")
    logger.info("Next steps:")
    logger.info("  1. Collect smartphone photos of plant diseases")
    logger.info("  2. Organize into real_world_train/ and real_world_test/")
    logger.info("  3. Add unknown/background images to unknown/")
    logger.info("  4. Run training with mixed dataset")


if __name__ == '__main__':
    main()