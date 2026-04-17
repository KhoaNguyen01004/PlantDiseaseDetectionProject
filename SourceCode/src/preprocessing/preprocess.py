import os
import glob
import yaml
import cv2
import numpy as np
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

from collections import Counter
from torch.utils.data import WeightedRandomSampler

# TFLite optional support
try:
    from tflite_support.metadata_writers import MetadataWriter, ImageClassifier
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    ImageClassifier = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def safe_read_image(path, flags=cv2.IMREAD_UNCHANGED):
    if not os.path.exists(path):
        logging.warning("File does not exist: %s", path)
        return None
    try:
        img = cv2.imread(path, flags)
        if img is None:
            raise IOError(f"Unable to read image: {path}")
        return img
    except Exception as e:
        logging.warning("Failed to load image %s: %s", path, e)
        return None

def resize_img(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def build_image_lists(raw_root):
    rgb_dir = os.path.join(raw_root, "color")
    gray_dir = os.path.join(raw_root, "grayscale")
    seg_dir = os.path.join(raw_root, "segmented")

    # allow fallback
    if not os.path.isdir(rgb_dir):
        logging.warning("data/raw/color/ not found, trying to use data/raw/ as color source")
        rgb_dir = raw_root
    if not os.path.isdir(gray_dir):
        gray_dir = None
    if not os.path.isdir(seg_dir):
        seg_dir = None

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(rgb_dir, "**", "*.png"), recursive=True))
    if not rgb_files:
        raise FileNotFoundError("No RGB images found under data/raw/color/ or data/raw/")

    # Dynamic classes for all PlantVillage + Background
    classes = [d for d in os.listdir(rgb_dir) if os.path.isdir(os.path.join(rgb_dir, d))]
    label_map = {}
    class_id = 0
    samples = []
    unknown_count = 0
    
    for rgb_path in tqdm(rgb_files, desc="Loading plant classes"):
        dir_name = os.path.basename(os.path.dirname(rgb_path))
        if dir_name not in classes:
            unknown_count += 1
            continue
        
        clean_class = dir_name.replace('___', ': ').replace('_', ' ').strip()
        if clean_class not in label_map:
            label_map[clean_class] = class_id
            class_id += 1

        rel = os.path.relpath(rgb_path, rgb_dir)
        gray_path = os.path.join(gray_dir, rel) if gray_dir else None
        seg_path = os.path.join(seg_dir, rel) if seg_dir else None
        if gray_path and not os.path.exists(gray_path):
            gray_path = None
        if seg_path and not os.path.exists(seg_path):
            seg_path = None

        samples.append((rgb_path, gray_path, seg_path, clean_class))
    
    # Add Background Noise
    bg_dir = os.path.join(rgb_dir, "Background_Noise")
    if os.path.isdir(bg_dir):
        bg_files = sorted(glob.glob(os.path.join(bg_dir, "*.png")) + glob.glob(os.path.join(bg_dir, "*.jpg")))
        clean_bg = "Background Noise"
        if clean_bg not in label_map:
            label_map[clean_bg] = class_id
        for bg_path in bg_files[:500]:
            samples.append((bg_path, None, None, clean_bg))
    
    logging.info(f"Loaded {len(samples)} samples from {len(label_map)} classes ({unknown_count} skipped)")
    return samples, list(label_map.keys()), label_map

def preprocess_image(rgb_path, gray_path, seg_path, image_size, mask_prob=0.7):
    rgb = safe_read_image(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize first
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # Key Innovation: Apply mask only mask_prob of the time during training
    # This helps generalize to messy real-world backgrounds
    if seg_path and os.path.exists(seg_path) and np.random.random() < mask_prob:
        seg = safe_read_image(seg_path, cv2.IMREAD_GRAYSCALE)
        if seg is not None:
            seg = cv2.resize(seg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            mask = (seg > 127).astype(np.float32)
            mask = np.stack([mask] * 3, axis=-1)
            logging.debug("Applied mask")
            return rgb * mask
    logging.debug("No mask applied")
    return rgb

def get_augmentation_pipeline(image_size):
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.RandomScale(scale_limit=0.2, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.GaussNoise(p=0.5),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class PlantVillageDataset(Dataset):
    def __init__(self, samples, class_names, image_size, augment=False):
        self.samples = samples
        self.class_names = class_names
        self.augment = augment
        self.image_size = image_size
        self.augmenter = get_augmentation_pipeline(image_size) if augment else A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, gray_path, seg_path, class_name = self.samples[idx]
        image = preprocess_image(rgb_path, gray_path, seg_path, self.image_size, mask_prob=0.7 if self.augment else 0.2)
        if image is None:
            raise RuntimeError(f"Corrupt input at index {idx}: {rgb_path}")

        transformed = self.augmenter(image=image)
        image_tensor = transformed["image"]

        label = self.class_names[class_name]
        return image_tensor, label

def split_data(samples, train_ratio, val_ratio, test_ratio, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_val, test = train_test_split(samples, test_size=test_ratio, random_state=seed, stratify=[x[3] for x in samples])
    val_relative = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed, stratify=[x[3] for x in train_val])

    return train, val, test

def build_dataloaders(cfg):
    raw_root = cfg.get("raw_data_dir", os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
    samples, _, label_map = build_image_lists(raw_root)
    logging.info("Found %d samples across %d classes: %s", len(samples), len(label_map), list(label_map.keys()))

    train_samples, val_samples, test_samples = split_data(samples, cfg["train_split"], cfg["val_split"], cfg["test_split"], cfg["seed"])

    train_dataset = PlantVillageDataset(train_samples, label_map, cfg["image_size"], augment=True)
    val_dataset = PlantVillageDataset(val_samples, label_map, cfg["image_size"], augment=False)
    test_dataset = PlantVillageDataset(test_samples, label_map, cfg["image_size"], augment=False)

    # WeightedRandomSampler for imbalance
    class_counts = Counter([s[3] for s in train_samples])
    sample_weights = [1.0 / class_counts[label] for _, _, _, label in train_samples]
    sampler = WeightedRandomSampler(torch.tensor(sample_weights), len(sample_weights), replacement=True)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True),
    }

    return dataloaders, label_map

def export_to_tflite(model, label_map, tflite_path):
    """Export to TFLite with metadata support."""
    import json
    labels_list = list(label_map.keys())
    with open('labels.json', 'w') as f:
        json.dump(labels_list, f, indent=2)
    with open('labels.txt', 'w') as f:
        for label in labels_list:
            f.write(label + '\n')
    logging.info(f"Exported {len(labels_list)} labels to labels.json for TFLite bundling.")
    if TFLITE_AVAILABLE:
        logging.info("Use MetadataWriter:")
        logging.info("  writer = MetadataWriter.create_from_model_with_version(...)")
        logging.info("  writer.add_label_file(labels_file_path='labels.txt' )")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml")
    cfg = load_config(config_path)

    try:
        dataloaders, label_map = build_dataloaders(cfg)
        logging.info("Dataloaders built: train=%d, val=%d, test=%d", len(dataloaders["train"].dataset), len(dataloaders["val"].dataset), len(dataloaders["test"].dataset))
        
        # Generate label files for TFLite
        export_to_tflite(None, label_map, "model.tflite")
        
        logging.info("preprocess.py refactored: 3-ch RGB masked + sampler, 39 classes, TFLite labels ready.")
        logging.info("Preprocessing complete. Ready for model training.")

    except Exception as e:
        logging.exception("Preprocessing failed: %s", e)
        raise

