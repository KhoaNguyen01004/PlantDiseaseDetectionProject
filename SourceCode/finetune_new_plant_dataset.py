#!/usr/bin/env python3
"""Preprocess a new field dataset and fine-tune the 39-class EfficientNet-B2 model.

This script keeps the original 39-label layout from labels.json, preprocesses only
new class folders that match that layout, and blends historical PlantVillage /
Unknown samples so full fine-tuning does not train only on the new domain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


LOGGER = logging.getLogger("finetune_new_plant_dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    class_name: str
    source: str


class HybridImageDataset(Dataset):
    def __init__(
        self,
        samples: list[ImageSample],
        new_transform: transforms.Compose,
        historical_transform: transforms.Compose,
    ):
        self.samples = samples
        self.new_transform = new_transform
        self.historical_transform = historical_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")

        transform = self.new_transform if sample.source == "new_preprocessed" else self.historical_transform
        return transform(image), sample.label


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_class_names(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    raw_labels = load_json(labels_path)
    class_names = [
        raw_labels[str(index)]
        for index in range(len(raw_labels))
        if str(index) in raw_labels
    ]

    if len(class_names) != len(raw_labels):
        raise ValueError(f"{labels_path} must contain contiguous integer string keys.")
    if len(class_names) != 39:
        raise ValueError(f"Expected 39 classes from labels.json, found {len(class_names)}.")
    if class_names[-1] != "Unknown":
        raise ValueError("The last label must be 'Unknown' to preserve the checkpoint layout.")

    return class_names


def iter_images(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return (
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images(directory: Path) -> int:
    return sum(1 for _ in iter_images(directory))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remove_tree(path: Path) -> None:
    def handle_remove_error(function, failed_path, _exc_info):
        os.chmod(failed_path, stat.S_IWRITE)
        function(failed_path)

    if path.exists():
        shutil.rmtree(path, onerror=handle_remove_error)


def split_root_name(seed: int, test_ratio: float) -> str:
    return f"split_seed{seed}_test{int(round(test_ratio * 100))}"


def split_root_name_with_val(seed: int, val_ratio: float, test_ratio: float) -> str:
    return f"split_seed{seed}_val{int(round(val_ratio * 100))}_test{int(round(test_ratio * 100))}"


def choose_fresh_split_root(base_root: Path) -> Path:
    if not base_root.exists():
        return base_root
    for index in range(2, 100):
        candidate = base_root.with_name(f"{base_root.name}_v{index}")
        if not candidate.exists():
            LOGGER.warning("Using fresh split directory because %s already exists: %s", base_root, candidate)
            return candidate
    raise RuntimeError(f"Could not find an available split directory near {base_root}")


def find_existing_split_root(preprocessed_root: Path, seed: int, val_ratio: float, test_ratio: float) -> Path:
    direct_has_split = (preprocessed_root / "train").exists() and (preprocessed_root / "val").exists()
    if direct_has_split:
        return preprocessed_root

    base_name = split_root_name_with_val(seed, val_ratio, test_ratio)
    candidates = sorted(
        path
        for path in preprocessed_root.glob(f"{base_name}*")
        if (path / "train").exists() and (path / "val").exists() and (path / "test").exists()
    )
    if candidates:
        return candidates[-1]

    legacy_base_name = split_root_name(seed, test_ratio)
    legacy_candidates = sorted(
        path
        for path in preprocessed_root.glob(f"{legacy_base_name}*")
        if (path / "train").exists() and (path / "test").exists()
    )
    if legacy_candidates:
        return legacy_candidates[-1]
    return preprocessed_root


def write_manifest(manifest_path: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "class_name",
        "label",
        "source_path",
        "preprocessed_path",
        "source_sha256",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_manifest_leakage(split_root: Path) -> None:
    sha_to_split: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for split_name in ("train", "val", "test"):
        manifest_path = split_root / f"{split_name}_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
        for row in read_manifest(manifest_path):
            source_sha = row["source_sha256"]
            previous_split = sha_to_split.get(source_sha)
            if previous_split and previous_split != split_name:
                duplicates.append((source_sha, previous_split, split_name))
            sha_to_split[source_sha] = split_name

    if duplicates:
        sample = ", ".join(f"{sha[:10]}:{left}->{right}" for sha, left, right in duplicates[:5])
        raise RuntimeError(f"Data leakage detected across new-domain splits: {sample}")
    LOGGER.info("Manifest leakage check passed for %s", split_root)


def validate_required_paths(
    old_root: Path,
    new_root: Path,
    unknown_root: Path,
    checkpoint_path: Path,
    labels_path: Path,
) -> None:
    LOGGER.info("Validating required directories and files...")
    for path, description in [
        (old_root, "baseline PlantVillage color dataset"),
        (new_root, "new in-the-wild color dataset"),
        (unknown_root, "historical Unknown dataset"),
        (checkpoint_path, "baseline checkpoint"),
        (labels_path, "39-class labels.json"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
        LOGGER.info("OK: %s -> %s", description, path)


def maybe_apply_segmentation(rgb: Image.Image, source_path: Path, segmented_root: Path | None, source_root: Path) -> Image.Image:
    if segmented_root is None or not segmented_root.exists():
        return rgb

    try:
        relative = source_path.relative_to(source_root)
    except ValueError:
        return rgb

    seg_path = segmented_root / relative
    if not seg_path.exists():
        return rgb

    try:
        with Image.open(seg_path) as mask_image:
            mask = ImageOps.exif_transpose(mask_image).convert("L").resize(rgb.size, Image.Resampling.NEAREST)
        black = Image.new("RGB", rgb.size)
        return Image.composite(rgb, black, mask.point(lambda value: 255 if value > 127 else 0))
    except (OSError, UnidentifiedImageError) as exc:
        LOGGER.warning("Skipping segmentation mask for %s: %s", source_path, exc)
        return rgb


def split_class_images(
    image_paths: list[Path],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_paths:
        grouped[sha256_file(image_path)].append(image_path)

    groups = list(grouped.values())
    rng.shuffle(groups)
    shuffled = [path for group in groups for path in group]
    group_count = len(groups)
    if group_count < len(image_paths):
        LOGGER.info(
            "Grouped %d images into %d unique file-hash groups before splitting.",
            len(image_paths),
            group_count,
        )

    total = len(shuffled)
    if total <= 1:
        return shuffled, [], []
    if group_count <= 1:
        return shuffled, [], []
    if group_count == 2:
        return groups[0], [], groups[1]

    val_group_count = max(1, int(round(group_count * val_ratio)))
    test_group_count = max(1, int(round(group_count * test_ratio)))
    if val_group_count + test_group_count >= group_count:
        overflow = val_group_count + test_group_count - group_count + 1
        if test_group_count >= val_group_count:
            test_group_count = max(1, test_group_count - overflow)
        else:
            val_group_count = max(1, val_group_count - overflow)

    val_groups = groups[:val_group_count]
    test_groups = groups[val_group_count: val_group_count + test_group_count]
    train_groups = groups[val_group_count + test_group_count:]
    val_paths = [path for group in val_groups for path in group]
    test_paths = [path for group in test_groups for path in group]
    train_paths = [path for group in train_groups for path in group]
    return train_paths, val_paths, test_paths


def preprocess_new_dataset(
    new_root: Path,
    output_root: Path,
    class_to_index: dict[str, int],
    image_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    segmented_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    direct_train_output_root = output_root / "train"
    direct_val_output_root = output_root / "val"
    direct_test_output_root = output_root / "test"

    if not 0.0 < val_ratio < 0.5:
        raise ValueError("--new-val-ratio must be between 0 and 0.5")
    if not 0.0 < test_ratio < 0.5:
        raise ValueError("--new-test-ratio must be between 0 and 0.5")
    if val_ratio + test_ratio >= 0.8:
        raise ValueError("--new-val-ratio + --new-test-ratio must leave at least 20% for training")

    split_exists = (
        count_images(direct_train_output_root) > 0
        and count_images(direct_val_output_root) > 0
        and count_images(direct_test_output_root) > 0
        and all((output_root / f"{split_name}_manifest.csv").exists() for split_name in ("train", "val", "test"))
    )
    if split_exists and not overwrite:
        LOGGER.info(
            "Reusing existing preprocessed new split: train=%d val=%d test=%d",
            count_images(direct_train_output_root),
            count_images(direct_val_output_root),
            count_images(direct_test_output_root),
        )
        validate_manifest_leakage(output_root)
        return output_root

    target_root = output_root / split_root_name_with_val(seed, val_ratio, test_ratio)
    direct_has_generated_images = (
        count_images(direct_train_output_root) > 0
        or count_images(direct_val_output_root) > 0
        or count_images(direct_test_output_root) > 0
    )
    if direct_has_generated_images and not overwrite:
        LOGGER.warning(
            "Existing direct preprocessed data is missing the audited train/val/test manifest set. "
            "Writing an isolated split under %s instead of deleting existing artifacts.",
            target_root,
        )
        target_root = choose_fresh_split_root(target_root)
    elif overwrite:
        target_root = output_root

    train_output_root = target_root / "train"
    val_output_root = target_root / "val"
    test_output_root = target_root / "test"
    for split_root in (train_output_root, val_output_root, test_output_root):
        remove_tree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)

    valid_class_names = set(class_to_index)
    present_counts: dict[str, dict[str, int]] = {}
    skipped_dirs: list[str] = []
    manifest_rows: dict[str, list[dict[str, str | int]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    LOGGER.info("Preprocessing new dataset from %s", new_root)
    for class_dir in sorted(path for path in new_root.iterdir() if path.is_dir()):
        class_name = class_dir.name
        if class_name not in valid_class_names:
            skipped_dirs.append(class_name)
            LOGGER.warning("Skipping unmatched new class folder: %s", class_name)
            continue

        image_paths = list(iter_images(class_dir))
        train_paths, val_paths, test_paths = split_class_images(image_paths, val_ratio, test_ratio, rng)
        class_counts = {"train": 0, "val": 0, "test": 0}

        for split_name, split_paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            output_class_dir = target_root / split_name / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in tqdm(split_paths, desc=f"Preprocess {split_name}/{class_name}", leave=False):
                output_path = output_class_dir / f"{image_path.stem}.jpg"
                try:
                    with Image.open(image_path) as image:
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image = maybe_apply_segmentation(image, image_path, segmented_root, new_root)
                        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
                        image.save(output_path, format="JPEG", quality=95, optimize=True)
                    class_counts[split_name] += 1
                    manifest_rows[split_name].append(
                        {
                            "split": split_name,
                            "class_name": class_name,
                            "label": class_to_index[class_name],
                            "source_path": str(image_path),
                            "preprocessed_path": str(output_path),
                            "source_sha256": sha256_file(image_path),
                        }
                    )
                except (OSError, UnidentifiedImageError) as exc:
                    LOGGER.warning("Skipping unreadable image %s: %s", image_path, exc)

        present_counts[class_name] = class_counts
        LOGGER.info(
            "Preprocessed %s: train=%d val=%d test=%d",
            class_name,
            class_counts["train"],
            class_counts["val"],
            class_counts["test"],
        )

    if skipped_dirs:
        LOGGER.warning("Skipped %d unmatched class folders: %s", len(skipped_dirs), ", ".join(skipped_dirs))

    for split_name, rows in manifest_rows.items():
        if split_name != "train" and not rows:
            raise RuntimeError(f"New dataset {split_name} split is empty; cannot measure new-domain generalization.")
        write_manifest(target_root / f"{split_name}_manifest.csv", rows)
        LOGGER.info("Wrote %s manifest with %d rows", split_name, len(rows))
    validate_manifest_leakage(target_root)

    LOGGER.info(
        "New preprocessed output ready: train=%s val=%s test=%s",
        train_output_root,
        val_output_root,
        test_output_root,
    )
    LOGGER.info("Matched new classes: %d", len(present_counts))
    return target_root


def collect_class_samples(root: Path, class_to_index: dict[str, int], source: str) -> dict[str, list[ImageSample]]:
    samples_by_class: dict[str, list[ImageSample]] = defaultdict(list)
    for class_name, label in class_to_index.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        samples_by_class[class_name].extend(
            ImageSample(path=path, label=label, class_name=class_name, source=source)
            for path in iter_images(class_dir)
        )
    return dict(samples_by_class)


def collect_unknown_samples(unknown_root: Path, unknown_label: int) -> list[ImageSample]:
    return [
        ImageSample(path=path, label=unknown_label, class_name="Unknown", source="historical_unknown")
        for path in iter_images(unknown_root)
    ]


def sample_balanced(
    candidates: list[ImageSample],
    count: int,
    rng: random.Random,
    allow_upsample: bool = True,
) -> list[ImageSample]:
    if count <= 0 or not candidates:
        return []
    if len(candidates) >= count:
        return rng.sample(candidates, count)
    if not allow_upsample:
        return list(candidates)
    return [rng.choice(candidates) for _ in range(count)]


def build_hybrid_samples(
    preprocessed_root: Path,
    old_root: Path,
    unknown_root: Path,
    class_names: list[str],
    historical_samples_per_class: int,
    max_new_samples_per_class: int,
    seed: int,
) -> list[ImageSample]:
    class_to_index = {name: index for index, name in enumerate(class_names)}
    disease_class_names = [name for name in class_names if name != "Unknown"]
    rng = random.Random(seed)

    new_by_class = collect_class_samples(preprocessed_root / "train", class_to_index, "new_preprocessed")
    old_by_class = collect_class_samples(old_root, class_to_index, "historical_plantvillage")
    unknown_candidates = collect_unknown_samples(unknown_root, class_to_index["Unknown"])

    samples: list[ImageSample] = []
    LOGGER.info("Building hybrid anti-forgetting sample set...")

    for class_name in disease_class_names:
        new_samples = list(new_by_class.get(class_name, []))
        if max_new_samples_per_class > 0 and len(new_samples) > max_new_samples_per_class:
            new_samples = rng.sample(new_samples, max_new_samples_per_class)

        historical = sample_balanced(
            old_by_class.get(class_name, []),
            historical_samples_per_class,
            rng,
            allow_upsample=True,
        )

        if not new_samples and not historical:
            raise FileNotFoundError(
                f"No samples available for required class '{class_name}' in new or old datasets."
            )

        samples.extend(new_samples)
        samples.extend(historical)
        LOGGER.info(
            "%s: new=%d historical=%d total=%d",
            class_name,
            len(new_samples),
            len(historical),
            len(new_samples) + len(historical),
        )

    unknown_historical = sample_balanced(
        unknown_candidates,
        historical_samples_per_class,
        rng,
        allow_upsample=True,
    )
    if not unknown_historical:
        raise FileNotFoundError(f"No Unknown images found under {unknown_root}")

    samples.extend(unknown_historical)
    LOGGER.info(
        "Unknown: available=%d sampled=%d",
        len(unknown_candidates),
        len(unknown_historical),
    )

    class_counts = Counter(sample.class_name for sample in samples)
    LOGGER.info("Unified 39-class sample layout built with %d total images.", len(samples))
    LOGGER.info("Per-class count range: min=%d max=%d", min(class_counts.values()), max(class_counts.values()))
    return samples


def stratified_train_val_split(
    samples: list[ImageSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample]]:
    by_label: dict[int, list[ImageSample]] = defaultdict(list)
    rng = random.Random(seed)
    for sample in samples:
        by_label[sample.label].append(sample)

    train_samples: list[ImageSample] = []
    val_samples: list[ImageSample] = []
    for label, label_samples in by_label.items():
        rng.shuffle(label_samples)
        val_count = max(1, int(round(len(label_samples) * val_ratio))) if len(label_samples) > 1 else 0
        val_samples.extend(label_samples[:val_count])
        train_samples.extend(label_samples[val_count:])
        LOGGER.debug("Label %d split into train=%d val=%d", label, len(label_samples) - val_count, val_count)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def get_transforms(image_size: int, mean: list[float], std: list[float]) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    new_train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), value="random"),
        ]
    )
    historical_train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return new_train_transform, historical_train_transform, val_transform


def compute_class_weights(
    train_samples: list[ImageSample],
    num_classes: int,
    beta: float,
    max_weight: float,
) -> torch.Tensor:
    label_counts = Counter(sample.label for sample in train_samples)
    if len(label_counts) != num_classes:
        missing_labels = sorted(set(range(num_classes)) - set(label_counts))
        raise ValueError(f"Training samples are missing labels required for class weights: {missing_labels}")

    if not 0.0 <= beta < 1.0:
        raise ValueError("class_weight_beta must be in [0, 1)")

    weights = torch.tensor(
        [
            (1.0 - beta) / (1.0 - beta ** label_counts[label]) if beta > 0 else 1.0 / label_counts[label]
            for label in range(num_classes)
        ],
        dtype=torch.float32,
    )
    weights = weights / weights.mean()
    if max_weight > 0:
        weights = torch.clamp(weights, max=max_weight)
        weights = weights / weights.mean()
    LOGGER.info(
        "Class-balanced loss enabled: beta=%.5f min_weight=%.4f max_weight=%.4f",
        beta,
        float(weights.min().item()),
        float(weights.max().item()),
    )
    return weights


def build_training_loader(
    samples: list[ImageSample],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    mean: list[float],
    std: list[float],
    class_weight_beta: float,
    max_class_weight: float,
    use_balanced_sampler: bool,
) -> tuple[DataLoader, torch.Tensor]:
    new_train_transform, historical_train_transform, _ = get_transforms(image_size, mean, std)

    train_dataset = HybridImageDataset(samples, new_train_transform, historical_train_transform)

    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if use_balanced_sampler:
        train_label_counts = Counter(sample.label for sample in samples)
        weights = torch.DoubleTensor([1.0 / train_label_counts[sample.label] for sample in samples])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        LOGGER.warning("Balanced sampler enabled together with class-balanced loss; monitor for overcorrection.")

    class_weights = compute_class_weights(
        samples,
        num_classes=len(set(sample.label for sample in samples)),
        beta=class_weight_beta,
        max_weight=max_class_weight,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )

    LOGGER.info("Training DataLoader ready: train=%d batch_size=%d", len(train_dataset), batch_size)
    LOGGER.info("New-domain augmentation applies only to source='new_preprocessed' training samples.")
    return train_loader, class_weights


def build_new_eval_loader(
    split_root: Path,
    class_to_index: dict[str, int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: list[float],
    std: list[float],
    split_name: str,
) -> DataLoader:
    _, _, val_transform = get_transforms(image_size, mean, std)
    samples_by_class = collect_class_samples(split_root / split_name, class_to_index, "new_preprocessed")
    samples = [sample for class_samples in samples_by_class.values() for sample in class_samples]
    if not samples:
        raise FileNotFoundError(f"No new-domain {split_name} samples found under {split_root / split_name}")

    dataset = HybridImageDataset(samples, val_transform, val_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    class_counts = Counter(sample.class_name for sample in samples)
    LOGGER.info("New-domain %s DataLoader ready: samples=%d classes=%d", split_name, len(dataset), len(class_counts))
    LOGGER.info("%s transform is strict Resize -> ToTensor -> Normalize.", split_name.capitalize())
    return loader


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = True
    return model


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint(model: nn.Module, checkpoint_path: Path, expected_num_classes: int, expected_class_names: list[str]) -> dict:
    LOGGER.info("Loading baseline checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a model state dict.")

    model.load_state_dict(normalize_state_dict_keys(state_dict), strict=True)

    if isinstance(checkpoint, dict):
        checkpoint_classes = checkpoint.get("class_names")
        checkpoint_num_classes = checkpoint.get("num_classes")
        if checkpoint_num_classes is not None and int(checkpoint_num_classes) != expected_num_classes:
            raise ValueError(f"Checkpoint has num_classes={checkpoint_num_classes}, expected {expected_num_classes}")
        if checkpoint_classes and list(checkpoint_classes) != expected_class_names:
            raise ValueError("Checkpoint class_names differ from labels.json; refusing to change classifier layout.")
        LOGGER.info(
            "Checkpoint confirmed: architecture=%s num_classes=%s best_acc=%s",
            checkpoint.get("architecture", "unknown"),
            checkpoint_num_classes,
            checkpoint.get("best_acc", "unknown"),
        )
    return checkpoint if isinstance(checkpoint, dict) else {}


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()
            predictions = outputs.argmax(dim=1)
            total += labels.numel()
            correct += (predictions == labels).sum().item()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    avg_loss = loss_total / max(len(data_loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    macro_f1 = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1


def freeze_backbone_for_head_training(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    LOGGER.info("Stage A: EfficientNet-B2 backbone frozen; classifier head trainable.")


def unfreeze_all_layers(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    LOGGER.info("Stage B: all EfficientNet-B2 layers unfrozen for full fine-tuning.")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    total_epochs: int,
    head_epochs: int,
    head_lr: float,
    full_lr: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    output_path: Path,
    class_names: list[str],
    device: torch.device,
) -> None:
    if total_epochs < 1:
        raise ValueError("total_epochs must be at least 1")
    if head_epochs < 0 or head_epochs >= total_epochs:
        raise ValueError("head_epochs must be >= 0 and less than total_epochs")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1,
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_val_macro_f1 = -1.0
    history: list[dict[str, float | int]] = []
    optimizer: torch.optim.Optimizer | None = None
    active_stage = ""

    for epoch in range(1, total_epochs + 1):
        if epoch <= head_epochs:
            stage_name = "Stage A - Head Stabilization"
            learning_rate = head_lr
            if active_stage != stage_name:
                freeze_backbone_for_head_training(model)
                optimizer = torch.optim.Adam(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
                active_stage = stage_name
                LOGGER.info(
                    "%s active for epochs 1-%d | lr=%s | trainable_params=%d",
                    stage_name,
                    head_epochs,
                    learning_rate,
                    count_trainable_parameters(model),
                )
        else:
            stage_name = "Stage B - Full Discriminative Fine-Tuning"
            learning_rate = full_lr
            if active_stage != stage_name:
                unfreeze_all_layers(model)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                )
                active_stage = stage_name
                LOGGER.info(
                    "%s active for epochs %d-%d | lr=%s | trainable_params=%d",
                    stage_name,
                    head_epochs + 1,
                    total_epochs,
                    learning_rate,
                    count_trainable_parameters(model),
                )

        if optimizer is None:
            raise RuntimeError("Optimizer was not initialized.")

        model.train()
        train_loss_total = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_total / max(len(train_loader), 1)
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        epoch_metrics = {
            "epoch": epoch,
            "stage": stage_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_macro_f1,
            "learning_rate": learning_rate,
        }
        history.append(epoch_metrics)

        LOGGER.info(
            "Epoch %d/%d | %s | Train Loss: %.4f | New Val Loss: %.4f | New Val Acc: %.2f%% | New Val Macro F1: %.2f%%",
            epoch,
            total_epochs,
            stage_name,
            train_loss,
            val_loss,
            val_acc,
            val_macro_f1,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": val_acc,
                    "best_macro_f1": best_val_macro_f1,
                    "best_epoch": epoch,
                    "architecture": "efficientnet_b2",
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "history": history,
                    "fine_tuned_from": "models/best_model.pth",
                    "optimizer": "Adam",
                    "head_epochs": head_epochs,
                    "head_learning_rate": head_lr,
                    "full_learning_rate": full_lr,
                    "class_weighted_loss": True,
                    "selection_metric": "new_val_macro_f1",
                },
                output_path,
            )
            LOGGER.info(
                "Saved best fine-tuned checkpoint to %s (New Val Macro F1: %.2f%% | Acc: %.2f%%)",
                output_path,
                best_val_macro_f1,
                val_acc,
            )


def parse_args() -> argparse.Namespace:
    root = resolve_project_root()
    parser = argparse.ArgumentParser(
        description="Preprocess NewPLantDataset and fine-tune EfficientNet-B2 with anti-forgetting historical sampling."
    )
    parser.add_argument("--old-root", type=Path, default=root / "data/plantvillage/plantvillage dataset/color")
    parser.add_argument("--new-root", type=Path, default=root / "data/NewPLantDataset/color")
    parser.add_argument("--preprocessed-root", type=Path, default=root / "data/NewPLantDataset_preprocessed")
    parser.add_argument("--unknown-root", type=Path, default=root / "data/unknown")
    parser.add_argument("--labels-path", type=Path, default=root / "labels.json")
    parser.add_argument("--checkpoint-path", type=Path, default=root / "models/best_model.pth")
    parser.add_argument("--output-path", type=Path, default=root / "models/best_model_finetuned.pth")
    parser.add_argument("--segmented-root", type=Path, default=None, help="Optional segmented folder matching the new color tree.")
    parser.add_argument("--image-size", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8, help="Total staged fine-tuning epochs.")
    parser.add_argument("--head-epochs", type=int, default=2, help="Initial epochs with the backbone frozen.")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for classifier-head stabilization.")
    parser.add_argument("--full-lr", "--lr", dest="full_lr", type=float, default=5e-6, help="Learning rate after unfreezing all layers.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--historical-samples-per-class", type=int, default=256)
    parser.add_argument("--max-new-samples-per-class", type=int, default=0, help="0 keeps all new images per matched class.")
    parser.add_argument("--new-val-ratio", type=float, default=0.15, help="Held-out fraction of each new class used for model selection.")
    parser.add_argument("--new-test-ratio", type=float, default=0.2, help="Held-out fraction of each new class reserved for final evaluation.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Deprecated compatibility alias; use --new-val-ratio.")
    parser.add_argument("--class-weight-beta", type=float, default=0.999, help="Class-balanced loss beta; set 0 for plain inverse-count weighting.")
    parser.add_argument("--max-class-weight", type=float, default=4.0, help="Clamp class-balanced weights before renormalization; 0 disables clamping.")
    parser.add_argument("--use-balanced-sampler", action="store_true", help="Also use a balanced sampler. Disabled by default to avoid overcorrection.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-preprocessed", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate preprocessing, sampling, loaders, and checkpoint without training.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.head_epochs < 0 or args.head_epochs >= args.epochs:
        raise ValueError("--head-epochs must be >= 0 and less than --epochs")
    if args.full_lr > 1e-4:
        LOGGER.warning("Requested full_lr=%s is larger than the conservative fine-tuning range.", args.full_lr)
    if not 0.0 < args.new_val_ratio < 0.5:
        raise ValueError("--new-val-ratio must be between 0 and 0.5")
    if not 0.0 < args.new_test_ratio < 0.5:
        raise ValueError("--new-test-ratio must be between 0 and 0.5")
    if args.new_val_ratio + args.new_test_ratio >= 0.8:
        raise ValueError("--new-val-ratio + --new-test-ratio must leave at least 20% for training")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    class_names = load_class_names(args.labels_path)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    LOGGER.info("Loaded %d target classes from %s", len(class_names), args.labels_path)

    validate_required_paths(args.old_root, args.new_root, args.unknown_root, args.checkpoint_path, args.labels_path)

    old_dirs = {path.name for path in args.old_root.iterdir() if path.is_dir()}
    missing_old = [name for name in class_names if name != "Unknown" and name not in old_dirs]
    if missing_old:
        raise FileNotFoundError("PlantVillage is missing required class directories: " + ", ".join(missing_old))
    LOGGER.info("PlantVillage class directories align with the first 38 labels.")

    split_preprocessed_root = args.preprocessed_root
    if not args.skip_preprocess:
        split_preprocessed_root = preprocess_new_dataset(
            new_root=args.new_root,
            output_root=args.preprocessed_root,
            class_to_index=class_to_index,
            image_size=args.image_size,
            val_ratio=args.new_val_ratio,
            test_ratio=args.new_test_ratio,
            seed=args.seed,
            segmented_root=args.segmented_root,
            overwrite=args.overwrite_preprocessed,
        )
    else:
        split_preprocessed_root = find_existing_split_root(
            args.preprocessed_root,
            args.seed,
            args.new_val_ratio,
            args.new_test_ratio,
        )
        LOGGER.info(
            "Skipping preprocessing; training uses %s, validation uses %s, evaluation should use %s",
            split_preprocessed_root / "train",
            split_preprocessed_root / "val",
            split_preprocessed_root / "test",
        )
        if (split_preprocessed_root / "train_manifest.csv").exists():
            validate_manifest_leakage(split_preprocessed_root)

    hybrid_samples = build_hybrid_samples(
        preprocessed_root=split_preprocessed_root,
        old_root=args.old_root,
        unknown_root=args.unknown_root,
        class_names=class_names,
        historical_samples_per_class=args.historical_samples_per_class,
        max_new_samples_per_class=args.max_new_samples_per_class,
        seed=args.seed,
    )

    train_loader, class_weights = build_training_loader(
        samples=hybrid_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        class_weight_beta=args.class_weight_beta,
        max_class_weight=args.max_class_weight,
        use_balanced_sampler=args.use_balanced_sampler,
    )
    val_loader = build_new_eval_loader(
        split_root=split_preprocessed_root,
        class_to_index=class_to_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        split_name="val",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    if torch.cuda.is_available():
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))

    model = build_model(num_classes=len(class_names))
    load_checkpoint(model, args.checkpoint_path, len(class_names), class_names)

    if args.dry_run:
        LOGGER.info("Dry run complete; checkpoint and hybrid DataLoaders are valid. Training was not started.")
        return

    model = model.to(device)

    LOGGER.info(
        "Starting staged fine-tuning: epochs=%d head_epochs=%d head_lr=%s full_lr=%s optimizer=Adam output=%s",
        args.epochs,
        args.head_epochs,
        args.head_lr,
        args.full_lr,
        args.output_path,
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        total_epochs=args.epochs,
        head_epochs=args.head_epochs,
        head_lr=args.head_lr,
        full_lr=args.full_lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        output_path=args.output_path,
        class_names=class_names,
        device=device,
    )
    LOGGER.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
