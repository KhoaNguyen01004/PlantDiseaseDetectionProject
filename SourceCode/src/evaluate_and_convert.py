"""Evaluate the fine-tuned model on the new dataset test split and export it.

The old version of this module reused the training dataloader builder, which
read configs/config.yaml and evaluated against the PlantVillage benchmark split.
This version is explicit about the checkpoint and evaluation dataset paths so a
fine-tuned model is tested on held-out in-the-wild images.
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
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a project dependency.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOGGER = logging.getLogger(__name__)


class LabelMappedImageFolder(Dataset):
    """ImageFolder-like dataset that preserves a fixed checkpoint label map."""

    def __init__(
        self,
        root: Path,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.root = root
        self.class_to_index = class_to_index
        self.transform = transform
        self.samples = self._collect_samples()

        if not self.samples:
            raise FileNotFoundError(f"No evaluation images found under {root}")

    def _collect_samples(self) -> list[tuple[Path, int, str]]:
        samples: list[tuple[Path, int, str]] = []
        unknown_dirs: list[str] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Evaluation split does not exist: {self.root}")

        for class_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            class_name = class_dir.name
            if class_name not in self.class_to_index:
                unknown_dirs.append(class_name)
                continue

            label = self.class_to_index[class_name]
            for image_path in iter_images(class_dir):
                samples.append((image_path, label, class_name))

        if unknown_dirs:
            LOGGER.warning(
                "Ignoring evaluation folders not present in labels.json/checkpoint: %s",
                ", ".join(unknown_dirs),
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, _ = self.samples[index]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), label


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


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
    if (preprocessed_root / "train").exists() and (preprocessed_root / "val").exists() and (preprocessed_root / "test").exists():
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
    fieldnames = ["split", "class_name", "label", "source_path", "preprocessed_path", "source_sha256"]
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


def load_config(config_path: Path) -> dict:
    if yaml is None:
        return {}
    if not config_path.exists():
        LOGGER.warning("Config not found: %s. Using script defaults.", config_path)
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as handle:
        raw_labels = json.load(handle)

    labels = [
        raw_labels[str(index)]
        for index in range(len(raw_labels))
        if str(index) in raw_labels
    ]
    if len(labels) != len(raw_labels):
        raise ValueError(f"{labels_path} must contain contiguous integer string keys.")
    return labels


def build_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
    elif architecture == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
    elif architecture == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint_model(
    checkpoint_path: Path,
    labels_path: Path,
    fallback_architecture: str,
    device: torch.device,
) -> tuple[nn.Module, list[str], dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train or pass --checkpoint-path models/best_model_finetuned.pth first."
        )

    LOGGER.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dict.")
    state_dict = normalize_state_dict_keys(state_dict)

    checkpoint_class_names = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    class_names = list(checkpoint_class_names) if checkpoint_class_names else load_labels(labels_path)
    classifier_weight = state_dict.get("classifier.1.weight")
    if classifier_weight is None:
        raise KeyError("Checkpoint is missing classifier.1.weight; cannot infer class count.")
    num_classes = int(classifier_weight.shape[0])

    if len(class_names) != num_classes:
        raise ValueError(
            f"Label count mismatch: checkpoint has {num_classes} outputs but labels provide {len(class_names)} names."
        )

    architecture = (
        checkpoint.get("architecture")
        if isinstance(checkpoint, dict) and checkpoint.get("architecture")
        else fallback_architecture
    )
    model = build_model(architecture, num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    LOGGER.info("Checkpoint confirmed: architecture=%s num_classes=%d", architecture, num_classes)
    if isinstance(checkpoint, dict) and checkpoint.get("best_acc") is not None:
        LOGGER.info("Checkpoint best validation accuracy metadata: %.4f", float(checkpoint["best_acc"]))
    return model, class_names, checkpoint if isinstance(checkpoint, dict) else {}


def split_class_images(
    image_paths: list[Path],
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for image_path in image_paths:
        grouped.setdefault(sha256_file(image_path), []).append(image_path)

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


def save_preprocessed_image(source_path: Path, destination_path: Path, image_size: int) -> bool:
    try:
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination_path, format="JPEG", quality=95, optimize=True)
        return True
    except (OSError, UnidentifiedImageError) as exc:
        LOGGER.warning("Skipping unreadable image %s: %s", source_path, exc)
        return False


def prepare_new_dataset_split(
    raw_new_root: Path,
    preprocessed_root: Path,
    class_names: list[str],
    image_size: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    """Create deterministic train/val/test splits for the new dataset.

    The generated test split is the only split used for final evaluation. The
    val split is reserved for model selection during fine-tuning.
    """

    if not raw_new_root.exists():
        raise FileNotFoundError(f"Raw new dataset not found: {raw_new_root}")
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("--new-val-ratio must be between 0 and 0.5")
    if not 0.0 < test_ratio < 0.5:
        raise ValueError("--new-test-ratio must be between 0 and 0.5")
    if val_ratio + test_ratio >= 0.8:
        raise ValueError("--new-val-ratio + --new-test-ratio must leave at least 20% for training")

    direct_train_root = preprocessed_root / "train"
    direct_val_root = preprocessed_root / "val"
    direct_test_root = preprocessed_root / "test"
    split_exists = (
        count_images(direct_train_root) > 0
        and count_images(direct_val_root) > 0
        and count_images(direct_test_root) > 0
        and all((preprocessed_root / f"{split_name}_manifest.csv").exists() for split_name in ("train", "val", "test"))
    )

    if split_exists and not overwrite:
        LOGGER.info(
            "Reusing existing new dataset split: train=%d val=%d test=%d",
            count_images(direct_train_root),
            count_images(direct_val_root),
            count_images(direct_test_root),
        )
        validate_manifest_leakage(preprocessed_root)
        return direct_train_root, direct_val_root, direct_test_root

    target_root = preprocessed_root / split_root_name_with_val(seed, val_ratio, test_ratio)
    direct_has_generated_images = (
        count_images(direct_train_root) > 0
        or count_images(direct_val_root) > 0
        or count_images(direct_test_root) > 0
    )
    if direct_has_generated_images and not overwrite:
        LOGGER.warning(
            "Found direct preprocessed output without audited train/val/test manifests. "
            "Writing an isolated split under %s instead of deleting existing artifacts.",
            target_root,
        )
        target_root = choose_fresh_split_root(target_root)
    elif overwrite:
        target_root = preprocessed_root

    train_root = target_root / "train"
    val_root = target_root / "val"
    test_root = target_root / "test"

    for split_root in [train_root, val_root, test_root]:
        remove_tree(split_root)
        split_root.mkdir(parents=True, exist_ok=True)

    valid_classes = set(class_names)
    rng = random.Random(seed)
    total_train = 0
    total_val = 0
    total_test = 0
    matched_classes = 0
    class_to_index = {name: index for index, name in enumerate(class_names)}
    manifest_rows: dict[str, list[dict[str, str | int]]] = {"train": [], "val": [], "test": []}

    for class_dir in sorted(path for path in raw_new_root.iterdir() if path.is_dir()):
        class_name = class_dir.name
        if class_name not in valid_classes:
            LOGGER.warning("Skipping unmatched new class folder: %s", class_name)
            continue

        image_paths = list(iter_images(class_dir))
        if not image_paths:
            LOGGER.warning("Skipping empty new class folder: %s", class_name)
            continue

        train_paths, val_paths, test_paths = split_class_images(image_paths, val_ratio, test_ratio, rng)
        matched_classes += 1

        for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            for source_path in tqdm(paths, desc=f"Preprocess {split_name}/{class_name}", leave=False):
                destination = target_root / split_name / class_name / f"{source_path.stem}.jpg"
                if save_preprocessed_image(source_path, destination, image_size):
                    if split_name == "train":
                        total_train += 1
                    elif split_name == "val":
                        total_val += 1
                    else:
                        total_test += 1
                    manifest_rows[split_name].append(
                        {
                            "split": split_name,
                            "class_name": class_name,
                            "label": class_to_index[class_name],
                            "source_path": str(source_path),
                            "preprocessed_path": str(destination),
                            "source_sha256": sha256_file(source_path),
                        }
                    )

        LOGGER.info("%s: train=%d val=%d test=%d", class_name, len(train_paths), len(val_paths), len(test_paths))

    if total_val == 0:
        raise RuntimeError("New dataset val split is empty; cannot select a model on new-domain validation.")
    if total_test == 0:
        raise RuntimeError("New dataset test split is empty; cannot evaluate real-world performance.")

    for split_name, rows in manifest_rows.items():
        write_manifest(target_root / f"{split_name}_manifest.csv", rows)
        LOGGER.info("Wrote %s manifest with %d rows", split_name, len(rows))
    validate_manifest_leakage(target_root)

    LOGGER.info(
        "Prepared new dataset split at %s with %d matched classes, train=%d, val=%d, test=%d",
        target_root,
        matched_classes,
        total_train,
        total_val,
        total_test,
    )
    return train_root, val_root, test_root


def build_eval_loader(
    test_root: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    mean: list[float],
    std: list[float],
) -> tuple[DataLoader, LabelMappedImageFolder]:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = LabelMappedImageFolder(
        root=test_root,
        class_to_index={name: index for index, name in enumerate(class_names)},
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    class_counts = Counter(class_name for _, _, class_name in dataset.samples)
    LOGGER.info("Evaluation dataset: %s", test_root)
    LOGGER.info("Evaluation images: %d across %d present classes", len(dataset), len(class_counts))
    for class_name, count in sorted(class_counts.items()):
        LOGGER.info("  %s: %d", class_name, count)
    return loader, dataset


def run_evaluation(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    report_dir: Path,
) -> dict[str, float]:
    LOGGER.info("Running held-out new dataset evaluation...")
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="New Test Eval"):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = outputs.argmax(dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())
            y_score.extend(probabilities.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred) * 100.0
    top3_accuracy = top_k_accuracy_score(
        y_true,
        y_score,
        k=min(3, len(class_names)),
        labels=list(range(len(class_names))),
    ) * 100.0
    LOGGER.info("New Dataset Test Accuracy: %.2f%%", accuracy)
    LOGGER.info("New Dataset Top-3 Accuracy: %.2f%%", top3_accuracy)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    LOGGER.info("\nClassification Report:\n%s", report_text)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    with (report_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with (report_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())

    per_class_rows = []
    for class_name in class_names:
        metrics = report_dict[class_name]
        support = int(metrics["support"])
        if support > 0:
            per_class_rows.append(
                {
                    "class_name": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": support,
                }
            )
    per_class_rows.sort(key=lambda row: (row["recall"], row["support"]))
    with (report_dir / "worst_recall_classes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    summary = {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "macro_f1": report_dict["macro avg"]["f1-score"] * 100.0,
        "weighted_f1": report_dict["weighted avg"]["f1-score"] * 100.0,
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Evaluation reports written to %s", report_dir)
    return summary


def export_onnx(model: nn.Module, image_size: int = 260, onnx_file: str | Path = "plant_model.onnx") -> str:
    """Export model to ONNX. Kept compatible with tests that call export_onnx(model)."""

    onnx_path = Path(onnx_file)
    LOGGER.info("Exporting ONNX: %s", onnx_path)
    model_cpu = model.cpu().eval()
    dummy_input = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model_cpu,
        dummy_input,
        str(onnx_path),
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        export_params=True,
    )
    LOGGER.info("ONNX export complete.")
    return str(onnx_path)


def convert_to_tflite_float32(onnx_file: str | Path, output_dir: Path, overwrite: bool) -> Path | None:
    """Convert ONNX to float32 TFLite via onnx2tf."""

    try:
        if output_dir.exists() and overwrite:
            remove_tree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = ["onnx2tf", "-i", str(onnx_file), "-o", str(output_dir), "--non_verbose"]
        LOGGER.info("Converting ONNX to float32 TFLite: %s", " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            LOGGER.error("onnx2tf failed with code %d:\n%s", result.returncode, result.stderr)
            return None

        candidates = sorted(output_dir.rglob("*.tflite"))
        if not candidates:
            LOGGER.error("onnx2tf completed but no .tflite file was found under %s", output_dir)
            return None

        LOGGER.info("Float32 TFLite conversion complete: %s", candidates[0])
        return candidates[0]
    except FileNotFoundError:
        LOGGER.error("onnx2tf command not found. Install onnx2tf or skip TFLite conversion.")
        return None
    except Exception as exc:  # pragma: no cover - external converter failures vary by environment.
        LOGGER.error("Float32 TFLite conversion failed: %s", exc)
        return None


def verify_tflite(tflite_path: Path | None) -> None:
    if tflite_path is None or not tflite_path.exists():
        LOGGER.warning("TFLite verification skipped: %s", tflite_path)
        return

    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        size_mb = tflite_path.stat().st_size / (1024**2)
        LOGGER.info("TFLite verified: %s (%.1f MB)", tflite_path, size_mb)
    except Exception as exc:  # pragma: no cover - TensorFlow availability is environment-specific.
        LOGGER.warning("TFLite verification failed: %s", exc)


def calibration_dataset(data_loader: DataLoader, mean: list[float], std: list[float], num_calib: int = 100):
    """Representative dataset for optional INT8 conversion."""

    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    yielded = 0

    for inputs, _ in data_loader:
        for image in inputs:
            image_np = image.permute(1, 2, 0).numpy()
            image_np = np.clip((image_np * std_array + mean_array), 0.0, 1.0).astype(np.float32)
            image_np = np.expand_dims(image_np, axis=0)
            yield [image_np]
            yielded += 1
            if yielded >= num_calib:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate best_model_finetuned.pth on the new held-out test split and export ONNX/TFLite."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "models/best_model_finetuned.pth")
    parser.add_argument("--labels-path", type=Path, default=PROJECT_ROOT / "labels.json")
    parser.add_argument("--raw-new-root", type=Path, default=PROJECT_ROOT / "data/NewPLantDataset/color")
    parser.add_argument("--preprocessed-root", type=Path, default=PROJECT_ROOT / "data/NewPLantDataset_preprocessed")
    parser.add_argument("--new-val-ratio", type=float, default=0.15)
    parser.add_argument("--new-test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--overwrite-preprocessed", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--onnx-path", type=Path, default=PROJECT_ROOT / "plant_model.onnx")
    parser.add_argument("--tflite-output-dir", type=Path, default=PROJECT_ROOT / "plant_model_tflite_float32")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/new_dataset_evaluation")
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--verify-tflite", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg = load_config(args.config)

    image_cfg = cfg.get("image", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    image_size = args.image_size or int(image_cfg.get("size", 260))
    batch_size = args.batch_size or int(training_cfg.get("batch_size", 32))
    num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 0))
    mean = image_cfg.get("mean", [0.485, 0.456, 0.406])
    std = image_cfg.get("std", [0.229, 0.224, 0.225])
    architecture = model_cfg.get("architecture", "efficientnet_b2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    labels = load_labels(args.labels_path)
    if not args.skip_preprocess:
        _, _, test_root = prepare_new_dataset_split(
            raw_new_root=args.raw_new_root,
            preprocessed_root=args.preprocessed_root,
            class_names=labels,
            image_size=image_size,
            val_ratio=args.new_val_ratio,
            test_ratio=args.new_test_ratio,
            seed=args.seed,
            overwrite=args.overwrite_preprocessed,
        )
    else:
        split_root = find_existing_split_root(
            args.preprocessed_root,
            args.seed,
            args.new_val_ratio,
            args.new_test_ratio,
        )
        test_root = split_root / "test"
        LOGGER.info("Skipping preprocessing. Evaluating existing split: %s", test_root)
        manifest_root = test_root.parent
        if (manifest_root / "test_manifest.csv").exists() and (manifest_root / "val_manifest.csv").exists():
            validate_manifest_leakage(manifest_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)
    if torch.cuda.is_available():
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))

    model, class_names, _ = load_checkpoint_model(
        checkpoint_path=args.checkpoint_path,
        labels_path=args.labels_path,
        fallback_architecture=architecture,
        device=device,
    )

    eval_loader, _ = build_eval_loader(
        test_root=test_root,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        mean=mean,
        std=std,
    )
    run_evaluation(model, eval_loader, device, class_names, args.report_dir)

    if args.skip_onnx:
        LOGGER.info("Skipping ONNX export by request.")
        return

    onnx_file = export_onnx(model, image_size=image_size, onnx_file=args.onnx_path)
    if args.skip_tflite:
        LOGGER.info("Skipping TFLite conversion by request.")
        return

    tflite_path = convert_to_tflite_float32(
        onnx_file=onnx_file,
        output_dir=args.tflite_output_dir,
        overwrite=True,
    )
    if args.verify_tflite:
        verify_tflite(tflite_path)

    LOGGER.info("Evaluation/export pipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        sys.exit(1)
