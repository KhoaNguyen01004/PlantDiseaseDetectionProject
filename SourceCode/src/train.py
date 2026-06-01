import os
import json
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    ConcatDataset
)

from torchvision import (
    datasets,
    transforms,
    models
)

from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts
)

from tqdm import tqdm

import argparse

from .logging_config import get_logger

logger = get_logger(__name__)

# =========================================================
# CUDA OPTIMIZATION
# =========================================================

torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================================================
# UNKNOWN DATASET
# =========================================================

class UnknownDataset(Dataset):

    def __init__(self, samples, transform):

        self.samples = samples

        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = datasets.folder.default_loader(path)

        image = self.transform(image)

        return image, label


# =========================================================
# FIND DATASET ROOT
# =========================================================

def find_dataset_root(base_dir="data"):

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ]

    for root, dirs, files in os.walk(base_dir):

        if len(dirs) < 5:
            continue

        image_found = False

        for subdir in dirs[:5]:

            subdir_path = os.path.join(
                root,
                subdir
            )

            try:

                for file in os.listdir(subdir_path):

                    if (
                        Path(file)
                        .suffix
                        .lower()
                        in image_extensions
                    ):

                        image_found = True

                        break

            except:
                pass

            if image_found:
                break

        if image_found:
            return root

    raise FileNotFoundError(
        f"Could not find dataset inside '{base_dir}'"
    )


# =========================================================
# EXPORT LABELS (Deprecated locally - using src.metadata)
# =========================================================


# =========================================================
# BUILD DATALOADERS
# =========================================================

def build_dataloaders(
    dataset_root,
    batch_size=16,
    image_size=260,
    config=None
):
    # Load configuration parameters
    cfg_aug = config.get("augmentation", {}) if config else {}
    cfg_data = config.get("data", {}) if config else {}
    
    # Set random seeds for reproducibility
    seed = cfg_data.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # =====================================================
    # STRONG MOBILE CAMERA AUGMENTATION
    # =====================================================

    transform_train = transforms.Compose([

        transforms.RandomResizedCrop(
            image_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomVerticalFlip(
            p=0.3
        ),

        transforms.RandomRotation(
            degrees=35
        ),

        transforms.RandomPerspective(
            distortion_scale=0.3,
            p=0.4
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),

        transforms.ColorJitter(
            brightness=0.45,
            contrast=0.45,
            saturation=0.35,
            hue=0.08
        ),

        transforms.RandomGrayscale(
            p=cfg_aug.get("grayscale_prob", 0.05)
        ),

        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        transforms.RandomErasing(
            p=cfg_aug.get("random_erasing_prob", 0.25),
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])

    # =====================================================
    # VALIDATION
    # =====================================================

    transform_val = transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =====================================================
    # MAIN DATASET
    # =====================================================

    main_dataset = datasets.ImageFolder(
        root=dataset_root,
        transform=None
    )

    class_names = main_dataset.classes.copy()
    samples = list(main_dataset.samples)

    # =====================================================
    # UNKNOWN DATASET
    # =====================================================

    unknown_dir = "data/unknown"

    if os.path.exists(unknown_dir):
        logger.info("Unknown dataset detected")

        unknown_class_index = len(class_names)
        class_names.append("Unknown")

        unknown_images = []

        for file in os.listdir(unknown_dir):
            path = os.path.join(unknown_dir, file)
            if Path(file).suffix.lower() in [".jpg", ".jpeg", ".png"]:
                unknown_images.append((path, unknown_class_index))

        unknown_limit = config.get("data", {}).get("unknown_limit", 6000)
        if unknown_limit > 0:
            unknown_images = unknown_images[:unknown_limit]
        logger.info(f"Using {len(unknown_images)} unknown images")
        samples.extend(unknown_images)
    else:
        logger.info("No unknown dataset found")

    # =====================================================
    # DATASET WRAPPER
    # =====================================================

    class ImagePathDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            image = datasets.folder.default_loader(path)
            if self.transform is not None:
                image = self.transform(image)
            return image, label

    # =====================================================
    # SPLIT
    # =====================================================

    total_size = len(samples)

    train_ratio = cfg_data.get("train_split", 0.8)
    val_ratio = cfg_data.get("val_split", 0.1)
    test_ratio = cfg_data.get("test_split", 0.1)

    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size: train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = ImagePathDataset(
        [samples[i] for i in train_indices],
        transform_train
    )
    val_dataset = ImagePathDataset(
        [samples[i] for i in val_indices],
        transform_val
    )
    test_dataset = ImagePathDataset(
        [samples[i] for i in test_indices],
        transform_val
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}, class_names


# =========================================================
# TRAIN
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    config=None
):
    cfg_train = config.get("training", {}) if config else {}

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg_train.get("label_smoothing", 0.1)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(cfg_train.get("weight_decay", 1e-4))
    )

    # Dynamic scheduler loading
    sched_cfg = cfg_train.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")
    if sched_type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 5),
            T_mult=sched_cfg.get("T_mult", 2)
        )
    elif sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 5),
            gamma=sched_cfg.get("gamma", 0.1)
        )
    elif sched_type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=sched_cfg.get("gamma", 0.9)
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2
        )

    scaler = torch.cuda.amp.GradScaler(
        enabled=torch.cuda.is_available()
    )

    best_acc = 0.0

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        train_correct = 0

        train_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}"
        )

        for images, labels in pbar:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            # gradient clipping
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=cfg_train.get("gradient_clip", 1.0)
            )

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            _, predicted = outputs.max(1)

            train_total += labels.size(0)

            train_correct += (
                predicted
                .eq(labels)
                .sum()
                .item()
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_correct = 0

        val_total = 0

        val_loss = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                _, predicted = outputs.max(1)

                val_total += labels.size(0)

                val_correct += (
                    predicted
                    .eq(labels)
                    .sum()
                    .item()
                )

        train_acc = (
            100.0
            * train_correct
            / train_total
        )

        val_acc = (
            100.0
            * val_correct
            / val_total
        )

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # =================================================
        # SAVE BEST
        # =================================================

        if val_acc > best_acc:

            best_acc = val_acc

            os.makedirs(
                "models",
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                },
                "models/best_model.pth",
            )

            logger.info(f"Best model saved (Val Acc: {best_acc:.2f}%)")

    logger.info("Training completed")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to centralized config file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None
    )
    parser.add_argument(
        "--unknown-limit",
        type=int,
        default=None,
        help="Max unknown/background images to use (0 = no limit)"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        fallback = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        if os.path.exists(fallback):
            config_path = fallback
            
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config with parser arguments if provided
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.unknown_limit is not None:
        config.setdefault("data", {})["unknown_limit"] = args.unknown_limit

    configured_root = config.get("data", {}).get("raw_data_dir", "data")
    try:
        dataset_root = find_dataset_root(configured_root)
    except FileNotFoundError:
        fallback_paths = [
            "data/plantvillage/plantvillage dataset/color",
            "data/plantvillage/plantvillage dataset",
            "data/plantvillage",
        ]
        dataset_root = None
        for candidate in fallback_paths:
            if os.path.exists(candidate):
                try:
                    dataset_root = find_dataset_root(candidate)
                    logger.warning(
                        f"Configured dataset root '{configured_root}' not found; falling back to '{dataset_root}'"
                    )
                    break
                except FileNotFoundError:
                    continue

        if dataset_root is None:
            raise

    logger.info(f"Dataset found: {dataset_root}")

    batch_size = config["training"]["batch_size"]
    image_size = config["image"]["size"]
    epochs = config["training"]["epochs"]
    learning_rate = float(config["training"]["learning_rate"])

    dataloaders, class_names = (
        build_dataloaders(
            dataset_root,
            batch_size=batch_size,
            image_size=image_size,
            config=config
        )
    )

    num_classes = len(class_names)

    logger.info(f"Total classes: {num_classes}")
    logger.info("Classes list available; use DEBUG to view each class")
    for idx, class_name in enumerate(class_names):
        logger.debug(f"{idx}: {class_name}")

    # Use centralized src.metadata module for unified export of labels and metadata
    from src.metadata import export_metadata_for_model
    label_map = {idx: name for idx, name in enumerate(class_names)}
    export_metadata_for_model(label_map, output_dir=".", config=config)

    # =====================================================
    # MODEL (Dynamic Model Selection)
    # =====================================================
    arch = config.get("model", {}).get("architecture", "efficientnet_b2")
    pretrained = config.get("model", {}).get("pretrained", True)
    
    logger.info(f"Initializing model architecture: {arch}")
    
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")

    # FULL FINETUNING
    for param in model.parameters():

        param.requires_grad = True

    model = model.to(DEVICE)

    logger.info("Starting training...")

    train_model(
        model,
        dataloaders["train"],
        dataloaders["val"],
        epochs=epochs,
        learning_rate=learning_rate,
        config=config
    )


if __name__ == "__main__":

    main()