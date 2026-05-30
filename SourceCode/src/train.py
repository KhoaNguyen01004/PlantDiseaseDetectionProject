import os
import json
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

print(f"\nUsing device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


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
# EXPORT LABELS
# =========================================================

def export_labels(class_names):

    label_map = {}

    for idx, class_name in enumerate(class_names):

        label_map[idx] = class_name

    with open("labels.json", "w") as f:

        json.dump(
            label_map,
            f,
            indent=4
        )

    with open("labels.txt", "w") as f:

        for class_name in class_names:

            f.write(class_name + "\n")

    print("\nLabels exported")


# =========================================================
# BUILD DATALOADERS
# =========================================================

def build_dataloaders(
    dataset_root,
    batch_size=16,
    image_size=260
):

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
            p=0.05
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
            p=0.25,
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
        transform=transform_train
    )

    class_names = main_dataset.classes.copy()

    # =====================================================
    # UNKNOWN DATASET
    # =====================================================

    unknown_dir = "data/unknown"

    if os.path.exists(unknown_dir):

        print("\nUnknown dataset detected")

        unknown_class_index = len(class_names)

        class_names.append("Unknown")

        unknown_images = []

        for file in os.listdir(unknown_dir):

            path = os.path.join(
                unknown_dir,
                file
            )

            if (
                Path(file)
                .suffix
                .lower()
                in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]
            ):

                unknown_images.append(
                    (
                        path,
                        unknown_class_index
                    )
                )

        unknown_images = unknown_images[:6000]

        print(
            f"Using {len(unknown_images)} unknown images"
        )

        unknown_dataset = UnknownDataset(
            unknown_images,
            transform_train
        )

        full_dataset = ConcatDataset([
            main_dataset,
            unknown_dataset
        ])

    else:

        print("\nNo unknown dataset found")

        full_dataset = main_dataset

    # =====================================================
    # SPLIT
    # =====================================================

    total_size = len(full_dataset)

    train_size = int(total_size * 0.8)

    val_size = int(total_size * 0.1)

    test_size = (
        total_size
        - train_size
        - val_size
    )

    train_dataset, val_dataset, test_dataset = (
        random_split(
            full_dataset,
            [
                train_size,
                val_size,
                test_size
            ]
        )
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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }, class_names


# =========================================================
# TRAIN
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate
):

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

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
                max_norm=1.0
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

        print(
            f"\nEpoch {epoch+1}"
            f"\nTrain Loss: "
            f"{train_loss/len(train_loader):.4f}"
            f"\nTrain Acc : "
            f"{train_acc:.2f}%"
            f"\nVal Loss  : "
            f"{val_loss/len(val_loader):.4f}"
            f"\nVal Acc   : "
            f"{val_acc:.2f}%"
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
                    "model_state_dict":
                        model.state_dict(),

                    "best_acc":
                        best_acc
                },
                "models/best_model.pth"
            )

            print(
                f"\nBest model saved "
                f"(Val Acc: {best_acc:.2f}%)"
            )

    print("\nTraining completed")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()

    dataset_root = find_dataset_root(
        "data"
    )

    print(f"\nDataset found:")
    print(dataset_root)

    dataloaders, class_names = (
        build_dataloaders(
            dataset_root,
            batch_size=args.batch_size
        )
    )

    num_classes = len(class_names)

    print(
        f"\nTotal classes: {num_classes}"
    )

    print("\nClasses:")

    for idx, class_name in enumerate(class_names):

        print(f"{idx}: {class_name}")

    export_labels(class_names)

    # =====================================================
    # MODEL
    # =====================================================

    model = models.efficientnet_b2(
        weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
    )

    # FULL FINETUNING
    for param in model.parameters():

        param.requires_grad = True

    in_features = (
        model.classifier[1]
        .in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    model = model.to(DEVICE)

    print("\nStarting training...\n")

    train_model(
        model,
        dataloaders["train"],
        dataloaders["val"],
        epochs=args.epochs,
        learning_rate=args.lr
    )


if __name__ == "__main__":

    main()