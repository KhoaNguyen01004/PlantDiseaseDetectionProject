import os
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR100


OUTPUT_DIR = "data/unknown"


def save_dataset(dataset, start_index=0):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count = start_index

    for image, label in dataset:

        filename = os.path.join(
            OUTPUT_DIR,
            f"unknown_{count}.jpg"
        )

        image.save(filename)

        count += 1

    return count


def main():

    print("=" * 60)
    print("Preparing UNKNOWN dataset from CIFAR100")
    print("=" * 60)

    train_dataset = CIFAR100(
        root="./temp_cifar",
        train=True,
        download=True
    )

    test_dataset = CIFAR100(
        root="./temp_cifar",
        train=False,
        download=True
    )

    print("\nSaving train images...")

    count = save_dataset(train_dataset)

    print("\nSaving test images...")

    count = save_dataset(test_dataset, count)

    print("\nDone")
    print(f"Total unknown images: {count}")

    print(f"\nSaved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
