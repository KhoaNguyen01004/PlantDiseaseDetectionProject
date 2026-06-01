import os
import json
import argparse
import subprocess
import warnings
import yaml

import torch
import torch.nn as nn

from torchvision import models

# =========================================================
# DEPRECATION WARNING
# =========================================================
warnings.warn(
    "\n" + "="*60 + "\n"
    "DEPRECATED: export_to_tflite.py\n"
    "="*60 + "\n"
    "This script is deprecated and will be removed in a future version.\n"
    "Please use the centralized evaluate and convert pipeline instead:\n"
    "  - python -m src.evaluate_and_convert\n"
    "\n"
    "The old export_to_tflite.py script is no longer maintained.\n"
    "="*60 + "\n",
    DeprecationWarning,
    stacklevel=2
)

# =========================================================
# CONFIG
# =========================================================

# Dynamically load image size from config.yaml
try:
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    if not os.path.exists(config_path):
        config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    IMAGE_SIZE = config["image"]["size"]
except Exception as e:
    IMAGE_SIZE = 260

DEVICE = torch.device("cpu")


# =========================================================
# LOAD LABELS
# =========================================================

def load_labels():

    with open("labels.txt", "r") as f:

        labels = [
            line.strip()
            for line in f.readlines()
            if line.strip()
        ]

    return labels


# =========================================================
# BUILD MODEL
# =========================================================

def build_model(num_classes):
    # Must match train.py architecture (EfficientNet-B2)
    model = models.efficientnet_b2(
        weights=None
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# =========================================================
# LOAD MODEL
# =========================================================

def load_trained_model(model_path):

    labels = load_labels()

    num_classes = len(labels)

    print(f"\nClasses: {num_classes}")

    model = build_model(num_classes)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    print("\nPyTorch model loaded")

    return model


# =========================================================
# EXPORT ONNX
# =========================================================

def export_onnx(
    model,
    onnx_path="plant_model.onnx"
):

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

    print(f"\nONNX exported:")
    print(onnx_path)


# =========================================================
# ONNX2TF
# =========================================================

def convert_to_tflite():

    cmd = [
        "onnx2tf",
        "-i",
        "plant_model.onnx",
        "-o",
        "saved_model",
        "-osd",
        "-nuo"
    ]

    print("\nRunning onnx2tf...")

    subprocess.run(
        cmd,
        check=True
    )

    print("\nTFLite export completed")


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="models/best_model.pth"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Export PyTorch -> TFLite")
    print("=" * 60)

    model = load_trained_model(
        args.model
    )

    export_onnx(model)

    convert_to_tflite()

    print("\nDone")

    print("\nGenerated files:")

    print(
        "saved_model/"
    )

    print(
        "saved_model/model_float32.tflite"
    )

    print(
        "\nCopy to Android:"
    )

    print(
        "agrilens/app/src/main/assets/"
    )


if __name__ == "__main__":
    main()