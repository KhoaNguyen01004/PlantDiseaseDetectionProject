import torch
import torch.nn as nn
import warnings

from torchvision import models


# =========================================================
# DEPRECATION WARNING
# =========================================================
warnings.warn(
    "\n" + "="*60 + "\n"
    "DEPRECATED: export_torchscript.py\n"
    "="*60 + "\n"
    "This script is deprecated and will be removed in a future version.\n"
    "Please use the ONNX -> TFLite pipeline instead:\n"
    "  - python -m src.evaluate_and_convert\n"
    "  - or python export_to_tflite.py\n"
    "\n"
    "The TorchScript export path is no longer maintained.\n"
    "="*60 + "\n",
    DeprecationWarning,
    stacklevel=2
)

# =========================================================
# CONFIG
# =========================================================

IMAGE_SIZE = 260

NUM_CLASSES = 39

MODEL_PATH = "models/best_model.pth"

EXPORT_PATH = "plant_model.pt"


# =========================================================
# BUILD MODEL
# =========================================================

def build_model(num_classes):

    model = models.efficientnet_b2(
        weights=None
    )

    in_features = (
        model.classifier[1]
        .in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nBuilding model...")

    model = build_model(NUM_CLASSES)

    print("Loading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Creating example input...")

    example_input = torch.rand(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    print("Tracing TorchScript model...")

    traced_model = torch.jit.trace(
        model,
        example_input
    )

    print("Saving TorchScript file...")

    traced_model.save(
        EXPORT_PATH
    )

    print("\n===================================")
    print("TorchScript export successful!")
    print(f"Saved to: {EXPORT_PATH}")
    print("===================================")


if __name__ == "__main__":

    main()