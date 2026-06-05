import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn

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

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model_finetuned.pth"

DEFAULT_EXPORT_PATH = PROJECT_ROOT / "plant_model.pt"


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


def parse_args():

    parser = argparse.ArgumentParser(
        description="Export the fine-tuned EfficientNet-B2 checkpoint to TorchScript."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Checkpoint to export. Defaults to models/best_model_finetuned.pth "
            "when present, otherwise the newest models/best_model_finetuned*.pth."
        ),
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help="Output TorchScript .pt path. Defaults to plant_model.pt.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Square input size used for tracing.",
    )

    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_path):

    if checkpoint_path is not None:
        resolved = checkpoint_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH.resolve()

    candidates = sorted(
        (path for path in (PROJECT_ROOT / "models").glob("best_model_finetuned*.pth") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        selected = candidates[0].resolve()
        print(
            "Default checkpoint models/best_model_finetuned.pth was not found; "
            f"using newest fine-tuned checkpoint: {selected}"
        )
        return selected

    raise FileNotFoundError(
        f"No fine-tuned checkpoint found under {PROJECT_ROOT / 'models'}. "
        "Run finetune_new_plant_dataset.py first or pass --checkpoint-path."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    export_path = args.export_path.resolve()

    print("\nBuilding model...")

    model = build_model(NUM_CLASSES)

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    if isinstance(checkpoint, dict):
        print(
            "Checkpoint metadata: "
            f"architecture={checkpoint.get('architecture', 'unknown')} "
            f"num_classes={checkpoint.get('num_classes', NUM_CLASSES)} "
            f"best_acc={checkpoint.get('best_acc', 'unknown')}"
        )

    model.eval()

    print("Creating example input...")

    example_input = torch.rand(
        1,
        3,
        args.image_size,
        args.image_size
    )

    print("Tracing TorchScript model...")

    traced_model = torch.jit.trace(
        model,
        example_input
    )

    print("Saving TorchScript file...")

    traced_model.save(
        export_path
    )

    print("\n===================================")
    print("TorchScript export successful!")
    print(f"Exported checkpoint: {checkpoint_path}")
    print(f"Saved to: {export_path}")
    print("===================================")


if __name__ == "__main__":

    main()
