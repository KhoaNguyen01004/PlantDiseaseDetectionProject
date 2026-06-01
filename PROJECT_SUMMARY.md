# PlantDiseaseDetectionProject — Current Project Summary

## Purpose
This repository implements a plant leaf disease classification system with two deployment targets:
- Python CLI inference using TensorFlow Lite
- Android on-device inference using PyTorch Mobile (TorchScript)

The backend is written in Python under `SourceCode/`, and the Android prototype lives in `agrilens/`.

## Current architecture and implementation
- **Model backbone**: `efficientnet_b2` by default, configured in `SourceCode/configs/config.yaml`.
- **Input size**: 260×260 pixels for the primary inference pipeline.
- **Configuration source**: `SourceCode/configs/config.yaml` is the single source of truth for model, image, training, quality validation, inference, export, and metadata settings.
- **Metadata export**: `SourceCode/src/metadata.py` exports `metadata.json`, `labels.json`, and `labels.txt` for deployment consistency.
- **Deployment**:
  - Python CLI: TFLite models produced via `SourceCode/src/evaluate_and_convert.py`.
  - Android: TorchScript `.pt` produced via `SourceCode/export_torchscript.py` and consumed by `agrilens/app/src/main/java/com/example/ImageClassifierHelper.java`.

## Repository structure
- `README.md`: high-level repository overview and setup guidance.
- `PROJECT_SUMMARY.md`: this summary document.
- `CHANGELOG.md`: recorded refactor and hardening history.
- `VALIDATION_GUIDE.md`: verification and validation procedures.
- `version.json`: lightweight manifest describing current versioning and compatibility.
- `docs/adr/`: architecture decision records.
- `SourceCode/`: Python backend implementation.
- `agrilens/`: Android application prototype.
- `PlantDiseaseDetectionKnowledge/`: knowledge and review notes.

## SourceCode backend
- `SourceCode/src/train.py`
  - Builds and trains the classifier.
  - Reads `SourceCode/configs/config.yaml` for model architecture, image size, training hyperparameters, and dataset splits.
  - Supports `efficientnet_b0`, `efficientnet_b2`, and `efficientnet_v2_s`, with `efficientnet_b2` as default.
  - Uses `torchvision` pretrained weights and exports the best checkpoint to `SourceCode/models/best_model.pth`.
- `SourceCode/src/evaluate_and_convert.py`
  - Loads the checkpoint and evaluation dataset.
  - Produces ONNX and TFLite exports for Python inference.
  - Supports float32 and INT8 conversion.
- `SourceCode/src/inference.py`
  - Provides a Python CLI for TFLite inference.
  - Detects the TFLite model's input dtype and quantization parameters dynamically.
  - Applies ImageNet normalization and uses softmax to compute probabilities.
- `SourceCode/src/gradcam.py`
  - Implements Grad-CAM for EfficientNet-B2.
  - Generates heatmaps and overlay visualizations from a trained checkpoint.
- `SourceCode/src/quality_validator.py`
  - Validates image quality for blur, brightness, and resolution.
  - Provides user guidance for low-quality input.
- `SourceCode/src/metadata.py`
  - Exports unified metadata and labels for deployment.

## Android integration
- Android app is located under `agrilens/`.
- The app uses PyTorch Mobile (`org.pytorch:pytorch_android`) and loads `plant_model.pt` from assets.
- `ImageClassifierHelper.java` resizes input to 260×260 and applies ImageNet mean/std normalization.
- It uses a softmax confidence threshold to label low-confidence predictions as `Unknown`.
- The Android path is integrated for TorchScript inference, while Python uses TFLite.

## Feature status
- **Implemented**:
  - Python-side training, evaluation, TFLite export, and metadata export.
  - Grad-CAM explainability via `SourceCode/src/gradcam.py`.
  - Image quality validation via `SourceCode/src/quality_validator.py`.
  - Optional Unknown dataset support in `SourceCode/src/train.py` when `SourceCode/data/unknown` exists.
- **Integrated**:
  - Android TorchScript inference path in `agrilens/`.
  - Shared label metadata assets across Python and Android deployments.
- **Evaluated**:
  - Evaluation and export pipelines are documented and should be verified through generated artifacts from `SourceCode/src/evaluate_and_convert.py`.
  - The repository does not publish a fixed accuracy claim in documentation without reproducible output artifacts.
- **Planned**:
  - Full Android production packaging and user-facing deployment validation.
  - Additional end-to-end cross-platform evaluation artifacts under the `SourceCode/` export pipeline.

## Feature Status Matrix

| Feature | Status | Details | Python | Android |
|---------|--------|---------|--------|---------|
| Model Training | ✅ Implemented | EfficientNet-B2, AdamW, configurable architectures (B0/B2/V2-S) | SourceCode/src/train.py | N/A |
| TFLite Export | ✅ Implemented | Float32 + INT8 PTQ quantization via onnx2tf | SourceCode/src/evaluate_and_convert.py | N/A (Python CLI only) |
| TorchScript Export | ✅ Implemented | .pt format for PyTorch Mobile | SourceCode/export_torchscript.py | Loads plant_model.pt |
| Grad-CAM | ✅ Implemented | Feature visualization on EfficientNet-B2 (features.8.0 layer) | SourceCode/src/gradcam.py | Not ported (Python only) |
| Quality Validator | ✅ Implemented | Blur, brightness, resolution checks; user guidance | SourceCode/src/quality_validator.py | ImageClassifierHelper checks dimensions |
| Unknown Detection | ✅ Implemented | Softmax confidence threshold (0.65) for Unknown class | train.py + inference.py | ImageClassifierHelper.java |
| Metadata Management | ✅ Implemented | Version tracking, labels export, dataset hashing | SourceCode/src/metadata.py | Loads labels.txt + labels.json |
| TFLite Android Inference | 🔄 Evaluated | TFLite delegates (GPU/XNNPACK) available but not integrated | Available in inference.py | Not used (using PyTorch Mobile instead) |
| End-to-end Testing | ✅ Implemented | 21 pytest tests covering config, export, inference, preprocessing | SourceCode/tests/ | Not covered by CI |
| CI/CD Pipeline | 🔲 Planned | GitHub Actions for pytest, linting, validation | TODO | TODO |
| Production Android Build | 🔲 Planned | Signed APK, Play Store submission | N/A | TODO |
| Cross-platform Evaluation Artifacts | 🔲 Planned | Confusion matrices, Grad-CAM examples, benchmark reports | TODO | TODO |

**Status Legend:**
- ✅ **Implemented**: Code exists, tested, and functional
- 🔄 **Integrated**: Available in code but not actively used in primary deployment
- 📊 **Evaluated**: Concept validated, artifacts pending
- 🔲 **Planned**: Roadmap item, not yet started

## Notes
- `SourceCode/export_to_tflite.py` is deprecated and retained for historical compatibility; the recommended pipeline is `SourceCode/src/evaluate_and_convert.py`.
- The repo avoids hardcoded benchmark claims in documentation and refers readers to evaluation artifacts instead.
- The version manifest is available in `version.json`.
