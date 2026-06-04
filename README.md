# Plant Disease Detection Project

This repository contains a plant leaf disease detection system with two main parts:

- `SourceCode/`: Python training, evaluation, metadata, and export utilities.
- `agrilens/`: Android application that runs on-device inference with PyTorch Mobile.

Runtime metrics such as accuracy, latency, model size, memory use, and dataset counts are intentionally left as placeholders until they are measured from the current trained artifacts.

---

## Current Architecture

The training pipeline uses a configurable EfficientNet family model. The current default in `SourceCode/configs/config.yaml` is:

```yaml
model:
  architecture: "efficientnet_b2"
image:
  size: 260
```

The Android app uses:

- PyTorch Mobile runtime.
- `plant_model.pt` from `agrilens/app/src/main/assets/`.
- `labels.txt` from `agrilens/app/src/main/assets/`.
- CameraX for camera input.
- English and Vietnamese UI strings.

The Python export pipeline can also generate ONNX and TFLite artifacts for experimentation or Python-side deployment, but the Android app currently loads the TorchScript `.pt` asset.

---

## Repository Layout

```text
.
├── SourceCode/
│   ├── configs/config.yaml
│   ├── src/
│   │   ├── train.py
│   │   ├── evaluate_and_convert.py
│   │   ├── inference.py
│   │   ├── metadata.py
│   │   ├── gradcam.py
│   │   ├── quality_validator.py
│   │   └── preprocessing/
│   ├── models/
│   │   └── best_model.pth
│   ├── TRAINING_PIPELINE.md
│   └── README.md
├── agrilens/
│   ├── app/src/main/assets/
│   │   ├── plant_model.pt
│   │   └── labels.txt
│   └── README.md
├── docs/adr/
├── PlantDiseaseDetectionKnowledge/
├── VALIDATION_GUIDE.md
├── PROJECT_SUMMARY.md
├── CHANGELOG.md
└── report.md
```

Generated export files may appear in `SourceCode/`, such as `plant_model.onnx`, `plant_model.onnx.data`, `plant_model.pt`, and `plant_model_tflite_*` folders. These are reproducible artifacts if `SourceCode/models/best_model.pth` is preserved.

---

## Important Artifacts

Keep:

```text
SourceCode/models/best_model.pth
SourceCode/labels.txt
SourceCode/labels.json
SourceCode/metadata.json
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Regeneratable export artifacts:

```text
SourceCode/plant_model.onnx
SourceCode/plant_model.onnx.data
SourceCode/plant_model.pt
SourceCode/plant_model_tflite_float32/
SourceCode/plant_model_tflite_int8/
```

Do not delete `SourceCode/models/best_model.pth` unless you are prepared to retrain.

---

## Training

Run from `SourceCode/`:

```bash
python -m src.train
```

Optional overrides:

```bash
python -m src.train --epochs {{EPOCHS}}
python -m src.train --batch-size {{BATCH_SIZE}}
python -m src.train --lr {{LEARNING_RATE}}
python -m src.train --unknown-limit {{UNKNOWN_LIMIT}}
```

The current training flow is documented in detail in:

```text
SourceCode/TRAINING_PIPELINE.md
```

---

## Evaluation And Export

Run from `SourceCode/`:

```bash
python -m src.evaluate_and_convert
```

Expected generated artifacts:

```text
plant_model.onnx
plant_model_tflite_float32/
plant_model_tflite_int8/
```

TorchScript export for Android is produced separately and copied to:

```text
agrilens/app/src/main/assets/plant_model.pt
```

Keep `labels.txt` synchronized with the model used by Android.

---

## Android App

Run a debug build from `agrilens/`:

```bash
./gradlew.bat :app:assembleDebug
```

The app supports:

- Live camera detection.
- Capture-first detection.
- Gallery image analysis.
- Pause-on-known-result behavior so users can read results.
- Care guidance for known diseases.
- English and Vietnamese settings.
- Adjustable text size.

See:

```text
agrilens/README.md
```

---

## Runtime Results

Fill these after running the current model and app benchmarks:

```text
Test accuracy: {{TEST_ACCURACY}}
Macro F1: {{MACRO_F1}}
Weighted F1: {{WEIGHTED_F1}}
TorchScript model size: {{TORCHSCRIPT_MODEL_SIZE}}
ONNX model size: {{ONNX_MODEL_SIZE}}
TFLite model size: {{TFLITE_MODEL_SIZE}}
Android average inference latency: {{ANDROID_AVG_LATENCY}}
Android memory usage: {{ANDROID_MEMORY_USAGE}}
Dataset version/hash: {{DATASET_VERSION_OR_HASH}}
```

Do not replace placeholders with estimates.

---

## Documentation Map

- `SourceCode/TRAINING_PIPELINE.md`: authoritative training/export pipeline.
- `VALIDATION_GUIDE.md`: validation checklist and commands.
- `PROJECT_SUMMARY.md`: current project summary.
- `report.md`: academic report draft with placeholders.
- `docs/adr/`: architecture decision records.
- `PlantDiseaseDetectionKnowledge/`: focused technical notes.
