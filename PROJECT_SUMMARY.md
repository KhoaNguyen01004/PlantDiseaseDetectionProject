# Project Summary

This project implements plant leaf disease detection with a Python training/export pipeline and an Android on-device app.

No runtime performance numbers are asserted in this document. Fill placeholders after validating the current artifacts.

---

## Current Scope

Included:

- PyTorch training pipeline.
- Staged in-the-wild fine-tuning pipeline with audited train/validation/test manifests.
- Config-driven EfficientNet model selection.
- Metadata and label export.
- ONNX/TFLite export pipeline for Python/deployment experiments.
- TorchScript model asset for Android PyTorch Mobile.
- Android scanner with live, capture, and gallery workflows.
- English and Vietnamese UI.
- Offline guide and treatment guidance.

Not confirmed until measured:

- Current model accuracy.
- Field robustness.
- Device latency.
- Memory and battery impact.
- Runtime model size comparisons.

---

## Architecture

```text
Dataset folders
  -> SourceCode/src/train.py
  -> SourceCode/models/best_model.pth
  -> metadata.json / labels.json / labels.txt
  -> SourceCode/src/evaluate_and_convert.py
  -> ONNX / TFLite generated artifacts

best_model.pth
  -> SourceCode/finetune_new_plant_dataset.py
  -> SourceCode/models/best_model_finetuned.pth
  -> new-domain evaluation reports

best_model.pth or best_model_finetuned.pth
  -> TorchScript export
  -> agrilens/app/src/main/assets/plant_model.pt
  -> Android PyTorch Mobile inference
```

Android runtime:

```text
CameraX frame or user-selected image
  -> resize to 260 x 260
  -> ImageNet normalization
  -> PyTorch Mobile model
  -> label + confidence
  -> result + treatment guidance
```

---

## Main Components

Python:

- `SourceCode/src/train.py`
- `SourceCode/finetune_new_plant_dataset.py`
- `SourceCode/src/evaluate_and_convert.py`
- `SourceCode/src/inference.py`
- `SourceCode/src/metadata.py`
- `SourceCode/src/gradcam.py`
- `SourceCode/src/quality_validator.py`
- `SourceCode/src/preprocessing/preprocess.py`

Android:

- `agrilens/app/src/main/java/com/example/MainActivity.java`
- `agrilens/app/src/main/java/com/example/ImageClassifierHelper.java`
- `agrilens/app/src/main/java/com/example/DiseaseTreatmentRepository.java`
- `agrilens/app/src/main/java/com/example/GuideActivity.kt`
- `agrilens/app/src/main/res/layout/activity_main.xml`

---

## Configuration Facts

From `SourceCode/configs/config.yaml`:

```text
Default architecture: efficientnet_b2
Input image size: 260
Pretrained weights: true
Unknown data path: data/unknown
Training split: 0.8
Validation split: 0.1
Test split: 0.1
```

These are configuration values, not measured results.

The real-world fine-tune path uses explicit script arguments for the new-domain split. Current defaults are:

```text
New-domain validation split: 0.15
New-domain test split: 0.2
Fine-tune selection metric: new validation macro F1
```

---

## Artifact Policy

Primary model checkpoint:

```text
SourceCode/models/best_model.pth
SourceCode/models/best_model_finetuned.pth
```

Android runtime assets:

```text
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Generated and regeneratable:

```text
SourceCode/plant_model.onnx
SourceCode/plant_model.onnx.data
SourceCode/plant_model.pt
SourceCode/plant_model_tflite_float32/
SourceCode/plant_model_tflite_int8/
SourceCode/reports/new_dataset_evaluation/
```

---

## Runtime Result Placeholders

```text
Dataset size: {{DATASET_SIZE}}
Class count from latest run: {{CLASS_COUNT}}
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Test accuracy: {{TEST_ACCURACY}}
Macro F1: {{MACRO_F1}}
Weighted F1: {{WEIGHTED_F1}}
Top-3 accuracy: {{TOP3_ACCURACY}}
Android average latency: {{ANDROID_AVG_LATENCY}}
Android memory usage: {{ANDROID_MEMORY_USAGE}}
Field test result: {{FIELD_TEST_RESULT}}
```

---

## Current Risks

- Runtime quality depends on field-photo coverage, not only PlantVillage-style images.
- Hybrid validation accuracy can overestimate field performance; new-domain validation macro-F1 and locked test metrics should be reported separately.
- Unknown detection depends on the quality and diversity of unknown/background samples.
- Android predictions require strict label/model synchronization.
- TFLite artifacts are not the Android runtime path unless the app is changed to use TFLite.
- Treatment guidance should be reviewed by a domain expert before operational use.
