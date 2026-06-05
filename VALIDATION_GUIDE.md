# Validation Guide

This guide describes how to validate the current training, export, and Android app pipeline without inventing runtime results.

Fill placeholders only after running the commands against the current repository state.

---

## 1. Environment

Record the environment before validation:

```text
Python version: {{PYTHON_VERSION}}
PyTorch version: {{PYTORCH_VERSION}}
Torchvision version: {{TORCHVISION_VERSION}}
TensorFlow version: {{TENSORFLOW_VERSION}}
ONNX version: {{ONNX_VERSION}}
Android Studio version: {{ANDROID_STUDIO_VERSION}}
Gradle/JDK version: {{GRADLE_JDK_VERSION}}
Test device: {{ANDROID_TEST_DEVICE}}
```

---

## 2. Python Config Validation

Run from `SourceCode/`:

```bash
python -m pytest tests/test_config.py
```

Expected:

```text
Result: {{TEST_CONFIG_RESULT}}
```

Manual checks:

- `configs/config.yaml` exists.
- `data.train_split + data.val_split + data.test_split == 1.0`.
- `image.size` matches training and Android preprocessing.
- `image.mean` and `image.std` match training and Android preprocessing.
- `data.raw_data_dir` points to image class folders.

---

## 3. Training Validation

Run:

```bash
python -m src.train
```

Expected artifacts:

```text
models/best_model.pth
metadata.json
labels.json
labels.txt
```

Record:

```text
Discovered classes: {{DISCOVERED_CLASS_COUNT}}
Configured classes: {{CONFIGURED_CLASS_COUNT}}
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Best epoch: {{BEST_EPOCH}}
Training duration: {{TRAINING_DURATION}}
Dataset hash/version: {{DATASET_HASH_OR_VERSION}}
```

Do not use historical results unless they were produced by this exact run.

---

## 4. In-The-Wild Fine-Tuning Validation

Run from `SourceCode/`:

```bash
python finetune_new_plant_dataset.py --dry-run
```

Expected artifacts:

```text
data/NewPLantDataset_preprocessed/split_seed*_val*_test*/
data/NewPLantDataset_preprocessed/split_seed*_val*_test*/train_manifest.csv
data/NewPLantDataset_preprocessed/split_seed*_val*_test*/val_manifest.csv
data/NewPLantDataset_preprocessed/split_seed*_val*_test*/test_manifest.csv
```

Manual checks:

- Manifest leakage check passes.
- Duplicate source hashes are grouped into the same split.
- `train/`, `val/`, and `test/` all contain images.
- Training uses only the new-domain `train/` split plus historical replay.
- Model selection uses new-domain validation macro-F1, not hybrid validation accuracy.

Full fine-tune command:

```bash
python finetune_new_plant_dataset.py --skip-preprocess --epochs 8 --head-epochs 2 --head-lr 1e-3 --full-lr 5e-6
```

Record:

```text
Split root: {{NEW_SPLIT_ROOT}}
Train manifest rows: {{NEW_TRAIN_MANIFEST_ROWS}}
Val manifest rows: {{NEW_VAL_MANIFEST_ROWS}}
Test manifest rows: {{NEW_TEST_MANIFEST_ROWS}}
Best new validation macro F1: {{BEST_NEW_VAL_MACRO_F1}}
Best new validation accuracy: {{BEST_NEW_VAL_ACCURACY}}
Fine-tuned checkpoint: {{FINE_TUNED_CHECKPOINT_PATH}}
```

---

## 5. Evaluation And Export Validation

Run:

```bash
python -m src.evaluate_and_convert --skip-preprocess
```

Expected artifacts may include:

```text
plant_model.onnx
plant_model.onnx.data
plant_model_tflite_float32/
plant_model_tflite_int8/
reports/new_dataset_evaluation/classification_report.txt
reports/new_dataset_evaluation/classification_report.json
reports/new_dataset_evaluation/confusion_matrix.csv
reports/new_dataset_evaluation/worst_recall_classes.csv
reports/new_dataset_evaluation/summary.json
```

Record:

```text
Test accuracy: {{TEST_ACCURACY}}
Macro F1: {{MACRO_F1}}
Weighted F1: {{WEIGHTED_F1}}
Top-3 accuracy: {{TOP3_ACCURACY}}
ONNX export result: {{ONNX_EXPORT_RESULT}}
TFLite float32 export result: {{TFLITE_FLOAT32_EXPORT_RESULT}}
TFLite int8 export result: {{TFLITE_INT8_EXPORT_RESULT}}
```

Notes:

- ONNX and TFLite files are generated artifacts.
- They can be removed if the checkpoint used to create them is kept.
- The Android app currently uses TorchScript, not these TFLite artifacts.
- Final model quality should be reported from the locked new-domain `test` split, not the hybrid validation split.

---

## 6. TorchScript Android Asset Validation

Android uses:

```text
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Check:

```text
Model asset exists: {{ANDROID_MODEL_ASSET_EXISTS}}
Labels asset exists: {{ANDROID_LABELS_ASSET_EXISTS}}
Model and labels are from same export run: {{ANDROID_ASSET_PAIR_VERIFIED}}
Model hash: {{ANDROID_MODEL_HASH}}
Labels hash: {{ANDROID_LABELS_HASH}}
```

---

## 7. Android Build Validation

Run from `agrilens/`:

```bash
./gradlew.bat :app:assembleDebug
```

Record:

```text
Debug build result: {{ANDROID_DEBUG_BUILD_RESULT}}
Warnings: {{ANDROID_BUILD_WARNINGS}}
APK path: {{ANDROID_DEBUG_APK_PATH}}
```

---

## 8. Android Runtime Validation

Manual test checklist:

- Camera permission prompt appears correctly.
- Live mode scans continuously for unknown results.
- Live mode pauses when a known class is detected.
- `Scan again` resumes detection.
- Capture mode waits for the user to capture before analyzing.
- Gallery mode analyzes selected images.
- Treatment guidance appears for known classes.
- Guide screen opens without crashing.
- English and Vietnamese settings work.
- Text size settings work.
- Navigation/status bars are hidden while inside scanner/guide screens.

Record:

```text
Live mode result: {{LIVE_MODE_RESULT}}
Capture mode result: {{CAPTURE_MODE_RESULT}}
Gallery mode result: {{GALLERY_MODE_RESULT}}
Guide screen result: {{GUIDE_SCREEN_RESULT}}
Vietnamese UI review result: {{VIETNAMESE_UI_REVIEW_RESULT}}
```

---

## 9. Performance Placeholders

Fill only after measuring:

```text
Average inference latency: {{AVG_INFERENCE_LATENCY}}
P95 inference latency: {{P95_INFERENCE_LATENCY}}
Average memory use: {{AVG_MEMORY_USE}}
Peak memory use: {{PEAK_MEMORY_USE}}
Battery impact: {{BATTERY_IMPACT}}
Model file size: {{MODEL_FILE_SIZE}}
```

---

## 10. Cleanup Validation

Safe-to-remove generated artifacts when not needed:

```text
SourceCode/plant_model.onnx
SourceCode/plant_model.onnx.data
SourceCode/plant_model.pt
SourceCode/plant_model_tflite_float32/
SourceCode/plant_model_tflite_int8/
SourceCode/reports/new_dataset_evaluation/
```

Keep:

```text
SourceCode/models/best_model.pth
SourceCode/models/best_model_finetuned*.pth
SourceCode/labels.txt
SourceCode/labels.json
SourceCode/metadata.json
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```
