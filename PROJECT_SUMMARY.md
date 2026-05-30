# PlantDiseaseDetectionProject — Project Summary (Read from repository)

## 1) What the project is
This repository is an end-to-end **plant leaf disease classifier** with deployment paths for both **mobile inference (Android)** and **Python CLI inference**.

It contains:
- **ML Backend (PyTorch)** under `SourceCode/`
  - Dataset building + augmentation + (optional) segmentation-mask masking
  - Model training (EfficientNet-B2)
  - Evaluation metrics
  - Export pipeline to **ONNX** and then to **TensorFlow SavedModel / TFLite** (float32 and INT8 via PTQ)
  - Export pipeline to **TorchScript (.pt)** for PyTorch Mobile
- **Inference CLI** for running the exported **TFLite** model under `SourceCode/src/inference.py`
- **Android app** under `agrilens/` using **PyTorch Mobile (TorchScript)** for on-device inference
- **A knowledge base** under `PlantDiseaseDetectionKnowledge/` (Markdown entity map + code review notes)

> **⚠️ CRITICAL ARCHITECTURE NOTE:** The project has a **model architecture mismatch**:
> - `train.py` trains **EfficientNet-B2**
> - `evaluate_and_convert.py` and `export_to_tflite.py` expect **EfficientNet-B0**
> - This mismatch will cause export/inference failures if the trained B2 model is used with B0 export code.

> **⚠️ DUAL DEPLOYMENT PATHS:** The project has TWO separate inference implementations:
> - **Android app**: Uses PyTorch Mobile (TorchScript .pt format)
> - **Python CLI**: Uses TFLite (float32 and INT8 quantized)

---

## 2) Repository structure (detailed)
### Root
- `README.md`
  - High-level description and quick setup.
- `PROJECT_SUMMARY.md`
  - This file.
- `PlantDiseaseDetectionKnowledge/`
  - Markdown pages describing inference/training concepts.
- `SourceCode/`
  - Backend scripts, configs, and Python package.
- `agrilens/`
  - Android app using PyTorch Mobile.

### `SourceCode/`
- `README.md`
  - Backend overview and intended command usage.
- `requirements.txt`, `pyproject.toml`
  - Python dependencies.
- `config.yaml` and `configs/config.yaml`
  - Training/export hyperparameters.
- `labels.json`, `labels.txt`
  - Generated label files.
- Python entry scripts and conversion utilities:
  - `download_plantvillage.py`
  - `prepare_unknown_dataset.py`
  - `export_to_tflite.py` — PyTorch → ONNX → TFLite (float32 only)
  - `export_torchscript.py` — PyTorch → TorchScript (.pt) for mobile
  - `TODO.md`
- ML model artifacts:
  - `plant_model.pt` — PyTorch Mobile model (TorchScript format)
  - `models/` — Training checkpoint directory
  - `saved_model/` — TFLite export output directory
- `src/` (main Python package)
  - `train.py` — training pipeline (EfficientNet-B2)
  - `evaluate_and_convert.py` — evaluation + ONNX + float32 TFLite + INT8 PTQ export (EfficientNet-B0)
  - `inference.py` — TFLite inference CLI with preprocessing + delegate loading
  - `__init__.py`
  - `preprocessing/`
    - `preprocess.py` — dataset indexing, masking augmentation, dataloader building, labels export
    - `generate_background_noise.py` — generates 500 background noise images

### `agrilens/` (Android App)
- **Framework**: Android with Kotlin/Java, Jetpack Compose UI
- **ML Runtime**: PyTorch Mobile (TorchScript)
- **Dependencies**: 
  - `org.pytorch:pytorch_android:1.13.1`
  - `org.pytorch:pytorch_android_torchvision:1.13.1`
- **Key files**:
  - `app/src/main/java/com/example/ImageClassifierHelper.java` — Core ML inference logic
  - `app/src/main/java/com/example/MainActivity.java` — Main UI
  - `app/src/main/java/com/example/GuideActivity.kt` — Disease treatment guide
  - `app/src/main/java/com/example/DiseaseTreatmentRepository.java` — Treatment data
  - `app/src/main/java/com/example/OverlayView.java` — Camera overlay
  - `app/build.gradle.kts` — Gradle configuration
- **Model assets**: Expects `plant_model.pt` and `labels.txt` in `app/src/main/assets/`

---

## 3) Backend: components and behavior

### 3.1 Dataset building & preprocessing (`SourceCode/src/preprocessing/preprocess.py`)
Key functions/classes:
- `load_config(cfg_path)`
- `build_image_lists(raw_root)`
  - Scans `raw_data_dir` (defaults to `SourceCode/data/raw`) for:
    - RGB images under `color/` (or falls back to `raw_root/` if `color/` missing)
    - Optional grayscale under `grayscale/`
    - Optional segmentation masks under `segmented/`
  - Builds a dynamic `label_map` from directory names.
  - Adds **Background Noise** class and samples the first **500** noise images under `Background_Noise/`.
- `preprocess_image(rgb_path, gray_path, seg_path, image_size, mask_prob=0.7)`
  - Reads RGB image → converts to RGB float32 `[0..1]`.
  - Resizes to `image_size`.
  - If `seg_path` exists and `rand < mask_prob`, applies a stochastic mask:
    - mask derived from `(seg > 127)`
    - expands to 3 channels
    - output becomes `rgb * mask`
- `get_augmentation_pipeline(image_size)`
  - Albumentations augmentation suite (rotate/scale/brightness/noise)
  - `A.Resize(image_size, image_size)`
  - ImageNet normalization (mean/std)
  - `ToTensorV2()`
- `PlantVillageDataset`
  - Uses `preprocess_image` then applies Albumentations.
  - Training uses `mask_prob=0.7`; non-training uses `mask_prob=0.2`.
- `split_data(samples, train_ratio, val_ratio, test_ratio, seed)`
  - Stratified split via `sklearn.model_selection.train_test_split`.
- `build_dataloaders(cfg)`
  - Builds train/val/test dataloaders.
  - Uses `WeightedRandomSampler` for imbalance correction.
  - Returns `(dataloaders, label_map)`.

#### Label export helper (in this file)
- `export_to_tflite(model, label_map, tflite_path)`
  - Despite the name, it only writes label artifacts:
    - `labels.json`
    - `labels.txt`
  - If `tflite_support.metadata_writers` is available, it logs guidance but does **not** write real TFLite metadata.

---

### 3.2 Training (`SourceCode/src/train.py`)
This script trains a classifier using torchvision's EfficientNet.

Major parts:
- `find_dataset_root(base_dir="data")`
  - Walks filesystem under `base_dir` to locate a dataset directory containing image files.
- `export_labels(class_names)`
  - Writes `labels.json` and `labels.txt` into the current working directory.
- `build_dataloaders(dataset_root, batch_size=16, image_size=260)`
  - Uses `torchvision.datasets.ImageFolder` (directory-driven labels).
  - Applies a "strong mobile camera augmentation" training pipeline:
    - RandomResizedCrop, horizontal/vertical flips, rotation, perspective, affine
    - ColorJitter, grayscale, blur, RandomErasing
    - ImageNet mean/std normalization
  - Validation pipeline is Resize + Normalize.
  - Optional integration of an "unknown dataset" at `data/unknown/`:
    - adds a final class name `Unknown`
    - caps unknown images to 6000
  - Splits dataset: 80% train / 10% val / 10% test using `random_split`.

- `train_model(model, train_loader, val_loader, epochs, learning_rate)`
  - Loss: `CrossEntropyLoss(label_smoothing=0.1)`
  - Optimizer: `AdamW(weight_decay=1e-4)`
  - Scheduler: `CosineAnnealingWarmRestarts(T_0=5, T_mult=2)`
  - AMP enabled when CUDA is available.
  - Saves best model by validation accuracy to `models/best_model.pth`.

#### ⚠️ MODEL ARCHITECTURE: EfficientNet-B2
- The training code instantiates `models.efficientnet_b2(...)` (line 668)
- This is the **actual trained model architecture**

---

### 3.3 Evaluation + conversion/export (`SourceCode/src/evaluate_and_convert.py`)
This script:
1. Loads config via `configs/config.yaml` (default `cfg_path='configs/config.yaml'`)
2. Calls `build_dataloaders(cfg)` and loads `models/best_model.pth`
3. Evaluates on `dataloaders['test']` using:
   - accuracy_score
   - sklearn `classification_report`
4. Exports to ONNX:
   - `opset_version=15`
   - dynamic batch axes
   - input/output names: `input`/`output`

5. Converts ONNX to float32 TFLite:
   - uses external `onnx2tf` command
   - output directory: `plant_model_tflite_float32`
   - expected output file: `plant_model.tflite`

6. Converts ONNX to INT8 TFLite (PTQ):
   - exports ONNX to TensorFlow SavedModel via `onnx_tf`
   - builds a representative dataset from test images:
     - denormalizes using ImageNet mean/std
     - scales to `[0..255]` and casts to `uint8`
     - converts to `CHW`
   - sets converter:
     - `converter.optimizations = [tf.lite.Optimize.DEFAULT]`
     - `converter.inference_input_type = tf.uint8`
     - `converter.inference_output_type = tf.float32`

7. `verify_tflite(tflite_path)`
   - allocates tensors and logs file size.

#### ⚠️ MODEL ARCHITECTURE MISMATCH: EfficientNet-B0
- This script instantiates `models.efficientnet_b0(...)` (line 36)
- **This conflicts with train.py which uses EfficientNet-B2**
- If you train with train.py (B2) and then run evaluate_and_convert.py (B0), the checkpoint weights will not match the model architecture, causing errors.

---

### 3.4 Inference CLI (`SourceCode/src/inference.py`)
Implements TFLite inference with:
- constants:
  - `MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]`, `IMAGE_SIZE=224`
- `preprocess_image(img_path_or_frame, image_size=IMAGE_SIZE)`:
  - BGR→RGB
  - resize to 224×224
  - float32 normalize then transpose to CHW
  - **casts to `uint8` at the end**
  - returns shape `[1,3,224,224]` uint8

- `create_delegates(model_path)`:
  - attempts to load `libgpu_delegate.so` (GPU delegate) then loads `libxnnpack.so`
  - does not verify existence/compatibility of shared libraries (brittle/portability risk)

- `run_inference(model_path, img_path=None, webcam=False, labels=None)`:
  - always uses delegates in `tf.lite.Interpreter(... experimental_delegates=...)`
  - runs a single pass and:
    - `pred_id = argmax(output_data)`
    - `conf = max(output_data)`
    - prints `conf*100` as percent

#### Known issues:
- The preprocessing always normalizes using mean/std and then casts to `uint8` **without applying TFLite's quantization parameters**.
- If the model is truly float32, casting to `uint8` is incorrect.
- If the model is INT8 quantized, correct preprocessing must quantize using the model's scale/zero_point parameters.
- Output confidence logic assumes output values behave like probabilities. If outputs are logits, confidence is not meaningful unless softmax is applied.

---

### 3.5 Alternate export script (`SourceCode/export_to_tflite.py`)
This is a separate script that:
- loads PyTorch checkpoint from `models/best_model.pth`
- exports ONNX (`opset_version=13`) to `plant_model.onnx`
- runs `onnx2tf` with options:
  - `-osd -nuo` (as configured)
- expects generated float32 tflite at:
  - `saved_model/model_float32.tflite`

This script appears to be a simpler float32-only path (no INT8 PTQ shown here).

#### ⚠️ MODEL ARCHITECTURE: EfficientNet-B0
- This script also uses `models.efficientnet_b0(...)` (line 44)
- Same architecture mismatch issue as evaluate_and_convert.py

---

### 3.6 Android App (`agrilens/`)
The Android app uses **PyTorch Mobile (TorchScript)** for on-device inference.

#### Key Components:
- **ImageClassifierHelper.java**:
  - Loads `plant_model.pt` (TorchScript format) from assets
  - Loads `labels.txt` from assets
  - Uses `org.pytorch.Module` and `org.pytorch.Tensor` for inference
  - Properly applies ImageNet normalization: `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`
  - **Correctly applies softmax** to convert logits to probabilities (line 199-222)
  - Implements unknown class detection with threshold (0.65)
  - Has demo mode fallback for testing

- **MainActivity.java**: Main UI with camera integration
- **GuideActivity.kt**: Shows disease treatment information
- **DiseaseTreatmentRepository.java**: Provides treatment data
- **OverlayView.java**: Camera preview overlay with classification results

#### Model Format:
- Uses **TorchScript (.pt)** format, NOT TFLite
- Expected model file: `plant_model.pt` in `app/src/main/assets/`
- Expected labels file: `labels.txt` in `app/src/main/assets/`

---

## 4) What is needed to be fixed (current issues)

### A) CRITICAL: Model Architecture Mismatch (EfficientNet-B2 vs B0)
- `train.py` trains **EfficientNet-B2**
- `evaluate_and_convert.py` and `export_to_tflite.py` expect **EfficientNet-B0**
- This mismatch will cause runtime errors when trying to load B2 weights into B0 model

✅ Required fix:
- Align all scripts to use the same architecture (either all B0 or all B2)
- Update documentation to reflect the chosen architecture

### B) Inference preprocessing vs TFLite input contract
- `SourceCode/src/inference.py` preprocesses to float->normalize->transpose->cast to `uint8`.
- It does not:
  - inspect the interpreter input dtype
  - use quantization parameters (scale/zero_point)
  - branch float32 vs int8 behavior

✅ Required fix:
- Inspect `interpreter.get_input_details()[0]` and branch:
  - float32 model: keep float32 tensor (post mean/std), no casting to uint8
  - uint8/int8 model: quantize using `(x/scale + zero_point)` and cast to expected dtype

### C) Output confidence semantics
- Confidence uses `max(output_data)`.
- If outputs are logits, confidence is wrong.

✅ Required fix:
- Determine if output is already probabilities; if not, apply softmax before confidence display.
- Note: The Android app correctly implements softmax (see ImageClassifierHelper.java line 199-222)

### D) Delegate loading robustness
- Hardcoded `.so` names in `inference.py` without validation.

✅ Required fix:
- Only load delegates if files exist and catch errors per delegate.
- Avoid adding XNNPACK unconditionally.

### E) Label/class mapping consistency
- Labels are exported into `labels.json`/`labels.txt` at different stages.
- Android/Inference must use labels from the same training/export run.

✅ Required fix:
- Bundle the exact label file used to export the chosen model.

### F) Misleading function name in preprocessing
- `export_to_tflite()` in `preprocess.py` only exports labels.

✅ Required fix:
- Rename to something like `export_labels_for_tflite()` or implement full metadata writing.

### G) Dual Deployment Path Confusion
- The project has TWO separate inference paths:
  - Android: PyTorch Mobile (TorchScript .pt)
  - Python CLI: TFLite (float32/INT8)
- These use different model formats and preprocessing

✅ Required fix:
- Document both paths clearly
- Ensure consistency between training and both export paths
- Consider consolidating to a single deployment format

---

## 5) Knowledge base alignment
`PlantDiseaseDetectionKnowledge/PlantDiseaseDetection_CodeReview.md` already flags several of these issues:
- inference dtype/quantization mismatch
- output confidence semantics
- brittle delegate loading
- preprocessing/export confusion around `export_to_tflite()`

However, the code review does NOT mention:
- The critical EfficientNet-B2 vs B0 architecture mismatch
- The dual deployment path (PyTorch Mobile vs TFLite)

---

## 6) Current state of models for deployment

### PyTorch Mobile (Android)
- **Format**: TorchScript (.pt)
- **Expected file**: `plant_model.pt` in `agrilens/app/src/main/assets/`
- **Status**: The file `SourceCode/plant_model.pt` exists in the repository
- **Architecture**: Should match train.py (EfficientNet-B2)

### TFLite (Python CLI)
- **Formats**: float32 and INT8 quantized
- **Expected files**:
  - float32: `plant_model_tflite_float32/plant_model.tflite`
  - INT8: `plant_model_tflite_int8/plant_model_int8.tflite`
- **Status**: These files are NOT present in the repository (need to be generated)
- **Architecture**: Export scripts expect EfficientNet-B0 (MISMATCH with training)

---

## 7) Practical structure walkthrough (how to navigate the repo)
- Training path:
  1. `SourceCode/src/train.py` (build dataloaders + train + export labels) — **Uses EfficientNet-B2**
  2. Checkpoint saved to `models/best_model.pth`
- Export paths (TWO OPTIONS):
  - **Option A - PyTorch Mobile**: Use `export_torchscript.py` to convert to .pt format
  - **Option B - TFLite**: 
    1. `SourceCode/src/evaluate_and_convert.py` (load best checkpoint + evaluate + export) — **Uses EfficientNet-B0**
    2. ONNX: `export_onnx()` inside `evaluate_and_convert.py`
    3. float32 TFLite: `convert_to_tflite_float32()` (calls `onnx2tf`)
    4. INT8 PTQ: `convert_to_tflite_int8()` (onnx->tf saved model + representative dataset)
- Inference paths:
  - **Android**: Uses PyTorch Mobile, see `agrilens/app/src/main/java/com/example/ImageClassifierHelper.java`
  - **Python CLI**: `SourceCode/src/inference.py` (TFLite)

---

## 8) Summary checklist
- [ ] **CRITICAL**: Fix architecture mismatch (align train.py with export scripts - choose B0 or B2)
- [ ] Fix inference preprocessing to use interpreter input dtype + quantization params
- [ ] Fix confidence output semantics (apply softmax if needed) - Android already does this correctly
- [ ] Make delegate loading optional and robust
- [ ] Ensure label consistency across training and both deployment paths
- [ ] Rename or implement `preprocess.export_to_tflite()` to match its behavior
- [ ] Document both deployment paths clearly (PyTorch Mobile vs TFLite)
- [ ] Update knowledge base to reflect architecture mismatch and dual deployment paths