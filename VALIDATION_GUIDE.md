# Plant Disease Detection - Validation Guide & Testing Procedures

This document provides comprehensive instructions for validating the entire pipeline after the refactor.

---

## Table of Contents

1. [Quick Start Commands](#quick-start-commands)
2. [Expected Outputs](#expected-outputs)
3. [Architecture Verification](#architecture-verification)
4. [TFLite Verification](#tflite-verification)
5. [Grad-CAM Verification](#grad-cam-verification)
6. [Configuration Verification](#configuration-verification)
7. [Validation Report Template](#validation-report-template)

---

## Quick Start Commands

### 1. Training

```bash
# Navigate to SourceCode directory
cd SourceCode

# Train the model (requires data in data/raw/)
python -m src.train --epochs 20 --batch-size 32 --lr 1e-4

# Expected output:
# - models/best_model.pth (saved checkpoint)
# - Console output showing training progress
```

### 2. Evaluation & Export

```bash
# Evaluate model and export to ONNX/TFLite
python -m src.evaluate_and_convert

# Expected output:
# - plant_model.onnx (ONNX model)
# - plant_model_tflite_float32/plant_model.tflite (Float32 TFLite)
# - plant_model_tflite_int8/plant_model_int8.tflite (INT8 TFLite)
# - Classification report printed to console
```

### 3. Standalone ONNX Export

```bash
# Export only (using export_to_tflite.py)
python export_to_tflite.py --model models/best_model.pth

# Expected output:
# - plant_model.onnx
# - saved_model/ directory with TFLite models
```

### 4. Float32 TFLite Inference

```bash
# Test inference with float32 model
python -m src.inference \
    --model plant_model_tflite_float32/plant_model.tflite \
    --image test_image.jpg \
    --labels labels.json

# Expected output:
# - Prediction with confidence score
# - Top-3 predictions
```

### 5. INT8 TFLite Inference

```bash
# Test inference with INT8 model
python -m src.inference \
    --model plant_model_tflite_int8/plant_model_int8.tflite \
    --image test_image.jpg \
    --labels labels.json

# Expected output:
# - Prediction with confidence score
# - Top-3 predictions
```

### 6. Grad-CAM Generation

```bash
# Generate Grad-CAM heatmap
python -m src.gradcam \
    --image test_image.jpg \
    --model models/best_model.pth \
    --labels labels.json \
    --output gradcam_result.png

# Expected output:
# - gradcam_result.png (3-panel visualization)
# - original.png (original image)
# - overlay.png (heatmap overlay)
```

### 7. Quality Validation

```bash
# Validate image quality
python -m src.quality_validator \
    --image test_image.jpg \
    --verbose

# Expected output:
# - Validation results (pass/fail)
# - Detailed scores for blur, brightness, resolution
# - User guidance messages
```

### 8. Metadata Export

```bash
# Export all metadata
python -m src.metadata \
    --labels labels.json \
    --config configs/config.yaml \
    --output-dir .

# Expected output:
# - metadata.json
# - labels.json (updated)
# - labels.txt
```

### 9. Dataset Preparation

```bash
# Prepare real-world dataset structure
python prepare_real_world_dataset.py \
    --base-dir data \
    --create-classes \
    --summary

# Expected output:
# - Directory structure created
# - Dataset statistics printed
```

---

## Expected Outputs

### After Training
```
models/
└── best_model.pth          # Trained model checkpoint
```

### After Evaluation & Export
```
SourceCode/
├── plant_model.onnx                    # ONNX model
├── plant_model_tflite_float32/
│   └── plant_model.tflite              # Float32 TFLite model
├── plant_model_tflite_int8/
│   └── plant_model_int8.tflite         # INT8 TFLite model
├── plant_model_tf/                     # Intermediate TF model (for INT8)
└── labels.json / labels.txt            # Class labels
```

### After Grad-CAM
```
SourceCode/
├── gradcam_result.png                  # 3-panel visualization
├── original.png                        # Original image
└── overlay.png                         # Heatmap overlay only
```

### After Metadata Export
```
SourceCode/
├── metadata.json                       # Complete model metadata
├── labels.json                         # Class labels (JSON format)
└── labels.txt                          # Class labels (text format)
```

### After Dataset Preparation
```
data/
├── plantvillage/                       # Existing PlantVillage data
├── real_world_train/                   # Future smartphone training data
│   ├── class1/
│   ├── class2/
│   └── ...
├── real_world_test/                    # Future smartphone testing data
│   ├── class1/
│   ├── class2/
│   └── ...
└── unknown/                            # Unknown/background images
```

---

## Architecture Verification

### Files That Instantiate EfficientNet-B2

All of the following files should use `efficientnet_b2`:

1. **`SourceCode/src/train.py`** (line 668)
   ```python
   model = models.efficientnet_b2(
       weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
   )
   ```

2. **`SourceCode/src/evaluate_and_convert.py`** (line 36)
   ```python
   model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
   ```

3. **`SourceCode/export_to_tflite.py`** (line 44)
   ```python
   model = models.efficientnet_b2(weights=None)
   ```

4. **`SourceCode/export_torchscript.py`** (line 46)
   ```python
   model = models.efficientnet_b2(weights=None)
   ```

5. **`SourceCode/src/gradcam.py`** (line 188)
   ```python
   model = models.efficientnet_b2(weights=None)
   ```

### Verification Commands

```bash
# Search for any remaining B0 references
grep -r "efficientnet_b0" SourceCode/ --include="*.py"

# Search for B2 references (should find 5+ occurrences)
grep -r "efficientnet_b2" SourceCode/ --include="*.py"

# Check image size consistency
grep -r "IMAGE_SIZE\|image_size\|260\|224" SourceCode/ --include="*.py" | grep -v "test"
```

### Expected Image Size: 260

| Component | Expected Size | File Reference |
|-----------|---------------|----------------|
| Training | 260 | `src/train.py` line 175 |
| Evaluation | 260 | `configs/config.yaml` → `image.size` |
| Export | 260 | `export_to_tflite.py` line 16 (should be 260, currently 224 - see note below) |
| Inference | 260 | `src/inference.py` line 18 |
| Grad-CAM | 260 | `src/gradcam.py` line 166 |

⚠️ **Note:** `export_to_tflite.py` still has `IMAGE_SIZE = 224` hardcoded. This needs to be updated to 260 for consistency. This is a known issue that should be fixed.

---

## TFLite Verification

### Float32 Model Handling

**How it works:**
1. PyTorch model exported to ONNX (float32 weights)
2. ONNX converted to TFLite using `onnx2tf`
3. Input type: float32
4. Output type: float32

**Inference flow (`src/inference.py`):**
```python
# Detect input type
input_dtype, input_scale, input_zero_point = detect_input_details(input_details)

# For float32 models:
if input_dtype == np.float32:
    # Keep normalized float32 values
    input_data = img.astype(np.float32)
```

### INT8 Model Handling

**How it works:**
1. ONNX converted to TensorFlow SavedModel
2. Post-training quantization (PTQ) with 100 calibration images
3. Input type: uint8 (0-255 range)
4. Output type: float32

**Inference flow:**
```python
# For uint8 models with quantization:
if input_dtype == np.uint8 and (input_scale != 1.0 or input_zero_point != 0):
    # Apply quantization: dequantize then requantize
    input_data = (input_data / input_scale) + input_zero_point
    input_data = np.clip(input_data, 0, 255).astype(np.uint8)
```

### Quantization Parameter Logic

```python
def detect_input_details(input_details):
    """Extract quantization parameters from TFLite model."""
    input_details = input_details[0]
    input_dtype = input_details['dtype']
    
    # quantization = (scale, zero_point)
    input_scale = input_details.get('quantization', (1.0, 0))[0]
    input_zero_point = input_details.get('quantization', (1.0, 0))[1]
    
    return input_dtype, input_scale, input_zero_point
```

### When Softmax is Applied

**Always applied after getting model output:**
```python
# Get raw logits from model
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Apply softmax to convert logits to probabilities
probabilities = softmax(output_data)  # scipy.special.softmax

# Get prediction and confidence
pred_id = np.argmax(probabilities)
conf = probabilities[pred_id]
```

### Verification Test

```bash
# Verify float32 model
python -m src.inference \
    --model plant_model_tflite_float32/plant_model.tflite \
    --image test.jpg

# Expected console output:
# Model input dtype: <class 'numpy.float32'>
# Model quantization: scale=1.0, zero_point=0

# Verify INT8 model
python -m src.inference \
    --model plant_model_tflite_int8/plant_model_int8.tflite \
    --image test.jpg

# Expected console output:
# Model input dtype: <class 'numpy.uint8'>
# Model quantization: scale=<value>, zero_point=<value>
```

---

## Grad-CAM Verification

### Why Target Layer `features.8.0` is Correct for EfficientNet-B2

EfficientNet-B2 architecture structure:
```
EfficientNet-B2:
├── features (Sequential)
│   ├── 0: Conv2dNormActivation (initial conv)
│   ├── 1-7: MBConv blocks (repeated)
│   └── 8: MBConv block (final conv block)
│       └── 0: Conv2dNormActivation ← TARGET LAYER
├── avgpool (AdaptiveAvgPool2d)
└── classifier (Linear)
```

The final convolutional layer (`features.8.0`) is the correct target because:
1. It has the highest-level semantic features
2. It has appropriate spatial resolution (8x8 for 260x260 input)
3. It's the last layer before global average pooling
4. Gradients flow through this layer to all previous layers

### How to Verify Heatmaps are Meaningful

1. **Visual Inspection:**
   - Heatmap should highlight diseased areas of the leaf
   - Healthy leaf predictions should show diffuse or edge-focused activation
   - Background should have low activation

2. **Quantitative Check:**
   ```python
   # Load Grad-CAM and check activation distribution
   import numpy as np
   heatmap = ...  # Generated heatmap
   
   # Check that activation is concentrated (not uniform)
   activation_ratio = np.sum(heatmap > 0.5) / heatmap.size
   print(f"Focused activation ratio: {activation_ratio:.3f}")
   # Expected: 0.05 - 0.30 (5-30% of image has high activation)
   ```

3. **Consistency Check:**
   - Same image should produce similar heatmaps across runs
   - Different disease classes should show different activation patterns

### Verification Command

```bash
# Generate Grad-CAM and inspect
python -m src.gradcam \
    --image test_healthy_leaf.jpg \
    --model models/best_model.pth \
    --output healthy_gradcam.png

# Check output files exist and are valid images
ls -la gradcam_result.png original.png overlay.png
```

---

## Configuration Verification

### New config.yaml Parameters

| Parameter | Location | Default | Used By |
|-----------|----------|---------|---------|
| `model.architecture` | Line 12 | "efficientnet_b2" | train.py, evaluate_and_convert.py |
| `model.num_classes` | Line 13 | 39 | All model building functions |
| `image.size` | Line 19 | 260 | All image processing |
| `image.mean` | Line 20 | [0.485, 0.456, 0.406] | All normalization |
| `image.std` | Line 21 | [0.229, 0.224, 0.225] | All normalization |
| `training.batch_size` | Line 27 | 32 | train.py, preprocess.py |
| `training.learning_rate` | Line 28 | 1e-4 | train.py |
| `training.epochs` | Line 30 | 20 | train.py |
| `augmentation.*` | Lines 52-66 | Various | preprocess.py |
| `quality_validation.*` | Lines 70-82 | Various | quality_validator.py |
| `inference.*` | Lines 86-92 | Various | inference.py |
| `metadata.version` | Line 100 | "1.0.0" | metadata.py |

### Files That Consume config.yaml

1. **`src/train.py`** - Reads via `build_dataloaders()` from preprocess
2. **`src/evaluate_and_convert.py`** - Reads directly via `load_config()`
3. **`src/preprocessing/preprocess.py`** - Reads via `load_config()`
4. **`src/quality_validator.py`** - Can read for thresholds
5. **`src/metadata.py`** - Reads for metadata export

### Verification Command

```bash
# Check that config is being read correctly
python -c "
import yaml
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f\"Architecture: {config['model']['architecture']}\")
print(f\"Image size: {config['image']['size']}\")
print(f\"Num classes: {config['model']['num_classes']}\")
print(f\"Batch size: {config['training']['batch_size']}\")
"
```

---

## Validation Report Template

Copy this template and fill it in during testing:

```markdown
# Validation Report - Plant Disease Detection Pipeline

**Date:** _______________
**Tester:** _______________
**Environment:** _______________

---

## 1. Training

- [ ] Command executed successfully
- [ ] `models/best_model.pth` created
- [ ] Training loss decreased over epochs
- [ ] Validation accuracy > 80%
- [ ] No NaN/Inf values in loss

**Notes:**
_______________________________________________

## 2. Evaluation & Export

- [ ] `plant_model.onnx` created
- [ ] `plant_model_tflite_float32/plant_model.tflite` created
- [ ] `plant_model_tflite_int8/plant_model_int8.tflite` created
- [ ] Test accuracy matches training report
- [ ] No errors during ONNX→TFLite conversion

**File sizes:**
- ONNX: _______ MB
- Float32 TFLite: _______ MB
- INT8 TFLite: _______ MB

**Notes:**
_______________________________________________

## 3. Architecture Verification

- [ ] No `efficientnet_b0` references found
- [ ] All 5 files use `efficientnet_b2`
- [ ] Image size is 260 everywhere (except known issue in export_to_tflite.py)

**grep results:**
```
efficientnet_b0: _______ occurrences
efficientnet_b2: _______ occurrences
```

**Notes:**
_______________________________________________

## 4. Float32 Inference

- [ ] Model loads successfully
- [ ] Inference runs without errors
- [ ] Prediction is reasonable
- [ ] Confidence is between 0-1
- [ ] Top-3 predictions displayed

**Sample output:**
```
Model input dtype: _______________
Prediction: _______________
Confidence: _______________%
```

**Notes:**
_______________________________________________

## 5. INT8 Inference

- [ ] Model loads successfully
- [ ] Inference runs without errors
- [ ] Prediction matches float32 model
- [ ] Quantization parameters detected
- [ ] Confidence is between 0-1

**Sample output:**
```
Model input dtype: _______________
Model quantization: scale=_______, zero_point=_______
Prediction: _______________
Confidence: _______________%
```

**Notes:**
_______________________________________________

## 6. Grad-CAM

- [ ] `gradcam_result.png` created
- [ ] `original.png` created
- [ ] `overlay.png` created
- [ ] Heatmap shows focused activation on leaf
- [ ] Visualization is clear and interpretable

**Activation ratio:** _______ (expected: 0.05-0.30)

**Notes:**
_______________________________________________

## 7. Quality Validation

- [ ] Validator runs without errors
- [ ] Blur detection works
- [ ] Brightness detection works
- [ ] Resolution check works
- [ ] Guidance messages are helpful

**Test results:**
- Good image: PASS / FAIL
- Blurry image: PASS / FAIL
- Dark image: PASS / FAIL
- Low res image: PASS / FAIL

**Notes:**
_______________________________________________

## 8. Metadata Export

- [ ] `metadata.json` created
- [ ] `labels.json` updated
- [ ] `labels.txt` created
- [ ] All files have correct content
- [ ] Version is set correctly

**Notes:**
_______________________________________________

## 9. Configuration

- [ ] `configs/config.yaml` is valid YAML
- [ ] All expected parameters present
- [ ] Values are reasonable
- [ ] No hardcoded values remain in code

**Notes:**
_______________________________________________

## Overall Assessment

- [ ] Pipeline works end-to-end
- [ ] All critical issues resolved
- [ ] Ready for Android deployment

**Critical Issues Found:**
_______________________________________________

**Recommendations:**
_______________________________________________
```

---

## Common Failure Symptoms & Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ImportError: No module named torch` | PyTorch not installed | `pip install -r requirements.txt` |
| `FileNotFoundError: data/raw/` | Dataset not downloaded | Run `download_plantvillage.py` or place data manually |
| `RuntimeError: size mismatch` | Architecture mismatch | Ensure all files use same EfficientNet version |
| `ONNX export failed` | Opset version issue | Check ONNX version compatibility |
| `onnx2tf failed` | Missing dependencies | Install `onnx2tf` and TensorFlow |
| `TFLite inference gives wrong results` | Input preprocessing mismatch | Verify normalization matches training |
| `Grad-CAM heatmap is uniform` | Wrong target layer | Verify layer name matches model architecture |
| `Quality validator always fails` | Thresholds too strict | Adjust thresholds in config.yaml |

---

**End of Validation Guide**