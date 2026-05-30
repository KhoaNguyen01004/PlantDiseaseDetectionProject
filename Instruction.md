# Plant Disease Detection Project — Refactor & Improvement Roadmap

## Project Vision

Build a fully offline Android application for plant disease diagnosis that works in real agricultural environments without internet access.

### Target Pipeline

```text
Farmer takes photo
        ↓
Photo quality validation
        ↓
Leaf validation / detection
        ↓
Disease classification
        ↓
Explainability heatmap
        ↓
Treatment recommendation
```

Requirements:

- Fully offline
- Android-first
- No cloud inference
- TFLite deployment
- Mobile optimized
- Designed for future real-world farm photos

---

## Key Strategic Decisions

### Keep

- Mobile-first approach
- Offline inference
- Android deployment
- Lightweight CNN-based classification
- Future real-world data collection

### Avoid

- Cloud APIs
- Large Vision-Language Models
- LocateAnything integration for production
- Drones and flycam-based workflows
- Internet-dependent features

---

## Highest Priority Fixes

### 1. Fix Architecture Mismatch

Current issue:

- `train.py` uses EfficientNet-B2
- `evaluate_and_convert.py` uses EfficientNet-B0
- `export_to_tflite.py` uses EfficientNet-B0

Choose one architecture and use it everywhere.

Recommended:

- EfficientNetV2-S

Alternative:

- EfficientNet-B2

Store architecture inside configuration instead of hardcoding.

---

### 2. Unify Deployment Pipeline

Current state:

```text
PyTorch
 ├── TorchScript
 └── ONNX → TFLite
```

Target:

```text
PyTorch
    ↓
ONNX
    ↓
TFLite
    ↓
Android
```

Benefits:

- Easier maintenance
- Single deployment path
- Smaller technical debt

---

### 3. Fix TFLite Inference

Current issues:

- Always casts input to uint8
- Ignores quantization parameters
- Confidence calculation is unreliable

Required:

- Detect model input type dynamically
- Support float32 models correctly
- Support INT8 models correctly
- Apply softmax before confidence reporting

---

## Real-World Robustness Improvements

### 4. Photo Quality Validation

Add a validation step before classification.

Checks:

- Blur detection
- Brightness validation
- Resolution validation

Return clear user guidance:

- Image too blurry
- Lighting too dark
- Move closer to leaf

---

### 5. Leaf Presence Validation

Before disease classification:

```text
Image
 ↓
Leaf Validator
 ↓
Disease Classifier
```

Use a lightweight MobileNet-based classifier:

- Leaf
- Not Leaf

Do not use large grounding models.

---

### 6. Future Leaf Detector Interface

Create a detector abstraction.

Current implementation can be a placeholder.

Future replacement:

- YOLOv11n

Design the system so detector upgrades do not require classifier changes.

---

## Training Improvements

### 7. Stronger Data Augmentation

Add:

- RandomShadow
- RandomFog
- RandomRain
- MotionBlur
- GaussNoise
- ImageCompression
- RandomBrightnessContrast

Goal:

Simulate real farming conditions and mobile photography.

---

### 8. Prepare for Real-World Data

Future dataset structure:

```text
data/
├── plantvillage/
├── real_world/
├── unknown/
```

Training pipeline should support mixing datasets automatically.

Important:

Improving dataset quality is expected to produce more gains than switching to newer model architectures.

---

## Explainability

### 9. Implement Grad-CAM

Generate:

- Original image
- Heatmap
- Overlay

Display inside Android app:

```text
Disease: Early Blight
Confidence: 92%
Highlighted affected area
```

Purpose:

- Improve user trust
- Improve interpretability

---

## Unknown Class Handling

### 10. Improve Open-Set Recognition

Current approach:

```text
confidence < threshold
```

Short-term:

- Maximum softmax probability

Future-ready design:

- Energy-based detection
- Entropy-based detection

Make the implementation modular.

---

## Metadata & Label Management

### 11. Single Source of Truth

Every exported model should include:

```text
labels.json
labels.txt
metadata.json
```

Metadata example:

```json
{
  "architecture": "...",
  "num_classes": 0,
  "image_size": 224,
  "export_date": "..."
}
```

Android and Python inference must use the same labels.

---

## Configuration Refactor

### 12. Centralize Configuration

Move hardcoded values into config files.

Examples:

- Architecture
- Image size
- Thresholds
- Quality validation settings

Remove magic numbers from source code.

---

## Code Cleanup

### 13. Rename Misleading Functions

Example:

`export_to_tflite()` currently exports labels only.

Rename it to reflect actual behavior or implement true TFLite export.

Function names should match functionality.

---

## Android Roadmap

Target Android flow:

```text
Photo Capture
↓
Quality Validation
↓
Leaf Validation
↓
Disease Classification
↓
Grad-CAM
↓
Treatment Guide
```

Features:

- Offline operation
- Confidence display
- Retake image prompts
- Explainability visualization

---

## Honest Assessment

The current mobile-first offline strategy is realistic and appropriate for the target users.

The largest project risk is not model architecture but dataset domain gap:

- PlantVillage images are clean and controlled.
- Real farm images are noisy and unpredictable.

Therefore:

1. Fix architecture and deployment issues.
2. Improve robustness and validation.
3. Collect real smartphone photos later.
4. Continue using lightweight mobile-friendly models.

The future success of the project will depend more on real-world data quality than on adopting larger or newer foundation models.
