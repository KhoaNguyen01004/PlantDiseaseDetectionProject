# Plant Disease Detection Project - Changelog

This document tracks all changes made during the project refactor and improvement roadmap implementation.

---

## [1.0.1] - 2026-05-30

### Summary
Documentation cleanup and project maintenance. Updated all documentation to reflect the current EfficientNet-B2 architecture and deployment pipeline. Removed obsolete configuration files.

---

### 📝 Documentation Updates

#### Files Modified:
- **`README.md`**
  - Updated model architecture from EfficientNet-B0 to EfficientNet-B2
  - Fixed directory reference from `AndroidApp/` to `agrilens/`
  - Updated technology stack section
  - Enhanced project structure diagram with all new components
  - Updated roadmap with current status emojis
  - Improved quick setup instructions

- **`SourceCode/README.md`**
  - Updated model architecture references (B0 → B2)
  - Added comprehensive feature list including Grad-CAM, quality validation, metadata management
  - Expanded usage examples with all new CLI commands
  - Updated configuration section to reference `configs/config.yaml`
  - Enhanced outputs section with all generated artifacts
  - Updated detailed structure diagram
  - Expanded dependencies section with categorization
  - Added TFLite deployment instructions

- **`VALIDATION_GUIDE.md`**
  - Updated training command examples (removed redundant arguments)
  - Clarified metadata export command
  - Updated dataset preparation command examples
  - Enhanced troubleshooting section with additional error scenarios
  - Added scipy dependency to common issues

#### Files Removed:
- **`SourceCode/config.yaml`**
  - Old simple configuration file (9 lines)
  - Superseded by `SourceCode/configs/config.yaml` (115 lines)
  - Verified: NOT referenced by any Python code
  - Removal eliminates confusion about configuration source

- **`SourceCode/TODO.md`**
  - Outdated task list with single completed item
  - Tasks now tracked in CHANGELOG.md
  - No longer relevant for project maintenance

---

### 📚 Documentation Standards

#### Updated Guidelines:
- All architecture references standardized to EfficientNet-B2
- All directory references standardized to `agrilens/`
- Image size consistently documented as 260×260
- Configuration source clearly identified as `configs/config.yaml`
- Deployment pipeline clearly documented as ONNX→TFLite

---

## [1.0.0] - 2026-05-30

### Summary
Major refactor to fix architecture mismatches, improve robustness, and prepare for real-world deployment. All changes align with the Instruction.md roadmap.

---

### 🔧 Phase 1: Fixed Architecture Mismatch

#### Files Modified:
- **`SourceCode/src/evaluate_and_convert.py`**
  - Changed model from `efficientnet_b0` to `efficientnet_b2` (line 36)
  - Added comment clarifying architecture must match train.py
  - This ensures consistency with the trained model checkpoint

- **`SourceCode/export_to_tflite.py`**
  - Changed `build_model()` to use `efficientnet_b2` instead of `efficientnet_b0` (line 44)
  - Added comment clarifying architecture match requirement
  - Ensures TFLite export uses correct architecture

- **`SourceCode/export_torchscript.py`**
  - Added deprecation warning at module import (lines 8-24)
  - Warning informs users to use ONNX→TFLite pipeline instead
  - Script retained for backward compatibility but marked for future removal

---

### 📱 Phase 2: Fixed TFLite Inference Engine

#### Files Modified:
- **`SourceCode/src/inference.py`**
  - Complete rewrite for robust TFLite inference
  - **New Features:**
    - Dynamic input type detection (`detect_input_details()` function)
    - Proper handling of both float32 and uint8 models
    - Correct quantization parameter application
    - Softmax applied before confidence reporting (using `scipy.special.softmax`)
    - Top-3 predictions displayed
    - Updated image size to 260 (matching EfficientNet-B2)
  - **Fixed Issues:**
    - No longer hardcodes uint8 input
    - Properly detects model input dtype from TFLite interpreter
    - Applies correct normalization based on model type
    - Confidence scores now represent proper probabilities

---

### 🎯 Phase 3: Added Grad-CAM Explainability

#### New File:
- **`SourceCode/src/gradcam.py`**
  - Full Grad-CAM implementation for EfficientNet-B2
  - **Features:**
    - `GradCAM` class with hook-based gradient extraction
    - Target layer: `features.8.0` (final conv layer of EfficientNet-B2)
    - Heatmap generation with proper normalization
    - Batch processing support
    - Visualization utilities (`apply_heatmap_to_image()`, `save_visualization()`)
    - Command-line interface for easy use
  - **Output:**
    - Generates 3-panel visualization (original, heatmap, overlay)
    - Saves individual PNG files for integration

---

### 📸 Phase 4: Added Photo Quality Validation

#### New File:
- **`SourceCode/src/quality_validator.py`**
  - Complete photo quality validation system
  - **Checks Implemented:**
    - **Blur Detection:** Laplacian variance method (threshold: 100.0)
    - **Brightness Validation:** Min/max thresholds (30-220 on 0-255 scale)
    - **Resolution Validation:** Min/max dimensions (224-4000 pixels)
  - **Features:**
    - `QualityValidator` class with configurable thresholds
    - User-friendly guidance messages
    - Detailed validation results in JSON format
    - Command-line interface for testing
  - **Use Case:**
    - Validate images before classification
    - Provide actionable feedback to users
    - Integrate into Android app preprocessing pipeline

---

### 🌈 Phase 5: Enhanced Data Augmentation

#### Files Modified:
- **`SourceCode/src/preprocessing/preprocess.py`**
  - Completely rewrote `get_augmentation_pipeline()` function
  - **New Augmentations Added:**
    - **Weather Simulations:**
      - `RandomShadow` (p=0.3) - Simulates shadows on leaves
      - `RandomFog` (p=0.2) - Simulates foggy conditions
      - `RandomRain` (p=0.2) - Simulates rain droplets
    - **Camera Artifacts:**
      - `MotionBlur` (p=0.3) - Simulates camera shake
      - `GaussNoise` (p=0.4) - Simulates ISO noise
      - `ImageCompression` (p=0.3) - Simulates JPEG compression
      - `GaussianBlur` (p=0.2) - Simulates out-of-focus
    - **Enhanced Geometric Transforms:**
      - `RandomPerspective` (p=0.4)
      - `Affine` transforms (translation, shear)
    - **Enhanced Color Transforms:**
      - `HueSaturationValue` (p=0.3)
      - Stronger `RandomBrightnessContrast` (±0.4)
    - **Occlusion Simulation:**
      - `RandomErasing` (p=0.25)
  - **Goal:** Simulate real-world farming conditions and mobile photography

---

### ⚙️ Phase 6: Centralized Configuration

#### Files Modified:
- **`SourceCode/configs/config.yaml`**
  - Complete rewrite with comprehensive configuration
  - **New Sections:**
    - `model`: Architecture, num_classes, pretrained flag
    - `image`: Size, mean, std normalization values
    - `training`: All training hyperparameters
    - `data`: Data splits, workers, seed
    - `augmentation`: All augmentation probabilities
    - `quality_validation`: Blur, brightness, resolution thresholds
    - `inference`: Confidence thresholds, delegation settings
    - `export`: ONNX opset, TFLite quantization options
    - `metadata`: Version, description, author
  - **Benefits:**
    - Single source of truth for all configuration
    - No more hardcoded magic numbers
    - Easy to tune hyperparameters
    - Clear documentation of all parameters

---

### 📋 Phase 7: Metadata & Model Versioning

#### New File:
- **`SourceCode/src/metadata.py`**
  - Unified metadata management system
  - **Features:**
    - `ModelMetadata` class for managing model metadata
    - Exports: `metadata.json`, `labels.json`, `labels.txt`
    - Includes model architecture, image size, normalization params
    - Training metrics tracking
    - Version information with export date
    - Deployment target information (Android API level)
  - **Convenience Functions:**
    - `export_metadata_for_model()` - One-call export of all files
    - `get_label_map()` - Retrieve label mapping
    - `get_label_list()` - Get ordered label list
  - **Ensures:**
    - Consistent labels across Python and Android
    - Proper model versioning
    - Complete deployment metadata

---

### 🗂️ Phase 8: Real-World Dataset Preparation

#### New File:
- **`SourceCode/prepare_real_world_dataset.py`**
  - Dataset structure preparation for future real-world data
  - **Creates Directory Structure:**
    - `data/plantvillage/` - Existing clean PlantVillage data
    - `data/real_world_train/` - Future smartphone photos (training)
    - `data/real_world_test/` - Future smartphone photos (testing)
    - `data/unknown/` - Images that don't belong to any class
  - **Features:**
    - Automatic class directory creation based on PlantVillage
    - Dataset statistics reporting
    - Image count validation
    - Command-line interface
  - **Purpose:**
    - Prepare for future real-world smartphone data collection
    - Support mixed dataset training (PlantVillage + real-world)
    - Organize unknown/background images

---

### 📝 Summary of All Changes (v1.0.0)

| Category | Files Created | Files Modified | Lines Changed |
|----------|--------------|----------------|---------------|
| Architecture Fix | 0 | 3 | ~15 |
| TFLite Inference | 0 | 1 | ~100 |
| Grad-CAM | 1 | 0 | ~280 |
| Quality Validation | 1 | 0 | ~220 |
| Augmentation | 0 | 1 | ~50 |
| Configuration | 0 | 1 | ~100 |
| Metadata | 1 | 0 | ~240 |
| Dataset Prep | 1 | 0 | ~200 |
| **TOTAL** | **4** | **5** | **~1,205** |

### 📝 Summary of Documentation Cleanup (v1.0.1)

| Category | Files Created | Files Modified | Files Removed | Lines Changed |
|----------|--------------|----------------|---------------|---------------|
| Documentation Updates | 0 | 3 | 0 | ~150 |
| Obsolete File Removal | 0 | 0 | 2 | N/A |
| **TOTAL** | **0** | **3** | **2** | **~150** |

---

### 🎯 Roadmap Completion Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| Fix architecture mismatch (B2 everywhere) | ✅ Complete | All files now use EfficientNet-B2 |
| Unify deployment pipeline (ONNX→TFLite) | ✅ Complete | Single path: PyTorch→ONNX→TFLite |
| Fix TFLite inference | ✅ Complete | Dynamic dtype detection, softmax |
| Add photo quality validation | ✅ Complete | Blur, brightness, resolution checks |
| Add leaf presence validation | ⏸️ Deferred | Not implemented (per user: no YOLO yet) |
| Implement Grad-CAM | ✅ Complete | Python implementation ready |
| Enhance data augmentation | ✅ Complete | Weather, camera, occlusion effects |
| Centralize configuration | ✅ Complete | All params in configs/config.yaml |
| Metadata & versioning | ✅ Complete | Unified export system |
| Prepare for real-world data | ✅ Complete | Dataset structure ready |
| Remove TorchScript path | ⏸️ Deferred | Marked deprecated, kept for backup |
| **Documentation cleanup** | ✅ **Complete** | **All docs updated, obsolete files removed** |

---

### 🚀 Next Steps

1. **Test the complete pipeline:**
   ```bash
   # Train model
   python -m src.train
   
   # Evaluate and export
   python -m src.evaluate_and_convert
   
   # Test inference
   python -m src.inference --model plant_model_tflite_float32/plant_model.tflite --image test.jpg
   
   # Generate Grad-CAM
   python -m src.gradcam --image test.jpg --model models/best_model.pth
   
   # Validate image quality
   python -m src.quality_validator --image test.jpg
   
   # Prepare real-world dataset
   python prepare_real_world_dataset.py --create-classes --summary
   ```

2. **Deploy to Android:**
   - Copy TFLite model to `agrilens/app/src/main/assets/`
   - Copy metadata files (metadata.json, labels.json, labels.txt)
   - Integrate quality validation into Android camera flow
   - Integrate Grad-CAM visualization (future)

3. **Collect real-world data:**
   - Use smartphone to capture plant disease photos
   - Organize into `data/real_world_train/` and `data/real_world_test/`
   - Retrain with mixed dataset for improved robustness

---

### 📦 Dependencies Added (v1.0.0)

- `scipy` - For softmax in inference
- `matplotlib` - For Grad-CAM visualization
- `albumentations` - Already present, now fully utilized

### 📦 Dependencies Added (v1.0.1)

- No new dependencies added
- All cleanup was documentation-only

---

### ⚠️ Breaking Changes (v1.0.0)

- **Image size changed from 224 to 260** (EfficientNet-B2 default)
  - Update any existing code that assumes 224x224 input
  - Retrain models if using old checkpoints with new code

- **Inference API updated**
  - Now requires proper TFLite model with correct input type
  - Confidence scores now use softmax (proper probabilities)

### ⚠️ Breaking Changes (v1.0.1)

- **Removed `SourceCode/config.yaml`**
  - Old simple config file deleted
  - All configuration now in `SourceCode/configs/config.yaml`
  - No code was using the old file, so no functional impact

- **Documentation structure updated**
  - README.md now references `agrilens/` instead of `AndroidApp/`
  - Update any bookmarks or links to project structure

---

### 📚 Documentation Updated (v1.0.0)

- This CHANGELOG.md created
- Code comments added throughout
- Docstrings added to all new functions
- Configuration file fully documented

### 📚 Documentation Updated (v1.0.1)

- **README.md** - Complete overhaul with current architecture
- **SourceCode/README.md** - Comprehensive update with all features
- **VALIDATION_GUIDE.md** - Updated commands and troubleshooting
- **PROJECT_SUMMARY.md** - Kept as historical reference (unchanged)
- All architecture references standardized to EfficientNet-B2
- All directory references standardized to `agrilens/`

---

**End of Changelog**