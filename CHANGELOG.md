# Plant Disease Detection Project - Changelog

This document tracks all changes made during the project refactor and improvement roadmap implementation.

---

## [1.0.8] - 2026-06-01

### Summary
Enhanced unknown dataset training support: made unknown image limit configurable via `configs/config.yaml` and command-line arguments. Allows flexible scaling of background/unknown images during training without code changes.

### Changes
- **`SourceCode/configs/config.yaml`**: Added `unknown_limit` parameter to `data` section (default: 0 for no limit)
- **`SourceCode/src/train.py`**: 
  - Added `--unknown-limit` CLI argument for runtime override
  - Modified unknown dataset loading to respect config value
  - Now supports unlimited unknown images by default
- **`SourceCode/prepare_unknown_dataset.py`**: Replaced CIFAR100-based unknown dataset creation with Leafsnap field-image import
  - Copies all images from `source_dir/images/field/`
  - Added optional Kaggle download support via `--download`
  - Cleans up temporary downloads after extraction
  - Preserves only field images from the Leafsnap dataset

### Usage Examples
```bash
# Default: use all available unknown images
python -m src.train

# Override config: use all available unknown images explicitly
python -m src.train --unknown-limit 0

# Override config: use 10000 unknown images
python -m src.train --unknown-limit 10000

# Prepare the unknown dataset from a local Leafsnap dataset directory
python prepare_unknown_dataset.py --source-dir path/to/leafsnap/dataset

# Prepare the unknown dataset by downloading Leafsnap from Kaggle (requires Kaggle CLI auth)
python prepare_unknown_dataset.py --download
```

### Benefits
- Flexible dataset scaling without code modification
- Support for larger datasets as more unknown/background images are collected
- Easy per-training customization via CLI override
- Enables using Leafsnap field images as the unknown training set

### Files modified
- `SourceCode/configs/config.yaml` (added `unknown_limit: 0`)
- `SourceCode/src/train.py` (added CLI arg, config loading, dynamic limit)
- `SourceCode/prepare_unknown_dataset.py` (Leafsnap field-image import)

---

## [1.0.7] - 2026-06-01

### Summary
Production & portfolio readiness updates: added GitHub portfolio entry, centralized structured logging, replaced ad-hoc prints with structured logs in core scripts, and validated CI workflows. Also fixed config schema alignment, training/validation data handling, centralized inference config loading, and CI config validation logic.

### Highlights
- Added `.github/README.md` with a concise portfolio overview, ASCII architecture diagrams, quick-start, and feature matrix.
- Introduced `SourceCode/src/logging_config.py` for centralized structured logging.
- Replaced `print()` calls with `logger` calls in key modules: `train.py`, `metadata.py`, `quality_validator.py`, and `evaluate_and_convert.py`.
- Fixed training pipeline and data split handling so validation/test datasets use separate normalization transforms.
- Aligned preprocessing and export scripts with the current `SourceCode/configs/config.yaml` schema.
- Updated Python inference to load shared config values for image size and normalization.
- Confirmed and/or added GitHub Actions workflows for testing, linting, and config validation under `.github/workflows/`.
- Updated `CHANGELOG.md` to reflect portfolio and infrastructure improvements.

### Files added / modified
- `.github/README.md` (portfolio overview)
- `SourceCode/src/logging_config.py` (centralized logger)
- `SourceCode/src/train.py` (logging replacements; `print()` → `logger`; train/validation transform fix)
- `SourceCode/src/metadata.py` (logging replacements; `print()` → `logger`)
- `SourceCode/src/quality_validator.py` (logging replacements; `print()` → `logger`)
- `SourceCode/src/evaluate_and_convert.py` (logging replacements; `print()` → `logger`; config schema alignment for export)
- `SourceCode/src/inference.py` (centralized config loading for image size and normalization)
- `SourceCode/src/preprocessing/preprocess.py` (config schema alignment)
- `SourceCode/README.md` (clarified export paths)
- `.github/workflows/pytest.yml` (tests CI)
- `.github/workflows/lint.yml` (lint & formatting checks)
- `.github/workflows/config-validation.yml` (config schema & version checks)
- `.github/README.md` (performance placeholders added)
- `CHANGELOG.md` (this file updated)

### Notes on placeholders and runtime data
- Replaced hardcoded performance numbers (model sizes, latency, memory) with placeholders in `.github/README.md` and `SourceCode/README.md`. These must be populated from measured artifacts produced by `SourceCode/src/evaluate_and_convert.py` and device benchmarks.
- Kept any dataset- or hardware-dependent values as placeholders (`{{...}}`) to avoid hardcoding runtime numbers.
- `report.md` TODO items were consolidated into clear 'Future Work' markers; where runtime statistics are required, placeholders were left for manual population after benchmarking.

### Rationale
- Improve observability and structured diagnostics by centralizing logging and replacing ad-hoc prints.
- Make CI explicit and reproducible by adding workflows for tests, linting, and config validation.
- Keep documentation honest and reproducible by not hardcoding runtime-specific numbers—placeholders direct reviewers to the evaluation artifacts for authoritative metrics.

## [1.0.6] - 2026-05-31

### Summary
Final repository hardening phase. Regenerated `PROJECT_SUMMARY.md` from verified source code state, removed obsolete documentation references, eliminated unsupported hardcoded accuracy claims, and added architecture decision records plus a version manifest.

### Documentation hardening
- Regenerated `PROJECT_SUMMARY.md` to reflect current repo structure, model architecture, deployment paths, and feature status.
- Removed stale references to `SourceCode/config.yaml` and legacy `SourceCode/TODO.md` in documentation.
- Replaced hardcoded performance claims with references to generated evaluation artifacts.
- Clarified status for Grad-CAM, Quality Validator, Unknown Detection, and Android integration.
- Added ADRs under `docs/adr/` and `version.json`.
- Updated knowledge notes in `PlantDiseaseDetectionKnowledge/` to remove outdated EfficientNet-B0 and 224×224 preprocessing claims.

### Files modified
- `PROJECT_SUMMARY.md`
- `README.md`
- `SourceCode/README.md`
- `CHANGELOG.md`
- `PlantDiseaseDetectionKnowledge/PlantDiseaseDetection_CodeReview.md`
- `PlantDiseaseDetectionKnowledge/02 Inference_Preprocess.md`
- `PlantDiseaseDetectionKnowledge/04 Training_Loop.md`
- `docs/adr/ADR-001-use-efficientnet-b2.md`
- `docs/adr/ADR-002-dual-deployment-strategy.md`
- `docs/adr/ADR-003-centralized-config-and-metadata.md`
- `docs/adr/ADR-004-testing-strategy.md`
- `version.json`

---

## [1.0.5] - 2026-05-31

### Summary
Full documentation consistency audit. Fixed critical contradictions regarding deployment runtime and export paths. All documentation now accurately reflects the dual deployment architecture: TFLite for Python CLI and PyTorch Mobile (TorchScript) for Android.

---

### 🔍 Documentation Consistency Audit

#### Critical Issues Fixed:

**1. Android Deployment Runtime Contradiction**
- **Issue**: README.md and SourceCode/README.md incorrectly stated that Android uses TFLite
- **Reality**: Android app uses PyTorch Mobile (TorchScript .pt format)
- **Evidence**:
  - `agrilens/app/build.gradle.kts` (lines 81-82): Uses `org.pytorch:pytorch_android:1.13.1`
  - `ImageClassifierHelper.java`: Uses `Module.load()` and PyTorch tensors
  - Expects `plant_model.pt` file, not `.tflite`
- **Fix**: Updated all documentation to accurately reflect dual deployment paths

**2. Export Pipeline Clarification**
- **Issue**: Documentation suggested single TFLite export path for all deployment
- **Reality**: Two separate export paths exist:
  - Python CLI: PyTorch → ONNX → TFLite (via `evaluate_and_convert.py`)
  - Android: PyTorch → TorchScript .pt (via `export_torchscript.py`)
- **Fix**: Clarified dual export paths in all documentation

#### Files Modified:

**README.md**
- Updated Overview section to mention dual export paths (ONNX/TFLite for Python CLI, TorchScript for Android)
- Updated Technology Used section to include PyTorch Mobile as Android runtime
- Updated Quick Setup to include TorchScript export command
- Updated Project Structure to clarify Python vs Android inference paths
- Updated Roadmap to reflect PyTorch Mobile integration status

**SourceCode/README.md**
- Updated description to mention dual export paths
- Updated Features section to explicitly list dual export paths
- Updated Usage section to include TorchScript export command
- Updated Outputs section to include plant_model.pt file
- Updated Detailed Structure to clarify export script purposes
- Updated Deployment section to separate Python CLI (TFLite) from Android (PyTorch Mobile)
- Updated Android Integration section to clarify PyTorch Mobile usage

**VALIDATION_GUIDE.md**
- Added note at top explaining dual deployment paths
- Updated Table of Contents to include TorchScript Verification section
- Added new section 4: TorchScript Export (Android)
- Renumbered subsequent sections (5-10)
- Added new section: TorchScript Verification (Android)
- Updated Expected Outputs to include plant_model.pt
- Added verification steps for PyTorch Mobile integration

#### Verification Against Source Code:

**Model Architecture**: ✅ Consistent
- All docs: EfficientNet-B2
- Source code: `configs/config.yaml`, `train.py`, `evaluate_and_convert.py` all use EfficientNet-B2
- Image size: 260×260 everywhere (matches EfficientNet-B2)

**Image Size**: ✅ Consistent
- All docs: 260×260
- Source code: config.yaml, train.py, evaluate_and_convert.py, inference.py, Android all use 260

**Deployment Runtime**: ✅ Fixed
- Before: Contradictory (TFLite vs PyTorch Mobile)
- After: Accurate (TFLite for Python CLI, PyTorch Mobile for Android)

**Export Pipeline**: ✅ Fixed
- Before: Single TFLite path implied
- After: Dual paths clearly documented

**Labels**: ✅ Consistent
- All docs: labels.json and labels.txt
- Source code: metadata.py exports both formats

**Metadata**: ✅ Consistent
- All docs: metadata.json with model info
- Source code: metadata.py generates comprehensive metadata

**Configuration**: ✅ Consistent
- All docs: configs/config.yaml as single source of truth
- Source code: All scripts read from config.yaml

**Android Integration**: ✅ Fixed
- Before: Implied TFLite integration
- After: Accurately describes PyTorch Mobile (TorchScript) integration

---

### 📝 Summary of Changes (v1.0.5)

| Category | Files Modified | Lines Changed |
|----------|----------------|---------------|
| Documentation Fixes | 3 | ~50 |
| **TOTAL** | **3** | **~50** |

---

### ✅ Verification Status

| Check | Status |
|-------|--------|
| Model architecture consistency (EfficientNet-B2) | ✅ Verified |
| Image size consistency (260) | ✅ Verified |
| Deployment runtime accuracy | ✅ Fixed |
| Export pipeline clarity | ✅ Fixed |
| Dual deployment paths documented | ✅ Complete |
| Configuration centralization | ✅ Verified |

---

## [1.0.4] - 2026-05-31

### Summary
Comprehensive verification of codebase consistency following Phase 1 and Phase 2 completion. Fixed minor image size inconsistencies and updated documentation to reflect current state. All 21 tests passing successfully.

---

### 🔍 Phase 1: Codebase Verification

#### Verification Results:
- **Architecture Consistency**: All files now consistently use EfficientNet-B2 (configurable via config.yaml)
- **Image Size Consistency**: All components use 260×260 input size (matches EfficientNet-B2)
- **Test Suite**: All 21 tests passing successfully
- **Configuration**: Centralized config.yaml is the single source of truth

#### Files Modified:
- **`SourceCode/src/preprocessing/generate_background_noise.py`**
  - Updated image size from 224 to 260 (line 13)
  - Now matches EfficientNet-B2 input size from config.yaml
  - Ensures background noise images are consistent with model input

- **`SourceCode/src/evaluate_and_convert.py`**
  - Updated default parameter from 224 to 260 (line 72)
  - export_onnx() function now defaults to correct image size
  - Note: Actual value is overridden by config at line 190, so this is a safety fallback

#### Documentation Updates:
- **`PROJECT_SUMMARY.md`**
  - Updated architecture status from "CRITICAL MISMATCH" to "CONSISTENT"
  - Marked architecture mismatch as FIXED in v1.0.0
  - Updated checklist to reflect completed fixes
  - Removed outdated warnings about B0/B2 mismatch

- **`VALIDATION_GUIDE.md`**
  - Updated verification commands to reflect current state
  - Updated image size table to show all components use 260
  - Changed note from "known issue" to "fixed in v1.0.0"
  - Updated architecture verification checklist to show completed items
  - Updated failure symptoms table to remove outdated B0/B2 mismatch reference

#### Verification Commands Run:
```bash
# Ran all tests
pytest tests/ -v
# Result: 21 passed in 76.98s

# Checked for architecture consistency
grep -r "efficientnet_b0" SourceCode/ --include="*.py"
# Result: Only found in config comment and train.py factory (acceptable)

# Checked for image size consistency
grep -r "224" SourceCode/ --include="*.py"
# Result: Only found in ImageNet std values and quality thresholds (correct)
```

---

### 📝 Summary of Changes (v1.0.4)

| Category | Files Modified | Lines Changed |
|----------|----------------|---------------|
| Code Fixes | 2 | ~2 |
| Documentation Updates | 2 | ~20 |
| **TOTAL** | **4** | **~22** |

---

### ✅ Verification Status

| Check | Status |
|-------|--------|
| Architecture consistency (EfficientNet-B2) | ✅ Verified |
| Image size consistency (260) | ✅ Verified |
| Test suite passing | ✅ 21/21 tests |
| Documentation accuracy | ✅ Updated |
| Configuration centralization | ✅ Verified |

---

## [1.0.3] - 2026-05-31

### Summary
Completed Phase 2 of the Improvement Roadmap. Built a comprehensive testing infrastructure including full conftest fixtures, pytest configuration, and unit tests covering all critical components: config loading, metadata export, model exporting, inference preprocessing, quality validation, and data loaders. All 21 tests execute and pass successfully.

---

### 🧪 Phase 2: Testing Infrastructure

#### Files Added:
- **`SourceCode/tests/__init__.py`**
  - Package initialization for the test suite.
- **`SourceCode/tests/conftest.py`**
  - Established pytest fixtures including project root detection, config loading, PIL/OpenCV mock images, and dynamic temporary datasets for preprocessing tests.
- **`SourceCode/pytest.ini`**
  - Directs pytest execution paths, search formats, and silences irrelevant environment/deprecation warnings.
- **`SourceCode/tests/test_config.py`**
  - Unit tests verifying config existence, schema, required sections, learning rate conversions, and training split sum-to-one validation.
- **`SourceCode/tests/test_metadata.py`**
  - Tests ModelMetadata instantiation, metrics configuration, deterministic folder hashing, and label formats.
- **`SourceCode/tests/test_export.py`**
  - Tests PyTorch-to-ONNX model compilation and export using a custom lightweight mock network.
- **`SourceCode/tests/test_inference.py`**
  - Validates image pre-processing dimensions, quantization levels (float32 and uint8), and label json loading.
- **`SourceCode/tests/test_quality_validator.py`**
  - Tests Laplacian sharpness boundaries, resolution minimum/maximum checks, solid image brightness checks (dark, normal, bright), and formatting of user-facing feedback messages.
- **`SourceCode/tests/test_preprocessing.py`**
  - Tests building dataloaders with standard splits, reproducible random seeding, and isolates environment directories using monkeypatching.

---

## [1.0.2] - 2026-05-31

### Summary
Completed Phase 1 of the Improvement Roadmap. Fully centralized model configuration, consolidated label/metadata export pipelines under a unified API, introduced reproducible dataset hashing, added dynamic model architectures, and cleaned up/pinned all project dependencies.

---

### 🔧 Phase 1: Critical Consistency & Centralization Fixes

#### Files Modified:
- **`SourceCode/src/metadata.py`**
  - Added a deterministic and reproducible dataset hashing helper (`calculate_dataset_hash`) that walks folders deterministically to compute hashes based on file sizes/names (extremely fast, avoiding slow read-operations).
  - Standardized metadata dictionary schema to dynamically read from the hierarchical layout of `configs/config.yaml`.
  - Added `dataset_version` and `dataset_hash` tracking in the `training` metadata block.
  
- **`SourceCode/src/train.py`**
  - Fully refactored training entry point to load all parameters directly from `configs/config.yaml` with argparse overrides.
  - Eliminated the local duplicated `export_labels` method in favor of the unified `src.metadata.export_metadata_for_model` pipeline.
  - Introduced dynamic model factory mapping supporting `efficientnet_b0`, `efficientnet_b2`, and `efficientnet_v2_s` configurations.
  - Integrated deterministic seeds for training and dataset random splitting.
  - Added support for multiple customizable learning rate schedulers (`cosine`, `step`, `exponential`) loaded dynamically from configuration.

- **`SourceCode/src/preprocessing/preprocess.py`**
  - Consolidated label export logic by replacing local `export_to_tflite` implementation with the centralized `src.metadata.export_metadata_for_model` utility.
  - Enabled dynamic config loading for data pre-processing runs.

- **`SourceCode/export_to_tflite.py`**
  - Added a formal deprecation warning directing users to `src.evaluate_and_convert`.
  - Refactored script to read input target image size dynamically from the centralized `config.yaml`.

- **`SourceCode/requirements.txt`**
  - Added missing `matplotlib` package and pinned all dependencies to exact active environment versions.
  - Categorized and structured the packages clearly into core, deployment, tensorflow, and general scientific/utility stacks.

---

## [1.0.1] - 2026-05-30

### Summary
Documentation cleanup, project maintenance, and academic report creation. Updated all documentation to reflect the current EfficientNet-B2 architecture and deployment pipeline. Removed obsolete configuration files. Created comprehensive academic report for course "Computer Vision in Human-Robot Interaction".

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

#### Files Created:
- **`report.md`**
  - Comprehensive academic report for course "Computer Vision in Human-Robot Interaction"
  - **Structure:**
    - Cover page with university, department, course, and student information
    - Introduction (research context, motivation, objectives)
    - Chapter 1: Project Overview (problem statement, objectives, scope, significance)
    - Chapter 2: Theoretical Background (Computer Vision, Deep Learning, EfficientNet-B2, TFLite, Grad-CAM, Android AI)
    - Chapter 3: System Analysis and Design (architecture, requirements, data flow, metadata design)
    - Chapter 4: System Implementation (most detailed chapter - dataset, preprocessing, augmentation, model building, training, quality validation, Grad-CAM, metadata management, export pipeline, Android app)
    - Chapter 5: Experiments and Evaluation (with TODO placeholders for runtime results)
    - Chapter 6: Conclusion and Future Work
    - References (placeholder)
    - Appendices (configuration, commands, directory structure)
  - **Key Features:**
    - Written in formal academic Vietnamese (third person)
    - Based on actual source code analysis
    - No fabricated experimental results
    - Uses TODO placeholders for runtime-dependent metrics
    - ~800+ lines of detailed technical documentation
  - **Technical Content:**
    - Detailed explanation of EfficientNet-B2 architecture
    - Complete data preprocessing and augmentation pipeline
    - Grad-CAM implementation and usage
    - Quality validation system (blur, brightness, resolution)
    - ONNX/TFLite export pipeline
    - Android deployment with PyTorch Mobile

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
| Academic Report | 1 | 0 | 0 | ~800 |
| Obsolete File Removal | 0 | 0 | 2 | N/A |
| **TOTAL** | **1** | **3** | **2** | **~950** |

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
- **report.md** - NEW comprehensive academic report for course submission
- All architecture references standardized to EfficientNet-B2
- All directory references standardized to `agrilens/`

---

**End of Changelog**