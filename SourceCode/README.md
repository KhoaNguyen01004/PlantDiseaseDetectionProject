# Plant Disease Detection

Plant leaf disease classification using EfficientNet-B2 (PyTorch). Achieves ~99.67% test accuracy on PlantVillage-like data. Supports ONNX/TFLite export for deployment.

## Features
- **Model**: EfficientNet-B2 (260×260 input images)
- **Data preprocessing & augmentation** (real-world robustness: weather, camera artifacts)
- **Training** with early stopping, cosine learning rate scheduler
- **Evaluation & metrics** (accuracy, classification report)
- **Export to ONNX/TFLite** (float32 and INT8 quantized models)
- **Grad-CAM explainability** (heatmap visualization)
- **Photo quality validation** (blur, brightness, resolution checks)
- **Metadata management** (versioning, labels export)

## Setup
1. Activate venv: `py3_10\\Scripts\\activate` (Windows)
2. Install deps: `pip install -r requirements.txt`
3. Data in `data/raw/` (PlantVillage-like dataset)
4. Configuration in `configs/config.yaml` (single source of truth)

## Usage
```bash
# Train model (EfficientNet-B2)
python -m src.train

# Evaluate & Export to ONNX/TFLite
python -m src.evaluate_and_convert

# Test inference with TFLite model
python -m src.inference --model plant_model_tflite_float32/plant_model.tflite --image test.jpg

# Generate Grad-CAM heatmap
python -m src.gradcam --image test.jpg --model models/best_model.pth

# Validate image quality
python -m src.quality_validator --image test.jpg

# Export model metadata
python -m src.metadata

# Prepare real-world dataset structure
python prepare_real_world_dataset.py --create-classes --summary
```

## Configuration
All hyperparameters are centralized in `configs/config.yaml`:
- Model architecture (efficientnet_b2)
- Image size (260×260)
- Training hyperparameters
- Data augmentation settings
- Quality validation thresholds
- Inference configuration
- Export options

## Outputs
After running the full pipeline:
- `models/best_model.pth` - Trained PyTorch checkpoint
- `plant_model.onnx` - ONNX model for cross-platform deployment
- `plant_model_tflite_float32/plant_model.tflite` - Float32 TFLite model
- `plant_model_tflite_int8/plant_model_int8.tflite` - INT8 quantized TFLite model
- `metadata.json`, `labels.json`, `labels.txt` - Model metadata and class labels
- Grad-CAM visualizations (when using gradcam.py)

## Detailed Structure
```
SourceCode/
├── src/
│   ├── __init__.py
│   ├── train.py              # Training (EfficientNet-B2, weighted sampler)
│   ├── evaluate_and_convert.py # Evaluation + ONNX/TFLite export (float32/INT8)
│   ├── inference.py          # TFLite inference CLI (dynamic dtype detection)
│   ├── gradcam.py            # Grad-CAM explainability (target: features.8.0)
│   ├── quality_validator.py  # Photo quality validation (blur, brightness, resolution)
│   ├── metadata.py           # Model metadata management & export
│   └── preprocessing/
│       ├── __init__.py
│       ├── preprocess.py     # Data prep, augmentation, dataloader building
│       └── generate_background_noise.py  # Background noise generation
├── configs/
│   ├── __init__.py
│   └── config.yaml           # Centralized configuration (single source of truth)
├── export_to_tflite.py       # Alternative ONNX→TFLite export script
├── export_torchscript.py     # TorchScript export (DEPRECATED, kept for compatibility)
├── prepare_real_world_dataset.py  # Dataset structure preparation
├── prepare_unknown_dataset.py # Unknown class dataset preparation
├── download_plantvillage.py  # Dataset download utility
├── data/ (ignored)           # Raw/processed PlantVillage dataset
├── models/ (ignored)         # Trained checkpoints (.pth)
├── saved_model/ (generated)  # TFLite export output
├── plant_model.onnx          # Exported ONNX model
├── plant_model_tflite_float32/  # Float32 TFLite export
├── plant_model_tflite_int8/     # INT8 quantized TFLite export
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package configuration
└── README.md
```

## Dependencies
See `requirements.txt`:
- **Core**: PyTorch 2.4+, torchvision 0.19.1
- **Export**: ONNX 1.16+, TensorFlow 2.17+, onnx2tf
- **Inference**: TensorFlow Lite, OpenCV 4.10+
- **Augmentation**: albumentations 1.4.8+
- **Utilities**: scipy (softmax), matplotlib (Grad-CAM), PyYAML, scikit-learn

## Deployment
### TFLite Deployment (Recommended)
```bash
# Export to TFLite (float32 and INT8)
python -m src.evaluate_and_convert

# Copy to Android assets
cp plant_model_tflite_float32/plant_model.tflite ../agrilens/app/src/main/assets/
cp labels.json ../agrilens/app/src/main/assets/
cp labels.txt ../agrilens/app/src/main/assets/
```

### Android Integration
- See root `README.md` and `agrilens/README.md` for Android setup
- TFLite models are ready for mobile deployment
- Metadata files ensure label consistency across platforms

---
*Last updated: 2026-05-30 | Version 1.0.0*
