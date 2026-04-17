# Plant Disease Detection

Plant leaf disease classification using EfficientNet-B0 (PyTorch). Achieves ~99.67% test accuracy. Supports ONNX/TFLite export for deployment.

## Features
- Data preprocessing & augmentation
- Training w/ early stopping, cosine LR
- Evaluation & metrics
- Export to ONNX/TFLite

## Setup
1. Activate venv: `py3_10\\Scripts\\activate` (Windows)
2. Install deps: `pip install -r requirements.txt`
3. Data in `data/raw/` (PlantVillage-like)

## Usage
```bash
# Train
python -m src.train

# Evaluate & Export
python -m src.evaluate
```

## Config
Edit `configs/config.yaml` for hparams.

## Outputs
- `models/best_model.pth`
- `plant_model.onnx`
- `plant_model_tflite/float32/plant_model.tflite` etc.

## Structure
```
├── src/          # Code
├── configs/      # Configs
├── data/         # Datasets, labels
├── models/       # Checkpoints
├── notebooks/    # Explores
├── tests/        # Tests
├── requirements.txt
└── ...
```

*Cleaned/organized by BLACKBOXAI*
