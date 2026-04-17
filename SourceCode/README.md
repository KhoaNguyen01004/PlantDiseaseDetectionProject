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

## Detailed Structure
```
SourceCode/
├── src/
│   ├── __init__.py
│   ├── train.py              # Training w/ early stopping
│   ├── evaluate_and_convert.py # Eval metrics + ONNX/TFLite export
│   ├── inference.py          # Model inference
│   ├── models/               # EfficientNet-B0
│   │   └── __init__.py
│   └── preprocessing/
│       ├── __init__.py
│       ├── preprocess.py     # Data prep/aug
│       └── generate_background_noise.py
├── configs/config.yaml       # Hyperparams
├── data/ (ignored)           # Raw/processed PlantVillage
├── models/ (ignored)         # .pth, .onnx, .tflite
├── notebooks/                # EDA
├── tests/                    # Unit tests
├── requirements.txt          # Torch, ONNX, TF, OpenCV
├── pyproject.toml            # Hatch build
└── README.md
```

## Dependencies
See `requirements.txt`: PyTorch 2.4+, TensorFlow 2.17+, ONNX 1.16+, OpenCV.

## Deployment
- Export: `python -m src.evaluate_and_convert`
- TFLite ready for AndroidApp integration.
- See root README.md & AndroidApp/README.md

*Enhanced by BLACKBOXAI*
