# Smart Crop Disease Diagnosis Project

Bridging the 2050 Food Security Gap with ML-powered plant disease detection.

## Overview
This project builds a plant-leaf disease classifier and a mobile UI prototype to demonstrate real-world diagnosis.

- **Backend (`SourceCode/`)**
  - **Model**: EfficientNet-B2 (PyTorch)
  - **Accuracy**: ~99.67% test accuracy on PlantVillage-like data
  - **Pipeline**: preprocessing + augmentation → training → evaluation → export to **ONNX/TFLite** for deployment

- **Frontend (`agrilens/`)**
  - **App**: *AgriLens* Android app (Kotlin + Gradle)
  - **Status**: currently a lightweight skeleton; designed for future integration with the exported **TFLite** model

- **Documentation / proposals**
  - See `Project Proposal.*` and `PlantDiseaseDetectionKnowledge/` for deeper technical notes.

## Technology Used
- **PyTorch**: model training (EfficientNet-B2)
- **ONNX / TFLite**: model export for cross-platform deployment
- **TensorFlow Lite Interpreter**: intended runtime path for mobile inference
- **Android (Kotlin)**: mobile frontend implementation

## Quick Setup
### Backend
> Prerequisite: a Python environment (venv) and installed dependencies.

```bash
cd SourceCode
py3_10/Scripts/activate  # Windows venv
pip install -r requirements.txt
python -m src.download_plantvillage  # Data (if needed)
python -m src.train  # Train (EfficientNet-B2, 260x260 images)
python -m src.evaluate_and_convert  # Evaluate + Export to ONNX/TFLite
```

### Frontend
```bash
cd agrilens
./gradlew build
# Run with Android Studio on an emulator/device
```

## Project Structure
```
.
├── README.md              # This file
├── CHANGELOG.md           # Project history and changes
├── Instruction.md         # Refactor roadmap and requirements
├── PROJECT_SUMMARY.md     # Historical reference document
├── VALIDATION_GUIDE.md    # Testing and validation procedures
├── SourceCode/            # ML Backend
│   ├── src/               # Main Python package
│   │   ├── train.py       # Training pipeline (EfficientNet-B2)
│   │   ├── inference.py   # TFLite inference CLI
│   │   ├── evaluate_and_convert.py  # Evaluation + ONNX/TFLite export
│   │   ├── gradcam.py     # Grad-CAM explainability
│   │   ├── quality_validator.py  # Photo quality validation
│   │   ├── metadata.py    # Model metadata management
│   │   └── preprocessing/ # Data preprocessing & augmentation
│   ├── configs/config.yaml  # Centralized configuration
│   ├── data/              # Dataset (ignored by git)
│   ├── models/            # Trained checkpoints (ignored by git)
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # Backend details
├── agrilens/              # Android App
│   ├── app/src/main/      # MainActivity, resources
│   └── README.md          # App setup details
└── PlantDiseaseDetectionKnowledge/  # Technical documentation
```

## Roadmap
1. ✅ Train/export model (EfficientNet-B2, ONNX→TFLite pipeline)
2. 🔄 Integrate TFLite in AgriLens (pending validation)
3. 📸 Add photo quality validation (implemented, pending integration)
4. 🎯 Add Grad-CAM explainability (implemented in Python, pending Android)
5. 📱 Camera → Real-time detection
6. 🚀 Deploy (Play Store?)

See sub-READMEs for details.

