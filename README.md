# Smart Crop Disease Diagnosis Project

Bridging the 2050 Food Security Gap with ML-powered plant disease detection.

## Overview
This project builds a plant-leaf disease classifier and a mobile UI prototype to demonstrate real-world diagnosis.

- **Backend (`SourceCode/`)**
  - **Model**: EfficientNet-B0 (PyTorch)
  - **Accuracy**: ~99.67% test accuracy on PlantVillage-like data
  - **Pipeline**: preprocessing + augmentation → training → evaluation → export to **ONNX/TFLite** for deployment

- **Frontend (`AndroidApp/`)**
  - **App**: *AgriLens* Android app (Kotlin + Gradle)
  - **Status**: currently a lightweight skeleton; designed for future integration with the exported **TFLite** model

- **Documentation / proposals**
  - See `Project Proposal.*` and `PlantDiseaseDetectionKnowledge/` for deeper technical notes.

## Technology Used
- **PyTorch**: model training (EfficientNet-B0)
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
python -m src.download_plantvillage  # Data
python -m src.train  # Train
python -m src.evaluate_and_convert  # TFLite export (and related exports/metrics)
```

### Frontend
```bash
cd AndroidApp
./gradlew build
# Run with Android Studio on an emulator/device
```

## Project Structure
```
.
├── README.md              # This
├── TODO.md               # Tasks
├── SourceCode/           # ML Backend
│   ├── src/ (train.py, inference.py, models/, preprocessing/)
│   ├── data/, models/     # Generated/ignored (by convention)
│   ├── requirements.txt
│   └── README.md         # Backend details
├── AndroidApp/           # Mobile App
│   ├── app/src/main/ (MainActivity.kt, res/)
│   └── README.md         # App details
└── Proposals/ docs
```

## Roadmap
1. Train/export model ✅
2. Integrate TFLite in AgriLens
3. Camera → Real-time detection
4. Deploy (Play Store?)

See sub-READMEs for details.

