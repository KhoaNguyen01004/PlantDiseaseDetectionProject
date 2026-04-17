# Smart Crop Disease Diagnosis Project

Bridging 2050 Food Security Gap with ML-powered plant disease detection.

## Overview
- **Backend** (`SourceCode/`): PyTorch EfficientNet-B0 classifier (~99.67% acc on PlantVillage-like data). Train, export TFLite/ONNX.
- **Frontend** (`AndroidApp/`): AgriLens Kotlin app (basic; future TFLite integration).
- **Proposals**: See `Project Proposal.*` docs.

## Quick Setup
### Backend
```bash
cd SourceCode
py3_10/Scripts/activate  # Windows venv
pip install -r requirements.txt
python -m src.download_plantvillage  # Data
python -m src.train  # Train
python -m src.evaluate_and_convert  # TFLite
```

### Frontend
```bash
cd AndroidApp
./gradlew build
# Android Studio run on device
```

## Project Structure
```
.
├── README.md              # This
├── TODO.md               # Tasks
├── SourceCode/           # ML Backend
│   ├── src/ (train.py, inference.py, models/, preprocessing/)
│   ├── data/, models/     # Generated/ignored
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
3. Camera -> Real-time detection
4. Deploy (Play Store?)

See sub-READMEs for details. Organized by BLACKBOXAI.

