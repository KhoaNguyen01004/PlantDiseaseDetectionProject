# AgriLens Android App

AgriLens is the Android application for on-device plant leaf disease detection.

The app currently uses PyTorch Mobile with a TorchScript model:

```text
app/src/main/assets/plant_model.pt
app/src/main/assets/labels.txt
```

It does not load the TFLite artifacts generated under `SourceCode/plant_model_tflite_*`.

---

## Current Features

- Live camera detection with CameraX.
- Capture-first mode.
- Gallery image analysis.
- Pause-on-known-result behavior.
- Continued scanning when the model returns `Unknown`.
- Disease care guidance for known labels.
- English and Vietnamese language setting.
- Adjustable text size.
- Fullscreen/immersive scanner and guide screens.
- Native offline guide screen.

---

## Build

Run from `agrilens/`:

```bash
./gradlew.bat :app:assembleDebug
```

Runtime benchmark placeholders:

```text
Debug build result: {{DEBUG_BUILD_RESULT}}
Release build result: {{RELEASE_BUILD_RESULT}}
Device used: {{ANDROID_TEST_DEVICE}}
Average inference latency: {{ANDROID_AVG_LATENCY}}
Memory usage: {{ANDROID_MEMORY_USAGE}}
Camera FPS during live mode: {{CAMERA_FPS}}
```

---

## Assets

Required:

```text
app/src/main/assets/plant_model.pt
app/src/main/assets/labels.txt
```

Rules:

- `plant_model.pt` and `labels.txt` must come from the same training/export run.
- Label order must match the model output order.
- Android preprocessing must match training image size and normalization.

Current Android preprocessing facts:

```text
Input size: 260 x 260
Mean: [0.485, 0.456, 0.406]
Std: [0.229, 0.224, 0.225]
Runtime: PyTorch Mobile
```

---

## Scan Modes

Live:

- The camera is analyzed continuously.
- Unknown predictions keep scanning.
- Known predictions pause so the user can read the result and guidance.

Capture:

- The user captures one image.
- The app analyzes only the captured frame.
- This is useful when the user needs a stable frame.

Gallery:

- The user selects a saved image.
- The app analyzes that image and displays care guidance when available.

---

## Localization

Supported UI languages:

```text
English
Vietnamese
```

Translation policy:

- Use natural Vietnamese wording.
- Keep technical disease or chemical names in English when translation would reduce clarity.
- Do not invent local treatment recommendations without verification.

---

## Known Runtime Placeholders

Fill these after device testing:

```text
Test device: {{ANDROID_TEST_DEVICE}}
Android version: {{ANDROID_VERSION}}
Model asset hash: {{MODEL_ASSET_HASH}}
Labels asset hash: {{LABELS_ASSET_HASH}}
Average live inference latency: {{LIVE_INFERENCE_LATENCY}}
Average capture inference latency: {{CAPTURE_INFERENCE_LATENCY}}
Battery impact: {{BATTERY_IMPACT}}
```
