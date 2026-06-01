# ADR-002 — Dual deployment strategy (Python TFLite + Android TorchScript)

## Status
Accepted

## Context
The repository targets both Python CLI inference and Android on-device deployment. The Python backend uses TensorFlow Lite, while the Android application is implemented with PyTorch Mobile.

## Decision
Maintain a dual deployment strategy:
- Python CLI inference via TFLite (float32 and INT8 quantized models)
- Android inference via TorchScript `.pt` using PyTorch Mobile

## Alternatives considered
- Force a single unified deployment format across Python and Android.
- Use only TorchScript for both desktop/Python and Android inference.
- Use only TFLite for both Python and Android inference.

## Tradeoffs
- Dual-format support increases maintenance effort and requires separate validation flows.
- It avoids pushing Android into an unsupported or brittle conversion chain.
- TorchScript on Android is more stable for the existing PyTorch model export path.
- TFLite remains a useful Python-native inference path with delegate support.

## Consequences
- The repo now treats Python CLI and Android as separate but related deployment targets.
- `SourceCode/src/evaluate_and_convert.py` is the recommended TFLite export pipeline.
- `export_torchscript.py` remains the Android TorchScript export path.
- The Android app expects `plant_model.pt` and `labels.txt` in assets.
- Validation artifacts and documentation must explicitly distinguish the two targets.
