# Entity: Inference (TFLite)

**Source:** `SourceCode/src/inference.py`

## Purpose
CLI inference for the TFLite plant disease model, supporting:
- single image (`--image`)
- webcam loop (`--webcam`)

## Responsibilities
- load the TFLite model via `tf.lite.Interpreter(...)`
- optionally attach delegates
- preprocess the image/frame
- run `interpreter.invoke()`
- decode the output and return/display prediction

## Critical issues
- Preprocessing likely mismatches the model input contract (especially dtype/quantization)
- “confidence” uses `max(output)` without guaranteeing the output is probabilities
- Delegate loading is not robust/portable (hardcoded `.so` names)

## Links
- [[02 Inference_Preprocess]]
- [[03 Delegates]]

