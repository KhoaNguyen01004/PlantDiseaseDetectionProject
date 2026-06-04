# Inference And TFLite

TFLite is supported by the Python export/inference path, but it is not the current Android runtime.

---

## Current Use

Python-side TFLite inference:

```bash
cd SourceCode
python -m src.inference --model {{TFLITE_MODEL_PATH}} --image {{IMAGE_PATH}} --labels labels.json
```

Android inference:

```text
PyTorch Mobile + plant_model.pt
```

---

## TFLite Artifacts

Generated artifacts may include:

```text
SourceCode/plant_model_tflite_float32/
SourceCode/plant_model_tflite_int8/
```

These can be regenerated if `SourceCode/models/best_model.pth` is kept.

---

## Placeholders

```text
TFLite float32 accuracy: {{TFLITE_FLOAT32_ACCURACY}}
TFLite int8 accuracy: {{TFLITE_INT8_ACCURACY}}
TFLite latency: {{TFLITE_LATENCY}}
```
