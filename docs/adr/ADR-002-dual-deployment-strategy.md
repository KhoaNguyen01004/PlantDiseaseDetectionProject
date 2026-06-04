# ADR-002: Separate Source Checkpoints From Deployment Artifacts

Status: Accepted

---

## Context

The project can produce several model formats:

- PyTorch checkpoint: `SourceCode/models/best_model.pth`
- TorchScript model: `plant_model.pt`
- ONNX export: `plant_model.onnx` and optional external data
- TFLite exports: `plant_model_tflite_*`

The Android app currently uses PyTorch Mobile with TorchScript:

```text
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

---

## Decision

Treat `SourceCode/models/best_model.pth` as the primary source checkpoint.

Treat ONNX, TFLite, and top-level `SourceCode/plant_model.pt` files as generated artifacts unless they are explicitly promoted for release.

---

## Consequences

Keep:

```text
SourceCode/models/best_model.pth
SourceCode/labels.txt
SourceCode/labels.json
SourceCode/metadata.json
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Regeneratable:

```text
SourceCode/plant_model.onnx
SourceCode/plant_model.onnx.data
SourceCode/plant_model.pt
SourceCode/plant_model_tflite_float32/
SourceCode/plant_model_tflite_int8/
```

Runtime placeholders:

```text
Model hash: {{MODEL_HASH}}
Labels hash: {{LABELS_HASH}}
Android latency: {{ANDROID_AVG_LATENCY}}
```
