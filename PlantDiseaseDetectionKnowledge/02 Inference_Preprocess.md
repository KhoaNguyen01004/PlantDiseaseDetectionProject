# Inference Preprocessing

Training, Python inference, export calibration, and Android inference must use compatible preprocessing.

---

## Current Configuration

From `SourceCode/configs/config.yaml` and Android helper code:

```text
Input size: 260 x 260
Mean: [0.485, 0.456, 0.406]
Std: [0.229, 0.224, 0.225]
```

These are configuration/code facts, not measured metrics.

---

## Android Path

```text
Bitmap
  -> resize to 260 x 260
  -> TensorImageUtils.bitmapToFloat32Tensor
  -> PyTorch Mobile module
```

---

## Validation

Record after checking:

```text
Android preprocessing verified: {{ANDROID_PREPROCESSING_VERIFIED}}
Python preprocessing verified: {{PYTHON_PREPROCESSING_VERIFIED}}
Export calibration preprocessing verified: {{EXPORT_PREPROCESSING_VERIFIED}}
```
