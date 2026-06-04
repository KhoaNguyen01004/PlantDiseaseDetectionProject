# Runtime Delegates

This note separates configured delegate options from the current Android runtime.

---

## Android

The current app uses PyTorch Mobile:

```text
org.pytorch:pytorch_android
org.pytorch:pytorch_android_torchvision
```

TFLite delegate settings in `configs/config.yaml` apply to Python/TFLite experimentation, not the current Android app.

---

## TFLite

Config includes:

```yaml
inference:
  use_gpu_delegate: true
  use_xnnpack_delegate: true
```

Actual delegate behavior must be measured when using TFLite.

Placeholders:

```text
XNNPACK result: {{XNNPACK_RESULT}}
GPU delegate result: {{GPU_DELEGATE_RESULT}}
PyTorch Mobile latency: {{PYTORCH_MOBILE_LATENCY}}
```
