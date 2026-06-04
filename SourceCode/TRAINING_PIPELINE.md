# Training Pipeline

This is the authoritative reference for the current Python training, evaluation, metadata, and export pipeline.

Runtime values must remain placeholders until measured from the current artifacts.

---

## 1. Overview

```text
Dataset folders
  -> src.train
  -> models/best_model.pth
  -> metadata.json / labels.json / labels.txt
  -> src.evaluate_and_convert
  -> ONNX / TFLite generated artifacts

models/best_model.pth
  -> TorchScript export
  -> agrilens/app/src/main/assets/plant_model.pt
```

The Android app currently uses PyTorch Mobile and TorchScript. TFLite artifacts are generated for Python/deployment experimentation and are not loaded by the current Android app.

---

## 2. Configuration

Config file:

```text
configs/config.yaml
```

Current config facts:

```text
Default architecture: efficientnet_b2
Supported architectures: efficientnet_b0, efficientnet_b2, efficientnet_v2_s
Input size: 260
Normalization: ImageNet mean/std from config
Unknown data path: data/unknown
```

Training uses the discovered dataset class count. If `model.num_classes` does not match the discovered class count, training logs a warning and uses the discovered count.

---

## 3. Data Input

Expected class-folder layout:

```text
data/
  plantvillage/
    plantvillage dataset/
      color/
        Class_A/
        Class_B/
```

Optional unknown images:

```text
data/unknown/
```

Unknown images are loaded recursively, shuffled with the configured seed, and added as the `Unknown` class when the folder exists.

---

## 4. Splitting

Training uses:

```text
data.train_split
data.val_split
data.test_split
data.seed
```

Behavior:

- Stratified split when every class has enough samples for the requested splits.
- Seeded random fallback for tiny datasets.
- Clear error when no samples are found.

Record after training:

```text
Train sample count: {{TRAIN_SAMPLE_COUNT}}
Validation sample count: {{VAL_SAMPLE_COUNT}}
Test sample count: {{TEST_SAMPLE_COUNT}}
Unknown sample count: {{UNKNOWN_SAMPLE_COUNT}}
```

---

## 5. Augmentation

Active train-time transforms in `src/train.py`:

- Random resized crop.
- Horizontal and vertical flips.
- Rotation.
- Perspective.
- Affine transform.
- Color jitter.
- Optional grayscale.
- Optional Gaussian blur controlled by config.
- Random erasing controlled by config.
- Config-driven normalization.

Validation/test transforms:

- Resize.
- Tensor conversion.
- Config-driven normalization.

---

## 6. Training

Run:

```bash
cd SourceCode
python -m src.train
```

Optional overrides:

```bash
python -m src.train --epochs {{EPOCHS}}
python -m src.train --batch-size {{BATCH_SIZE}}
python -m src.train --lr {{LEARNING_RATE}}
python -m src.train --unknown-limit {{UNKNOWN_LIMIT}}
```

Training uses:

- Configured EfficientNet architecture.
- ImageNet pretrained weights when enabled.
- Full fine-tuning.
- AdamW.
- CrossEntropyLoss with label smoothing.
- Configured scheduler.
- Gradient clipping.
- Mixed precision when CUDA is available.

Output:

```text
models/best_model.pth
```

Checkpoint stores:

- weights
- best validation accuracy
- best epoch
- architecture
- class names
- class count
- config snapshot
- history
- metrics

---

## 7. Metadata And Labels

Generated:

```text
metadata.json
labels.json
labels.txt
```

These files must stay paired with the checkpoint and deployed model.

---

## 8. Evaluation And Export

Run:

```bash
cd SourceCode
python -m src.evaluate_and_convert
```

Generated artifacts may include:

```text
plant_model.onnx
plant_model.onnx.data
plant_model_tflite_float32/
plant_model_tflite_int8/
```

These are generated artifacts. They can be removed when not needed if `models/best_model.pth` is kept.

Record after running:

```text
Test accuracy: {{TEST_ACCURACY}}
Macro F1: {{MACRO_F1}}
Weighted F1: {{WEIGHTED_F1}}
ONNX export result: {{ONNX_EXPORT_RESULT}}
TFLite float32 export result: {{TFLITE_FLOAT32_EXPORT_RESULT}}
TFLite int8 export result: {{TFLITE_INT8_EXPORT_RESULT}}
```

---

## 9. Android Deployment

Android runtime assets:

```text
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Rules:

- Copy model and labels from the same export run.
- Verify label order.
- Verify preprocessing size and normalization.
- Build the app after asset changes.

Build:

```bash
cd agrilens
./gradlew.bat :app:assembleDebug
```

Record:

```text
Android build result: {{ANDROID_BUILD_RESULT}}
Android average latency: {{ANDROID_AVG_LATENCY}}
Android memory usage: {{ANDROID_MEMORY_USAGE}}
```

---

## 10. Cleanup Policy

Keep:

```text
models/best_model.pth
labels.txt
labels.json
metadata.json
../agrilens/app/src/main/assets/plant_model.pt
../agrilens/app/src/main/assets/labels.txt
```

Safe to remove when not needed:

```text
plant_model.onnx
plant_model.onnx.data
plant_model.pt
plant_model_tflite_float32/
plant_model_tflite_int8/
```

Do not remove `models/best_model.pth` unless retraining is acceptable.

---

## 11. Placeholders To Fill Later

```text
Dataset version/hash: {{DATASET_VERSION_OR_HASH}}
Class count: {{CLASS_COUNT}}
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Best epoch: {{BEST_EPOCH}}
Test accuracy: {{TEST_ACCURACY}}
Macro F1: {{MACRO_F1}}
Weighted F1: {{WEIGHTED_F1}}
Android device: {{ANDROID_TEST_DEVICE}}
Android latency: {{ANDROID_AVG_LATENCY}}
Android memory: {{ANDROID_MEMORY_USAGE}}
```
