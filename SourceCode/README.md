# SourceCode

This directory contains the Python model pipeline for training, evaluation, metadata export, and deployment artifact generation.

Runtime-dependent values are not hardcoded in this documentation. Fill placeholders only after running the current pipeline.

---

## Main Entry Points

```text
src/train.py                 Train and save the best PyTorch checkpoint
src/evaluate_and_convert.py  Evaluate checkpoint and export ONNX/TFLite artifacts
finetune_new_plant_dataset.py Fine-tune the 39-class checkpoint on audited in-the-wild splits
src/inference.py             Run TFLite inference from Python
src/metadata.py              Export metadata.json, labels.json, labels.txt
src/gradcam.py               Generate offline Grad-CAM visualizations from the fine-tuned checkpoint
src/quality_validator.py     Check image quality before inference
```

See `../GRADCAM_EXPLAINABILITY.md` for the distinction between offline Grad-CAM and Android top-3 explanations.

---

## Configuration

The central config is:

```text
configs/config.yaml
```

Current default facts from config:

```yaml
model:
  architecture: "efficientnet_b2"
  pretrained: true
image:
  size: 260
training:
  batch_size: 32
  learning_rate: 1e-4
data:
  raw_data_dir: "data/plantvillage/plantvillage dataset/color"
  unknown_data_dir: "data/unknown"
```

The discovered dataset class count is used for the model head. If `model.num_classes` differs from the discovered labels, training logs a warning and uses the dataset-derived count.

---

## Training

Run from this directory:

```bash
python -m src.train
```

Optional overrides:

```bash
python -m src.train --epochs {{EPOCHS}}
python -m src.train --batch-size {{BATCH_SIZE}}
python -m src.train --lr {{LEARNING_RATE}}
python -m src.train --unknown-limit {{UNKNOWN_LIMIT}}
```

Primary checkpoint:

```text
models/best_model.pth
```

Real-world fine-tuning:

```bash
python finetune_new_plant_dataset.py --skip-preprocess --epochs 8 --head-epochs 2 --head-lr 1e-3 --full-lr 5e-6
```

Fine-tuning creates or reuses:

```text
data/NewPLantDataset_preprocessed/split_seed*_val*_test*/
  train/
  val/
  test/
  train_manifest.csv
  val_manifest.csv
  test_manifest.csv
models/best_model_finetuned.pth
```

The manifests include source SHA-256 values and are checked for split leakage. Fine-tuning trains on the new `train` split plus historical replay and saves the best checkpoint by new-domain validation macro-F1.

The checkpoint stores:

- model weights
- best validation accuracy
- best epoch
- architecture
- class names
- class count
- config snapshot
- training history
- best metrics

---

## Evaluation And Export

Run:

```bash
python -m src.evaluate_and_convert
```

By default this evaluates `models/best_model_finetuned.pth` on the audited new-domain `test` split. Use `--skip-preprocess` to reuse an existing split.

Generated artifacts can include:

```text
plant_model.onnx
plant_model.onnx.data
plant_model_tflite_float32/
plant_model_tflite_int8/
reports/new_dataset_evaluation/
```

These are generated deployment artifacts. They can be regenerated from the selected checkpoint. For the current real-world app deployment, export from the fine-tuned checkpoint.

---

## Android Export

The Android app currently uses PyTorch Mobile/TorchScript:

```text
../agrilens/app/src/main/assets/plant_model.pt
../agrilens/app/src/main/assets/labels.txt
```

To export the current fine-tuned TorchScript model directly into the Android asset:

```bash
python export_torchscript.py --export-path ../agrilens/app/src/main/assets/plant_model.pt
```

Keep the model and matching labels together. Do not update one without the other.

---

## Metadata

Training exports:

```text
metadata.json
labels.json
labels.txt
```

These files should match the checkpoint and deployment model.

---

## Runtime Placeholders

Record measured values here after running evaluation:

```text
Test accuracy: {{TEST_ACCURACY}}
Macro F1: {{MACRO_F1}}
Weighted F1: {{WEIGHTED_F1}}
Top-3 accuracy: {{TOP3_ACCURACY}}
Evaluation report directory: {{EVALUATION_REPORT_DIR}}
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Best epoch: {{BEST_EPOCH}}
ONNX model size: {{ONNX_MODEL_SIZE}}
TFLite float32 model size: {{TFLITE_FLOAT32_MODEL_SIZE}}
TFLite int8 model size: {{TFLITE_INT8_MODEL_SIZE}}
```

---

## Full Pipeline Reference

See:

```text
TRAINING_PIPELINE.md
```
