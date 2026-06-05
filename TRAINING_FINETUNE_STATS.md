# Training And Fine-Tuning Stats

Last updated: 2026-06-05

This file summarizes the measured training, fine-tuning, evaluation, and deployment stats available in this workspace. Metrics from the old PlantVillage benchmark and the new in-the-wild dataset are not directly comparable because they use different data distributions.

---

## Current Deployment Choice

Use the 256 historical replay fine-tuned model for the Android app deployment.

```text
Checkpoint: SourceCode/models/best_model_finetuned256.pth
TorchScript asset: agrilens/app/src/main/assets/plant_model.pt
Intermediate TorchScript export: SourceCode/plant_model.pt
TFLite export folder: SourceCode/plant_model_tflite_float32/
```

Reason: the 256 replay model has slightly lower top-1 accuracy than the 512 replay run, but better macro F1 and top-3 accuracy. That is usually preferable for the app because it is less dominated by the largest classes and gives more useful top-3 suggestions.

---

## Dataset Layout

Baseline historical dataset:

```text
SourceCode/data/plantvillage/plantvillage dataset/color/
```

New in-the-wild dataset:

```text
SourceCode/data/NewPLantDataset/color/
```

Audited preprocessed split:

```text
SourceCode/data/NewPLantDataset_preprocessed/split_seed42_val15_test20_v2/
```

Split manifest counts:

| Split | Images | Purpose |
|---|---:|---|
| Train | 577 | New-domain images used for fine-tuning with historical replay |
| Validation | 139 | New-domain model selection during fine-tuning |
| Test | 180 | Locked new-domain final evaluation |

The audited split uses manifests with source SHA-256 hashes. Duplicate source hashes are grouped before splitting, and manifest leakage validation passed.

---

## Baseline Model

Baseline checkpoint:

```text
SourceCode/models/best_model.pth
```

Known metadata:

| Item | Value |
|---|---:|
| Architecture | EfficientNet-B2 |
| Output classes | 39 |
| Includes `Unknown` class | Yes |
| Baseline checkpoint best accuracy metadata | 99.8388% |

Important note: this near-100% baseline metric comes from the old PlantVillage-style benchmark and should not be treated as real-world app accuracy.

---

## Evaluation Issue That Was Fixed

The earlier evaluator used `src.train.build_dataloaders()`, which read:

```text
SourceCode/configs/config.yaml -> data.raw_data_dir
```

That path pointed to the old PlantVillage dataset, so the test accuracy was near 100% because it was not testing on the new in-the-wild data.

The current evaluator:

```bash
python -m src.evaluate_and_convert --skip-preprocess
```

now evaluates `models/best_model_finetuned*.pth` against the locked audited new-domain test split.

---

## Fine-Tuning Pipeline

Fine-tuning script:

```text
SourceCode/finetune_new_plant_dataset.py
```

Current strategy:

- Start from `SourceCode/models/best_model.pth`.
- Keep the 39-class label layout from `SourceCode/labels.json`.
- Train on new-domain train split plus balanced historical replay from PlantVillage and `Unknown`.
- Validate only on the new-domain validation split.
- Select best checkpoint by new-domain validation macro F1.
- Use source-aware augmentation: strong outdoor/domain-shift augmentation only for new-domain training images.
- Use historical replay to reduce catastrophic forgetting.
- Use class-balanced cross-entropy loss.

Two-stage schedule:

| Stage | Epochs | Trainable layers | Learning rate |
|---|---:|---|---:|
| A: Head stabilization | 1-2 | Classifier head only | `1e-3` |
| B: Full fine-tuning | Remaining epochs | Entire EfficientNet-B2 | `5e-6` to `8e-6` |

---

## Fine-Tuning Runs

Locked new-domain test results:

| Historical replay per class | Epochs | Full LR | Test accuracy | Top-3 accuracy | Macro F1 | Weighted F1 | Notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 8 | `5e-6` | **60.00%** | 84.44% | 34.35% | **59.08%** | Best top-1 accuracy, stronger retention |
| 256 | 10 | `8e-6` | 58.33% | **86.11%** | **36.95%** | 58.32% | Current deployment choice |
| 128 | 10 | `8e-6` | 56.11% | 82.78% | 33.31% | 55.49% | Worse than 256 and 512 |

Available fine-tuned checkpoints currently on disk:

| Checkpoint | Notes |
|---|---|
| `SourceCode/models/best_model_finetuned128.pth` | 128 replay experiment |
| `SourceCode/models/best_model_finetuned256.pth` | 256 replay experiment and current deployment source |

The 512 replay checkpoint produced the highest recorded top-1 test accuracy, but it is not currently listed in `SourceCode/models/`. If it was not copied elsewhere, it was likely overwritten during later experiments.

---

## Current 256 Replay Evaluation Details

Evaluation command:

```bash
python -m src.evaluate_and_convert --skip-preprocess
```

Checkpoint evaluated:

```text
SourceCode/models/best_model_finetuned256.pth
```

Held-out test dataset:

```text
SourceCode/data/NewPLantDataset_preprocessed/split_seed42_val15_test20_v2/test
```

Summary metrics:

| Metric | Value |
|---|---:|
| Test accuracy | 58.33% |
| Top-3 accuracy | 86.11% |
| Macro F1 | 36.95% |
| Weighted F1 | 58.32% |
| Test images | 180 |
| Present test classes | 30 |

Report outputs:

```text
SourceCode/reports/new_dataset_evaluation/classification_report.txt
SourceCode/reports/new_dataset_evaluation/classification_report.json
SourceCode/reports/new_dataset_evaluation/confusion_matrix.csv
SourceCode/reports/new_dataset_evaluation/worst_recall_classes.csv
SourceCode/reports/new_dataset_evaluation/summary.json
```

---

## Export And Android Build Stats

TorchScript export script:

```text
SourceCode/export_torchscript.py
```

Current behavior:

- Prefers `SourceCode/models/best_model_finetuned.pth` if present.
- Otherwise selects the newest `SourceCode/models/best_model_finetuned*.pth`.
- In the latest export, it selected `SourceCode/models/best_model_finetuned256.pth`.

Latest exports:

| Artifact | Source checkpoint |
|---|---|
| `SourceCode/plant_model.pt` | `SourceCode/models/best_model_finetuned256.pth` |
| `agrilens/app/src/main/assets/plant_model.pt` | `SourceCode/models/best_model_finetuned256.pth` |
| `SourceCode/plant_model_tflite_float32/plant_model_float16.tflite` | Latest evaluated fine-tuned checkpoint during ONNX/TFLite conversion |

Android validation:

```text
Command: ./gradlew.bat :app:assembleDebug
Result: Passed
APK: agrilens/app/build/outputs/apk/debug/app-debug.apk
Warning: SDK XML version warning from Android command-line tooling
```

---

## Practical Interpretation

The realistic in-the-wild top-1 accuracy is currently around 58-60%, not the old near-100% PlantVillage number.

The next highest-value improvements are:

- Add more real-world images for weak and low-support classes.
- Keep saving experiment-specific checkpoints instead of overwriting.
- Show top-3 predictions in the app, since top-3 accuracy is much stronger than top-1.
- Re-evaluate only on the locked audited new-domain test split.
- Continue fine-tuning rather than training from scratch until the new real-world dataset is much larger and more balanced.
