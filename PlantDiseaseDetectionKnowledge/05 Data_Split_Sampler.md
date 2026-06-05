# Data Split And Sampling

Data loading is handled by `SourceCode/src/train.py`.

Real-world domain-adaptation splitting is handled by `SourceCode/finetune_new_plant_dataset.py`.

---

## Current Behavior

- Loads class folders with `torchvision.datasets.ImageFolder`.
- Adds optional unknown/background images from `data.unknown_data_dir`.
- Uses stratified train/validation/test splitting when the dataset is large enough.
- Falls back to seeded random splitting for tiny datasets.
- Uses configured `data.seed`.

For the in-the-wild dataset:

- Raw images are read from `SourceCode/data/NewPLantDataset/color/`.
- A deterministic `train/val/test` split is written under `SourceCode/data/NewPLantDataset_preprocessed/split_seed*_val*_test*/`.
- `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv` record source path, preprocessed path, label, split, and source SHA-256.
- Duplicate source hashes are grouped before splitting.
- A leakage check fails if the same source SHA-256 appears in more than one split.
- Fine-tuning trains on the new `train` split plus historical PlantVillage/Unknown replay.
- Model selection uses only the new `val` split.
- Final reporting uses only the locked new `test` split.

---

## Config

```yaml
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  seed: 42
  unknown_data_dir: "data/unknown"
  unknown_limit: 0
```

---

## Placeholders

```text
Train sample count: {{TRAIN_SAMPLE_COUNT}}
Validation sample count: {{VAL_SAMPLE_COUNT}}
Test sample count: {{TEST_SAMPLE_COUNT}}
Unknown sample count: {{UNKNOWN_SAMPLE_COUNT}}
Class imbalance summary: {{CLASS_IMBALANCE_SUMMARY}}
New train manifest rows: {{NEW_TRAIN_MANIFEST_ROWS}}
New val manifest rows: {{NEW_VAL_MANIFEST_ROWS}}
New test manifest rows: {{NEW_TEST_MANIFEST_ROWS}}
Manifest leakage result: {{MANIFEST_LEAKAGE_RESULT}}
```
