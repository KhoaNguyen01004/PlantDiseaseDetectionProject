# Data Split And Sampling

Data loading is handled by `SourceCode/src/train.py`.

---

## Current Behavior

- Loads class folders with `torchvision.datasets.ImageFolder`.
- Adds optional unknown/background images from `data.unknown_data_dir`.
- Uses stratified train/validation/test splitting when the dataset is large enough.
- Falls back to seeded random splitting for tiny datasets.
- Uses configured `data.seed`.

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
```
