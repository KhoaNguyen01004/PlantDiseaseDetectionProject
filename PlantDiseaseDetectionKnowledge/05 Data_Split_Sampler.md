# Entity: Data Split & Class Imbalance Sampler

**Source:** `SourceCode/src/preprocessing/preprocess.py`

## Purpose
Create train/val/test splits and mitigate class imbalance.

## Split behavior
- stratified train/val/test splitting using labels derived from directory names

## Sampler behavior
- compute per-class counts on the training set
- compute inverse-frequency weights per sample
- use `WeightedRandomSampler` with replacement

## Assessment
Overall appropriate for class imbalance.

## Links
- [[04 Training_Loop]]
- [[06 Augmentation_and_Masking]]

