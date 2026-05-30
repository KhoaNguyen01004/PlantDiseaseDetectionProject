# Entity: Augmentation & Segmentation Masking

**Source:** `SourceCode/src/preprocessing/preprocess.py`

## Purpose
Improve generalization by augmenting data and (optionally) masking background using segmented masks.

## Masking logic
- during preprocessing, if `seg_path` exists and random condition triggers:
  - read segmentation mask (grayscale)
  - threshold to binary mask
  - apply mask to RGB image

## Augmentation pipeline
Albumentations includes:
- rotations and scaling
- brightness/contrast
- Gaussian noise
- resize + ImageNet normalization
- convert to tensor

## Refactor note
- resizing occurs both in preprocessing and again inside the augmentation pipeline
- not always a correctness bug, but inefficient and can introduce slight differences

## Links
- [[04 Training_Loop]]
- [[02 Inference_Preprocess]]

