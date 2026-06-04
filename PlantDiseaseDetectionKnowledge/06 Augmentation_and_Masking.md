# Augmentation And Masking

The active training augmentations are defined in `SourceCode/src/train.py`.

---

## Current Training Augmentation

- Random resized crop.
- Random horizontal and vertical flips.
- Random rotation.
- Random perspective.
- Random affine transform.
- Color jitter.
- Optional grayscale.
- Optional Gaussian blur controlled by `augmentation.blur_prob`.
- Random erasing controlled by `augmentation.random_erasing_prob`.

---

## Masking

`configs/config.yaml` contains `data.mask_prob`, and `src/preprocessing/preprocess.py` includes masking utilities for preprocessing experiments.

The main `src.train` dataloader currently trains from image folders and does not apply segmentation masks.

---

## Placeholders

```text
Augmentation ablation result: {{AUGMENTATION_ABLATION_RESULT}}
Masking ablation result: {{MASKING_ABLATION_RESULT}}
```
