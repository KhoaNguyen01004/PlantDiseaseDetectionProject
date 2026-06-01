# Entity: Training Loop

**Source:** `SourceCode/src/train.py` (function `train_model`)

## Purpose
Fine-tune a configurable EfficientNet-family classifier for plant disease classification, with EfficientNet-B2 as the default architecture.

## Key steps
- choose optimizer (AdamW)
- choose scheduler (CosineAnnealingLR)
- run training epochs
- validate each epoch
- save best checkpoint by validation accuracy

## Structural notes
The loop itself is logically sound.
Primary risk is **pipeline mismatch**:
- training preprocessing/normalization contract must match inference preprocessing

## Links
- [[05 Data_Split_Sampler]]
- [[06 Augmentation_and_Masking]]
- [[02 Inference_Preprocess]]

