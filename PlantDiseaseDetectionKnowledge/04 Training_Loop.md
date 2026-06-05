# Training Loop

The main training loop is implemented in:

```text
SourceCode/src/train.py
```

The real-world fine-tuning loop is implemented in:

```text
SourceCode/finetune_new_plant_dataset.py
```

---

## Current Behavior

- Builds dataloaders from configured dataset paths.
- Uses configurable EfficientNet architecture.
- Uses AdamW.
- Uses CrossEntropyLoss with label smoothing.
- Uses mixed precision when CUDA is available.
- Clips gradients.
- Tracks validation accuracy.
- Saves the best checkpoint to `models/best_model.pth`.

The checkpoint stores weights, architecture, class names, config, metrics, and training history.

---

## In-The-Wild Fine-Tuning Behavior

- Preserves the 39-class `labels.json` mapping.
- Uses audited new-domain `train/val/test` manifests.
- Trains on new-domain `train` samples plus historical PlantVillage/Unknown replay.
- Applies stronger outdoor augmentations only to new-domain training samples.
- Uses class-balanced cross entropy from the compiled hybrid training pool.
- Stage A freezes EfficientNet-B2 features and trains only the classifier head.
- Stage B unfreezes all layers for conservative full fine-tuning.
- Saves `models/best_model_finetuned.pth` by new-domain validation macro-F1.

---

## Placeholders

```text
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Best epoch: {{BEST_EPOCH}}
Final train loss: {{FINAL_TRAIN_LOSS}}
Final validation loss: {{FINAL_VAL_LOSS}}
Best new validation macro F1: {{BEST_NEW_VAL_MACRO_F1}}
Training duration: {{TRAINING_DURATION}}
```
