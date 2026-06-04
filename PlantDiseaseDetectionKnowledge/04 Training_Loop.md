# Training Loop

The main training loop is implemented in:

```text
SourceCode/src/train.py
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

## Placeholders

```text
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Best epoch: {{BEST_EPOCH}}
Final train loss: {{FINAL_TRAIN_LOSS}}
Final validation loss: {{FINAL_VAL_LOSS}}
Training duration: {{TRAINING_DURATION}}
```
