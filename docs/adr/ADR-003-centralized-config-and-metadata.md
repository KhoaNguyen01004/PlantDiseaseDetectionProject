# ADR-003: Centralize Configuration, Labels, And Metadata

Status: Accepted

---

## Context

Training, evaluation, export, and Android inference must agree on image size, normalization, class order, and model architecture.

---

## Decision

Use `SourceCode/configs/config.yaml` as the central configuration file.

Export metadata and labels through `SourceCode/src/metadata.py`:

```text
metadata.json
labels.json
labels.txt
```

Store checkpoint metadata in `models/best_model.pth`, including:

- architecture
- class names
- class count
- config snapshot
- training history
- best metrics

---

## Consequences

Benefits:

- Easier reproducibility.
- Safer export and Android deployment.
- Clearer warnings when config and dataset disagree.

Requirements:

- Keep model and labels paired.
- Do not update Android `plant_model.pt` without updating `labels.txt`.
- Use placeholders for runtime metrics until measured.

Placeholders:

```text
Dataset hash: {{DATASET_HASH}}
Best epoch: {{BEST_EPOCH}}
Best validation accuracy: {{BEST_VAL_ACCURACY}}
```
