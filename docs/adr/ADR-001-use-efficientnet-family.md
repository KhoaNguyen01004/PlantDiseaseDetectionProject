# ADR-001: Use A Configurable EfficientNet Family Model

Status: Accepted

---

## Context

The project needs an image classification backbone suitable for plant leaf disease recognition and mobile deployment experiments.

The current default configuration is `efficientnet_b2`, but the training code supports:

```text
efficientnet_b0
efficientnet_b2
efficientnet_v2_s
```

The selected architecture is configured in:

```text
SourceCode/configs/config.yaml
```

---

## Decision

Use the EfficientNet family as the supported model family, with `efficientnet_b2` as the current default.

The model head is created from the discovered dataset class count, not blindly from the configured `model.num_classes`.

---

## Consequences

Benefits:

- Model architecture can be changed through config.
- Training, evaluation, and export can share the same model builder.
- Checkpoints now store architecture and class order for safer reuse.

Risks:

- Android assets must be regenerated when architecture or labels change.
- Runtime metrics must be measured after each model change.

Runtime placeholders:

```text
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Test accuracy: {{TEST_ACCURACY}}
Android latency: {{ANDROID_AVG_LATENCY}}
```
