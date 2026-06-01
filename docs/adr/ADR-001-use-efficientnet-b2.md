# ADR-001 — Use EfficientNet-B2 as the default architecture

## Status
Accepted

## Context
The repository supports multiple EfficientNet variants, but the current default training and deployment pipeline is based on EfficientNet-B2. A prior inconsistency between documentation and code created ambiguity about whether the project used EfficientNet-B0 or EfficientNet-B2.

## Decision
Adopt EfficientNet-B2 as the primary architecture for training, evaluation, and deployment.

## Alternatives considered
- Use EfficientNet-B0 as the primary architecture.
- Continue supporting separate architectures for Python/TorchScript export paths.
- Use EfficientNet-V2-S or another mobile-capable backbone.

## Tradeoffs
- EfficientNet-B2 provides a stronger capacity/accuracy balance than B0, which is desirable for a fine-grained plant disease classification task.
- B2 requires a larger input size (260×260) and slightly more compute than B0.
- Maintaining a single primary architecture reduces documentation drift and simplifies validation.

## Consequences
- The default config uses `efficientnet_b2` and `image.size: 260`.
- Training, evaluation, export, and Android inference are aligned around B2.
- Legacy references to EfficientNet-B0 are considered outdated and should be removed from documentation.
- The project remains architecturally flexible: `train.py` still supports `efficientnet_b0`, `efficientnet_b2`, and `efficientnet_v2_s` as configurable alternatives.
