# ADR-003 — Centralized configuration and metadata

## Status
Accepted

## Context
The repo contains configuration values, normalization constants, and metadata in multiple code locations. A single source of truth is required to keep training, export, and inference aligned.

## Decision
Use `SourceCode/configs/config.yaml` as the centralized configuration source, and use `SourceCode/src/metadata.py` to export unified model metadata.

## Alternatives considered
- Embed constants directly in each script.
- Use multiple per-script YAML or JSON files.
- Use environment variables for configuration.

## Tradeoffs
- A centralized config file improves consistency but requires all scripts to parse the same file format and path.
- It adds a single point of dependency for the repository setup.
- Unified metadata export simplifies deployment across Python and Android, but requires additional metadata-validation checks.

## Consequences
- Scripts such as `train.py`, `evaluate_and_convert.py`, `inference.py`, and `metadata.py` read from `SourceCode/configs/config.yaml`.
- The metadata export includes model architecture, image size, dataset hashing, label mapping, and export metadata.
- Android and Python deployment can share the same label export artifacts (`labels.json`, `labels.txt`).
- Legacy references to `SourceCode/config.yaml` are obsolete and must be removed.
