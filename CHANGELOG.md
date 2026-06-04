# Changelog

This changelog summarizes maintained project changes. Runtime results are not recorded here unless they were measured in the corresponding run.

---

## 2026-06-04 - Documentation Alignment

### Summary

Rewrote Markdown documentation to match the current repository structure, training pipeline, export artifacts, and Android runtime behavior.

### Changed

- Standardized documentation around PyTorch training and Android PyTorch Mobile inference.
- Clarified that ONNX/TFLite files are generated artifacts and not the current Android runtime path.
- Added placeholder policy for runtime values such as accuracy, latency, memory, and model size.
- Updated root, SourceCode, Android, validation, report, ADR, and knowledge docs.

### Runtime Results

```text
Not measured in this documentation pass.
```

---

## 2026-06-04 - Training Pipeline Audit

### Summary

Audited and improved the Python training/export pipeline.

### Changed

- Made dataloader workers, normalization, blur probability, unknown data path, and deterministic mode config-driven.
- Added stratified splitting when possible with seeded fallback.
- Added richer checkpoint metadata.
- Updated evaluation/export to use checkpoint/config architecture.
- Fixed standalone preprocessing config path.
- Added `SourceCode/TRAINING_PIPELINE.md`.

---

## 2026-06-04 - Android UI And Localization

### Summary

Improved Android scanner usability and localization.

### Changed

- Added settings for language, scan mode, and text size.
- Added live and capture scan modes.
- Restored capture-first analysis as an optional workflow.
- Expanded guide content.
- Improved Vietnamese UI copy.
- Hid Android system bars in scanner/guide screens.

---

## 2026-06-04 - Repository Cleanup

### Summary

Removed generated, duplicate, and personal workspace files from the repository where safe.

### Changed

- Added ignore rules for generated files.
- Removed duplicate Android generated tree under `agrilens/app/bin/`.
- Removed local/editor files and logs that are not required for maintenance.

---

## Earlier Work

Earlier project work included:

- EfficientNet-based training pipeline.
- Config and metadata centralization.
- Quality validation utilities.
- Grad-CAM utility.
- ONNX/TFLite export scripts.
- Android PyTorch Mobile integration.

Detailed historical claims should be checked against source code and measured artifacts before being reused.
