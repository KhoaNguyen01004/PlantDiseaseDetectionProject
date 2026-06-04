# Changelog

This changelog summarizes maintained project changes. Runtime results are not recorded here unless they were measured in the corresponding run.

---

## 2026-06-04 - Training Pipeline Multiprocessing Fixes

### Summary

Fixed Windows multiprocessing pickling errors, eliminated repeated logging from worker processes, and updated deprecated PyTorch AMP APIs in the training pipeline.

### Changed

- **Multiprocessing Pickling Fix**: Moved `ImagePathDataset` class and `seed_worker` function from local scope to module level in `src/train.py` to enable proper serialization on Windows spawn multiprocessing.
  - `ImagePathDataset` now defined at module level after `UnknownDataset` class.
  - `seed_worker` now uses `torch.initial_seed()` for worker-specific seed derivation instead of accessing local variables.
  - Resolves: `AttributeError: Can't pickle local object 'build_dataloaders.<locals>.ImagePathDataset'` and related pickling errors.

- **Logging Spam Fix**: Added main process guard to device initialization logging in `src/train.py`.
  - Device info (`"Using device:"` and GPU name) now logged only once in main process.
  - Prevents repeated log spam from DataLoader worker process initialization on Windows.
  - Uses `if __name__ != "__mp_main__":` guard to detect worker processes.

- **PyTorch AMP API Modernization**: Replaced deprecated `torch.cuda.amp` API with new `torch.amp` API in `src/train.py`.
  - Replaced `torch.cuda.amp.GradScaler()` with `torch.amp.GradScaler()` (compatible with PyTorch 1.10+)
  - Replaced `torch.cuda.amp.autocast()` with `torch.amp.autocast(device_type='cuda', ...)` (compatible with PyTorch 1.10+)
  - Eliminates FutureWarning messages during training.
  - Maintains backward compatibility with PyTorch 1.10+ while preparing for PyTorch 2.0+.

### Runtime Results

```text
Training now runs on Windows with num_workers > 0 without pickling errors.
Device logging no longer floods output with repeated messages.
FutureWarning messages eliminated from training output.
```

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
