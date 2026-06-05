# Changelog

This changelog summarizes maintained project changes. Runtime results are not recorded here unless they were measured in the corresponding run.

---

## 2026-06-05 - TorchScript Fine-Tuned Export Default

### Summary

Updated the TorchScript export helper so app-facing `.pt` exports use the fine-tuned checkpoint by default instead of the original PlantVillage baseline checkpoint.

### Changed

- Updated `SourceCode/export_torchscript.py`.
- Changed the default checkpoint path from `SourceCode/models/best_model.pth` to fine-tuned checkpoint discovery.
- Preferred `SourceCode/models/best_model_finetuned.pth` when present, with automatic fallback to the newest `SourceCode/models/best_model_finetuned*.pth` checkpoint.
- Kept the default TorchScript output at `SourceCode/plant_model.pt`.
- Added CLI arguments for `--checkpoint-path`, `--export-path`, and `--image-size`.
- Added explicit checkpoint existence validation and console logging for exported checkpoint metadata.
- Made checkpoint loading accept either a full checkpoint dictionary with `model_state_dict` or a raw state dictionary.
- Regenerated `SourceCode/plant_model.pt` from `SourceCode/models/best_model_finetuned256.pth`.
- Regenerated `agrilens/app/src/main/assets/plant_model.pt` from `SourceCode/models/best_model_finetuned256.pth` for Android deployment.
- Updated `README.md`, `SourceCode/README.md`, `SourceCode/TRAINING_PIPELINE.md`, and `VALIDATION_GUIDE.md` so deployment export documentation matches the fine-tuned checkpoint flow.

### Runtime Results

```text
CLI help passed for SourceCode/export_torchscript.py.
AST syntax validation passed without writing bytecode.
py_compile was not used because Windows denied replacing an existing __pycache__ file.
TorchScript export passed and wrote SourceCode/plant_model.pt from SourceCode/models/best_model_finetuned256.pth.
TorchScript export passed and wrote agrilens/app/src/main/assets/plant_model.pt from SourceCode/models/best_model_finetuned256.pth.
Android debug build passed with `./gradlew.bat :app:assembleDebug`; Gradle reported an SDK XML version warning.
```

---

## 2026-06-05 - New Dataset Fine-Tuning Orchestrator

### Summary

Added a production-oriented script to preprocess the new in-the-wild plant dataset and fine-tune the existing 39-class EfficientNet-B2 checkpoint while mitigating catastrophic forgetting with historical data blending.

### Changed

- Added `SourceCode/finetune_new_plant_dataset.py`.
- Preprocesses `SourceCode/data/NewPLantDataset/color/<class_name>/` into `SourceCode/data/NewPLantDataset_preprocessed/train/<class_name>/`.
- Preserves the existing `labels.json` 39-class layout, including the `Unknown` class.
- Validates that new dataset class folders match the historical PlantVillage class names and skips missing new classes gracefully.
- Builds a hybrid training set using all matched new classes plus balanced historical samples from PlantVillage for every disease/healthy class.
- Samples from `SourceCode/data/unknown/` for the `Unknown` class to preserve false-positive protection.
- Loads `SourceCode/models/best_model.pth` as an `efficientnet_b2` checkpoint with `num_classes=39`.
- Uses full fine-tuning with all parameters trainable, `torch.optim.Adam`, conservative learning rate defaults, validation metrics, and best-checkpoint output at `SourceCode/models/best_model_finetuned.pth`.
- Added `--dry-run`, `--skip-preprocess`, sampling controls, preprocessing overwrite control, and configurable path arguments.

### Runtime Results

```text
Syntax validation passed with Python AST parsing.
Argument parsing passed with --help.
Dry run passed through preprocessing, hybrid sampling, DataLoader construction, CUDA detection, and strict checkpoint loading.
Preprocessed 702 images into 9 matching class folders under SourceCode/data/NewPLantDataset_preprocessed/train/.
Confirmed baseline checkpoint metadata: architecture=efficientnet_b2, num_classes=39.
Actual fine-tuning was not started during validation.
```

---

## 2026-06-05 - New Dataset Evaluation And Export Fix

### Summary

Rewrote the evaluation/export path so the fine-tuned model is evaluated against a held-out split of the new in-the-wild dataset instead of the old PlantVillage benchmark split.

### Changed

- Rewrote `SourceCode/src/evaluate_and_convert.py`.
- Changed the default checkpoint from `models/best_model.pth` behavior to explicit evaluation of `SourceCode/models/best_model_finetuned.pth`.
- Removed the evaluator dependency on `src.train.build_dataloaders()` because that helper reads `configs/config.yaml`, whose `data.raw_data_dir` points to the old PlantVillage dataset.
- Added deterministic preprocessing of `SourceCode/data/NewPLantDataset/color/` into separate `train` and `test` splits.
- Added fixed-label evaluation using the checkpoint/`labels.json` 39-class layout while only scoring images present in the new test split.
- Added path-explicit ONNX and float32 TFLite export arguments.
- Added lazy TensorFlow import so importing `export_onnx` for tests does not require TensorFlow startup.
- Added robust handling for stale generated preprocessing output: if `NewPLantDataset_preprocessed/train` exists without a matching `test`, a clean isolated split is written under `NewPLantDataset_preprocessed/split_seed42_test20/`.
- Updated `SourceCode/finetune_new_plant_dataset.py` so future fine-tuning uses the new split `train` folder and leaves the split `test` folder for evaluation.
- Fixed `--skip-preprocess` so it automatically resolves the isolated split test folder when the direct `NewPLantDataset_preprocessed/test` folder does not exist.

### Issue Identified

```text
The previous evaluator loaded the dataset via configs/config.yaml -> data.raw_data_dir,
which points to data/plantvillage/plantvillage dataset/color.
Therefore dataloaders["test"] was an old PlantVillage split, not the new real-world dataset.
This explains the near-100% test accuracy.
```

### Runtime Results

```text
Syntax validation passed for src/evaluate_and_convert.py and finetune_new_plant_dataset.py.
CLI help passed for src.evaluate_and_convert.
Created isolated split: train=563 images, test=139 images under SourceCode/data/NewPLantDataset_preprocessed/split_seed42_test20/.
Evaluated SourceCode/models/best_model_finetuned.pth on the held-out new test split.
New Dataset Test Accuracy: 59.71%.
Fine-tune dry run confirmed training now uses split_seed42_test20/train and evaluation should use split_seed42_test20/test.
ONNX export smoke test passed with a tiny model.
Fine-tuned model ONNX export passed with --skip-tflite and wrote SourceCode/plant_model.onnx.
Could not run pytest because the current venv does not have pytest installed.
```

---

## 2026-06-05 - Staged Domain-Adaptation Fine-Tuning

### Summary

Improved the new-dataset fine-tuning pipeline with source-aware domain augmentations, dynamic class-weighted loss, and a two-stage EfficientNet-B2 training schedule.

### Changed

- Updated `SourceCode/finetune_new_plant_dataset.py`.
- Added source-aware training transforms so only `new_preprocessed` samples receive domain-adaptation augmentation.
- Added new-domain augmentations for outdoor domain shift: resize, horizontal/vertical flips, 30-degree rotation, brightness/contrast/saturation/hue jitter, and random erasing.
- Kept validation transforms strict: resize, tensor conversion, and ImageNet normalization only.
- Kept historical PlantVillage/Unknown samples on a conservative transform path instead of applying new-domain augmentations to them.
- Added dynamic inverse-frequency class weights computed from the compiled training pool and passed into `torch.nn.CrossEntropyLoss`.
- Changed the default fine-tuning schedule to 8 total epochs.
- Added Stage A head stabilization for epochs 1-2 with the EfficientNet-B2 feature extractor frozen and classifier head LR set to `1e-3`.
- Added Stage B full discriminative fine-tuning for epochs 3-8 with all layers unfrozen and LR set to `5e-6`.
- Added CLI controls for `--head-epochs`, `--head-lr`, and `--full-lr`; kept `--lr` as an alias for `--full-lr`.
- Added logs for frozen/unfrozen transitions, trainable parameter counts, class-weight ranges, and stage-specific epoch metrics.

### Runtime Results

```text
Syntax validation passed for SourceCode/finetune_new_plant_dataset.py.
CLI help passed and shows staged schedule arguments.
Dry run passed with class-weight computation and source-aware DataLoader construction.
Smoke run passed with 2 epochs: Stage A froze the backbone, Stage B unfroze all layers, and both stages completed train/validation loops.
Temporary smoke-test checkpoint models/finetune_smoke_test.pth was removed after validation.
Full 8-epoch fine-tuning was not run during this validation pass.
```

---

## 2026-06-05 - Audited New-Domain Validation Pipeline

### Summary

Upgraded the fine-tuning and evaluation workflow so model selection and final reporting are based on auditable in-the-wild train/validation/test splits instead of optimistic hybrid validation accuracy.

### Changed

- Updated `SourceCode/finetune_new_plant_dataset.py`.
- Added deterministic new-domain `train/val/test` preprocessing with split names that include seed, validation ratio, and test ratio.
- Added `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv` containing split, class, label, source path, preprocessed path, and source SHA-256.
- Added source-hash duplicate grouping before splitting so duplicate raw images remain in the same split.
- Added manifest leakage validation that fails if a source SHA-256 appears across multiple splits.
- Added fresh suffixed split directory creation when Windows refuses to delete stale generated files.
- Changed fine-tuning so historical PlantVillage/Unknown data is used only as replay training data.
- Changed validation to use only the new-domain `val` split.
- Changed checkpoint selection to use new-domain validation macro-F1 while still logging validation accuracy.
- Replaced raw inverse-frequency class weights with class-balanced weights using configurable `--class-weight-beta` and `--max-class-weight`.
- Disabled balanced sampling by default and added `--use-balanced-sampler` as an explicit opt-in.
- Updated `SourceCode/src/evaluate_and_convert.py` to use the same audited split discovery.
- Added evaluation report exports under `SourceCode/reports/new_dataset_evaluation/`: classification report text/JSON, confusion matrix CSV, worst-recall CSV, and summary JSON.
- Added top-3 accuracy reporting.
- Updated documentation in `README.md`, `SourceCode/README.md`, `SourceCode/TRAINING_PIPELINE.md`, `VALIDATION_GUIDE.md`, `PROJECT_SUMMARY.md`, `report.md`, and relevant `PlantDiseaseDetectionKnowledge/` notes.

### Runtime Results

```text
Syntax validation passed for finetune_new_plant_dataset.py and src/evaluate_and_convert.py.
CLI help passed for both scripts.
Created audited split: SourceCode/data/NewPLantDataset_preprocessed/split_seed42_val15_test20_v2/.
Manifest rows: train=577, val=139, test=180.
Manifest leakage check passed.
Detected and grouped duplicate source hashes before splitting.
Detected unreadable Apple healthy source images and skipped them during preprocessing.
Fine-tune dry run passed with new-domain validation loader and class-balanced loss.
Two-epoch smoke training passed Stage A/Stage B and saved by new validation macro-F1.
Temporary smoke-test checkpoint models/finetune_smoke_test.pth was removed after validation.
Evaluation dry run on existing fine-tuned checkpoint passed on the audited test split.
Audited test split metrics from existing checkpoint: accuracy=58.33%, top-3 accuracy=87.78%, macro F1=37.91%, weighted F1=59.21%.
Full 8-epoch fine-tuning was not run in this validation pass.
```

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
