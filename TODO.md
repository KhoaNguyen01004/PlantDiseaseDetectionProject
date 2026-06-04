# TODO

This file tracks remaining work without assuming runtime results.

---

## Validation

- [ ] Run full Python test suite in an environment with project dependencies installed.
- [ ] Run `python -m src.train` on the intended dataset.
- [ ] Run `python -m src.evaluate_and_convert` after training.
- [ ] Record current test metrics in `README.md`, `PROJECT_SUMMARY.md`, and `report.md`.
- [ ] Benchmark Android inference on target device.
- [ ] Verify Android model and labels hashes after every model update.

---

## Model And Data

- [ ] Confirm final dataset version/hash: `{{DATASET_VERSION_OR_HASH}}`.
- [ ] Add real field images for validation.
- [ ] Review unknown/background dataset diversity.
- [ ] Confirm class list and label order after the next training run.

---

## Android

- [ ] Test live mode, capture mode, and gallery mode on target device.
- [ ] Review Vietnamese copy with a native speaker familiar with agriculture terms.
- [ ] Verify treatment guidance for every known label.
- [ ] Decide whether Android will stay on PyTorch Mobile or migrate to TFLite.

---

## Documentation

- [ ] Replace all `{{PLACEHOLDER}}` values only with measured values.
- [ ] Update screenshots after UI review.
- [ ] Keep artifact policy synchronized with actual export scripts.
