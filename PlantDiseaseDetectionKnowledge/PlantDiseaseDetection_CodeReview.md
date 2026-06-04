# Code Review Notes

This file records current review findings without inventing runtime results.

---

## Current Strengths

- Central config exists in `SourceCode/configs/config.yaml`.
- Training checkpoint now stores class order, architecture, config, and metrics.
- Android uses local inference and does not require network inference.
- Android supports live and capture workflows.
- Metadata and label export are centralized.

---

## Current Risks

- Full Python tests require a dependency-complete environment.
- Android runtime quality still needs device validation.
- Treatment guidance should be reviewed by a domain expert.
- TFLite export artifacts are separate from the Android runtime path and should not be confused with app assets.
- Documentation placeholders must be replaced only with measured values.

---

## Required Measurements

```text
Python test result: {{PYTHON_TEST_RESULT}}
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Test accuracy: {{TEST_ACCURACY}}
Android latency: {{ANDROID_AVG_LATENCY}}
Android memory: {{ANDROID_MEMORY_USAGE}}
Field test result: {{FIELD_TEST_RESULT}}
```
