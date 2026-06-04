# ADR-004: Use Focused Tests And Explicit Runtime Validation

Status: Accepted

---

## Context

The repository contains Python ML code and an Android app. Some checks can be automated quickly, while training and device performance require runtime execution.

---

## Decision

Use focused automated tests for:

- config schema and bounds
- dataloader behavior
- metadata export
- inference preprocessing
- quality validation
- export utility smoke tests

Use manual/runtime validation for:

- full training quality
- test-set metrics
- Android inference latency
- Android memory/battery behavior
- real field-image behavior

---

## Commands

Python tests:

```bash
cd SourceCode
python -m pytest
```

Android build:

```bash
cd agrilens
./gradlew.bat :app:assembleDebug
```

---

## Placeholders

```text
Python test result: {{PYTHON_TEST_RESULT}}
Android build result: {{ANDROID_BUILD_RESULT}}
Device test result: {{DEVICE_TEST_RESULT}}
```
