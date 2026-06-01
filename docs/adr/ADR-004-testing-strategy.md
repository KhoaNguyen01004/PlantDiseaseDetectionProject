# ADR-004 — Testing strategy

## Status
Accepted

## Context
The repository requires reliable validation of preprocessing, export, inference, and metadata generation, especially because it supports multiple deployment targets and model export flows.

## Decision
Use pytest-based unit tests under `SourceCode/tests/` plus validation documentation in `VALIDATION_GUIDE.md` to cover critical components.

## Alternatives considered
- Rely solely on manual validation and ad hoc scripts.
- Use a lightweight script-only validation set without pytest.
- Create a full integration test suite with external device dependency.

## Tradeoffs
- Unit tests provide fast, repeatable checks without external hardware.
- They do not replace end-to-end Android or real-device validation.
- Documentation must clearly distinguish between code-level tests and deployment verification steps.

## Consequences
- Critical functionality is covered by tests for configuration, metadata export, export pipelines, inference preprocessing, and quality validation.
- Validation guidance documents the path to verify TorchScript Android deployment separately.
- The repo avoids claiming full device deployment as complete until Android runtime verification is explicitly performed.
