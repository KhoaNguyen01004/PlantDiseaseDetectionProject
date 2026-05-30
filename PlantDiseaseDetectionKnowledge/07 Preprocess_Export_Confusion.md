# Entity: Preprocess / Export Naming Confusion

**Source:** `SourceCode/src/preprocessing/preprocess.py` (function `export_to_tflite`)

## Purpose (as implied by name)
Intended to support TFLite packaging and metadata.

## Current behavior
- writes label files (`labels.json`, `labels.txt`)
- does not actually export the TFLite model
- metadata writer code is only logged as “how to do it”
- `tflite_path` argument is ignored
- called with `export_to_tflite(None, ...)` in `__main__`

## Impact
- Confusing pipeline for future work
- unclear whether metadata export is implemented

## Links
- [[01 Inference_TFLite]]
- [[02 Inference_Preprocess]]

