# Entity: Delegates (GPU/XNNPACK)

**Source:** `SourceCode/src/inference.py` (function `create_delegates`)

## Purpose
Try to accelerate TFLite inference using experimental delegates.

## Current logic summary
- attempt to load GPU delegate via `libgpu_delegate.so`
- always load XNNPACK delegate via `libxnnpack.so`

## Problems
- Hardcoded `.so` filenames break portability (Windows dev machines typically won’t have these)
- Delegate load failures may crash or silently degrade depending on runtime

## Best practice
- Only load delegates when the shared library exists in an expected location
- Wrap each delegate attempt independently and continue if unavailable
- Consider a `--no-delegates` CLI flag for reproducibility

## Links
- [[01 Inference_TFLite]]

