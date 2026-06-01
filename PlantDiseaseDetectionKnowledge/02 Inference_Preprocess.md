# Entity: Inference Preprocess

**Source:** `SourceCode/src/inference.py` (function `preprocess_image`) 

## Purpose
Convert OpenCV images/frames into the tensor that the TFLite model expects.

## Current logic summary
- BGR → RGB
- resize to 260×260
- float32 + ImageNet mean/std normalization
- transpose HWC → CHW
- expand batch
- cast to `uint8` only when the TFLite interpreter indicates a quantized input dtype

## Why this is risky
- If your TFLite model is float32, casting to `uint8` is incorrect.
- If your TFLite model is INT8 quantized, you must apply quantization using the model’s `input_details` scale/zero_point (not the ImageNet float normalization path + cast).

## Best practice contract
- Inspect `input_details[0]['dtype']`
- Inspect `input_details[0]['quantization']`
- Branch preprocessing accordingly:
  - float model: keep float32, don’t cast to uint8
  - quantized model: quantize using scale/zero_point and cast to the interpreter-required dtype

## Output decoding note
Your inference currently uses `conf = max(output)`.
- If outputs are logits: apply softmax first.
- If outputs are already probabilities: max is fine.

## Links
- [[01 Inference_TFLite]]
- [[04 Training_Loop]] (for preprocessing consistency)

