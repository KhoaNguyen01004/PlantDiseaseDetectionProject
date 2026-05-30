# Plant Disease Detection — Code Review (AndroidApp excluded)

## Scope
- Reviewed: `SourceCode/src/inference.py`, `SourceCode/src/train.py`, `SourceCode/src/preprocessing/preprocess.py`, `SourceCode/src/preprocessing/generate_background_noise.py`
- Excluded: everything under `AndroidApp/`

## High-level architecture
- **Training**: EfficientNet-B0 fine-tuned for `num_classes`.
- **Data**: directory-driven label mapping + optional mask-based augmentation (segmented mask).
- **Augmentation**: Albumentations with rotations/scales/brightness/noise + ImageNet normalize.
- **Inference**: runs a **TFLite** model with attempted delegate acceleration.

---

## Critical correctness issues

### 1) Inference preprocessing likely mismatches the TFLite model’s expected input type/range
In `SourceCode/src/inference.py`, preprocessing does:
1. Convert BGR→RGB
2. Resize
3. Convert to float, normalize with ImageNet mean/std
4. **Cast to `uint8`**:
   ```py
   img = np.expand_dims(img, 0).astype(np.uint8)
   ```

This is almost certainly wrong:
- If the TFLite model is **INT8 quantized**, you must quantize using the model’s **input quantization parameters** (scale/zero_point) and feed the dtype the model expects (commonly `uint8` for INT8 pipelines, but not always).
- If the model is **float32**, you must keep float32.

**Best practice**: use `interpreter.get_input_details()[0]` to branch on `dtype` and apply `quantization` correctly.

### 2) Output interpretation (“confidence”) assumes max(output) is probability
Current inference computes:
```py
conf = np.max(output_data)
```
But depending on export, `output_data` may be:
- logits (needs softmax)
- probabilities (already softmax)

**Best practice**: inspect output details and/or apply softmax when appropriate.

### 3) Delegate loading is not robust/portable
`create_delegates()` does:
- tries to load GPU delegate with a hardcoded `'libgpu_delegate.so'`
- then **always** loads XNNPACK delegate with `'libxnnpack.so'`

On Windows / non-Android runtime, these shared libraries will usually not exist or be loadable.

**Best practice**:
- only load delegates if `.so` exists and loads
- otherwise run without experimental delegates

---

## Training / preprocessing logic review

### 1) Masking logic is conceptually sound
In `preprocess.py`:
- images are read → converted to RGB float32 in `[0..1]`
- masked region (if seg mask exists) is applied stochastically
- then Albumentations normalization + ToTensorV2 happen

This matches your intent: improve robustness to real-world background clutter.

### 2) Double-resizing wastes compute and can introduce slight differences
You resize:
- once in `preprocess_image()`
- again in Albumentations (`A.Resize(...)`)

Not a logic-breaking bug, but it’s a refactor target: resize once.

### 3) WeightedRandomSampler is reasonable
You compute class counts from train split and sample with inverse frequency.

Potential improvement (not required): ensure tensor dtype/device choices match PyTorch expectations (usually safe as written, but could be refined).

---

## Code structure / maintainability issues

### 1) Training and inference preprocessing constants are duplicated
Mean/std/image size are hardcoded in multiple places:
- `inference.py`: `MEAN`, `STD`, `IMAGE_SIZE`
- `preprocess.py`: normalization mean/std inside Albumentations

If you change training normalization or image size, inference can silently drift.

**Best practice**: centralize constants in `config.yaml` and load in both training/inference.

### 2) `export_to_tflite()` is misleading / not implemented as stated
In `preprocess.py`, `export_to_tflite(model, label_map, tflite_path)`:
- writes `labels.json` and `labels.txt`
- but does **not** export any TFLite model or metadata
- ignores `tflite_path`

This creates confusion in the pipeline.

**Best practice**: either
- rename to `export_labels_for_tflite()`
- or implement actual model export + TFLite metadata writing.

---

## What I could not verify
- I attempted to inspect the deployed model input/output dtype/quantization, but the referenced file
  `SourceCode/plant_model_tflite_int8/plant_model_int8.tflite`
  is **not present in this repo snapshot**.

So I could not confirm whether the model expects:
- float32 vs uint8
- correct quantization scale/zero_point

---

## Condensed refactor / fix checklist

### Inference (`SourceCode/src/inference.py`)
- [ ] Determine input dtype + quantization params from TFLite interpreter
- [ ] Branch preprocessing:
  - float model → keep float32 after mean/std normalization
  - int8 model → quantize to `(x/scale + zero_point)` and cast to correct dtype
- [ ] Interpret outputs properly (probabilities vs logits)
- [ ] Make delegates optional/robust (don’t hard-load `.so` libraries)
- [ ] Centralize preprocessing constants with training config

### Training/data (`SourceCode/src/preprocessing/preprocess.py`)
- [ ] Remove redundant resizing
- [ ] Rename misleading “export_to_tflite” or implement real export
- [ ] Consider moving augmentation/preprocess parameters to config

---

## Files reviewed
- `SourceCode/src/inference.py`
- `SourceCode/src/train.py`
- `SourceCode/src/preprocessing/preprocess.py`
- `SourceCode/src/preprocessing/generate_background_noise.py`


