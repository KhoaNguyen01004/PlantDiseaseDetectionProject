# Preprocess And Export Confusion

This note clarifies which files are source artifacts and which are generated exports.

---

## Keep

```text
SourceCode/models/best_model.pth
SourceCode/models/best_model_finetuned.pth
SourceCode/labels.txt
SourceCode/labels.json
SourceCode/metadata.json
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

---

## Generated

```text
SourceCode/plant_model.onnx
SourceCode/plant_model.onnx.data
SourceCode/plant_model.pt
SourceCode/plant_model_tflite_float32/
SourceCode/plant_model_tflite_int8/
SourceCode/reports/new_dataset_evaluation/
SourceCode/reports/gradcam/
```

Generated files can be removed when not needed if the checkpoint is kept.

---

## Runtime Paths

Android:

```text
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Python TFLite:

```text
SourceCode/plant_model_tflite_*/
```

Real-world preprocessing:

```text
SourceCode/data/NewPLantDataset_preprocessed/split_seed*_val*_test*/
```

Those split folders are generated from `SourceCode/data/NewPLantDataset/color/`. Keep their manifests when reporting results because they prove the train/validation/test split and leakage checks.

Grad-CAM outputs under `SourceCode/reports/gradcam/` are generated explanation artifacts and can be regenerated from the checkpoint and source image.
