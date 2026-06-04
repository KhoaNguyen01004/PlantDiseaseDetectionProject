# Preprocess And Export Confusion

This note clarifies which files are source artifacts and which are generated exports.

---

## Keep

```text
SourceCode/models/best_model.pth
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
