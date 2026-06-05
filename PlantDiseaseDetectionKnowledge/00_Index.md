# Plant Disease Detection Knowledge Index

These notes summarize the current technical design. They do not contain measured runtime results.

---

## Notes

- `01 Inference_TFLite.md`: TFLite export and Python inference context.
- `02 Inference_Preprocess.md`: preprocessing alignment between training and inference.
- `03 Delegates.md`: runtime delegate notes.
- `04 Training_Loop.md`: training loop behavior.
- `05 Data_Split_Sampler.md`: data splitting and sampling.
- `06 Augmentation_and_Masking.md`: augmentation and masking context.
- `07 Preprocess_Export_Confusion.md`: common confusion between preprocessing/export paths.
- `PlantDiseaseDetection_CodeReview.md`: current review notes and risks.

---

## Current Runtime Truth

Android uses:

```text
PyTorch Mobile
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Python export can generate:

```text
plant_model.onnx
plant_model_tflite_float32/
plant_model_tflite_int8/
```

Real-world fine-tuning and evaluation use:

```text
SourceCode/finetune_new_plant_dataset.py
SourceCode/data/NewPLantDataset_preprocessed/split_seed*_val*_test*/
SourceCode/models/best_model_finetuned.pth
SourceCode/reports/new_dataset_evaluation/
```

Runtime metrics must be measured and recorded as placeholders until then.
