# Plant Disease Detection Report

This report is a maintained draft. Runtime metrics are placeholders until measured from the current model and Android app.

---

## 1. Introduction

Plant leaf disease detection can help users screen visible symptoms from leaf images. This project implements a local computer vision pipeline and an Android app for on-device inference.

The system is intended as a decision-support prototype. It does not replace expert agricultural diagnosis.

---

## 2. Objectives

- Train a leaf disease classification model.
- Export model artifacts for evaluation and deployment.
- Run inference locally on Android.
- Provide result confidence and care guidance.
- Support English and Vietnamese UI.
- Keep documentation and runtime artifacts reproducible.

---

## 3. System Architecture

```text
Image dataset
  -> preprocessing and augmentation
  -> EfficientNet training
  -> best_model.pth
  -> staged fine-tuning on audited in-the-wild train/val/test split
  -> best_model_finetuned.pth
  -> metadata and labels
  -> export artifacts
  -> Android model asset
  -> on-device inference
```

The Android app currently uses PyTorch Mobile and a TorchScript model asset.

---

## 4. Model Pipeline

Training is configured by:

```text
SourceCode/configs/config.yaml
```

Current configuration facts:

```text
Default architecture: efficientnet_b2
Input size: 260
Normalization: ImageNet mean/std from config
Optimizer: AdamW
Loss: CrossEntropyLoss with label smoothing
Checkpoint: SourceCode/models/best_model.pth
```

The training pipeline now stores model architecture, class names, config, history, and metrics in the checkpoint.

The in-the-wild fine-tuning pipeline uses:

```text
Script: SourceCode/finetune_new_plant_dataset.py
Split artifacts: SourceCode/data/NewPLantDataset_preprocessed/split_seed*_val*_test*/
Manifests: train_manifest.csv, val_manifest.csv, test_manifest.csv
Selection metric: new-domain validation macro F1
Fine-tuned checkpoint: SourceCode/models/best_model_finetuned.pth
```

The held-out test split should be used only for final reporting. Hybrid validation accuracy should not be presented as proof of field generalization.

---

## 5. Android Application

The Android app supports:

- Live camera detection.
- Capture-first detection.
- Gallery image analysis.
- Pause-on-known-result behavior.
- Disease care guidance.
- English and Vietnamese language setting.
- Adjustable text size.
- Offline guide.
- Top-3 closest model matches in the result panel.

Android runtime assets:

```text
agrilens/app/src/main/assets/plant_model.pt
agrilens/app/src/main/assets/labels.txt
```

Grad-CAM is available as an offline Python explanation tool through `SourceCode/src/gradcam.py`. The Android app does not run true Grad-CAM on device; it displays top-3 prediction probabilities for user-facing explanation.

---

## 6. Evaluation Plan

Fill the following after running validation:

```text
Dataset version/hash: {{DATASET_VERSION_OR_HASH}}
Class count: {{CLASS_COUNT}}
Training date: {{TRAINING_DATE}}
Best validation accuracy: {{BEST_VAL_ACCURACY}}
Best new validation macro F1: {{BEST_NEW_VAL_MACRO_F1}}
Test accuracy: {{TEST_ACCURACY}}
Top-3 accuracy: {{TOP3_ACCURACY}}
Macro F1: {{MACRO_F1}}
Weighted F1: {{WEIGHTED_F1}}
Confusion matrix path: {{CONFUSION_MATRIX_PATH}}
Worst-recall class report path: {{WORST_RECALL_CLASSES_PATH}}
```

Android benchmark placeholders:

```text
Device: {{ANDROID_TEST_DEVICE}}
Android version: {{ANDROID_VERSION}}
Average inference latency: {{ANDROID_AVG_LATENCY}}
P95 inference latency: {{ANDROID_P95_LATENCY}}
Memory usage: {{ANDROID_MEMORY_USAGE}}
Battery impact: {{BATTERY_IMPACT}}
```

---

## 7. Limitations

- Results depend on dataset quality and field-photo coverage.
- Unknown detection depends on unknown/background training data.
- Treatment guidance requires domain review before practical use.
- Lighting, blur, occlusion, and background clutter may affect predictions.
- Sparse per-class test support can make individual class metrics volatile.
- Android model and labels must stay synchronized.

---

## 8. Future Work

- Add measured evaluation results.
- Add more balanced real field validation and test images, especially for low-support classes.
- Add real unknown/background field images to evaluate false-positive protection.
- Review Vietnamese agricultural terminology with a domain speaker.
- Decide whether to keep PyTorch Mobile or migrate Android inference to TFLite.
- Add automated Android runtime tests where practical.

---

## 9. Conclusion

The project provides a working local training and Android inference prototype. The next step is to validate the current model with measured results and replace placeholders in this report.
