# Grad-CAM And Prediction Explainability

Last updated: 2026-06-05

This project now has two explanation paths:

1. Offline true Grad-CAM in Python for model debugging and report images.
2. On-device Android explanation using top-3 model probabilities shown in the result panel.

The Android app does not run true Grad-CAM on device. The current mobile runtime uses forward-only TorchScript inference, while Grad-CAM needs gradients from the PyTorch model.

---

## Offline Grad-CAM

Script:

```text
SourceCode/src/gradcam.py
```

Default checkpoint behavior:

- Prefer `SourceCode/models/best_model_finetuned.pth` if present.
- Otherwise use the newest `SourceCode/models/best_model_finetuned*.pth`.
- Fall back to `SourceCode/models/best_model.pth` only if no fine-tuned checkpoint exists.

Example:

```bash
cd SourceCode
python -m src.gradcam --image "data/NewPLantDataset_preprocessed/split_seed42_val15_test20_v2/test/Apple___Apple_scab/apple_scab1.jpg"
```

Optional explicit checkpoint:

```bash
python -m src.gradcam ^
  --image "data/NewPLantDataset_preprocessed/split_seed42_val15_test20_v2/test/Apple___Apple_scab/apple_scab1.jpg" ^
  --model "models/best_model_finetuned256.pth" ^
  --output "reports/gradcam/apple_scab_example.png"
```

Optional target layer comparison:

```bash
python -m src.gradcam --image "<path-to-image>" --target-layer "features.7" --output "reports/gradcam/features7_check.png"
```

The default target layer is `features.8`, the final EfficientNet-B2 convolution path.

Outputs:

```text
SourceCode/reports/gradcam/gradcam_output.png
SourceCode/reports/gradcam/gradcam_output_original.png
SourceCode/reports/gradcam/gradcam_output_overlay.png
```

The combined image contains:

- original image,
- Grad-CAM heatmap,
- overlay showing which regions influenced the predicted class.

---

## Android Explanation

The Android app now displays the model's top-3 closest matches below the confidence bar.

This helps users understand whether the model made a strong single-class decision or whether several classes were close. It is not a heatmap and should not be described as Grad-CAM.

Android files:

```text
agrilens/app/src/main/java/com/example/ImageClassifierHelper.java
agrilens/app/src/main/java/com/example/MainActivity.java
agrilens/app/src/main/res/layout/activity_main.xml
agrilens/app/src/main/res/values/strings.xml
agrilens/app/src/main/res/values-vi/strings.xml
```

Current behavior:

- The classifier computes softmax probabilities for all 39 classes.
- The result panel shows the selected label and confidence.
- The explanation line shows the top-3 closest model matches and their probabilities.
- If top-1 confidence is below the unknown threshold, the app still labels the result as `Unknown`, but the top-3 line shows the closest known model matches for review.

---

## Validation

Use this after changing Grad-CAM, TorchScript assets, or Android explanation UI:

```bash
cd SourceCode
python -m src.gradcam --help
python -m src.gradcam --image "<path-to-test-image>" --output "reports/gradcam/check.png"

cd ../agrilens
./gradlew.bat :app:assembleDebug
```

The Grad-CAM output should focus on leaf or lesion regions. If it consistently highlights background areas, the model is likely using spurious visual cues and the dataset/augmentation strategy should be revisited.
