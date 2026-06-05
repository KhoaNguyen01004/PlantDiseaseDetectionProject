package com.example;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ImageClassifierHelper {

    private static final String TAG = "AgriLensClassifier";

    private static final String MODEL_FILE = "plant_model.pt";
    private static final String LABELS_FILE = "labels.txt";

    // EfficientNet-B2 requires 260x260 input (see configs/config.yaml)
    private static final int IMAGE_SIZE = 260;

    // Unknown threshold
    private static final float UNKNOWN_THRESHOLD = 0.65f;

    private Module module;

    private final List<String> labels = new ArrayList<>();

    private boolean isDemoMode = false;

    // ImageNet normalization
    private static final float[] IMAGENET_MEAN = {
            0.485f,
            0.456f,
            0.406f
    };

    private static final float[] IMAGENET_STD = {
            0.229f,
            0.224f,
            0.225f
    };

    public static class Prediction {
        public final String label;
        public final float confidence;

        public Prediction(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }

    public interface ClassificationListener {
        void onClassificationResult(String label, float confidence, List<Prediction> topPredictions);

        void onClassificationError(String errorMessage);
    }

    public ImageClassifierHelper(Context context) {
        initLabels(context);
        initModel(context);
    }

    private void initLabels(Context context) {

        try (
                BufferedReader reader = new BufferedReader(
                        new InputStreamReader(
                                context.getAssets().open(LABELS_FILE)))) {

            String line;

            while ((line = reader.readLine()) != null) {

                line = line.trim();

                if (!line.isEmpty()) {
                    labels.add(line);
                }
            }

            Log.i(TAG, "Loaded labels: " + labels.size());

        } catch (Exception e) {

            Log.e(TAG, "Failed loading labels", e);

            isDemoMode = true;
        }
    }

    private void initModel(Context context) {

        try {

            String modelPath = assetFilePath(context, MODEL_FILE);

            module = Module.load(modelPath);

            Log.i(TAG, "PyTorch model loaded");

        } catch (Exception e) {

            Log.e(TAG, "Failed loading model", e);

            isDemoMode = true;
        }
    }

    public void classify(
            Bitmap bitmap,
            ClassificationListener listener) {

        if (bitmap == null) {

            listener.onClassificationError(
                    "Bitmap is null");

            return;
        }

        if (isDemoMode || module == null) {

            runSimulation(bitmap, listener);

            return;
        }

        try {

            Bitmap resizedBitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    true);

            Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                    resizedBitmap,
                    IMAGENET_MEAN,
                    IMAGENET_STD);

            Tensor outputTensor = module.forward(
                    IValue.from(inputTensor)).toTensor();

            float[] scores = outputTensor.getDataAsFloatArray();

            int maxIndex = 0;

            float maxScore = scores[0];

            for (int i = 1; i < scores.length; i++) {

                if (scores[i] > maxScore) {

                    maxScore = scores[i];

                    maxIndex = i;
                }
            }

            float confidence = softmaxConfidence(scores, maxIndex);

            String label = labels.get(maxIndex);
            List<Prediction> topPredictions = topPredictions(scores, 3);

            // UNKNOWN LOGIC
            if (confidence < UNKNOWN_THRESHOLD) {

                label = "Unknown";
            }

            listener.onClassificationResult(
                    label,
                    confidence,
                    topPredictions);

        } catch (Exception e) {

            Log.e(TAG, "Inference failed", e);

            listener.onClassificationError(
                    e.getMessage());
        }
    }

    private float softmaxConfidence(
            float[] logits,
            int index) {

        float max = Float.NEGATIVE_INFINITY;

        for (float v : logits) {

            if (v > max) {
                max = v;
            }
        }

        float sum = 0f;

        for (float v : logits) {

            sum += Math.exp(v - max);
        }

        return (float) (Math.exp(logits[index] - max) / sum);
    }

    private List<Prediction> topPredictions(float[] logits, int count) {
        float max = Float.NEGATIVE_INFINITY;

        for (float value : logits) {
            if (value > max) {
                max = value;
            }
        }

        float sum = 0f;
        float[] probabilities = new float[logits.length];

        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) Math.exp(logits[i] - max);
            sum += probabilities[i];
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < logits.length; i++) {
            indices.add(i);
        }

        Collections.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer left, Integer right) {
                return Float.compare(probabilities[right], probabilities[left]);
            }
        });

        List<Prediction> predictions = new ArrayList<>();
        int limit = Math.min(count, indices.size());

        for (int i = 0; i < limit; i++) {
            int index = indices.get(i);
            predictions.add(new Prediction(labels.get(index), probabilities[index] / sum));
        }

        return predictions;
    }

    private String assetFilePath(
            Context context,
            String assetName) throws IOException {

        File file = new File(
                context.getFilesDir(),
                assetName);

        if (file.exists() && file.length() > 0) {

            return file.getAbsolutePath();
        }

        try (
                InputStream is = context.getAssets().open(assetName);

                FileOutputStream os = new FileOutputStream(file)) {

            byte[] buffer = new byte[4 * 1024];

            int read;

            while ((read = is.read(buffer)) != -1) {

                os.write(buffer, 0, read);
            }

            os.flush();
        }

        return file.getAbsolutePath();
    }

    // Demo mode fallback
    private void runSimulation(
            Bitmap bitmap,
            ClassificationListener listener) {

        if (labels.isEmpty()) {

            listener.onClassificationError(
                    "Labels empty");

            return;
        }

        int index = Math.abs(bitmap.getWidth() + bitmap.getHeight())
                % labels.size();

        float confidence = 0.80f;
        List<Prediction> topPredictions = new ArrayList<>();
        topPredictions.add(new Prediction(labels.get(index), confidence));

        listener.onClassificationResult(
                labels.get(index),
                confidence,
                topPredictions);
    }
}
