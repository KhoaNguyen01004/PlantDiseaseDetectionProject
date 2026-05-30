package com.example;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import java.io.ByteArrayOutputStream;
import android.graphics.BitmapFactory;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "AgriLens";

    private static final int REQUEST_CAMERA_PERMISSION = 100;

    private PreviewView previewView;

    private ImageView imagePreview;

    private TextView resultText;

    private TextView confidenceText;

    private ProgressBar progressBar;

    private Button btnCapture;

    private Button btnGallery;

    private OverlayView overlayView;

    private ImageCapture imageCapture;

    private ImageClassifierHelper classifierHelper;

    private ExecutorService analysisExecutor;

    private final ActivityResultLauncher<Intent>
            galleryLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(),
                    result -> {

                        if (
                                result.getResultCode() == RESULT_OK
                                        && result.getData() != null
                        ) {

                            Uri imageUri =
                                    result.getData().getData();

                            try {

                                Bitmap bitmap =
                                        MediaStore.Images.Media.getBitmap(
                                                getContentResolver(),
                                                imageUri
                                        );

                                processBitmap(bitmap);

                            } catch (Exception e) {

                                Log.e(TAG,
                                        "Gallery image error",
                                        e);
                            }
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        EdgeToEdge.enable(this);

        setContentView(R.layout.activity_main);

        initViews();

        analysisExecutor = Executors.newSingleThreadExecutor();

        classifierHelper =
                new ImageClassifierHelper(this);

        if (hasCameraPermission()) {

            startCamera();

        } else {

            requestCameraPermission();
        }

        btnCapture.setOnClickListener(
                v -> captureImage()
        );

        btnGallery.setOnClickListener(
                v -> openGallery()
        );
    }

    private void initViews() {

        previewView =
                findViewById(R.id.previewView);

        imagePreview =
                findViewById(R.id.imagePreview);

        resultText =
                findViewById(R.id.resultText);

        confidenceText =
                findViewById(R.id.confidenceText);

        progressBar =
                findViewById(R.id.progressBar);

        btnCapture =
                findViewById(R.id.btnCapture);

        btnGallery =
                findViewById(R.id.btnGallery);

        overlayView =
                findViewById(R.id.overlayView);
    }

    // =====================================================
    // CAMERA
    // =====================================================

    private void startCamera() {

        ListenableFuture<ProcessCameraProvider>
                cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {

            try {

                ProcessCameraProvider cameraProvider =
                        cameraProviderFuture.get();

                androidx.camera.core.Preview preview =
                        new androidx.camera.core.Preview.Builder()
                                .build();

                preview.setSurfaceProvider(
                        previewView.getSurfaceProvider()
                );

                imageCapture =
                        new ImageCapture.Builder()
                                .build();

                ImageAnalysis imageAnalysis =
                        new ImageAnalysis.Builder()
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .setTargetRotation(previewView.getDisplay().getRotation())
                                .build();

                imageAnalysis.setAnalyzer(analysisExecutor, imageProxy -> {
                    try {
                        Bitmap bitmap = imageProxy.toBitmap();
                        if (bitmap != null) {
                            classifierHelper.classify(bitmap, new ImageClassifierHelper.ClassificationListener() {
                                @Override
                                public void onClassificationResult(String label, float confidence) {
                                    runOnUiThread(() -> {
                                        boolean detected = !"Unknown".equalsIgnoreCase(label);
                                        overlayView.setResults(detected);
                                        displayResult(label, confidence);
                                    });
                                }

                                @Override
                                public void onClassificationError(String error) {
                                    Log.e(TAG, "Analysis error: " + error);
                                }
                            });
                        }
                    } catch (Exception e) {
                        Log.e(TAG, "Error in analysis: " + e.getMessage());
                    } finally {
                        imageProxy.close();
                    }
                });

                CameraSelector cameraSelector =
                        CameraSelector.DEFAULT_BACK_CAMERA;

                cameraProvider.unbindAll();

                cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        preview,
                        imageCapture,
                        imageAnalysis
                );

            } catch (
                    ExecutionException
                    | InterruptedException e
            ) {

                Log.e(TAG,
                        "Camera start failed",
                        e);
            }

        }, ContextCompat.getMainExecutor(this));
    }

    private void captureImage() {

        if (imageCapture == null) {
            return;
        }

        File photoFile =
                new File(
                        getCacheDir(),
                        "captured.jpg"
                );

        ImageCapture.OutputFileOptions options =
                new ImageCapture.OutputFileOptions.Builder(photoFile)
                        .build();

        imageCapture.takePicture(
                options,
                ContextCompat.getMainExecutor(this),

                new ImageCapture.OnImageSavedCallback() {

                    @Override
                    public void onImageSaved(
                            @NonNull ImageCapture.OutputFileResults outputFileResults
                    ) {

                        Bitmap bitmap =
                                android.graphics.BitmapFactory.decodeFile(
                                        photoFile.getAbsolutePath()
                                );

                        processBitmap(bitmap);
                    }

                    @Override
                    public void onError(
                            @NonNull ImageCaptureException exception
                    ) {

                        Log.e(TAG,
                                "Capture failed",
                                exception);

                        Toast.makeText(
                                MainActivity.this,
                                "Capture failed",
                                Toast.LENGTH_SHORT
                        ).show();
                    }
                }
        );
    }

    // =====================================================
    // GALLERY
    // =====================================================

    private void openGallery() {

        Intent intent =
                new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                );

        galleryLauncher.launch(intent);
    }

    // =====================================================
    // CLASSIFICATION
    // =====================================================

    private void processBitmap(Bitmap bitmap) {

        if (bitmap == null) {
            return;
        }

        imagePreview.setImageBitmap(bitmap);

        progressBar.setVisibility(
                android.view.View.VISIBLE
        );

        resultText.setText("Processing...");

        confidenceText.setText("");

        classifierHelper.classify(
                bitmap,

                new ImageClassifierHelper.ClassificationListener() {

                    @Override
                    public void onClassificationResult(
                            String label,
                            float confidence
                    ) {

                        runOnUiThread(() -> {

                            progressBar.setVisibility(
                                    android.view.View.GONE
                            );

                            displayResult(
                                    label,
                                    confidence
                            );
                        });
                    }

                    @Override
                    public void onClassificationError(
                            String errorMessage
                    ) {

                        runOnUiThread(() -> {

                            progressBar.setVisibility(
                                    android.view.View.GONE
                            );

                            Toast.makeText(
                                    MainActivity.this,
                                    errorMessage,
                                    Toast.LENGTH_LONG
                            ).show();
                        });
                    }
                }
        );
    }

    private void displayResult(
            String label,
            float confidence
    ) {

        String lower =
                label.toLowerCase(Locale.US);

        if (lower.contains("unknown")) {
            resultText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_dark));
            resultText.setText("Result: " + label);
        } else if (lower.contains("healthy")) {
            resultText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_green_dark));
            resultText.setText("Result: " + label);
        } else {
            resultText.setTextColor(ContextCompat.getColor(this, android.R.color.holo_orange_dark));
            resultText.setText("Disease Detected: " + label);
        }

        confidenceText.setText(
                String.format(
                        Locale.US,
                        "Confidence: %.2f%%",
                        confidence * 100f
                )
        );
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (analysisExecutor != null) {
            analysisExecutor.shutdown();
        }
    }

    // =====================================================
    // PERMISSION
    // =====================================================

    private boolean hasCameraPermission() {

        return ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {

        ActivityCompat.requestPermissions(
                this,
                new String[]{
                        Manifest.permission.CAMERA
                },
                REQUEST_CAMERA_PERMISSION
        );
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            @NonNull String[] permissions,
            @NonNull int[] grantResults
    ) {

        super.onRequestPermissionsResult(
                requestCode,
                permissions,
                grantResults
        );

        if (
                requestCode == REQUEST_CAMERA_PERMISSION
                        && grantResults.length > 0
                        && grantResults[0]
                        == PackageManager.PERMISSION_GRANTED
        ) {

            startCamera();

        } else {

            Toast.makeText(
                    this,
                    "Camera permission denied",
                    Toast.LENGTH_LONG
            ).show();
        }
    }
}