package com.example;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "AgriLens";

    private static final int REQUEST_CAMERA_PERMISSION = 100;

    private static final long LIVE_ANALYSIS_INTERVAL_MS = 900L;

    private PreviewView previewView;

    private ImageView imagePreview;

    private ImageCapture imageCapture;

    private TextView resultText;

    private TextView confidenceText;

    private TextView topPredictionsText;

    private ProgressBar progressBar;

    private Button btnGallery;

    private Button btnGuide;

    private Button btnSettings;

    private Button btnCapture;

    private Button btnScanAgain;

    private Button btnEnglish;

    private Button btnVietnamese;

    private Button btnModeLive;

    private Button btnModeCapture;

    private Button btnTextNormal;

    private Button btnTextLarge;

    private Button btnTextXL;

    private TextView scannerStatusText;

    private TextView scanModeBadge;

    private TextView modeExplanationText;

    private ProgressBar confidenceBar;

    private LinearLayout settingsPanel;

    private LinearLayout treatmentPanel;

    private TextView treatmentTitleText;

    private TextView organicTreatmentText;

    private TextView conventionalTreatmentText;

    private OverlayView overlayView;

    private ImageClassifierHelper classifierHelper;

    private ExecutorService analysisExecutor;

    private long lastLiveAnalysisTimeMs = 0L;

    private volatile boolean isProcessingGalleryImage = false;

    private volatile boolean isLiveDetectionPaused = false;

    @Override
    protected void attachBaseContext(Context newBase) {
        super.attachBaseContext(LanguageManager.apply(newBase));
    }

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

                                Toast.makeText(
                                        MainActivity.this,
                                        R.string.toast_gallery_error,
                                        Toast.LENGTH_SHORT
                                ).show();
                            }
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        hideSystemBars();

        initViews();

        applyTextScale(findViewById(R.id.rootScroll), AppSettings.getTextScale(this));

        analysisExecutor = Executors.newSingleThreadExecutor();

        classifierHelper =
                new ImageClassifierHelper(this);

        if (hasCameraPermission()) {

            startCamera();

        } else {

            requestCameraPermission();
        }

        btnGallery.setOnClickListener(
                v -> openGallery()
        );

        btnGuide.setOnClickListener(
                v -> openGuide()
        );

        btnSettings.setOnClickListener(
                v -> toggleSettingsPanel()
        );

        btnCapture.setOnClickListener(
                v -> captureImage()
        );

        btnScanAgain.setOnClickListener(
                v -> resumeLiveScanning()
        );

        btnEnglish.setOnClickListener(
                v -> changeLanguage(LanguageManager.ENGLISH)
        );

        btnVietnamese.setOnClickListener(
                v -> changeLanguage(LanguageManager.VIETNAMESE)
        );

        btnModeLive.setOnClickListener(
                v -> changeScanMode(AppSettings.MODE_LIVE)
        );

        btnModeCapture.setOnClickListener(
                v -> changeScanMode(AppSettings.MODE_CAPTURE)
        );

        btnTextNormal.setOnClickListener(
                v -> changeTextSize(AppSettings.TEXT_NORMAL)
        );

        btnTextLarge.setOnClickListener(
                v -> changeTextSize(AppSettings.TEXT_LARGE)
        );

        btnTextXL.setOnClickListener(
                v -> changeTextSize(AppSettings.TEXT_XL)
        );

        updateLanguageButtons();
        updateSettingsButtons();
        resetScannerStatus();
        updateActionButtons();
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

        topPredictionsText =
                findViewById(R.id.topPredictionsText);

        progressBar =
                findViewById(R.id.progressBar);

        btnGallery =
                findViewById(R.id.btnGallery);

        btnGuide =
                findViewById(R.id.btnGuide);

        btnSettings =
                findViewById(R.id.btnSettings);

        btnCapture =
                findViewById(R.id.btnCapture);

        btnScanAgain =
                findViewById(R.id.btnScanAgain);

        btnEnglish =
                findViewById(R.id.btnEnglish);

        btnVietnamese =
                findViewById(R.id.btnVietnamese);

        btnModeLive =
                findViewById(R.id.btnModeLive);

        btnModeCapture =
                findViewById(R.id.btnModeCapture);

        btnTextNormal =
                findViewById(R.id.btnTextNormal);

        btnTextLarge =
                findViewById(R.id.btnTextLarge);

        btnTextXL =
                findViewById(R.id.btnTextXL);

        scannerStatusText =
                findViewById(R.id.scannerStatusText);

        scanModeBadge =
                findViewById(R.id.scanModeBadge);

        modeExplanationText =
                findViewById(R.id.modeExplanationText);

        confidenceBar =
                findViewById(R.id.confidenceBar);

        settingsPanel =
                findViewById(R.id.settingsPanel);

        treatmentPanel =
                findViewById(R.id.treatmentPanel);

        treatmentTitleText =
                findViewById(R.id.treatmentTitleText);

        organicTreatmentText =
                findViewById(R.id.organicTreatmentText);

        conventionalTreatmentText =
                findViewById(R.id.conventionalTreatmentText);

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
                                .setTargetRotation(previewView.getDisplay().getRotation())
                                .build();

                ImageAnalysis imageAnalysis =
                        new ImageAnalysis.Builder()
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .setTargetRotation(previewView.getDisplay().getRotation())
                                .build();

                imageAnalysis.setAnalyzer(analysisExecutor, imageProxy -> {
                    try {
                        if (
                                isProcessingGalleryImage
                                        || isLiveDetectionPaused
                                        || !AppSettings.isLiveMode(this)
                        ) {
                            return;
                        }

                        long now = System.currentTimeMillis();
                        if (now - lastLiveAnalysisTimeMs < LIVE_ANALYSIS_INTERVAL_MS) {
                            return;
                        }
                        lastLiveAnalysisTimeMs = now;

                        Bitmap bitmap = imageProxy.toBitmap();
                        if (bitmap != null) {
                            classifierHelper.classify(bitmap, new ImageClassifierHelper.ClassificationListener() {
                                @Override
                                public void onClassificationResult(
                                        String label,
                                        float confidence,
                                        List<ImageClassifierHelper.Prediction> topPredictions
                                ) {
                                    runOnUiThread(() -> {
                                        boolean detected = isKnownLabel(label);
                                        overlayView.setResults(detected);
                                        imagePreview.setImageBitmap(bitmap);
                                        displayResult(label, confidence, topPredictions, true);
                                        if (detected) {
                                            pauseLiveScanning();
                                        }
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

    private void captureImage() {

        if (imageCapture == null) {
            Toast.makeText(
                    this,
                    R.string.toast_capture_failed,
                    Toast.LENGTH_SHORT
            ).show();
            return;
        }

        progressBar.setVisibility(View.VISIBLE);
        scannerStatusText.setText(R.string.status_classifying);
        btnCapture.setEnabled(false);

        File photoFile =
                new File(
                        getCacheDir(),
                        "agrilens_capture.jpg"
                );

        ImageCapture.OutputFileOptions outputOptions =
                new ImageCapture.OutputFileOptions.Builder(photoFile)
                        .build();

        imageCapture.takePicture(
                outputOptions,
                ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        Bitmap bitmap =
                                BitmapFactory.decodeFile(photoFile.getAbsolutePath());

                        if (bitmap == null) {
                            progressBar.setVisibility(View.GONE);
                            btnCapture.setEnabled(true);
                            Toast.makeText(
                                    MainActivity.this,
                                    R.string.toast_capture_failed,
                                    Toast.LENGTH_SHORT
                            ).show();
                            return;
                        }

                        processBitmap(bitmap);
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "Capture failed", exception);
                        progressBar.setVisibility(View.GONE);
                        btnCapture.setEnabled(true);
                        Toast.makeText(
                                MainActivity.this,
                                R.string.toast_capture_failed,
                                Toast.LENGTH_SHORT
                        ).show();
                    }
                }
        );
    }

    private void openGuide() {

        Intent intent =
                new Intent(
                        this,
                        GuideActivity.class
                );

        startActivity(intent);
    }

    // =====================================================
    // CLASSIFICATION
    // =====================================================

    private void processBitmap(Bitmap bitmap) {

        if (bitmap == null) {
            return;
        }

        imagePreview.setImageBitmap(bitmap);

        progressBar.setVisibility(View.VISIBLE);

        scannerStatusText.setText(R.string.status_classifying);

        resultText.setText(R.string.status_classifying);

        confidenceText.setText(R.string.confidence_waiting);

        confidenceBar.setProgress(0);

        treatmentPanel.setVisibility(View.GONE);

        setControlsEnabled(false);
        isProcessingGalleryImage = true;
        isLiveDetectionPaused = true;

        analysisExecutor.execute(() ->
                classifierHelper.classify(
                        bitmap,

                        new ImageClassifierHelper.ClassificationListener() {

                            @Override
                            public void onClassificationResult(
                                    String label,
                                    float confidence,
                                    List<ImageClassifierHelper.Prediction> topPredictions
                            ) {

                                runOnUiThread(() -> {

                                    progressBar.setVisibility(View.GONE);

                                    setControlsEnabled(true);
                                    isProcessingGalleryImage = false;

                                    displayResult(
                                            label,
                                            confidence,
                                            topPredictions,
                                            false
                                    );
                                    pauseLiveScanning();
                                });
                            }

                            @Override
                            public void onClassificationError(
                                    String errorMessage
                            ) {

                                runOnUiThread(() -> {

                                    progressBar.setVisibility(View.GONE);

                                    setControlsEnabled(true);
                                    isProcessingGalleryImage = false;
                                    resumeLiveScanning();

                                    scannerStatusText.setText(R.string.status_error);

                                    resultText.setText(R.string.status_error);

                                    Toast.makeText(
                                            MainActivity.this,
                                            errorMessage,
                                            Toast.LENGTH_LONG
                                    ).show();
                                });
                            }
                        }
                )
        );
    }

    private void displayResult(
            String label,
            float confidence,
            List<ImageClassifierHelper.Prediction> topPredictions,
            boolean isLive
    ) {

        if (label == null) {
            label = "Unknown";
        }

        String lower =
                label.toLowerCase(Locale.US);
        String languageCode =
                LanguageManager.getLanguage(this);
        String displayLabel =
                DiseaseTreatmentRepository.getDisplayName(label, languageCode);

        if (lower.contains("unknown")) {
            resultText.setTextColor(ContextCompat.getColor(this, R.color.accent_red));
            resultText.setText(R.string.result_unknown);
            scannerStatusText.setText(
                    isLive
                            ? R.string.status_unknown_keep_scanning
                            : R.string.status_unknown_review
            );
            if (isLive) {
                treatmentPanel.setVisibility(View.GONE);
            } else {
                showTreatment(label);
            }
        } else if (lower.contains("healthy")) {
            resultText.setTextColor(ContextCompat.getColor(this, R.color.accent_green_dark));
            resultText.setText(getString(R.string.result_healthy_format, displayLabel));
            if (isLive) {
                scannerStatusText.setText(R.string.status_detection_paused);
            } else {
                scannerStatusText.setText(R.string.status_healthy_paused);
            }
            showTreatment(label);
        } else {
            resultText.setTextColor(ContextCompat.getColor(this, R.color.accent_amber));
            resultText.setText(getString(R.string.result_disease_format, displayLabel));
            if (isLive) {
                scannerStatusText.setText(R.string.status_detection_paused);
            } else {
                scannerStatusText.setText(R.string.status_gallery_done);
            }
            showTreatment(label);
        }

        confidenceText.setText(
                getString(
                        R.string.confidence_format,
                        confidence * 100f
                )
        );

        confidenceBar.setProgress(
                Math.round(confidence * 100f)
        );

        topPredictionsText.setText(formatTopPredictions(topPredictions));

    }

    private String formatTopPredictions(List<ImageClassifierHelper.Prediction> topPredictions) {

        if (topPredictions == null || topPredictions.isEmpty()) {
            return getString(R.string.explanation_waiting);
        }

        String languageCode = LanguageManager.getLanguage(this);
        StringBuilder builder = new StringBuilder(getString(R.string.explanation_top3_prefix));

        for (int i = 0; i < topPredictions.size(); i++) {
            ImageClassifierHelper.Prediction prediction = topPredictions.get(i);
            if (i > 0) {
                builder.append("  |  ");
            }
            builder.append(DiseaseTreatmentRepository.getDisplayName(prediction.label, languageCode));
            builder.append(" ");
            builder.append(String.format(Locale.US, "%.1f%%", prediction.confidence * 100f));
        }

        return builder.toString();
    }

    private boolean isKnownLabel(String label) {

        if (label == null) {
            return false;
        }

        return !label.toLowerCase(Locale.US).contains("unknown");
    }

    private void pauseLiveScanning() {

        isLiveDetectionPaused = true;
        progressBar.setVisibility(View.GONE);
        updateActionButtons();
    }

    private void resumeLiveScanning() {

        isLiveDetectionPaused = false;
        updateActionButtons();
        resetScannerStatus();
        resultText.setTextColor(ContextCompat.getColor(this, R.color.text_primary));
        resultText.setText(R.string.result_waiting);
        confidenceText.setText(R.string.confidence_waiting);
        confidenceBar.setProgress(0);
        topPredictionsText.setText(R.string.explanation_waiting);
        treatmentPanel.setVisibility(View.GONE);
        overlayView.setResults(false);
    }

    private void showTreatment(String label) {

        DiseaseTreatmentRepository.Treatment treatment =
                DiseaseTreatmentRepository.getTreatment(
                        label,
                        LanguageManager.getLanguage(this)
                );

        treatmentPanel.setVisibility(View.VISIBLE);

        if (treatment == null) {

            treatmentTitleText.setText(R.string.treatment_title_generic);

            organicTreatmentText.setText(R.string.treatment_unavailable);

            conventionalTreatmentText.setText("");

            return;
        }

        treatmentTitleText.setText(
                treatment.diseaseName
        );

        organicTreatmentText.setText(
                getString(R.string.treatment_organic)
                        + "\n"
                        + sanitizeTreatmentText(treatment.organicTreatment)
        );

        conventionalTreatmentText.setText(
                getString(R.string.treatment_conventional)
                        + "\n"
                        + sanitizeTreatmentText(treatment.conventionalTreatment)
        );
    }

    private String sanitizeTreatmentText(String text) {

        if (text == null) {
            return "";
        }

        return text.replace("\u00e2\u20ac\u00a2", "-");
    }

    private void setControlsEnabled(boolean enabled) {

        btnGallery.setEnabled(enabled);
        btnGuide.setEnabled(enabled);
        btnSettings.setEnabled(enabled);
        btnCapture.setEnabled(enabled);
        btnScanAgain.setEnabled(enabled);
        btnEnglish.setEnabled(enabled);
        btnVietnamese.setEnabled(enabled);
        btnModeLive.setEnabled(enabled);
        btnModeCapture.setEnabled(enabled);
        btnTextNormal.setEnabled(enabled);
        btnTextLarge.setEnabled(enabled);
        btnTextXL.setEnabled(enabled);

        if (enabled) {
            updateLanguageButtons();
            updateSettingsButtons();
        }
    }

    private void toggleSettingsPanel() {

        settingsPanel.setVisibility(
                settingsPanel.getVisibility() == View.VISIBLE
                        ? View.GONE
                        : View.VISIBLE
        );
    }

    private void changeScanMode(String scanMode) {

        if (scanMode.equals(AppSettings.getScanMode(this))) {
            return;
        }

        AppSettings.setScanMode(this, scanMode);
        resumeLiveScanning();
        updateSettingsButtons();
    }

    private void changeTextSize(String textSize) {

        if (textSize.equals(AppSettings.getTextSize(this))) {
            return;
        }

        AppSettings.setTextSize(this, textSize);
        recreate();
    }

    private void updateActionButtons() {

        if (isLiveDetectionPaused) {
            btnScanAgain.setVisibility(View.VISIBLE);
            btnCapture.setVisibility(View.GONE);
            btnGallery.setVisibility(View.GONE);
            return;
        }

        btnScanAgain.setVisibility(View.GONE);

        if (AppSettings.isLiveMode(this)) {
            btnCapture.setVisibility(View.GONE);
            btnGallery.setVisibility(View.VISIBLE);
        } else {
            btnCapture.setVisibility(View.VISIBLE);
            btnGallery.setVisibility(View.VISIBLE);
        }
    }

    private void resetScannerStatus() {

        if (AppSettings.isLiveMode(this)) {
            scannerStatusText.setText(R.string.status_ready_live);
            scanModeBadge.setText(R.string.live_scan);
        } else {
            scannerStatusText.setText(R.string.status_ready_capture);
            scanModeBadge.setText(R.string.capture_scan);
        }
    }

    private void updateSettingsButtons() {

        boolean isLive = AppSettings.isLiveMode(this);
        btnModeLive.setEnabled(!isLive);
        btnModeCapture.setEnabled(isLive);
        btnModeLive.setAlpha(isLive ? 0.55f : 1f);
        btnModeCapture.setAlpha(isLive ? 1f : 0.55f);
        modeExplanationText.setText(
                getString(R.string.mode_live_description)
                        + "\n\n"
                        + getString(R.string.mode_capture_description)
        );

        String textSize = AppSettings.getTextSize(this);
        updateTextSizeButton(btnTextNormal, AppSettings.TEXT_NORMAL.equals(textSize));
        updateTextSizeButton(btnTextLarge, AppSettings.TEXT_LARGE.equals(textSize));
        updateTextSizeButton(btnTextXL, AppSettings.TEXT_XL.equals(textSize));
    }

    private void updateTextSizeButton(Button button, boolean selected) {

        button.setEnabled(!selected);
        button.setAlpha(selected ? 0.55f : 1f);
    }

    private void changeLanguage(String languageCode) {

        if (languageCode.equals(LanguageManager.getLanguage(this))) {
            return;
        }

        LanguageManager.setLanguage(this, languageCode);
        recreate();
    }

    private void updateLanguageButtons() {

        String language = LanguageManager.getLanguage(this);
        boolean isEnglish = LanguageManager.ENGLISH.equals(language);

        btnEnglish.setEnabled(!isEnglish);
        btnVietnamese.setEnabled(isEnglish);
        btnEnglish.setAlpha(isEnglish ? 0.55f : 1f);
        btnVietnamese.setAlpha(isEnglish ? 1f : 0.55f);
    }

    private void applyTextScale(View view, float scale) {

        if (view instanceof TextView) {
            TextView textView = (TextView) view;
            float viewScale = view instanceof Button ? Math.min(scale, 1.05f) : scale;
            textView.setTextSize(
                    TypedValue.COMPLEX_UNIT_PX,
                    textView.getTextSize() * viewScale
            );
        }

        if (view instanceof ViewGroup) {
            ViewGroup group = (ViewGroup) view;
            for (int i = 0; i < group.getChildCount(); i++) {
                applyTextScale(group.getChildAt(i), scale);
            }
        }
    }

    private void hideSystemBars() {

        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        );
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            hideSystemBars();
        }
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
                    R.string.toast_camera_denied,
                    Toast.LENGTH_LONG
            ).show();
        }
    }
}
