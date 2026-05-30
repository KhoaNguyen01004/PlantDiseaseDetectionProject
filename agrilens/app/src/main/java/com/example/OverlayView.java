package com.example;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

public class OverlayView extends View {
    private Paint paint = new Paint();
    private boolean isDetected = false;

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(10f);
        paint.setAntiAlias(true);
    }

    public void setResults(boolean detected) {
        this.isDetected = detected;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        int width = getWidth();
        int height = getHeight();
        
        // Define the circular area in the center
        float size = Math.min(width, height) * 0.7f;
        float centerX = width / 2f;
        float centerY = height / 2f;
        float radius = size / 2f;

        if (isDetected) {
            paint.setColor(Color.GREEN);
        } else {
            paint.setColor(Color.RED);
            // Draw a subtle "scan" area even when not detected
            paint.setAlpha(100);
        }

        // Draw the "circle the box" - a circular guidance overlay
        canvas.drawCircle(centerX, centerY, radius, paint);
        
        // Draw corners or crosshairs to make it look like a scanner
        paint.setAlpha(255);
        float lineLength = radius * 0.2f;
        // Optional: add some "scanning" aesthetics
    }
}
