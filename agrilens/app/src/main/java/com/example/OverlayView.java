package com.example;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

public class OverlayView extends View {
    private final Paint paint = new Paint();
    private final Paint scrimPaint = new Paint();
    private final Paint cornerPaint = new Paint();
    private final Paint linePaint = new Paint();
    private boolean isDetected = false;

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4f);
        paint.setAntiAlias(true);

        scrimPaint.setColor(Color.argb(88, 17, 20, 15));
        scrimPaint.setStyle(Paint.Style.FILL);

        cornerPaint.setStyle(Paint.Style.STROKE);
        cornerPaint.setStrokeWidth(8f);
        cornerPaint.setStrokeCap(Paint.Cap.ROUND);
        cornerPaint.setAntiAlias(true);

        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(3f);
        linePaint.setStrokeCap(Paint.Cap.ROUND);
        linePaint.setAntiAlias(true);
    }

    public void setResults(boolean detected) {
        this.isDetected = detected;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        float width = getWidth();
        float height = getHeight();
        float horizontalInset = width * 0.12f;
        float verticalInset = height * 0.16f;
        RectF guide = new RectF(
                horizontalInset,
                verticalInset,
                width - horizontalInset,
                height - verticalInset
        );

        canvas.drawRect(0, 0, width, guide.top, scrimPaint);
        canvas.drawRect(0, guide.bottom, width, height, scrimPaint);
        canvas.drawRect(0, guide.top, guide.left, guide.bottom, scrimPaint);
        canvas.drawRect(guide.right, guide.top, width, guide.bottom, scrimPaint);

        int accent = isDetected
                ? Color.rgb(16, 185, 129)
                : Color.rgb(245, 158, 11);

        paint.setColor(Color.argb(140, Color.red(accent), Color.green(accent), Color.blue(accent)));
        canvas.drawRoundRect(guide, 18f, 18f, paint);

        cornerPaint.setColor(accent);
        float corner = Math.min(guide.width(), guide.height()) * 0.18f;

        drawCorner(canvas, guide.left, guide.top, corner, true, true);
        drawCorner(canvas, guide.right, guide.top, corner, false, true);
        drawCorner(canvas, guide.left, guide.bottom, corner, true, false);
        drawCorner(canvas, guide.right, guide.bottom, corner, false, false);

        linePaint.setColor(Color.argb(190, Color.red(accent), Color.green(accent), Color.blue(accent)));
        float scanY = guide.top + guide.height() * 0.58f;
        canvas.drawLine(guide.left + 24f, scanY, guide.right - 24f, scanY, linePaint);
    }

    private void drawCorner(
            Canvas canvas,
            float x,
            float y,
            float length,
            boolean startX,
            boolean startY
    ) {

        float xDirection = startX ? 1f : -1f;
        float yDirection = startY ? 1f : -1f;

        canvas.drawLine(x, y, x + (length * xDirection), y, cornerPaint);
        canvas.drawLine(x, y, x, y + (length * yDirection), cornerPaint);
    }
}
