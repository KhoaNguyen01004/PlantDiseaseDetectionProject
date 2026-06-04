package com.example;

import android.content.Context;
import android.content.SharedPreferences;

public final class AppSettings {

    public static final String MODE_LIVE = "live";
    public static final String MODE_CAPTURE = "capture";

    public static final String TEXT_NORMAL = "normal";
    public static final String TEXT_LARGE = "large";
    public static final String TEXT_XL = "xl";

    private static final String PREFS_NAME = "agrilens_app_settings";
    private static final String KEY_SCAN_MODE = "scan_mode";
    private static final String KEY_TEXT_SIZE = "text_size";

    private AppSettings() {
    }

    public static String getScanMode(Context context) {
        return prefs(context).getString(KEY_SCAN_MODE, MODE_LIVE);
    }

    public static void setScanMode(Context context, String scanMode) {
        prefs(context).edit().putString(KEY_SCAN_MODE, scanMode).apply();
    }

    public static boolean isLiveMode(Context context) {
        return MODE_LIVE.equals(getScanMode(context));
    }

    public static String getTextSize(Context context) {
        return prefs(context).getString(KEY_TEXT_SIZE, TEXT_LARGE);
    }

    public static void setTextSize(Context context, String textSize) {
        prefs(context).edit().putString(KEY_TEXT_SIZE, textSize).apply();
    }

    public static float getTextScale(Context context) {
        String textSize = getTextSize(context);
        if (TEXT_XL.equals(textSize)) {
            return 1.20f;
        }
        if (TEXT_NORMAL.equals(textSize)) {
            return 1.0f;
        }
        return 1.08f;
    }

    private static SharedPreferences prefs(Context context) {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }
}
