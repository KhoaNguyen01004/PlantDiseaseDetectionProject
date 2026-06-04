package com.example;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Configuration;

import java.util.Locale;

public final class LanguageManager {

    public static final String ENGLISH = "en";
    public static final String VIETNAMESE = "vi";

    private static final String PREFS_NAME = "agrilens_language";
    private static final String KEY_LANGUAGE = "language";

    private LanguageManager() {
    }

    public static Context apply(Context context) {
        String language = getLanguage(context);
        Locale locale = new Locale(language);
        Locale.setDefault(locale);

        Configuration configuration = new Configuration(context.getResources().getConfiguration());
        configuration.setLocale(locale);
        return context.createConfigurationContext(configuration);
    }

    public static String getLanguage(Context context) {
        SharedPreferences preferences =
                context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        return preferences.getString(KEY_LANGUAGE, ENGLISH);
    }

    public static void setLanguage(Context context, String language) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(KEY_LANGUAGE, language)
                .apply();
    }
}
