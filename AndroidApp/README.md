# AgriLens Android App

AgriLens is the mobile frontend for Plant Disease Detection project. Currently basic skeleton; future integration with TFLite model from `../SourceCode`.

## Structure
```
AndroidApp/
├── app/
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/example/agrilens/
│       │   ├── MainActivity.kt
│       │   └── ui/theme/ (Color.kt, Theme.kt, Type.kt)
│       └── res/ (drawables, mipmap icons, values/colors/strings/themes.xml)
│       └── xml/ (backup/data_extraction_rules.xml)
│   ├── test/ (unit tests)
│   └── androidTest/ (instrumented tests)
├── build.gradle.kts (root)
├── app/build.gradle.kts (module)
├── gradle.properties
├── settings.gradle.kts
└── gradlew(.bat)
```

## Setup & Run
1. Android Studio: Open folder or `code .`
2. `./gradlew build` or Studio Build > Make Project.
3. Run on emulator/device: `./gradlew installDebug` or Studio Run.

## Future
- Integrate plant_model.tflite from SourceCode.
- Camera input -> ML inference -> Disease UI.
- Update README as features added.

See root README.md for full project.

