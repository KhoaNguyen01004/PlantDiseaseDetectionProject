# .gitignore Rewrite Task TODO

## Steps:
1. [✅] Create TODO.md with plan steps.
2. [✅] Rewrite SourceCode/.gitignore for Python/ML generated data/venv/etc.
3. [✅] Rewrite AndroidApp/.gitignore for Android root builds/gradle.
4. [✅] Rewrite AndroidApp/app/.gitignore for Android module builds.
5. [✅] Update TODO.md with completion status.
6. [ ] Verify with `git status` and commit changes.

## Summary of Changes:
- **SourceCode/**: Python/ML best practices - ignores venvs (py3_10/, venv/), data/, models/*.tflite/*.h5/*.onnx/*.pth, plant_model_tflite/, notebooks checkpoints, caches, logs, IDE.
- **AndroidApp/**: Full Gradle/Android Studio root - .gradle/, local.properties, build/, captures/, .idea/.
- **AndroidApp/app/**: Module-specific - build/ variants, .externalNativeBuild, lint/, keystores.

## Verification Commands:
```
git status
git add TODO.md SourceCode/.gitignore AndroidApp/.gitignore AndroidApp/app/.gitignore
git commit -m "chore: rewrite .gitignore - ignore generated data, venvs, builds, IDE/logs"
```

Project now ignores all runtime/generated/unneeded files while keeping source/configs for running the code.

