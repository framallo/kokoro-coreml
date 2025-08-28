# Plan for iOS Inference Tester App

This document outlines the steps to create a simple iOS application for testing the inference speed of the Kokoro Core ML models on an iPhone.

## 1. Project Setup

*   **New Xcode Project:** The simplest approach is for you to create a new, empty SwiftUI iOS App project in Xcode named `Kokoro-iOS-Tester` and save it inside the `Swift/` directory.
*   **Add Package Dependency:** You will then add a local package dependency pointing to the `Swift/KokoroPhase2` directory. Xcode makes this straightforward.

## 2. Asset Bundling (Folder‑Synced)

Assets are located under `Swift/Kokoro-iOS-Tester/Resources` and are synchronized into the app bundle via Xcode's file‑system synchronized group. Do not drag these files into Xcode or add explicit Copy Bundle Resources entries.

- `Resources/kokoro_decoder_only_5s.mlpackage`
- `Resources/fixture_hi.json`
- `Resources/fixture_har_5s.json`

Notes:
- Xcode compiles `.mlpackage` into `kokoro_decoder_only_5s.mlmodelc` at build time; only the compiled `.mlmodelc` is typically present in the app bundle.
- The JSON fixtures are copied as‑is to the bundle root.

## 3. Application Logic

*   **ViewModel:** A new `InferenceViewModel.swift` class will be created to manage the state and logic.
    *   It will be responsible for loading the `DecoderOnly5sRunner` from the bundled model.
    *   It will expose `@Published` properties for displaying inference time and status messages to the SwiftUI view.
*   **Warm-up:**
    *   At app launch, the `InferenceViewModel` will be initialized.
    *   It will immediately run a single inference using the short "hi" fixture. This is a crucial step to warm up the model and the Apple Neural Engine, ensuring our measurements are accurate. The audio result of this run will be discarded.
*   **Main Inference Test:**
    *   The UI will have a button labeled "Run Inference Test".
    *   Tapping the button will trigger the main test run using the standard 5-second fixture.
    *   The view model will use the same timing logic as the command-line tool to measure the inference latency.
    *   The measured latency will be published to the UI for display.
    *   The audio will be played on the device speaker as soon as it's ready (0.35 sec buffer)

## 4. User Interface (`ContentView.swift`)

*   The UI will be ruthlessly simple:
    *   A text label to display the status (e.g., "Ready", "Warming up...", "Running test...").
    *   A text label to display the result of the last inference test (e.g., "Inference time: 123 ms").
    *   The "Run Inference Test" button.

This plan focuses on reusing the battle-tested `KokoroPhase2` library to get us a reliable performance number with minimal fuss.

## Quick Run (Xcode 16+)

1. Open `Swift/Kokoro-iOS-Tester/Kokoro-iOS-Tester.xcodeproj` in Xcode.
2. Product ▶︎ Clean Build Folder (Shift+Cmd+K).
3. File ▶︎ Packages ▶︎ Resolve Package Versions.
4. Select a physical iOS device as the run destination.
5. Build & Run.

Expected bundle contents (inspect via Products ▶︎ Show in Finder):
- `fixture_hi.json`, `fixture_har_5s.json`
- `kokoro_decoder_only_5s.mlmodelc/` (compiled model)

## Troubleshooting

- Model or fixtures duplicated in bundle
  - Remove any explicit files under target ▶︎ Build Phases ▶︎ Copy Bundle Resources. We rely on the folder‑synced `Resources` only.

- Fixture not found at runtime (e.g., `Fixture ... not found in bundle`)
  - Ensure `Swift/Kokoro-iOS-Tester/Resources` exists on disk and is visible in the Xcode Project navigator.
  - Confirm the target lists the `Resources` group under `fileSystemSynchronizedGroups` (already configured in the project).

- Model not found at runtime
  - Xcode bundles the compiled model as `kokoro_decoder_only_5s.mlmodelc`. If your code searches for `.mlpackage`, add a fallback to look for `.mlmodelc` or pass the compiled URL directly to `MLModel(contentsOf:)` inside the runner.

- Xcode version differences
  - Some Xcode versions treat file‑system synchronized groups differently. If resources do not appear in the app bundle, re‑add `Resources` via Add Files… and keep it as a folder (not individual files). Avoid adding explicit entries to Copy Bundle Resources.

