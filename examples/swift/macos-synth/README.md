# macOS Kokoro Synth App (Plan)

Skeleton plan for a macOS benchmarking app that loads Kokoro Core ML models, runs synthesis, measures timings, and verifies ANE usage.

## Features
- Load `coreml/kokoro_duration.mlpackage` and `coreml/kokoro_synthesizer_3s.mlpackage`
- Configure with `MLModelConfiguration(computeUnits: .all)`
- Provide text input, optional voice selector (future), Run button
- Show per-stage timings and total RTF
- Wrap prediction calls with `os_signpost` to measure durations
- Include a `bench.sh` script that launches the app and records an Instruments trace via `xcrun xctrace` for ANE verification

## Next Steps
- Create Xcode project targeting macOS 13+
- Add Swift package for a small UI (SwiftUI)
- Implement model loading and prediction bridging code
- Export signpost categories and hook them up to Instruments templates
