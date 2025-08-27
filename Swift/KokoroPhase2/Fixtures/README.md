Fixtures for Phase 2 Swift app.

Use the Python helper to export a single test fixture (JSON) containing flattened tensors and shapes for the 5s decoder-only model inputs (asr, f0_curve, n, s). The Swift CLI will read this file and run the Core ML model, writing a WAV to outputs/local/phase2/.
