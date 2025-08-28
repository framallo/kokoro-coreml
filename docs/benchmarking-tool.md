# 🎤 Kokoro TTS Benchmark & ANE Verification

This guide explains how to run the end‑to‑end Core ML harness, verify Apple Neural Engine (ANE) usage automatically, and interpret latency metrics. It also sketches a macOS Swift app plan for one‑click benchmarking.

## 🚦 Quick Start

```bash
# 1) Ensure Core ML packages exist
ls coreml/
# Expected: kokoro_duration.mlpackage, kokoro_synthesizer_3s.mlpackage

# 2) Create a WAV and print timings (no sudo needed)
python scripts/run_coreml_e2e.py --no-ane-check

# 3) Full run with ANE auto-verifier (requires sudo NOPASSWD for powermetrics)
python scripts/run_coreml_e2e.py
```

Makefile convenience:

```bash
make coreml_e2e
```

## 🧪 What the Harness Does

- Runs stage 1 `kokoro_duration.mlpackage` to produce: `pred_dur`, `d`, `t_en`, `s`
- Builds the alignment matrix on CPU from `pred_dur`
- Runs stage 2 `kokoro_synthesizer_3s.mlpackage` to produce waveform
- Saves `outputs/coreml_e2e.wav`
- Prints per‑stage timings and total RTF
- ANE Auto‑Verifier:
  - Primary: `sudo powermetrics -i 200 --samplers all` → parses lines containing "ANE Power:" (on macOS 14.6, `ane` sampler may not exist)
  - Fallback: Instructions to capture and export an `xctrace` Core ML trace

## 🔧 CLI Flags

```bash
python scripts/run_coreml_e2e.py \
  --text "This is Kokoro running on Apple Neural Engine." \
  --voice af_heart \            # or 'zeros' (default) to avoid HF download
  --repeat 3 \                   # repeat runs for stable timing
  --trace-length 64 \            # override duration trace tokens (pad/truncate)
  --no-ane-check                 # skip powermetrics (useful in CI)
```

Defaults:
- `--text`: "This is Kokoro running on Apple Neural Engine."
- `--voice`: `zeros` (uses zeroed `ref_s` to avoid downloads)
- `--repeat`: 3
- `--out`: `outputs/coreml_e2e.wav`

## 📦 Expected I/O Shapes

- Duration model inputs: `(trace_length,)` for `input_ids`, `attention_mask`; `(256,)` for `ref_s`; `(1,)` for `speed`
- Duration outputs:
  - `pred_dur`: `(trace_length,)` or `(1, trace_length)`
  - `d`, `t_en`: `(1, hidden, trace_length)`
  - `s`: `(1, 128)`
- Synthesizer inputs:
  - `d`, `t_en`: `(1, hidden, trace_length)`
  - `s`: `(1, 128)`
  - `ref_s`: `(1, 256)`
  - `pred_aln_trg`: `(trace_length, frame_count)`
- Waveform output: `(frame_count,)` at 24kHz (3s bucket → 72,000 samples)

> Note: The harness auto‑reads shapes from the synthesizer `.mlpackage` spec.

## ⚡ Interpreting Metrics

- `duration_ms`: Stage‑1 runtime
- `align_ms`: CPU time to build alignment
- `synth_ms`: Stage‑2 runtime (should dominate and use ANE)
- `total_ms`: End‑to‑end time
- `rtf`: Real‑time factor = `total_sec / audio_sec` (<1.0 is faster than real‑time)

## 🔋 ANE Auto‑Verification

### Primary: powermetrics (recommended)

- Configure passwordless sudo for powermetrics:

```bash
# Add a sudoers drop-in (admin users only)
# /etc/sudoers.d/powermetrics (via visudo -f)
%admin ALL=(root) NOPASSWD: /usr/bin/powermetrics
```

- Then run the harness without `--no-ane-check`. It samples "ANE Power" and asserts >0 W for N samples.

### Fallback: Xcode Instruments trace (no sudo)

- Record a short Core ML trace while running the harness:

```bash
xcrun xctrace record --template "Core ML" --time-limit 6s --output outputs/coreml_e2e.trace &
python scripts/run_coreml_e2e.py --no-ane-check --repeat 1
xcrun xctrace export --input outputs/coreml_e2e.trace --output outputs/coreml_e2e.json --format json
```

- Open the JSON and search for "Neural Engine" activity. Presence during Stage‑2 implies ANE usage.

## 🖥️ macOS Swift App Plan (skeleton)

- Xcode project `examples/swift/macos-synth/`
  - Load `coreml/kokoro_duration.mlpackage` and `coreml/kokoro_synthesizer_3s.mlpackage`
  - Use `MLModelConfiguration(computeUnits: .all)`
  - Wrap predictions with `os_signpost` for precise timing
  - Minimal UI: text field, voice selector, Run button, metrics table
  - Provide a shell script to `xctrace` the app for automated ANE verification

> Future work: add the project skeleton; current harness already covers E2E timing on macOS.

## 🧰 Makefile Target

Add to `Makefile`:

```make
coreml_e2e:
	python scripts/run_coreml_e2e.py
```

## 🧪 CI Note

- In CI, run the harness with `--no-ane-check` to avoid sudo. Still prints functional timings and saves the WAV.
- Reserve ANE verification for local bench scripts or release validation.
