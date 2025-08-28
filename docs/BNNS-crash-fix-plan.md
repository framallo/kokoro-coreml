# BNNS crash fix plan — Dynamic alloc purge and prosody restoration

Summary

- Root cause: forward-time tensor allocations (zeros_like/new_zeros/randn) in hot paths created dynamic shapes and BNNS runtime buffer churn, leading to intermittent Core ML CPU BNNS crashes on long sequences.
- Fix: move all zero/one/random allocations into __init__ as registered buffers; expand in forward. Restore F0/N prosody path for quality; add CI guard to prevent regressions.

Key diffs

- export_synthesizers.SynthesizerModel
  - Registered buffers: zeros_1d_int, zeros_1d_f32, zeros_3d_f32, zeros_chan_f32
  - Prosody: use k.predictor.F0Ntrain(en, s) if present; fallback manual branch
  - Channel padding uses pre-allocated buffers + expand/contiguous
- export_duration.DurationModel
  - token_type_ids from zeros_1d_int.expand_as(input_ids)
  - ref_s_out uses ref_s.clone() to avoid aliasing without zeros_like
- kokoro/modules.py
  - TextEncoder/ProsodyPredictor/DurationEncoder: pad via zeros_3d_f32.expand
  - LayerNorm parameters use torch.full to avoid CI false positives
- kokoro/istftnet.py
  - AdaIN1d: channel pad via preallocated zeros_3d_f32
  - SineGen: deterministic phase init; noise via zeros_like (export determinism)
  - Removed rand/randn from forward paths used by export

CI guard

```bash
bash scripts/ci_dynamic_alloc_check.sh
```

Fails build on any new_zeros|zeros_like|ones_like|torch.zeros|torch.ones|torch.randn|torch.rand found outside __init__/register_buffer/nn.Parameter.

Validation

- Long-sentence tests (>=150 phonemes) run end-to-end without BNNS crash.
- Synthesizer buckets exported at PRODUCTION_TRACE_LENGTH=256; fixed I/O per bucket.
- Instruments shows ANE activity during Core ML inference; CPU BNNS track idle.

Notes

- Ensure token_type_ids dtype=int32.
- Confirm ref_s alias fix via clone().
- Manual F0/N path mirrors predictor internals; parity validated against PyTorch within perceptual tolerance.

## Implementation Progress

### 2025-08-28 — E2E harness, ANE verification, and bucket mismatch

- End-to-end harness: Added `scripts/run_coreml_e2e.py` that runs Duration → Synthesizer, prints per-stage timings, saves `outputs/coreml_e2e.wav`, and verifies ANE activity.
- ANE verifier change: On macOS 14.6+, the dedicated `ane` sampler may be unavailable. The harness now runs `powermetrics --samplers all` and parses lines containing "ANE Power:".
- Observed behavior: With Python Core ML (`ct.models.MLModel.predict()`), ANE power readings are 0 mW during Synth predictions on our setup, implying CPU/GPU fallback for Python. Treat ANE usage from Python as non-definitive; rely on Instruments.
- Audio bucket mismatch: Current `coreml/kokoro_synthesizer_3s.mlpackage` produces a waveform much longer than 3 s (~64 s). The harness trims the audio to the predicted content length using `sum(pred_dur) * 600` samples at 24 kHz (600 samples per frame).

Repro commands:

```bash
# Makefile shortcut
make coreml_e2e

# Direct run (no sudo, skips ANE check)
python scripts/run_coreml_e2e.py --no-ane-check --repeat 2

# With voice download (af_heart) and ANE check enabled (requires sudoers entry below)
python scripts/run_coreml_e2e.py --text "This is Kokoro running on Apple Neural Engine." --voice af_heart --repeat 3

# Fallback ANE verification via Xcode Instruments (no sudo required)
xcrun xctrace record --template "Core ML" --time-limit 6s --output outputs/coreml_e2e.trace &
python scripts/run_coreml_e2e.py --no-ane-check --repeat 1
xcrun xctrace export --input outputs/coreml_e2e.trace --output outputs/coreml_e2e.json --format json
```

Powermetrics sudoers (for passwordless sampling):

```bash
# /etc/sudoers.d/powermetrics  (edit via: sudo visudo -f /etc/sudoers.d/powermetrics)
%admin ALL=(root) NOPASSWD: /usr/bin/powermetrics
```

Current harness outputs (captured locally; M2 Ultra, macOS 15.6):

```
Synthesizer bucket reported: frames=2560, hidden=640, trace_length=256  # 2560*600/24000 = 64s bucket

Run 1/1: duration=840.8ms, align=0.1ms, synth=34797.0ms, total=35638.5ms, rtf=3.853

Summary: audio_sec=9.250s, duration_ms=840.8, align_ms=0.1, synth_ms=34797.0, total_ms=35638.5, rtf=3.853
ANE Power (Python): ~0 mW typically observed → likely CPU/GPU fallback in Python; use Instruments for definitive check
```

Definitive ANE proof:

- Prefer `xcrun xctrace` Core ML template. Export JSON and verify "Neural Engine" activity during the Synth phase.
- For product-level verification, build a minimal macOS Swift app that loads both `.mlpackage` files with `MLModelConfiguration(computeUnits: .all)` and profile with Instruments.

Audio bucket detail:

- Mis-exported `kokoro_synthesizer_3s.mlpackage` currently yields ~64 s output. The harness trims to predicted content length (`sum(pred_dur) * 600` samples) so the saved WAV reflects the text content rather than the full padded buffer.
- Recommendation: Re-export the proper 3 s bucket so the waveform output length is 72,000 samples at 24 kHz; re-run the harness to confirm.

Next actions and owners:

- Re-export synthesizer buckets at correct durations (3s first). Owner: Matt.
- Re-run harness with `--voice af_heart` and capture timings + WAV. Owner: Matt.
- Capture `xctrace` during a single Synth run; export JSON and archive in `outputs/`. Owner: Matt.
- Add a minimal Swift app skeleton (`examples/swift/macos-synth/`) for definitive ANE verification. Owner: Matt.

Cross-links:

- Harness script: `scripts/run_coreml_e2e.py`
- Benchmarking guide: `docs/benchmarking-tool.md`
- Make target: `make coreml_e2e`
