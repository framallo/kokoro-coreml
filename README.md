# kokoro-coreml

**30 seconds of speech in 1.2 seconds. On a Mac Mini. No cloud, no API key, no network.**

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) running natively on the Apple Neural Engine via CoreML. Five compiled models, one Swift pipeline, zero Python at inference time. Every Mac with Apple Silicon is a TTS server.

> **Pre-converted `.mlpackage` files:** [huggingface.co/mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml)

## Why not just use PyTorch MPS?

"Just use the GPU" is the obvious move on Apple Silicon. Two problems:

1. **`aten::angle` doesn't exist on MPS.** The vocoder hits unsupported ops, forcing per-op CPU fallbacks. Every fallback is a round-trip data transfer that kills throughput.
2. **Python is the bottleneck.** Five `model.forward()` calls through the GIL and eager-mode dispatch. The interpreter overhead alone costs more than the M1 Mini's total inference time.

Swift+CoreML eliminates both: models run on the ANE with no fallback, orchestration is native Swift with no Python.

## Performance

An M1 Mac Mini with 16 GB of RAM — the cheapest Apple Silicon Mac you can buy — synthesizes 30 seconds of speech in 1.2 seconds. That's 22x realtime.

| Audio | M1 Mini (16 GB) | M2 Air (24 GB) | M2 Ultra (64 GB) |
| --- | --- | --- | --- |
| 3s | 157 ms | 200 ms | **59 ms** |
| 7s | 511 ms | 326 ms | **136 ms** |
| 15s | 691 ms | 783 ms | **278 ms** |
| 30s | **1,229 ms** | 1,829 ms | **422 ms** |

13-70x realtime across the lineup. The M2 Ultra finishes 30s of audio in 422 ms (70x RT), but the M1 Mini is the number that matters — it proves the pipeline ships on hardware people already own.

### vs alternatives

| Baseline | Speedup |
| --- | --- |
| PyTorch MPS (GPU) | **1.8-3.4x** faster |
| PyTorch CPU | **3.5-7.3x** faster |
| Python+CoreML hybrid | **1.1-2.0x** faster |

Counterbalanced ordering, 5 iterations, warm median. Full data: [bakeoff-results.md](README/Notes/bakeoff-results.md).

## Architecture

Five CoreML models chained with native Swift DSP. Zero Python at inference time.

```
Text --> Phonemes (tokenizer)
    |
    v
Duration CoreML [32/64/128/256/512 tokens] --> pred_dur, d, t_en, s, ref_s
    |
    v
Alignment (Swift) --> one-hot matrix from pred_dur
    |
    v
Matrix ops (Accelerate) --> en = d x alignment, asr = t_en x alignment
    |
    v
F0Ntrain CoreML --> F0_pred, N_pred (pitch + noise contours)
    |
    v
Pad to bucket geometry (Swift)
    |
    +--> DecoderPre CoreML --> x_pre (decoder features)
    |
    +--> hn-nsf (Swift/Accelerate, Double-precision phase) --> har (harmonics)
    |
    v
GeneratorFromHar CoreML --> waveform (24 kHz)
    |
    v
Trim (Swift) --> final audio
```

### Model inventory

| Model | Sizes | Input | Output | Role |
| --- | --- | --- | --- | --- |
| `kokoro_duration_t{N}` | T=32, 64, 128, 256, 512 | input_ids, attention_mask, ref_s, speed | pred_dur, d, t_en, s, ref_s_out | BERT + prosody prediction |
| `kokoro_f0ntrain_t{N}` | T=120, 400, 560, 1200, 2400 | en, s | F0_pred, N_pred | Pitch/noise from aligned features |
| `kokoro_decoder_pre_{N}s` | 3s, 7s, 10s, 15s, 30s | asr, f0, n_input, ref_s | x_pre | Decoder stack (Conv + AdaIN) |
| `kokoro_decoder_har_post_{N}s` | 3s, 7s, 10s, 15s, 30s | x_pre, ref_s, har | waveform | Generator (0 linear ops, all Conv1d for ANE) |

The Duration model uses per-size exports because E5RT (Apple Neural Engine runtime) cannot handle `RangeDim` or `EnumeratedShapes` with multiple variable inputs. The caller pads tokens to the nearest enumeration.

The GeneratorFromHar model has all `nn.Linear` replaced with `nn.Conv1d(kernel_size=1)` in `AdaIN1d`, reducing MIL linear ops from 48 to 0. This keeps the entire generator on the ANE path.

## Quick start

### One-command setup

```bash
bash scripts/setup_bakeoff.sh
```

This handles everything: Python deps, model downloads from HF, all model exports (Duration, F0Ntrain, DecoderPre, GeneratorFromHar), Swift binary build, and benchmark input preparation. Takes ~10 minutes. Use `--skip-download` if models are already local.

### Run the benchmark

```bash
BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python scripts/bakeoff_harness.py run \
  --configs a,d,e,f --iterations 5 --order-seed 0
```

Or use the `$bakeoff` skill — it walks through prerequisites, runs the harness, and updates performance-notes.md.

### Listen to Config F audio (bakeoff inputs)

The benchmark records timings only. To **render and hear** the same Swift+Core ML (Config F) outputs as the bakeoff, generate WAVs locally (not checked into git):

```bash
uv run python scripts/bakeoff_listen.py
```

**Output directory (repo root):** `outputs/bakeoff/listen/`

| Bakeoff key | WAV (24 kHz mono PCM) | Metrics JSON |
| --- | --- | --- |
| `3s` | `outputs/bakeoff/listen/config_f_3s.wav` | `outputs/bakeoff/listen/config_f_3s.json` |
| `7s` | `outputs/bakeoff/listen/config_f_7s.wav` | `outputs/bakeoff/listen/config_f_7s.json` |
| `15s` | `outputs/bakeoff/listen/config_f_15s.wav` | `outputs/bakeoff/listen/config_f_15s.json` |
| `30s` | `outputs/bakeoff/listen/config_f_30s.wav` | `outputs/bakeoff/listen/config_f_30s.json` |

**Play on macOS:** open a file in Finder, or from the repo root run `open outputs/bakeoff/listen/config_f_3s.wav`, or `afplay outputs/bakeoff/listen/config_f_3s.wav`. Prereqs: Swift `kokoro-bench` built and Core ML models present (see setup above); the script runs `prepare_swift_bench_inputs.py` if inputs are missing.

### Manual steps (if you prefer)

<details>
<summary>Individual setup commands</summary>

```bash
# 1. Python deps
uv sync

# 2. Download base models from HF
uv run python scripts/download_models.py --coreml

# 3. Export all models
uv run python export_duration.py
uv run python export_f0ntrain.py --t-frames 120 400 560 1200 2400
uv run python export_decoder_pre.py --buckets 3 7 10 15 30
uv run python -m export_synth.main --buckets "3s,7s,10s,15s,30s" --mode decoder-har --output_dir coreml

# 4. Build Swift binary
cd swift && swift build -c release --product kokoro-bench && cd ..

# 5. Prepare benchmark inputs
uv run python scripts/bakeoff_harness.py prepare-inputs
uv run python scripts/prepare_swift_bench_inputs.py
```

</details>

## Swift Package

The `swift/` directory contains a Swift Package (`KokoroPipeline`) with:

- **`KokoroPipeline.swift`** -- 9-stage orchestrator with per-stage timing
- **`HarmonicSource.swift`** -- hn-nsf in Swift/Accelerate (Double-precision phase accumulator)
- **`AlignmentBuilder.swift`** -- one-hot alignment matrix from phoneme durations
- **`MLMultiArrayHelpers.swift`** -- matrix multiply (cblas_sgemm), zero-padding, stride-safe MLMultiArray ops
- **`BucketSelector.swift`** -- smallest bucket >= ceil(audio_seconds)

```swift
import KokoroPipeline

let pipeline = try KokoroPipeline(
    modelsDirectory: coremlURL,
    buckets: [3, 7, 10, 15, 30],
    linearWeights: hnsfWeights,
    linearBias: hnsfBias
)

let result = try pipeline.synthesize(
    inputIds: tokenIds,
    attentionMask: mask,
    refS: voiceEmbedding,
    speed: 1.0
)

// result.audio: [Float] at 24 kHz
// result.timings: per-stage breakdown
// result.timings.total: end-to-end wall time
```

## Bakeoff configs

| Config | What it measures | Pipeline | Status |
| --- | --- | --- | --- |
| **F** | Swift + CoreML (winner) | 5 CoreML models + Swift hn-nsf DSP | **Production** |
| **A** | Python HAR-post hybrid | PyTorch prefix + CoreML GeneratorFromHar | Baseline |
| **D** | PyTorch MPS (GPU with CPU fallback) | Full PyTorch on MPS device | Comparison |
| **E** | PyTorch CPU | Full PyTorch on CPU | Comparison |
| B/C | ANE participation (diagnostic) | Decoder-only CoreML under .all / .cpuAndGPU | Diagnostic |

Config F is the recommended production path. The bakeoff harness uses a persistent batch subprocess for Config F — models compile once at startup and stay cached across all iterations. See [bakeoff-results.md](README/Notes/bakeoff-results.md) for the full cross-machine matrix.

## Repository structure

```
kokoro-coreml/
  coreml/                          # CoreML .mlpackage files (downloaded from HF)
  swift/                           # Swift Package (KokoroPipeline)
    Sources/KokoroPipeline/        # Pipeline library (production)
    Sources/KokoroBenchmark/       # Benchmark CLI with batch mode + model cache
    Tests/                         # Unit tests + benchmarks
  kokoro/                          # Python TTS library (PyTorch)
    coreml_pipeline.py             # Hybrid PyTorch+CoreML orchestrator
    synthesis_backends.py          # HAR-post / decoder-only backends
  export_duration.py               # Duration model export (enumerated sizes)
  export_f0ntrain.py               # F0Ntrain export
  export_decoder_pre.py            # DecoderPre export
  export_synth/                    # GeneratorFromHar / full synth export
  scripts/
    bakeoff_harness.py             # Controlled benchmark harness (persistent batch subprocess)
    bakeoff_summarize.py           # Results summary tables
    download_models.py             # HF Hub downloader
    prepare_swift_bench_inputs.py  # Pre-tokenize for Swift benchmark
    setup_bakeoff.sh               # One-command setup (deps, models, exports, build)
  README/
    Notes/bakeoff-results.md       # Final v5 cross-machine comparison
    Notes/performance-notes.md     # Full bakeoff history + stage breakdowns
    Notes/debug-notes.md           # Active issues + resolved investigations
    Plans/                         # Implementation plans (bakeoff, ANE opt, Swift rewrite)
    Guides/                        # Apple Silicon / CoreML guides
```

## Documentation

- **Bakeoff results:** [bakeoff-results.md](README/Notes/bakeoff-results.md) -- final corrected v5 cross-machine comparison (M2 Ultra, M2 Air, M1 Mini)
- **Performance data:** [performance-notes.md](README/Notes/performance-notes.md) -- all bakeoff history (v2-v5), stage breakdowns, raw timings
- **Bakeoff plan:** [kokoro-bakeoff-v2.md](README/Plans/kokoro-bakeoff-v2.md) -- benchmark methodology, Phases 0-7
- **ANE optimization:** [ane-optimization-v1.md](README/Plans/ane-optimization-v1.md) -- Linear→Conv1d swap, MIL audit (48→0 linear ops)
- **Swift pipeline plan:** [swift-prefix-rewrite-v1.md](README/Plans/swift-prefix-rewrite-v1.md) -- architecture, export strategy, per-stage validation
- **Debug notes:** [debug-notes.md](README/Notes/debug-notes.md) -- decoder-only quality issues, hn-nsf CoreML failure, M1 Mini OOM workarounds
- **Learnings:** [learnings.md](README/learnings.md) -- historical conversion challenges and solutions
- **CoreML scheduling guide:** [CoreML-Compute-Unit-Scheduling-guide.md](README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)

## Python usage

The original Kokoro Python library works independently of the CoreML pipeline:

```python
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')
for i, (gs, ps, audio) in enumerate(pipeline('Hello world!', voice='af_heart')):
    sf.write(f'{i}.wav', audio, 24000)
```

For MPS GPU acceleration on Apple Silicon:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python your_script.py
```

## Acknowledgements

- [@yl4579](https://huggingface.co/yl4579) for architecting StyleTTS 2
- [@Pendrokar](https://huggingface.co/Pendrokar) for TTS Spaces Arena
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for the open-weight model
- Apple's coremltools team
- Discord: https://discord.gg/QuGxSWBfQy
