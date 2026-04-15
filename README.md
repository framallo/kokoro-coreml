# kokoro-coreml

A production-ready PyTorch-to-CoreML conversion pipeline for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), enabling on-device text-to-speech on Apple Silicon with Apple Neural Engine acceleration.

> **Pre-converted `.mlpackage` files:** [huggingface.co/mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml)

## Performance (M2 Ultra)

The Swift+CoreML pipeline is **2.3-3.6x faster than PyTorch MPS** (GPU) and **43-79x realtime**:

| Audio length | PyTorch MPS | Swift+CoreML | Speedup vs MPS | Realtime factor |
| --- | --- | --- | --- | --- |
| 3s | 171 ms | **65 ms** | **2.6x** | 43x RT |
| 7s | 320 ms | **142 ms** | **2.3x** | 48x RT |
| 15s | 611 ms | **254 ms** | **2.4x** | 55x RT |
| 30s | 1247 ms | **349 ms** | **3.6x** | 79x RT |

Measured with counterbalanced ordering, 5 repetitions, warm median. Full results: `README/Notes/performance-notes.md`.

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
| `kokoro_decoder_har_post_{N}s` | 3s, 7s, 10s, 15s, 30s | x_pre, ref_s, har | waveform | Generator (ANE-optimized) |

The Duration model uses per-size exports because E5RT (Apple Neural Engine runtime) cannot handle `RangeDim` or `EnumeratedShapes` with multiple variable inputs. The caller pads tokens to the nearest enumeration.

### Why not just use MPS?

PyTorch MPS ("just use the GPU") is the obvious first choice on Apple Silicon, but it has two problems:

1. **`aten::angle` CPU fallback.** The vocoder's iSTFTNet uses operations that MPS doesn't support, forcing per-op fallbacks to CPU. This kills throughput with data transfers.
2. **Python interpreter overhead.** Every `model.forward()` call goes through Python's GIL and eager-mode dispatch. For a 5-model pipeline, this adds up.

The Swift+CoreML path solves both: models run on the ANE (no fallback), and orchestration is native Swift (no Python).

## Quick start

### 1. Download models

```bash
# From Hugging Face Hub
python scripts/download_models.py --coreml
```

### 2. Export models (if not using pre-built)

```bash
# Duration models (all enumerated token sizes)
uv run python export_duration.py

# HAR-post pipeline models (all bucket sizes)
uv run python -m export_synth.main --buckets "3s,7s,10s,15s,30s" --mode decoder-har --output_dir coreml

# F0Ntrain models
uv run python export_f0ntrain.py --t-frames 120 400 560 1200 2400

# DecoderPre models
uv run python export_decoder_pre.py --buckets 3 7 10 15 30
```

### 3. Build Swift pipeline

```bash
cd swift
swift build -c release --product kokoro-bench
```

### 4. Run benchmark

```bash
# Prepare inputs
uv run python scripts/prepare_swift_bench_inputs.py

# Run bakeoff (all configs, 5 iterations)
BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python scripts/bakeoff_harness.py run \
  --configs a,d,e,f --iterations 5 --order-seed 0
```

Or use the `$bakeoff` skill: it handles prerequisites, runs the harness, and updates performance-notes.md.

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

| Config | What it measures | Pipeline |
| --- | --- | --- |
| **A** | Python HAR-post hybrid | PyTorch prefix + CoreML GeneratorFromHar |
| **D** | PyTorch MPS (GPU with CPU fallback) | Full PyTorch on MPS device |
| **E** | PyTorch CPU | Full PyTorch on CPU |
| **F** | Swift + CoreML | 5 CoreML models + Swift hn-nsf DSP |
| B/C | ANE participation (diagnostic) | Decoder-only CoreML under .all / .cpuAndGPU |

## Repository structure

```
kokoro-coreml/
  coreml/                          # CoreML .mlpackage files (downloaded from HF)
  swift/                           # Swift Package (KokoroPipeline)
    Sources/KokoroPipeline/        # Pipeline library
    Sources/KokoroBenchmark/       # Benchmark CLI (kokoro-bench)
    Tests/                         # Unit tests + benchmarks
  kokoro/                          # Python TTS library (PyTorch)
    coreml_pipeline.py             # Hybrid PyTorch+CoreML orchestrator
    synthesis_backends.py          # HAR-post / decoder-only backends
  export_duration.py               # Duration model export (enumerated sizes)
  export_f0ntrain.py               # F0Ntrain export
  export_decoder_pre.py            # DecoderPre export
  export_synth/                    # GeneratorFromHar / full synth export
  scripts/
    bakeoff_harness.py             # Controlled benchmark harness
    bakeoff_summarize.py           # Results summary tables
    download_models.py             # HF Hub downloader
    prepare_swift_bench_inputs.py  # Pre-tokenize for Swift benchmark
  README/
    Notes/performance-notes.md     # All benchmark results
    Plans/                         # Implementation plans
    Guides/                        # Apple Silicon / CoreML guides
```

## Documentation

- **Performance data:** `README/Notes/performance-notes.md` -- all bakeoff results, stage breakdowns, cross-machine comparisons
- **Bakeoff plan:** `README/Plans/kokoro-bakeoff-v2.md` -- benchmark methodology, Phases 0-7
- **Swift pipeline plan:** `README/Plans/swift-prefix-rewrite-v1.md` -- architecture, export strategy, per-stage validation
- **Debug notes:** `README/Notes/debug-notes.md` -- decoder-only quality issues, hn-nsf CoreML failure
- **Learnings:** `README/learnings.md` -- historical conversion challenges and solutions
- **CoreML scheduling guide:** `README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md`

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
