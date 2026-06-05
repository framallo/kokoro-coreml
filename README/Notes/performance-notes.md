# Performance Notes

This note tracks the performance numbers that matter for users: **end-to-end wall time for one `pipe.synthesize(...)` request** using the current repo HAR-post packages versus the baseline packages downloaded from [mattmireles/kokoro-coreml on Hugging Face](https://huggingface.co/mattmireles/kokoro-coreml).

## What was measured

- **Candidate:** local repo packages in `coreml/`
- **Baseline:** downloaded HF packages in `outputs/hf_baseline/coreml/`
- **Artifacts compared:** `kokoro_decoder_har_post_3s.mlpackage` and `kokoro_decoder_har_post_10s.mlpackage`
- **Metric:** full wall clock around `pipe.synthesize(text, voice="af_heart", speed=1.0)`

These numbers include:

- text processing / `extract_vocoder_inputs()`
- CPU-side tensor prep
- Core ML dispatch and waiting inside the HAR-post call
- trim and Python orchestration
- final waveform returned to the caller

These numbers do **not** include:

- process startup
- model download
- application-level audio playback

## External Bakeoff: surgical Core ML vs MLX and iOS/Core ML Kokoro

**Collected:** 2026-06-05
**Status:** Phase 2 collection complete; waveform sanity complete; human
listening and privileged hardware-placement traces still pending.

This bakeoff compares the current Swift + Core ML Config F reference against
popular Apple Silicon Kokoro implementations:

- **MLX:** `Blaizzy/mlx-audio`, pinned clone
  `862dfbe5338e91df6f74ac986b4df8bede7961a6`, package version
  `mlx-audio 0.4.3`, model `mlx-community/Kokoro-82M-bf16`.
- **Primary iOS/Core ML comparator:** `soniqo/speech-swift`, pinned clone
  `0d09a2ed5464c7c94cf4545be59043c21f8775ea`, using
  `KokoroTTSModel.fromPretrained(computeUnits: .all)`.
- **Long-bucket Core ML backup:** `laishere/kokoro-coreml`, pinned clone
  `484907db6a8347a6afb6e7b86850ea2878c6a3fb`.

`mlalma/kokoro-ios` was excluded from the primary table because its public
package is MLX Swift, not Core ML. ONNX, GGML, browser, cloud, and non-Kokoro
engines were out of scope.

### Method

The shared manifest uses the shipped runtime buckets: `3s`, `7s`, `10s`,
`15s`, and `30s`. All adapters requested voice `af_heart`. Each cell records
one cold call followed by five warm calls. The warm table reports the median of
the five warm wall times. RTF uses the observed emitted audio duration, not the
nominal bucket label.

The intended timing boundary is from immediately before the implementation's
synthesis call or CLI invocation until full PCM audio is materialized in memory.
Config F, MLX, and Soniqo follow that boundary. laishere's public benchmark
boundary excludes G2P and feed preparation and times only the seven-stage Core
ML chain; those numbers are therefore useful as a Core ML chain comparison, but
not a fully equivalent end-to-end TTS boundary.

The paper-facing comparison is warmed inference. For Config F 30s on m2-air and
irvine-m1, the original same-window run let Core ML AOT compile/load work leak
into the recorded 30s calls. Those two warm cells were rerun with three
discarded preflight calls, `KOKORO_USE_EXACT_DURATION_MODELS=1`, and 20 recorded
warm calls. The old compile-inclusive cells remain cold-start/operational
evidence, but they are not used for warmed inference comparison.

### Machine Provenance

| Machine | Hardware model | Memory | macOS |
| --- | --- | ---: | --- |
| m2-studio | Mac14,14 | 64 GiB | 26.5 / 25F71 |
| m2-air | Mac14,15 | 24 GiB | 15.7.7 / 24G720 |
| irvine-m1 | Macmini9,1 | 16 GiB | 15.7.7 / 24G720 |

### iPhone Status

The connected iPhone 12 Pro is visible to CoreDevice as `Webcam`
(`F383FC46-FD64-5346-AEC6-59E3E2F8C9CA`, model `iPhone13,3`) and is available
and paired. No iPhone result is included in the table because physical-device
execution is blocked by local signing state: `security find-identity -v -p
codesigning` reports `0 valid identities found`. The minimal Soniqo Kokoro iOS
runner remains in `scripts/external_bakeoff/SoniqoKokoroIOSRunner/` for the
next signed-device pass.

Whisper, ASR, VAD, and echo-demo dependencies are not part of this bakeoff
boundary. The iOS runner is intentionally Kokoro TTS only.

### Consolidated Warm Median and RTF by Platform

Each cell is `warm median wall time / observed RTF`.

#### m2-studio

Mac14,14, 64 GiB, macOS 26.5 / 25F71.

| Impl | 3s | 7s | 10s | 15s | 30s | Caveats |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Config F | 131.5 ms / 0.047 | 284.2 ms / 0.042 | 548.2 ms / 0.057 | 632.9 ms / 0.046 | 1191.8 ms / 0.043 | - |
| MLX | error | 223.9 ms / 0.033 | 288.8 ms / 0.030 | 376.3 ms / 0.027 | 762.7 ms / 0.028 | 3s broadcast-shape failure |
| Soniqo | 71.7 ms / 0.027 | 69.3 ms / 0.014 | 71.0 ms / 0.014 | 68.1 ms / 0.014 | 69.5 ms / 0.014 | Long buckets emit 5.0s public artifact |
| laishere | 212.3 ms / 0.077 | 403.3 ms / 0.059 | 626.3 ms / 0.065 | 429.8 ms / 0.031 | 925.1 ms / 0.034 | Excludes G2P/feed prep |

#### m2-air

Mac14,15, 24 GiB, macOS 15.7.7 / 24G720.

| Impl | 3s | 7s | 10s | 15s | 30s | Caveats |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Config F | 317.4 ms / 0.113 | 808.1 ms / 0.120 | 1373.3 ms / 0.143 | 2052.4 ms / 0.148 | 3944.0 ms / 0.144 | 30s is post-preflight exact-duration N=20 |
| MLX | error | 685.6 ms / 0.102 | 835.8 ms / 0.087 | 1521.0 ms / 0.109 | 2600.3 ms / 0.095 | 3s broadcast-shape failure |
| Soniqo | 1097.4 ms / 0.406 | 1135.8 ms / 0.227 | 1123.0 ms / 0.225 | 1125.5 ms / 0.225 | 1123.5 ms / 0.225 | Long buckets emit 5.0s public artifact |
| laishere | 142.0 ms / 0.051 | 316.9 ms / 0.046 | 450.2 ms / 0.047 | 657.3 ms / 0.047 | 1476.4 ms / 0.054 | Excludes G2P/feed prep |

#### irvine-m1

Macmini9,1, 16 GiB, macOS 15.7.7 / 24G720.

| Impl | 3s | 7s | 10s | 15s | 30s | Caveats |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Config F | 304.6 ms / 0.109 | 696.1 ms / 0.103 | 1348.5 ms / 0.140 | 1672.7 ms / 0.120 | 2076.1 ms / 0.076 | 30s is post-preflight exact-duration N=20 |
| MLX | error | 824.0 ms / 0.122 | 1124.3 ms / 0.117 | 1589.5 ms / 0.114 | 3077.9 ms / 0.112 | 3s broadcast-shape failure |
| Soniqo | 1330.9 ms / 0.493 | 1343.6 ms / 0.269 | 1313.9 ms / 0.263 | 1343.6 ms / 0.269 | 1351.2 ms / 0.270 | Long buckets emit 5.0s public artifact |
| laishere | 176.3 ms / 0.064 | 394.6 ms / 0.058 | 593.9 ms / 0.062 | 912.0 ms / 0.065 | 2135.1 ms / 0.078 | Excludes G2P/feed prep |

### Cold Wall Time

| Machine | Impl | 3s | 7s | 10s | 15s | 30s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m2-studio | Config F | 125.5 ms | 309.5 ms | 570.9 ms | 647.3 ms | 1389.1 ms |
| m2-studio | MLX | error | 195.9 ms | 4737.1 ms | 438.1 ms | 930.2 ms |
| m2-studio | Soniqo | 615.2 ms | 433.0 ms | 398.0 ms | 411.7 ms | 414.3 ms |
| m2-studio | laishere | 237.0 ms | 359.0 ms | 839.7 ms | 676.1 ms | 1955.1 ms |
| m2-air | Config F | 313.8 ms | 683.1 ms | 1054.0 ms | 2134.6 ms | 9447.1 ms |
| m2-air | MLX | error | 670.4 ms | 20802.8 ms | 1636.8 ms | 2851.4 ms |
| m2-air | Soniqo | 1189.8 ms | 1210.7 ms | 1237.2 ms | 1273.7 ms | 1233.1 ms |
| m2-air | laishere | 289.9 ms | 330.7 ms | 710.6 ms | 746.8 ms | 1616.2 ms |
| irvine-m1 | Config F | 286.0 ms | 647.7 ms | 1035.0 ms | 1372.5 ms | 9114.3 ms |
| irvine-m1 | MLX | error | 807.2 ms | 20027.8 ms | 1662.4 ms | 3293.4 ms |
| irvine-m1 | Soniqo | 1395.3 ms | 1391.9 ms | 1413.1 ms | 1461.0 ms | 1431.8 ms |
| irvine-m1 | laishere | 1102.5 ms | 1239.3 ms | 1877.0 ms | 1659.6 ms | 2791.8 ms |

### Warm Median Wall Time

| Machine | Impl | 3s | 7s | 10s | 15s | 30s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m2-studio | Config F | 131.5 ms | 284.2 ms | 548.2 ms | 632.9 ms | 1191.8 ms |
| m2-studio | MLX | error | 223.9 ms | 288.8 ms | 376.3 ms | 762.7 ms |
| m2-studio | Soniqo | 71.7 ms | 69.3 ms | 71.0 ms | 68.1 ms | 69.5 ms |
| m2-studio | laishere | 212.3 ms | 403.3 ms | 626.3 ms | 429.8 ms | 925.1 ms |
| m2-air | Config F | 317.4 ms | 808.1 ms | 1373.3 ms | 2052.4 ms | 3944.0 ms |
| m2-air | MLX | error | 685.6 ms | 835.8 ms | 1521.0 ms | 2600.3 ms |
| m2-air | Soniqo | 1097.4 ms | 1135.8 ms | 1123.0 ms | 1125.5 ms | 1123.5 ms |
| m2-air | laishere | 142.0 ms | 316.9 ms | 450.2 ms | 657.3 ms | 1476.4 ms |
| irvine-m1 | Config F | 304.6 ms | 696.1 ms | 1348.5 ms | 1672.7 ms | 2076.1 ms |
| irvine-m1 | MLX | error | 824.0 ms | 1124.3 ms | 1589.5 ms | 3077.9 ms |
| irvine-m1 | Soniqo | 1330.9 ms | 1343.6 ms | 1313.9 ms | 1343.6 ms | 1351.2 ms |
| irvine-m1 | laishere | 176.3 ms | 394.6 ms | 593.9 ms | 912.0 ms | 2135.1 ms |

### Observed RTF

| Machine | Impl | 3s | 7s | 10s | 15s | 30s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m2-studio | Config F | 0.047 | 0.042 | 0.057 | 0.046 | 0.043 |
| m2-studio | MLX | error | 0.033 | 0.030 | 0.027 | 0.028 |
| m2-studio | Soniqo | 0.027 | 0.014 | 0.014 | 0.014 | 0.014 |
| m2-studio | laishere | 0.077 | 0.059 | 0.065 | 0.031 | 0.034 |
| m2-air | Config F | 0.113 | 0.120 | 0.143 | 0.148 | 0.144 |
| m2-air | MLX | error | 0.102 | 0.087 | 0.109 | 0.095 |
| m2-air | Soniqo | 0.406 | 0.227 | 0.225 | 0.225 | 0.225 |
| m2-air | laishere | 0.051 | 0.046 | 0.047 | 0.047 | 0.054 |
| irvine-m1 | Config F | 0.109 | 0.103 | 0.140 | 0.120 | 0.076 |
| irvine-m1 | MLX | error | 0.122 | 0.117 | 0.114 | 0.112 |
| irvine-m1 | Soniqo | 0.493 | 0.269 | 0.263 | 0.269 | 0.270 |
| irvine-m1 | laishere | 0.064 | 0.058 | 0.062 | 0.065 | 0.078 |

### Observed Audio Duration

| Machine | Impl | 3s | 7s | 10s | 15s | 30s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m2-studio | Config F | 2.800s | 6.750s | 9.625s | 13.900s | 27.400s |
| m2-studio | MLX | error | 6.750s | 9.600s | 13.900s | 27.375s |
| m2-studio | Soniqo | 2.700s | 5.000s | 5.000s | 5.000s | 5.000s |
| m2-studio | laishere | 2.775s | 6.800s | 9.625s | 13.975s | 27.375s |
| m2-air | Config F | 2.800s | 6.750s | 9.625s | 13.900s | 27.400s |
| m2-air | MLX | error | 6.750s | 9.600s | 13.900s | 27.375s |
| m2-air | Soniqo | 2.700s | 5.000s | 5.000s | 5.000s | 5.000s |
| m2-air | laishere | 2.775s | 6.825s | 9.650s | 13.925s | 27.350s |
| irvine-m1 | Config F | 2.800s | 6.750s | 9.625s | 13.900s | 27.400s |
| irvine-m1 | MLX | error | 6.750s | 9.600s | 13.900s | 27.375s |
| irvine-m1 | Soniqo | 2.700s | 5.000s | 5.000s | 5.000s | 5.000s |
| irvine-m1 | laishere | 2.775s | 6.750s | 9.625s | 13.950s | 27.375s |

### Config F Speed Ratio

Values are comparator warm median divided by Config F warm median. Values above
`1.0x` mean Config F was faster. Values below `1.0x` mean the comparator was
faster.

| Machine | Comparator | 3s | 7s | 10s | 15s | 30s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m2-studio | MLX / Config F | n/a | 0.79x | 0.53x | 0.59x | 0.64x |
| m2-studio | Soniqo / Config F | 0.55x | 0.24x | 0.13x | 0.11x | 0.06x |
| m2-studio | laishere / Config F | 1.61x | 1.42x | 1.14x | 0.68x | 0.78x |
| m2-air | MLX / Config F | n/a | 0.85x | 0.61x | 0.74x | 0.66x |
| m2-air | Soniqo / Config F | 3.46x | 1.41x | 0.82x | 0.55x | 0.28x |
| m2-air | laishere / Config F | 0.45x | 0.39x | 0.33x | 0.32x | 0.37x |
| irvine-m1 | MLX / Config F | n/a | 1.18x | 0.83x | 0.95x | 1.48x |
| irvine-m1 | Soniqo / Config F | 4.37x | 1.93x | 0.97x | 0.80x | 0.65x |
| irvine-m1 | laishere / Config F | 0.58x | 0.57x | 0.44x | 0.55x | 1.03x |

### Hardware Placement Evidence

This bakeoff records framework and compute-unit evidence, not privileged
per-run power traces:

- Config F ran the Swift `kokoro-bench` path over Core ML packages with
  `--compute-units all`. The result records include per-stage Core ML timings
  for Duration, F0Ntrain, DecoderPre, and generator calls.
- MLX ran through the `mlx-audio` Python package and MLX model
  `mlx-community/Kokoro-82M-bf16`; prior host setup recorded MLX default device
  as `gpu` on M2 Air, and MLX routes array kernels through Metal on Apple
  Silicon.
- Soniqo ran Swift `KokoroTTSModel.fromPretrained(computeUnits: .all)` and
  loaded Core ML through `MLModel` via the public `speech-swift` KokoroTTS
  surface.
- laishere ran seven `.mlpackage` Core ML models converted from its public repo.
  Its timed boundary is the Core ML chain only.

`powermetrics` was available on the local host, but it requires superuser
privileges in this environment. No Instruments trace or privileged
`powermetrics` capture was collected during this external bakeoff, so the data
does not prove ANE residency. It proves the requested framework paths and
compute-unit settings.

### Quality Caveats

Every successful cell wrote a mono 24 kHz spot-check WAV and passed the
lightweight waveform sanity gate: duration, RMS, active fraction,
zero-crossing rate, speech-band energy, clipping, sample rate, and channel
count. Human listening is still required before interpreting any latency cell
as quality parity.

Known caveats:

- **MLX 3s:** every machine failed the shared 3s input with
  `ValueError: [broadcast_shapes] Shapes (1,67200,1) and (1,67500,9) cannot be broadcast.`
  This is recorded as public-implementation behavior for the current pinned
  clone and manifest text.
- **Soniqo long buckets:** the public Soniqo Kokoro artifact emits 5.0s audio
  for the 7s, 10s, 15s, and 30s inputs because the upstream public Core ML repo
  only publishes `kokoro_5s.mlmodelc`. These cells are implementation behavior,
  not long-bucket quality-parity evidence.
- **laishere boundary:** laishere is the long-bucket Core ML backup, but its
  numbers exclude G2P and feed preparation.

### Interpretation

The current data does **not** support the broad claim that Config F is faster
than every popular way to run Kokoro on Apple Silicon. MLX is faster than
Config F on most successful same-boundary cells, while failing the shared 3s
input. laishere's Core ML chain is also faster on many cells, but it is not an
end-to-end TTS boundary because feed preparation is excluded. Soniqo is a
popular iOS/Core ML comparator, but its public artifact is 5s-only and cannot
support long-bucket parity as collected.

The defensible paper claim after this bakeoff is narrower: Config F is a
reproducible Swift + Core ML reference with durable bucketed outputs and strong
internal-vs-PyTorch performance, but the external Apple Silicon comparison
requires either a revised hypothesis or additional work that makes the timing
boundaries and quality parity stricter across MLX and Core ML competitors.

## Method

- Forced the synthesis path to `decoder_har_post_bucket_impl` only
- Swapped only the HAR-post Core ML packages between local repo and HF download
- Used identical text, voice, speed, and pipeline code on both sides
- `torch.manual_seed(0)` before each timed call
- Measured:
  - **cold call:** first `synthesize()` after pipeline construction
  - **warm call:** median of 5 additional `synthesize()` calls

Inputs used:

- `tiny`: `"Hello world!"`
- `long`: bakeoff-style longer sentence routed to the 10s HAR-post bucket

## Bakeoff v10: PyTorch baselines on M1 Mini (Configs A/F initially blocked)

**First collected:** 2026-04-17
**Status:** Partial timed results — Configs D and E complete; the A/F export blocker was resolved later on 2026-04-17 and needs a timed rerun

### Summary

Ran the controlled bakeoff on an Apple M1 Mini (16 GB, macOS 15.7.5). During
setup the pre-existing `kokoro_decoder_har_post_<N>s.mlpackage` artifacts (dated
Apr 15) were found to emit half the advertised samples (e.g. 3s bucket →
`waveform (1, 1, 36000)` instead of 72000). This made Config F trip the
canonical-duration-agreement guard: observed 1.5s vs canonical 2.8s at every
bucket. Attempting to re-export with
`uv run --no-sync python -m export_synth.main --mode decoder-har --buckets 3s,7s,10s,15s,30s -o coreml`
failed with `AttributeError: 'MaskedBidirectionalLSTM' object has no attribute 'num_layers'`,
so the HAR-post buckets could not be rebuilt in this session.

Configs D (PyTorch MPS) and E (PyTorch CPU) ran cleanly and are published
below. Config A (Python HAR-post) and Config F (Swift + Core ML) were marked
`config_unavailable` in the original v10 result file.

### Follow-up Resolution

On 2026-04-17, `export_synth/wrappers.py` was made idempotent for an already
masked `kmodel.predictor.lstm`, and the full HAR-post set was regenerated:

```bash
uv run --no-sync python -m export_synth.main --mode decoder-har --buckets 3s,7s,10s,15s,30s -o coreml
```

The saved package specs now advertise the expected waveform lengths:

| Bucket | Waveform samples |
| --- | ---: |
| 3s | 72000 |
| 7s | 168000 |
| 10s | 240000 |
| 15s | 360000 |
| 30s | 720000 |

Config A and Config F both loaded as `READY` in a zero-iteration smoke run, and
Config F passed the canonical-duration-agreement guard for all four frozen
inputs. The smoke result is
`outputs/bakeoff/results_debug_af_smoke.json`. The timed v10 A/F numbers still
need to be collected in a fresh bakeoff run.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | — | 456 ms | 768 ms | — |
| 7s | 6.75s | — | 960 ms | 1998 ms | — |
| 15s | 13.90s | — | 1847 ms | 4007 ms | — |
| 30s | 27.38s | — | 3680 ms | 8074 ms | — |

### RTF

| Input | D RTF | E RTF |
| --- | ---: | ---: |
| 3s | 0.163 | 0.274 |
| 7s | 0.142 | 0.296 |
| 15s | 0.133 | 0.288 |
| 30s | 0.134 | 0.295 |

### Cross-machine comparison (D and E only)

Config D (PyTorch MPS), 30s input wall time: M2 Ultra v9 1602 ms → M1 Mini
3680 ms (**2.3× slower**). Config E (PyTorch CPU), 30s: M2 Ultra v9 2714 ms →
M1 Mini 8074 ms (**3.0× slower**).

| Input | D (M2 Ultra, v9) | D (M1 Mini) | Mini/Ultra | E (M2 Ultra, v9) | E (M1 Mini) | Mini/Ultra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 225 ms | 456 ms | 2.0× | 409 ms | 768 ms | 1.9× |
| 7s | 412 ms | 960 ms | 2.3× | 811 ms | 1998 ms | 2.5× |
| 15s | 673 ms | 1847 ms | 2.7× | 1467 ms | 4007 ms | 2.7× |
| 30s | 1602 ms | 3680 ms | 2.3× | 2714 ms | 8074 ms | 3.0× |

### Interpretation

1. **M1 Mini is ~2–3× slower than M2 Ultra on the PyTorch baselines.** Gap
   widens slightly on long inputs and on CPU, consistent with the Ultra's
   larger GPU and memory bandwidth.
2. **E/D ratio on M1 Mini is ~2.2× at long inputs** (8074 / 3680), roughly
   matching M2 Ultra's E/D ratio (~1.7× at 30s). MPS still pays off on the
   Mini, but less dramatically.
3. **No Config F numbers here.** The half-samples `decoder_har_post` artifacts
   are an upstream export bug, not a Swift-pipeline regression. Once
   `export_synth` is fixed and the HAR-post mlpackages are re-exported, this
   machine should be re-bakedoff for A and F.

### Provenance

- Machine: Apple M1 Mini, 16 GB, macOS 15.7.5
- Command: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs a,d,e --iterations 5 --order-seed 0 --machine-id m1_mini`
- Result: `outputs/bakeoff/results_m1_mini.json` (60 records; 20 × `config_unavailable` for A, 20 × ok for D, 20 × ok for E)
- Order seed: 0, iterations: 5
- Known blockers:
  - `coreml/kokoro_decoder_har_post_*.mlpackage` produced at
    `bucket_samples / 2` output length; re-export needed.
  - `export_synth.main --mode decoder-har` raises
    `AttributeError: 'MaskedBidirectionalLSTM' object has no attribute 'num_layers'`
    on the current repo; fix required before the HAR-post mlpackages can be
    regenerated.

## Bakeoff v9: Config F host-materialization fix on M2 Ultra

**First collected:** 2026-04-17
**Status:** Complete

### Summary

Reran the full counterbalanced A/D/E/F bakeoff after fixing Config F's Swift
host-side materialization costs. F now beats the fixed Config A HAR-post path at
every canonical input length. The long-form loss in v8 was not the exact
Duration graph; it was two avoidable Swift/MLMultiArray costs after Core ML
prediction:

- building a sparse one-hot alignment matrix and multiplying through zeros
- extracting a strided `Float16` waveform through boxed `MLMultiArray`
  subscripting during trim

Config F now expands token states to frames directly and reads typed
`MLMultiArray` storage with stride-aware `Float32`/`Float16` paths. Tensor dumps
still materialize the full alignment/waveform only when requested.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | 333 ms | 225 ms | 409 ms | **57 ms** |
| 7s | 6.75s | 329 ms | 412 ms | 811 ms | **124 ms** |
| 15s | 13.90s | 486 ms | 673 ms | 1467 ms | **239 ms** |
| 30s | 27.38s | 870 ms | 1602 ms | 2714 ms | **476 ms** |

### RTF

| Input | A RTF | D RTF | E RTF | F RTF |
| --- | ---: | ---: | ---: | ---: |
| 3s | 0.119 | 0.080 | 0.146 | **0.020** |
| 7s | 0.049 | 0.061 | 0.120 | **0.018** |
| 15s | 0.035 | 0.048 | 0.106 | **0.017** |
| 30s | 0.032 | 0.059 | 0.099 | **0.017** |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | ---: | ---: | ---: |
| 3s | **5.9x** | **4.0x** | **7.2x** |
| 7s | **2.7x** | **3.3x** | **6.5x** |
| 15s | **2.0x** | **2.8x** | **6.1x** |
| 30s | **1.8x** | **3.4x** | **5.7x** |

### Config F stage medians

| Input | Duration | F0Ntrain | DecoderPre | Matrix | hn-sf | Trim | Core ML total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 10.0 ms | 4.4 ms | 2.8 ms | 0.1 ms | 9.3 ms | 0.2 ms | 28.5 ms |
| 7s | 14.3 ms | 18.9 ms | 8.3 ms | 0.3 ms | 23.1 ms | 0.4 ms | 56.6 ms |
| 15s | 28.8 ms | 38.5 ms | 9.7 ms | 0.7 ms | 46.9 ms | 0.7 ms | 111.6 ms |
| 30s | 52.1 ms | 76.8 ms | 16.6 ms | 1.4 ms | 99.6 ms | 1.6 ms | 224.7 ms |

### Before/after bottleneck proof

| Config F stage | v8 30s median | v9 30s median | Change |
| --- | ---: | ---: | ---: |
| Matrix/materialization | 125.5 ms | 1.4 ms | 90x faster |
| Trim/waveform extraction | 449.1 ms | 1.6 ms | 281x faster |
| End-to-end wall | 1025 ms | 476 ms | 2.2x faster |

### Config F exact Duration proof

| Input | Duration model | Predicted frames |
| --- | --- | ---: |
| 3s | `exact_t44` | `112` |
| 7s | `exact_t105` | `270` |
| 15s | `exact_t219` | `556` |
| 30s | `exact_t476` | `1095` |

### Audio gate

The regenerated F listen samples passed the waveform health gate and remain
available under `outputs/bakeoff/listen/`:

| Sample | Duration | RMS | Active32 | ZCR | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `config_f_3s.wav` | 2.800s | 4600.6 | 78.821% | 9.116% | `needs_listening` |
| `config_f_7s.wav` | 6.750s | 4661.0 | 84.963% | 10.037% | `needs_listening` |
| `config_f_15s.wav` | 13.900s | 5200.3 | 86.447% | 10.781% | `needs_listening` |
| `config_f_30s.wav` | 27.375s | 4301.8 | 86.326% | 10.974% | `needs_listening` |

### Interpretation

1. **F now beats A at every tested length.** In the full controlled run, F is
   `1.8-5.9x` faster than the fixed HAR-post Config A path.
2. **The v8 long-form loss was host work, not model work.** Exact Duration
   remains `10-52 ms`; the old 30s loss came mostly from matrix and trim
   materialization.
3. **Typed, stride-aware access matters.** The Swift waveform output is a
   strided `Float16` `MLMultiArray`, so a contiguous `Float32` fast path alone
   does not help trim.
4. **Load-time Core ML warnings are separate from timed medians.** The combined
   run still emitted occasional `manifest.plist` load warnings while Core ML
   revalidated compiled assets, but the subprocess exited cleanly and produced
   80/80 ok records.

### Provenance

- Machine: Apple M2 Ultra Mac Studio, 64 GB
- Rerun command after `scripts/setup_bakeoff.sh`: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs a,d,e,f --iterations 5 --order-seed 0 --machine-id m2_ultra_parity_final_20260417`
- Result: `outputs/bakeoff/results_m2_ultra_parity_final_20260417.json`
- Result provenance note: the JSON was collected before the final cleanup
  commit and records `git_dirty: true`; the Config F hot-path fixes were
  already present in that working tree. The later shared-executor audit refactor
  was verified separately with
  `outputs/bakeoff/results_shared_executor_smoke_20260417.json`.
- F-only confirmation: `outputs/bakeoff/results_m2_ultra_f_stride_float16_final_20260417.json`
- Listen samples:
  - `outputs/bakeoff/listen/config_f_3s.wav`
  - `outputs/bakeoff/listen/config_f_7s.wav`
  - `outputs/bakeoff/listen/config_f_15s.wav`
  - `outputs/bakeoff/listen/config_f_30s.wav`
- Quality report: `outputs/bakeoff/listen/quality/audio_quality_summary.md`
- Order seed: 0, iterations: 5

## Bakeoff v8: Exact Duration rerun on M2 Ultra

**First collected:** 2026-04-16
**Completed:** 2026-04-17
**Status:** Complete; superseded by v9 for Config F performance

### Summary

Reran the bakeoff after integrating exact fixed-shape native Duration packages
into the normal export/setup flow. The first combined `a,d,e,f` run exposed a
Config A setup bug: A initialized `HybridTTSPipeline()` and auto-loaded every
Core ML package discoverable under `coreml/`, then loaded the HAR-post buckets
again explicitly. That made A spend minutes in unrelated Core ML AOT
respecialization before benchmark availability output.

The harness now initializes Config A with `HybridTTSPipeline(force_engine="pytorch")`
and explicitly loads only the intended HAR-post bucket set with
`compute_units=ALL`. Config A, D, E, and F all completed 20/20 ok records.
Config F uses exact native Duration packages for all canonical inputs and keeps
the corrected frame counts: `112`, `270`, `556`, and `1095`. The v8 long-form
loss was later traced to Swift host materialization and fixed in v9.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | ---: | ---: | ---: | ---: |
| 3s | 2.80s | 180 ms | 208 ms | 461 ms | **111 ms** |
| 7s | 6.75s | 266 ms | 381 ms | 781 ms | **250 ms** |
| 15s | 13.90s | **450 ms** | 736 ms | 1470 ms | 518 ms |
| 30s | 27.38s | **786 ms** | 1628 ms | 2604 ms | 1025 ms |

### RTF

| Input | A RTF | D RTF | E RTF | F RTF |
| --- | ---: | ---: | ---: | ---: |
| 3s | 0.064 | 0.074 | 0.165 | **0.040** |
| 7s | 0.039 | 0.056 | 0.116 | **0.037** |
| 15s | **0.032** | 0.053 | 0.106 | 0.037 |
| 30s | **0.029** | 0.059 | 0.095 | 0.037 |

### Config A stage medians

| Input | Prefix | HAR CPU | Core ML | Trim |
| --- | ---: | ---: | ---: | ---: |
| 3s | 82.5 ms | 42.6 ms | 48.9 ms | 0.0 ms |
| 7s | 119.4 ms | 65.4 ms | 83.5 ms | 0.0 ms |
| 15s | 184.4 ms | 119.4 ms | 135.0 ms | 0.0 ms |
| 30s | 325.0 ms | 214.7 ms | 235.2 ms | 0.0 ms |

### Config F stage medians

| Input | Duration | F0Ntrain | DecoderPre | Matrix | hn-sf | Trim | Core ML total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 9.8 ms | 4.5 ms | 3.1 ms | 11.8 ms | 9.3 ms | 45.1 ms | 27.7 ms |
| 7s | 13.2 ms | 18.6 ms | 8.5 ms | 27.3 ms | 21.0 ms | 103.6 ms | 55.9 ms |
| 15s | 27.3 ms | 39.2 ms | 9.7 ms | 58.2 ms | 47.0 ms | 224.9 ms | 109.6 ms |
| 30s | 47.1 ms | 73.5 ms | 15.4 ms | 125.5 ms | 100.1 ms | 449.1 ms | 213.3 ms |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | ---: | ---: | ---: |
| 3s | **1.6x** | **1.9x** | **4.1x** |
| 7s | **1.1x** | **1.5x** | **3.1x** |
| 15s | 0.9x | **1.4x** | **2.8x** |
| 30s | 0.8x | **1.6x** | **2.5x** |

### Config F exact Duration proof

| Input | Duration model | Predicted frames |
| --- | --- | ---: |
| 3s | `exact_t44` | `112` |
| 7s | `exact_t105` | `270` |
| 15s | `exact_t219` | `556` |
| 30s | `exact_t476` | `1095` |

### Config A setup fix

The combined `a,d,e,f` run and the standalone `a` run both stayed in Python
Core ML model load for several minutes before any Config A availability output.
Process sampling showed:

```text
MLE5ProgramLibraryOnDeviceAOTCompilationImpl createProgramLibraryHandleWithRespecialization
_ANEClient compileModel
```

Root cause: Config A was auto-loading unrelated Core ML packages through
`HybridTTSPipeline()` before replacing the bucket dictionary with explicit
HAR-post packages. The fix skips auto Core ML discovery for A and loads only
`kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage`.

### Interpretation

1. **Config A is fixed as a benchmark harness path.** It now reaches
   availability and completes all canonical inputs instead of blocking in
   unrelated package AOT compilation.
2. **Config F wins at short lengths and remains the fastest non-A baseline.**
   F beats A at 3s and 7s, but A is still faster at 15s and 30s.
3. **F's remaining long-form cost is not Duration.** Exact Duration medians are
   `9.8/13.2/27.3/47.1 ms`; the larger F costs are matrix/HN-SF/trim
   materialization.
4. **D/E are no longer competitive on this run.** F beats PyTorch MPS by
   `1.4-1.9x` and CPU by `2.5-4.1x`.

### Provenance

- Machine: Apple M2 Ultra Mac Studio, 64 GB
- Swift warmup: completed; selected `exact_t44`, `exact_t105`, `exact_t219`,
  and `exact_t476`
- Full command attempted: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs a,d,e,f --iterations 5 --order-seed 0 --machine-id m2_ultra_exact_duration_rerun_20260416`
- Completed A command: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs a --iterations 5 --order-seed 0 --machine-id m2_ultra_exact_duration_a_fixed_20260417`
- Completed F command: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs f --iterations 5 --order-seed 0 --machine-id m2_ultra_exact_duration_fonly_20260416`
- Completed D/E command: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs d,e --iterations 5 --order-seed 0 --machine-id m2_ultra_exact_duration_de_20260416`
- Results:
  - `outputs/bakeoff/results_m2_ultra_exact_duration_a_fixed_20260417.json`
  - `outputs/bakeoff/results_m2_ultra_exact_duration_fonly_20260416.json`
  - `outputs/bakeoff/results_m2_ultra_exact_duration_de_20260416.json`
- Order seed: 0, iterations: 5

## Bakeoff v7: Duration-correct Config F on M2 Ultra

**First collected:** 2026-04-16
**Status:** Complete

### Summary

Reran the complete A/D/E/F bakeoff after fixing Config F Duration padding
semantics for all enum shapes. This run proves Config F now produces
canonical-duration audio, but it also shows the corrected Swift/Core ML path is
slower than Config A at every length on this M2 Ultra.

The fix replaced padded bidirectional LSTM execution in the Duration export
with mask-aware recurrent unrolls. That restores audio parity, but it makes the
large static Duration graphs expensive, especially `T=512`.

### Audio gate

Before the bakeoff, Config F listen samples were regenerated with:

```bash
uv run --no-sync python scripts/bakeoff_listen.py --quality-plots
```

All four samples passed the waveform health gate:

| Input | Observed duration | RMS | Active32 | ZCR | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 3s | `2.800s` | `4607.8` | `78.863%` | `9.128%` | `needs_listening` |
| 7s | `6.750s` | `4500.8` | `84.739%` | `10.063%` | `needs_listening` |
| 15s | `13.900s` | `5147.8` | `86.396%` | `10.785%` | `needs_listening` |
| 30s | `27.400s` | `4309.6` | `86.300%` | `10.937%` | `needs_listening` |

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | ---: | ---: | ---: | ---: |
| 3s | 2.80s | **162 ms** | 188 ms | 349 ms | 169 ms |
| 7s | 6.75s | **247 ms** | 345 ms | 642 ms | 368 ms |
| 15s | 13.90s | **482 ms** | 640 ms | 1224 ms | 766 ms |
| 30s | 27.38s | **713 ms** | 1340 ms | 2231 ms | 1595 ms |

### RTF

| Input | A RTF | D RTF | E RTF | F RTF |
| --- | ---: | ---: | ---: | ---: |
| 3s | **0.058** | 0.067 | 0.125 | 0.060 |
| 7s | **0.037** | 0.051 | 0.095 | 0.054 |
| 15s | **0.035** | 0.046 | 0.088 | 0.055 |
| 30s | **0.026** | 0.049 | 0.082 | 0.058 |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | ---: | ---: | ---: |
| 3s | 1.0x | **1.1x** | **2.1x** |
| 7s | 0.7x | 0.9x | **1.7x** |
| 15s | 0.6x | 0.8x | **1.6x** |
| 30s | 0.4x | 0.8x | **1.4x** |

### Interpretation

1. **Config A is not cheating.** It produces valid, canonical-duration audio and
   remains the fastest path after Config F is made duration-correct.
2. **Config F is now correct, not fast.** The old shorter outputs were invalid
   for performance comparison. After padding semantics are fixed, F no longer
   wins against A.
3. **The fixed Duration stage is a new bottleneck.** Median Config F duration
   time is `78 ms`, `157 ms`, `342 ms`, and `751 ms` for 3s/7s/15s/30s.
4. **Waveform materialization remains expensive.** Median Config F trim/output
   materialization is `43 ms`, `99 ms`, `220 ms`, and `433 ms` across the same
   lengths.
5. **The largest hidden cost is first-use compile/load.** T=512 Duration
   warmup requires minutes inside Core ML's E5/ANE compiler. The harness timeout
   was raised so publication runs do not falsely mark F as dead during warmup.

### Provenance

- Machine: Apple M2 Ultra Mac Studio, 64 GB
- Command: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs a,d,e,f --iterations 5 --order-seed 0 --machine-id m2_ultra_v7`
- Git recorded in results: `c251622b458d`, dirty tree with Duration masking,
  exporter controls, harness timeout/recovery, and notes updates
- Python: 3.12.13
- Torch: 2.6.0 / coremltools: 8.3.0
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m2_ultra_v7.json`
- Summary: `outputs/bakeoff/summary.md`
- Quality report: `outputs/bakeoff/listen/quality/audio_quality_report.json`

## Bakeoff v6: Audio-fixed Config F on M2 Ultra

**First collected:** 2026-04-16
**Status:** Complete

### Summary

Reran the complete A/D/E/F bakeoff on the Mac Studio M2 Ultra after fixing the
Config F waveform extraction bug, replacing the weak listen gate, and tightening
the benchmark so Config F always materializes the trimmed waveform inside the
timed path. This run is the first post-fix M2 Ultra bakeoff where Config F
listen samples were generated with the stride-safe Swift runtime, all four
enumerated listen shapes passed the objective audio gate, and the short/medium
samples were human-confirmed before performance numbers were accepted.

**Config F produces human-sounding audio and beats PyTorch CPU/MPS at every
duration**, but it is not the universal winner against Config A once waveform
extraction is included in the timed Swift path. Relative to Config A (Python
HAR-post), the Swift/Core ML path is `1.5x` faster at 3s, roughly tied at 7s,
and slower at 15s/30s. Relative to PyTorch CPU, it remains `2.5-3.3x` faster.

### Audio gate

Before this bakeoff, the listen samples were regenerated with:

```bash
uv run --no-sync python scripts/bakeoff_listen.py --keys 3s,7s,15s,30s
```

All four listen samples recorded `quality_pass=true` and
`quality_decision=needs_listening`:

| Input | WAV | RMS | Active32 | ZCR |
| --- | --- | ---: | ---: | ---: |
| 3s | `outputs/bakeoff/listen/config_f_3s.wav` | `4708.0` | `75.668%` | `8.578%` |
| 7s | `outputs/bakeoff/listen/config_f_7s.wav` | `4885.2` | `83.437%` | `10.078%` |
| 15s | `outputs/bakeoff/listen/config_f_15s.wav` | `5282.6` | `87.239%` | `10.858%` |
| 30s | `outputs/bakeoff/listen/config_f_30s.wav` | `4204.1` | `86.033%` | `11.146%` |

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | Bucket | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 3s | 2.80s | 3s | 148 ms | 183 ms | 319 ms | **98 ms** |
| 7s | 6.75s | 7s | 228 ms | 327 ms | 612 ms | **216 ms** |
| 15s | 13.90s | 15s | **422 ms** | 618 ms | 1195 ms | 453 ms |
| 30s | 27.38s | 30s | **692 ms** | 1288 ms | 2190 ms | 881 ms |

### RTF and realtime factor

| Input | Audio | A RTF | D RTF | E RTF | F RTF | F realtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | 0.053 | 0.065 | 0.114 | **0.035** | **29x RT** |
| 7s | 6.75s | 0.034 | 0.049 | 0.091 | **0.032** | **31x RT** |
| 15s | 13.90s | **0.030** | 0.045 | 0.086 | 0.033 | 31x RT |
| 30s | 27.38s | **0.025** | 0.047 | 0.080 | 0.032 | 31x RT |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | ---: | ---: | ---: |
| 3s | **1.5x** | **1.9x** | **3.3x** |
| 7s | **1.1x** | **1.5x** | **2.8x** |
| 15s | 0.9x | **1.4x** | **2.6x** |
| 30s | 0.8x | **1.5x** | **2.5x** |

### Interpretation

1. **The speed result is now tied to human-audible output.** The previous
   bakeoff winner was invalid because its listen samples were near-silent. This
   v6 run was only accepted after objective gates passed and short/medium clips
   were human-confirmed.

2. **Config F remains a valid fast path, but not always the fastest path.** The
   Swift/Core ML pipeline beats PyTorch CPU and PyTorch MPS at every enumerated
   shape, but Config A is faster at 15s and 30s on this M2 Ultra run.

3. **The benchmark now includes waveform extraction.** The earlier post-fix
   timing that showed Config F winning everywhere did not materialize the
   waveform in the timed Config F path. The corrected measurement is stricter
   and more product-representative.

4. **MPS fallback remains slower than Config F.** The PyTorch MPS path emitted
   the expected `aten::angle` CPU fallback warning and was 1.4-1.9x slower than
   Config F across the enumerated shapes.

### Provenance

- Machine: Apple M2 Ultra Mac Studio, 64 GB, macOS 26.4.1
- Command: `BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --no-sync python scripts/bakeoff_harness.py run --configs a,d,e,f --iterations 5 --order-seed 0 --machine-id m2_ultra_v6`
- Git recorded in results: `b722798e9995`, with a dirty tree containing the
  stride-safety audit fix and notes updates from the active recovery workflow
- Python: 3.12.13
- Torch: 2.6.0 / coremltools: 8.3.0
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m2_ultra_v6.json`
- Summary: `outputs/bakeoff/summary.md`
- Quality report: `outputs/bakeoff/listen/quality/audio_quality_report.json`

### Plan reference

Audio parity recovery plan Phase 7:
`README/Plans/kokoro-audio-parity-recovery-v1.md`

## End-to-end latency

| Preset | Audio returned | Repo warm median | HF warm median | Repo vs HF |
| --- | --- | --- | --- | --- |
| `tiny` | `1.5s` | `121 ms` | `108 ms` | Repo is `12.5%` slower |
| `long` | `5.0s` | `303 ms` | `262 ms` | Repo is `15.4%` slower |

Equivalent steady-state RTF from the same run:

| Preset | Repo warm RTF | HF warm RTF |
| --- | --- | --- |
| `tiny` | `0.081` | `0.072` |
| `long` | `0.061` | `0.052` |

## First-call latency

These are the first measured `synthesize()` calls after the pipeline object was created:

| Preset | Repo cold wall | HF cold wall |
| --- | --- | --- |
| `tiny` | `529 ms` | `337 ms` |
| `long` | `529 ms` | `574 ms` |

Treat these as directional only. The `long` case was measured after the `tiny` case in the same session, so it is not a pure fresh-process cold start.

## Pipeline init in this harness

Python-side pipeline construction in this benchmark took:

- repo init: `198.3s`
- HF init: `190.6s`

This is real for this script, but it should **not** be treated as the final app-level startup number without a separate startup-focused benchmark.

## Takeaway

For the metric we actually care about, **the current local HAR-post packages are slower than the HF baseline** on both tested end-to-end requests.

- `tiny`: local `121 ms` vs HF `108 ms`
- `long`: local `303 ms` vs HF `262 ms`

So the current answer is: **the new version is not faster in end-to-end latency on this run**.

## Where the slowdown shows up

To explain the latency gap, I reran the same local-vs-HF comparison with stage timing around the HAR-post path:

1. `extract_vocoder_inputs()`
2. bucket pick
3. CPU tensor build via `build_decoder_har_post_inputs_np`
4. Core ML `predict()`
5. trim
6. residual / orchestration remainder

This stage replay uses the same pipeline code and package swap, but times the HAR-post path directly instead of only wrapping `pipe.synthesize(...)`.

### Warm stage breakdown: `tiny`

| Stage | Repo | HF | Delta |
| --- | --- | --- | --- |
| extract vocoder inputs | `49.9 ms` | `41.5 ms` | repo `+8.4 ms` |
| bucket pick | `0.021 ms` | `0.020 ms` | noise |
| build inputs | `36.0 ms` | `33.7 ms` | repo `+2.3 ms` |
| Core ML `predict()` | `19.9 ms` | `19.1 ms` | repo `+0.8 ms` |
| trim | `0.016 ms` | `0.013 ms` | noise |
| residual | `0.001 ms` | `0.001 ms` | noise |
| total | `107.9 ms` | `94.0 ms` | repo `+13.9 ms` |

### Warm stage breakdown: `long`

| Stage | Repo | HF | Delta |
| --- | --- | --- | --- |
| extract vocoder inputs | `128.6 ms` | `119.4 ms` | repo `+9.2 ms` |
| bucket pick | `0.022 ms` | `0.021 ms` | noise |
| build inputs | `77.0 ms` | `83.8 ms` | repo `-6.8 ms` |
| Core ML `predict()` | `43.9 ms` | `42.0 ms` | repo `+1.9 ms` |
| trim | `0.015 ms` | `0.015 ms` | noise |
| residual | `0.001 ms` | `0.002 ms` | noise |
| total | `250.6 ms` | `244.6 ms` | repo `+6.0 ms` |

### Interpretation

The slowdown is **not** coming from one massive regression inside Core ML. On these runs:

- The largest repeated penalty is **`extract_vocoder_inputs()`**:
  - about `+8.4 ms` on `tiny`
  - about `+9.2 ms` on `long`
- There is also a smaller but real **Core ML `predict()`** penalty:
  - about `+0.8 ms` on `tiny`
  - about `+1.9 ms` on `long`
- The CPU-side HAR-post tensor build is:
  - a little slower on `tiny`
  - actually faster on `long`

So the current regression appears to be **mostly in the shared prefix path**, with a smaller contribution from the Core ML inference itself.

## Artifacts

- End-to-end results: `outputs/bakeoff/local_vs_hf_har_post_e2e.json`
- Older `predict()`-only micro-bench: `outputs/bakeoff/local_vs_hf_har_post_predict.json`
- Stage breakdown: `outputs/bakeoff/local_vs_hf_har_post_stage_breakdown.json`

---

## ANE optimization experiment: nn.Linear → nn.Conv1d in AdaIN1d — Resolved (reverted)

**First spotted:** 2026-04-14
**Resolved:** 2026-04-14
**Status:** Resolved — hypothesis disproved, Conv1d change reverted, dead code cleanup kept

### Summary

Cross-referenced the [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) against the production `GeneratorFromHar` ANE path. The Orion reverse-engineering project (Constraint #17) claims matmul executes 3x slower than 1x1 convolution on the ANE. We replaced `nn.Linear` with `nn.Conv1d(kernel_size=1)` in `AdaIN1d.fc` — dozens of instances in the hot ANE path. **Result: no improvement.** Core ML `predict()` time was unchanged or marginally worse. CoreML's MIL compiler very likely lowers `linear` ops to conv internally (inferred from identical predict times, not MIL-dump verified), making the source-level change redundant. The largest measured latency gap (vs HF baseline packages) appears in `extract_vocoder_inputs()`, the PyTorch CPU prefix path — though this needs a dedicated prefix-only A/B to rule out measurement noise (see caveats below). Note: the CPU-side `build inputs` stage was actually 6.8 ms *faster* for the repo on the `long` input, partially offsetting the predict penalty — the overall picture is mixed, not uniformly worse.

### What we did

1. **Audited the GeneratorFromHar traced graph** for ANE-incompatible ops per the scheduling guide:
   - `torch.cat` (Orion Constraint #1: banned on ANE) — found in `AdaIN1d.forward()` at `istftnet.py:154-155`, but confirmed it was **dead code**: the padding branch only fires when `C != num_features`, which never happens because `AdaIN1d` is always constructed with `num_features == channels`.
   - `nn.Linear` (Orion Constraint #17: matmul 3x slower than Conv on ANE) — found in `AdaIN1d.fc` at `istftnet.py:129`. Each `AdaINResBlock1` has 6 `AdaIN1d` instances. The Generator has `num_upsamples * num_kernels` resblocks plus `num_upsamples` noise_res blocks — dozens of Linear forward calls on the ANE per inference.
   - No `nn.GELU` (clean — uses LeakyReLU and Snake activations).
   - Tensor layout already `(B, C, T)` mapping to ANE's preferred `(B, C, 1, S)`.

2. **Removed dead `torch.cat` code** in `AdaIN1d.forward()` — replaced the slice/pad branch with `assert C == self.num_features`. This cleanup is kept (not reverted) because it removes code that could never execute and would have introduced an ANE-banned concat op if it somehow did.

3. **Replaced `nn.Linear` with `nn.Conv1d(kernel_size=1)`** in `AdaIN1d.__init__`:
   ```python
   # Before
   self.fc = nn.Linear(style_dim, num_features * 2)
   # After
   self.fc = nn.Conv1d(style_dim, num_features * 2, kernel_size=1)
   ```
   Adjusted `forward()` to unsqueeze style input for Conv1d. Added `register_load_state_dict_pre_hook` to reshape pretrained Linear weights `(out, in)` → `(out, in, 1)` for checkpoint compatibility.

4. **Re-exported decoder HAR post buckets** (3s, 10s) with the Conv1d-based AdaIN1d.

5. **Benchmarked** local repo packages vs HF baseline packages (which still use nn.Linear) using identical text, voice, and pipeline code.

### What we learned

**Core ML predict() — no improvement:**

| Input | Repo (Conv1d) | HF (Linear) | Delta |
| --- | --- | --- | --- |
| tiny | 19.9 ms | 19.1 ms | +0.8 ms (worse) |
| long | 43.9 ms | 42.0 ms | +1.9 ms (worse) |

**Conclusion:** CoreML's MIL compiler very likely optimizes `linear` → conv internally during the `.mlpackage` compilation/specialization step — inferred from the identical predict() times, not directly verified with a MIL before/after dump. The source-level Conv1d change appears redundant. Orion Constraint #17 likely applies to **direct ANE programming** (bypassing CoreML), not to the CoreML conversion pipeline. A definitive proof would require dumping the MIL graph (e.g. via `ct.models.MLModel._mil_program`) for both variants and comparing the lowered ops.

**The real regression is in the PyTorch CPU prefix:**

| Input | Stage | Repo | HF | Delta |
| --- | --- | --- | --- | --- |
| tiny | extract_vocoder_inputs | 49.9 ms | 41.5 ms | +8.4 ms |
| long | extract_vocoder_inputs | 128.6 ms | 119.4 ms | +9.2 ms |

This +8-9ms penalty is consistent across inputs and dwarfs the Core ML predict delta. It lives in the shared PyTorch path (duration model + alignment + hn-nsf), which the Conv1d change does not touch.

**Caveat:** The stage breakdown swapped only decoder `.mlpackage` files between repo and HF; the prefix extraction code was identical in both runs. A consistent +8-9ms delta on a shared code path is suspicious — it may reflect run-to-run thermal/cache variance, process ordering effects (repo always ran first), or genuine codebase drift. A dedicated prefix-only A/B with interleaved runs and higher iteration count is needed to confirm this as a real regression vs. measurement noise.

### Key takeaways for future work

1. **Don't optimize what CoreML likely already optimizes.** The MIL compiler's internal lowering passes very likely handle Linear → Conv conversion (inferred from identical predict times). Source-level changes to match Orion constraints likely only matter when programming the ANE directly (bypassing CoreML). Verify with a MIL dump if this assumption becomes load-bearing.
2. **Profile before optimizing.** The stage breakdown showed the bottleneck was in the PyTorch prefix, not the ANE decoder. Without the breakdown, we'd have spent more time on the wrong problem.
3. **The dead `torch.cat` removal was valid.** Even though it was dead code, removing it prevents future accidental activation and eliminates an ANE-banned op from the source.
4. **The `extract_vocoder_inputs()` gap needs a dedicated A/B.** A +8-9ms delta across inputs is suggestive but not conclusive — the stage breakdown only swapped decoder packages while prefix code was identical, so the delta may reflect thermal/ordering effects rather than a real regression. A prefix-only interleaved A/B benchmark is needed before committing to an optimization effort there.

### Related Guides

- [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) — Orion Constraints #1 (concat) and #17 (matmul vs conv); verification techniques
- [Apple: Deploying Transformers on the ANE](https://machinelearning.apple.com/research/neural-engine-transformers) — Linear-to-Conv2d recommendation (applies to direct ANE, not CoreML pipeline)
- [Orion paper](https://arxiv.org/abs/2603.06728) — reverse-engineered ANE constraints

### Files changed (then reverted)

- `kokoro/istftnet.py:129` — `AdaIN1d.fc`: Linear → Conv1d + `register_load_state_dict_pre_hook` for weight reshaping (both reverted)
- `kokoro/istftnet.py:131-152` — `AdaIN1d.forward`: input reshape for Conv1d (reverted)
- `kokoro/istftnet.py:146-155` — dead `torch.cat` padding branch removed (kept)

### Plan reference

Full experiment design: `README/Plans/ane-optimization-v1.md`

---

## Bakeoff v2: Controlled five-config benchmark on M2 Ultra

**First collected:** 2026-04-15
**Status:** Complete (powermetrics telemetry deferred; M1 Mini data collected — see below)

### Summary

Controlled benchmark of the shipping HAR-post path against PyTorch CPU/MPS baselines and a naive decoder-only Core ML control artifact. Five configs, four frozen inputs, five counterbalanced repetitions on Apple M2 Ultra (64 GB). The shipping hybrid path (Config A) is **2.6–3.5x faster than PyTorch CPU** on medium-to-long inputs and **18–30x realtime**, but CPU-side overhead still consumes ~80% of wall time.

### What was measured

- **Config A:** Shipping hybrid HAR-post path (`coreml/kokoro_decoder_har_post_{3,10}s.mlpackage`)
- **Config B:** Naive decoder-only 10s artifact, `compute_units=ALL`
- **Config C:** Naive decoder-only 10s artifact, `compute_units=CPU_AND_GPU`
- **Config D:** PyTorch end-to-end on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- **Config E:** PyTorch end-to-end on CPU

All configs used identical frozen inputs, `voice=af_heart`, `speed=1.0`, `torch.manual_seed(0)`.

### Method

- Harness: `scripts/bakeoff_harness.py` with `run --configs a,b,c,d,e --iterations 5 --order-seed 0`
- All models preloaded and warmed before timed iterations
- Config A uses explicit-path artifact loading with SHA256 recorded
- Counterbalanced: config and input order independently shuffled per repetition via `random.Random(order_seed + rep)`
- Timer: `time.perf_counter()` wall clock, MPS sync before stop for Config D
- Each iteration is one full text-to-waveform pass (text processing through final numpy array)

### Inputs

| Key | Text | Audio duration | Bucket |
| --- | --- | --- | --- |
| `tiny` | `"Hello world!"` | `1.55s` | `3s` |
| `short` | `"The quick brown fox jumps over the dog."` | `2.80s` | `3s` |
| `medium` | `"This is a longer sentence...running on the Apple GPU."` | `6.58s` | `10s` |
| `long` | `"This is a longer sentence...A few more words added here."` | `8.35s` | `10s` |

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (HAR-post) | B (.all) | C (.cpuAndGPU) | D (MPS) | E (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `1.55s` | `151 ms` | `137 ms` | `182 ms` | `215 ms` | `258 ms` |
| `short` | `2.80s` | `155 ms` | `176 ms` | `177 ms` | `287 ms` | `396 ms` |
| `medium` | `6.58s` | `283 ms` | `189 ms` | `184 ms` | `351 ms` | `782 ms` |
| `long` | `8.35s` | `274 ms` | `214 ms` | `238 ms` | `436 ms` | `966 ms` |

### RTF (canonical audio duration / wall time)

| Input | A (HAR-post) | B (.all) | C (.cpuAndGPU) | D (MPS) | E (CPU) |
| --- | --- | --- | --- | --- | --- |
| `tiny` | `0.097` (10x RT) | `0.089` | `0.117` | `0.139` | `0.167` |
| `short` | `0.055` (18x RT) | `0.063` | `0.063` | `0.103` | `0.142` |
| `medium` | `0.043` (23x RT) | `0.029` | `0.028` | `0.053` | `0.119` |
| `long` | `0.033` (30x RT) | `0.026` | `0.029` | `0.052` | `0.116` |

### Speedup: Config A vs PyTorch baselines

| Input | Audio | A vs E (CPU) | A vs D (MPS) |
| --- | --- | --- | --- |
| `tiny` | `1.55s` | `1.7x` | `1.4x` |
| `short` | `2.80s` | `2.6x` | `1.9x` |
| `medium` | `6.58s` | `2.8x` | `1.2x` |
| `long` | `8.35s` | `3.5x` | `1.6x` |

The advantage grows with sequence length because Core ML predict time scales sublinearly while PyTorch CPU scales linearly.

### Config A stage breakdown (warm median)

| Input | Bucket | Prefix extract | HAR builder (CPU) | CoreML predict | Orchestration | Total |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `3s` | `52.7 ms` (35%) | `40.9 ms` (27%) | `57.0 ms` (38%) | `2.0 ms` | `151 ms` |
| `short` | `3s` | `92.4 ms` (60%) | `39.9 ms` (26%) | `19.1 ms` (12%) | `2.0 ms` | `155 ms` |
| `medium` | `10s` | `109.5 ms` (39%) | `80.4 ms` (28%) | `84.0 ms` (30%) | `2.0 ms` | `283 ms` |
| `long` | `10s` | `127.0 ms` (46%) | `85.7 ms` (31%) | `47.5 ms` (17%) | `1.9 ms` | `274 ms` |

### Interpretation

1. **Config A is 18–30x realtime on M2 Ultra.** Even the shortest input (`tiny`, 1.55s audio) completes in 151 ms. The longest input (`long`, 8.35s audio) completes in 274 ms.

2. **CPU-side overhead dominates.** Across all inputs, `extract_vocoder_inputs()` + `build_decoder_har_post_inputs_np()` together consume 62–86% of wall time. Core ML `predict()` is only 12–38% of wall time — already fast, with limited room for further ANE optimization to improve end-to-end latency.

3. **The speedup scales with duration.** At `tiny` (1.55s), Config A is only 1.7x faster than CPU because the fixed prefix overhead dominates. At `long` (8.35s), the speedup reaches 3.5x because Core ML predict scales sublinearly while the prefix cost grows slowly.

4. **MPS is worse than expected.** Config D (PyTorch MPS with fallback) shows high variance and only modest improvement over CPU. This is consistent with known `aten::angle` fallback overhead on MPS — treat Config D as the path-of-least-resistance MPS baseline, not the GPU ceiling.

5. **Configs B and C (decoder-only) have similar latency.** Without powermetrics telemetry, Gate 1 (ANE participation under `.all`) is **indeterminate** from timing alone. B and C differ by <15% on most inputs, which is within thermal/scheduling noise. Telemetry loops with `sudo powermetrics` are needed for a definitive answer.

6. **The predict-time variance on `tiny` is notable.** Config A's Core ML predict shows `57 ms` on the 3s bucket for `tiny` but only `19 ms` for `short` (also 3s bucket). This likely reflects first-bucket compilation/warmup effects even after the general warmup pass, since `tiny` and `short` may not always warm the same bucket.

### Comparison to prior anecdotal numbers

The earlier section of this document reported repo HAR-post warm medians of `121 ms` (tiny) and `303 ms` (long) in a less controlled two-input comparison. The bakeoff numbers (`151 ms` tiny, `274 ms` long) are in the same ballpark but not directly comparable:

- The bakeoff uses counterbalanced ordering (prior test ran sequentially)
- The bakeoff uses a different `long` text (`~8.35s` vs `~5.0s` in the prior test)
- The bakeoff `tiny` is slightly slower, consistent with counterbalanced ordering disrupting cache locality

The +12–15% gap vs HF baseline packages reported earlier is **not re-tested** in this bakeoff because all five configs use the same local repo artifacts. The gap remains a known issue (see above).

### Provenance

- Machine: Apple M2 Ultra, 64 GB
- Git: `d123bee9ecbb`
- Torch: `2.6.0` / coremltools: `8.3.0` / numpy: `1.26.4`
- Order seed: `0`, iterations: `5`
- Results: `outputs/bakeoff/results_m2_ultra.json`
- Summary: `outputs/bakeoff/summary.md`

### Plan reference

Full experiment design: `README/Plans/kokoro-bakeoff-v2.md`

---

## Swift prefix rewrite: per-stage latency measurements

**First collected:** 2026-04-15
**Status:** Phase 3 complete (per-stage benchmarks; DecoderPre bridge; full bakeoff comparison deferred)

### Summary

Measured every stage of the proposed Swift prefix rewrite pipeline on M2 Ultra. The Duration model (BERT + predictor) runs 3.8x faster as CoreML vs PyTorch. The F0Ntrain prosody model exports to CoreML with 0.999995 correlation and runs in 3-23 ms depending on bucket. The hn-nsf harmonic source (SineGen + STFT) runs in native Swift at 14-45 ms with Double-precision phase accumulation, down from 166 ms after optimizing bulk noise generation. The DecoderPre stage remains a PyTorch bridge pending CoreML export (Phase 4, conditional on AdaIN).

### Individual stage latencies (warm median, M2 Ultra)

| Stage | 3s bucket | 10s bucket | Notes |
| --- | --- | --- | --- |
| Duration CoreML | 13.6 ms | 13.6 ms | Fixed 128-token input, `compute_units=ALL`. 3.8x vs ~50 ms PyTorch. |
| Alignment + matrix ops | ~1 ms | ~1 ms | Swift. Two matrix multiplies on small tensors. |
| F0Ntrain CoreML | 3.1 ms | 23.4 ms | T=120 (3s) / T=400 (10s). Correlation 0.9999+. |
| Padding | ~0.5 ms | ~0.5 ms | Zero-fill to bucket geometry. |
| DecoderPre (PyTorch bridge) | 44.7 ms | 103.8 ms | **Bottleneck.** PyTorch decoder stack. Phase 4 target. |
| hn-nsf Swift (Accelerate) | 14 ms | 45 ms | Double-precision phase, bulk noise gen, vvsin. |
| GeneratorFromHar CoreML | 16.7 ms | 41.0 ms | `compute_units=ALL`. Same models as bakeoff v2. |

### Estimated pipeline totals

| Configuration | 3s bucket | 10s bucket |
| --- | --- | --- |
| **Swift + bridge** (current) | ~94 ms | ~228 ms |
| **Swift + CoreML DecoderPre** (Phase 4) | ~49 ms | ~125 ms |
| **Python Config A** (bakeoff v2) | 151 ms | 274 ms |
| **PyTorch MPS** (bakeoff v2) | 215–287 ms | 351–436 ms |

### Speedup vs baselines

| Configuration | vs Python Config A (3s / 10s) | vs MPS (3s / 10s) |
| --- | --- | --- |
| Swift + bridge | 1.6x / 1.2x | 2.3x / 1.5x |
| Swift + CoreML DecoderPre | 3.1x / 2.2x | **4.4x / 2.8x** |

MPS is the natural "just use the GPU" baseline. The Swift pipeline beats it by 2–4x because MPS suffers from `aten::angle` CPU fallback overhead and Python interpreter costs, while Swift+CoreML routes compute to the ANE with zero Python overhead.

### hn-nsf optimization log

The initial naive Swift implementation ran at 166 ms (3s) and 554 ms (10s) — **slower than PyTorch**. Profiling identified per-sample Gaussian RNG (`Float.gaussianRandom` via Box-Muller with `RandomNumberGenerator` protocol dispatch) as the bottleneck at 154 ms / 648k calls. Replacing with bulk `generateGaussianNoise()` (pre-allocated buffer, direct xorshift64, no protocol dispatch) reduced hn-nsf to 14 ms (3s) and 45 ms (10s) — a **12x speedup**.

### Key findings

1. **DecoderPre is the new bottleneck.** At 44-104 ms, it's 48-46% of total pipeline time (with bridge). The rest of the pipeline is already fast enough.

2. **Duration CoreML is 3.8x faster than PyTorch.** 13.6 ms vs ~50 ms. This was an unknown prior to this measurement — the plan's latency budget had it marked as UNKNOWN.

3. **F0Ntrain exports cleanly to CoreML.** Despite containing AdainResBlk1d (the same module type that caused issues in the decoder-only export), F0Ntrain achieves correlation 0.9999+. The AdaIN broadcast issue may be graph-size dependent.

4. **hn-nsf Swift matches PyTorch speed** after optimization. The 14-45 ms range is comparable to the 30-40 ms PyTorch HAR builder that includes decoder-pre + hn-nsf. The Swift hn-nsf alone is slightly slower than PyTorch hn-nsf alone, but eliminates Python interpreter overhead.

5. **The "< 30 ms pre-decoder" target from the plan is not achievable with the bridge.** Pre-decoder with bridge is ~77 ms (3s) / ~187 ms (10s). Without the bridge (Phase 4), it would be ~32 ms (3s) / ~83 ms (10s) — close to target for 3s, over for 10s.

### Provenance

- Machine: Apple M2 Ultra, 64 GB
- Git: current branch (swift-prefix-rewrite-v1 work)
- Swift: Apple Swift version 6.1 (swiftlang-6.1.0.110.21 clang-1700.0.13.3)
- CoreML stage results: `outputs/swift_prefix_stage_bench.json`
- Duration model: `coreml/kokoro_duration.mlpackage`
- F0Ntrain models: `coreml/kokoro_f0ntrain_t120.mlpackage`, `coreml/kokoro_f0ntrain_t400.mlpackage`

### Phase 4 update: DecoderPre CoreML export succeeded

The AdaIN export risk did NOT materialize. DecoderPre (F0_conv + N_conv + encode + decode blocks) exports to CoreML with correlation 1.000000 (3s) and 0.999999 (10s). The debug-notes decoder-only issue was specific to SourceModuleHnNSF, not the decode blocks.

**DecoderPre CoreML predict latency:**

| Bucket | CoreML | PyTorch bridge | Speedup |
| --- | --- | --- | --- |
| 3s | 2.63 ms | 44.7 ms | 17x |
| 10s | 6.49 ms | 103.8 ms | 16x |

**Updated pipeline totals (all CoreML, no bridge):**

| Stage | 3s bucket | 10s bucket |
| --- | --- | --- |
| Duration CoreML | 13.6 ms | 13.6 ms |
| Alignment + matrix ops | ~1 ms | ~1 ms |
| F0Ntrain CoreML | 3.1 ms | 23.4 ms |
| Padding | ~0.5 ms | ~0.5 ms |
| DecoderPre CoreML | 2.6 ms | 6.5 ms |
| hn-nsf Swift | 14 ms | 45 ms |
| GeneratorFromHar CoreML | 16.7 ms | 41.0 ms |
| **Total** | **~51.5 ms** | **~131 ms** |

**Speedup vs baselines (bakeoff v2):**

| Bucket | Swift pipeline | Python Config A | Speedup vs A | MPS (Config D) | Speedup vs MPS |
| --- | --- | --- | --- | --- | --- |
| 3s | ~51.5 ms | 151 ms | 2.9x | 215–287 ms | **4.2–5.6x** |
| 10s | ~131 ms | 274 ms | 2.1x | 351–436 ms | **2.7–3.3x** |

### Plan reference

Full plan: `README/Plans/swift-prefix-rewrite-v1.md`

---

## Bakeoff v3: Swift pipeline (Config F) vs Python baselines

**First collected:** 2026-04-15
**Status:** Complete (M2 Ultra; M2 Air deferred to Phase 7)

### Summary

Controlled counterbalanced comparison of the Swift+CoreML pipeline (Config F) against Python HAR-post (Config A), PyTorch MPS (Config D), and PyTorch CPU (Config E). Same methodology as bakeoff v2: 4 frozen inputs, 5 counterbalanced repetitions, warm median. Config F is **1.5-2.7x faster than PyTorch MPS**, **1.4-1.7x faster than Python HAR-post**, and **2.7-5.1x faster than PyTorch CPU**, achieving **18-51x realtime** on M2 Ultra.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- |
| tiny | 1.55s | 122 ms | 127 ms | 233 ms | **86 ms** |
| short | 2.80s | 136 ms | 190 ms | 358 ms | **84 ms** |
| medium | 6.58s | 232 ms | 348 ms | 655 ms | **166 ms** |
| long | 8.35s | 286 ms | 449 ms | 848 ms | **165 ms** |

### RTF (wall time / audio duration)

| Input | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- |
| tiny | 0.079 (13x RT) | 0.082 | 0.150 | **0.055 (18x RT)** |
| short | 0.048 (21x RT) | 0.068 | 0.128 | **0.030 (34x RT)** |
| medium | 0.035 (28x RT) | 0.053 | 0.099 | **0.025 (40x RT)** |
| long | 0.034 (29x RT) | 0.054 | 0.102 | **0.020 (51x RT)** |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | --- | --- | --- |
| tiny | 1.4x | 1.5x | 2.7x |
| short | 1.6x | 2.3x | 4.3x |
| medium | 1.4x | 2.1x | 3.9x |
| long | 1.7x | 2.7x | 5.1x |

### Gate 6: How much faster is the Swift pipeline vs MPS and Python?

**vs MPS (the "just use the GPU" baseline):** Config F is **1.5-2.7x faster** than PyTorch MPS. MPS suffers from `aten::angle` CPU fallback and Python interpreter overhead. The Swift+CoreML path routes compute to the ANE with zero Python overhead. The gap grows with input length because MPS scales linearly while CoreML scales sublinearly.

**vs Python HAR-post:** Config F is **1.4-1.7x faster** than Config A. The speedup is modest because both pipelines share the same GeneratorFromHar CoreML predict call — the irreducible floor. The savings come from replacing PyTorch CPU inference (Duration, F0Ntrain, DecoderPre) with CoreML and eliminating Python orchestration overhead.

### Interpretation

1. **Config F beats MPS by 1.5-2.7x everywhere.** This is the key result. MPS is what a developer reaches for first on Apple Silicon ("just use the GPU"), and the Swift+CoreML pipeline decisively beats it across all input lengths. The gap grows with duration because MPS scales linearly while CoreML scales sublinearly.

2. **Config F achieves 18-51x realtime.** The fastest result is `long` at 165ms for 8.35s audio = 51x realtime. Even `tiny` (86ms for 1.55s) is 18x realtime.

3. **The speedup vs Python HAR-post is consistent (1.4-1.7x)** rather than scaling dramatically with input length. Both pipelines share the same GeneratorFromHar CoreML predict call, which is the irreducible floor.

4. **Config F vs CPU scales strongly with length.** 2.7x at tiny, 5.1x at long — because PyTorch CPU scales linearly with sequence length while the CoreML models scale sublinearly.

### Provenance

- Machine: Apple M2 Ultra, 64 GB
- Git: current main branch
- Swift: Apple Swift version 6.1
- Torch: 2.6.0 / coremltools: 8.3.0
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m2_ultra_v3.json`

### Plan reference

Bakeoff plan Phase 6: `README/Plans/kokoro-bakeoff-v2.md`
Swift pipeline plan: `README/Plans/swift-prefix-rewrite-v1.md`

---

## Bakeoff v4: Extended duration range (3s-30s) on M2 Ultra

**First collected:** 2026-04-15
**Status:** Complete

### Summary

Extended the bakeoff to cover the full range of practical audio durations: 3s, 7s, 15s, and 30s. Required exporting new bucket models (7s, 15s, 30s) and Duration models at enumerated token sizes [32, 64, 128, 256, 512]. The Swift pipeline (Config F) scales sublinearly — 30s of audio in 349 ms = **79x realtime** — and the speedup vs PyTorch CPU grows to **6.2x** at 30s.

### What changed from v3

- **Inputs:** Renamed from tiny/short/medium/long to 3s/7s/15s/30s targeting specific audio durations
- **Duration model:** Per-size exports [32, 64, 128, 256, 512] (E5RT can't handle RangeDim/EnumeratedShapes)
- **New bucket models:** 7s, 15s, 30s for F0Ntrain, DecoderPre, and GeneratorFromHar (all pass 0.99+ correlation)
- **New bucket models:** 7s, 15s, 30s for F0Ntrain, DecoderPre, and GeneratorFromHar (all pass 0.99+ correlation)
- **Config D (MPS)** added in a separate pass with `PYTORCH_ENABLE_MPS_FALLBACK=1`

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | Bucket | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 3s | 161 ms | 171 ms | 324 ms | **65 ms** |
| 7s | 6.75s | 7s | 219 ms | 320 ms | 588 ms | **142 ms** |
| 15s | 13.90s | 15s | 276 ms | 611 ms | 1133 ms | **254 ms** |
| 30s | 27.38s | 30s | 435 ms | 1247 ms | 2162 ms | **349 ms** |

### RTF and realtime factor

| Input | Audio | A RTF | D RTF | E RTF | F RTF | F realtime |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 0.058 | 0.061 | 0.116 | **0.023** | **43x RT** |
| 7s | 6.75s | 0.032 | 0.047 | 0.087 | **0.021** | **48x RT** |
| 15s | 13.90s | 0.020 | 0.044 | 0.082 | **0.018** | **55x RT** |
| 30s | 27.38s | 0.016 | 0.046 | 0.079 | **0.013** | **79x RT** |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | --- | --- | --- |
| 3s | **2.5x** | **2.6x** | **5.0x** |
| 7s | **1.5x** | **2.3x** | **4.1x** |
| 15s | **1.1x** | **2.4x** | **4.5x** |
| 30s | **1.2x** | **3.6x** | **6.2x** |

### Interpretation

1. **F vs MPS grows from 2.6x to 3.6x with duration.** This is the headline result. MPS is the natural "just use the GPU" baseline, and the Swift+CoreML pipeline beats it decisively — and the gap widens with longer inputs. At 30s, MPS takes 1.25s vs Swift's 349ms. The `aten::angle` CPU fallback on MPS and Python interpreter overhead are the primary reasons MPS falls behind.

2. **Config F scales sublinearly with duration.** 3s audio -> 65 ms, 30s audio -> 349 ms. A 10x increase in audio duration costs only 5.4x more wall time. At 30s, Config F achieves 79x realtime. MPS and CPU both scale linearly.

3. **F vs A (Python HAR-post) is largest at short durations (2.5x at 3s)** where Python orchestration overhead dominates. At 15s the gap narrows to 1.1x — both pipelines are CoreML-predict-bound at that point.

4. **F vs CPU scales strongly with duration (5.0x -> 6.2x).** At 30s, CPU takes 2.2 seconds vs Swift's 349 ms.

5. **The 30s input (476 tokens) validates the Duration model expansion.** The T=512 Duration model handles 476 tokens correctly. Without the enumerated model export, this input would have been impossible (old T=128 model only supported ~120 real tokens after BOS/EOS).

### Provenance

- Machine: Apple M2 Ultra, 64 GB
- Git: current main branch
- Swift: Apple Swift version 6.1
- Configs: A, E, F (5 reps each), D (5 reps, separate pass with MPS fallback)
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m2_ultra_v4.json`, `outputs/bakeoff/results_m2_ultra_v4_mps.json`

---

## Bakeoff v4: Extended duration range (3s-30s) on M2 MacBook Air

**First collected:** 2026-04-15
**Status:** Complete

### Summary

Same bakeoff v4 harness and frozen inputs as the M2 Ultra run, now on a base M2 MacBook Air (8-core CPU, 8-core GPU, 16-core ANE, 24 GB). Config A (Python HAR-post) remains the fastest pipeline on the M2 Air at longer durations, while Config F (Swift + CoreML) is faster only at short inputs (3s: 1.3x, 7s: 1.4x). At 15s and 30s, Config F is **slower** than Config A — a reversal of the M2 Ultra result. The bottleneck is GeneratorFromHar `predict()` via Swift's `MLModel.prediction()`, which runs **7.4x slower** than the equivalent Python `coremltools` call on the same 30s model.

### What was measured

- **Config A:** Shipping Python HAR-post hybrid (PyTorch prefix + CoreML decoder)
- **Config D:** PyTorch end-to-end on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- **Config E:** PyTorch end-to-end on CPU
- **Config F:** Swift + CoreML pipeline (5 models + Swift hn-nsf DSP)

All configs used identical frozen inputs, `voice=af_heart`, `speed=1.0`, `torch.manual_seed(0)`.

### Method

`scripts/bakeoff_harness.py` with `run --configs a,d,e,f --iterations 5 --order-seed 0`. Counterbalanced ordering, models preloaded and warmed. `PYTORCH_ENABLE_MPS_FALLBACK=1` set. Config F ran in a separate pass due to warmup key fix; all models warmed before timed runs.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | Bucket | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 3s | 322 ms | 334 ms | 759 ms | **256 ms** |
| 7s | 6.75s | 7s | 507 ms | 689 ms | 1920 ms | **371 ms** |
| 15s | 13.90s | 15s | **554 ms** | 1362 ms | 3966 ms | 833 ms |
| 30s | 27.38s | 30s | **854 ms** | 2880 ms | 8098 ms | 2280 ms |

### RTF and realtime factor

| Input | Audio | A RTF | D RTF | E RTF | F RTF | F realtime |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 0.115 | 0.119 | 0.271 | **0.091** | **11x RT** |
| 7s | 6.75s | 0.075 | 0.102 | 0.285 | **0.055** | **18x RT** |
| 15s | 13.90s | **0.040** | 0.098 | 0.285 | 0.060 | 17x RT |
| 30s | 27.38s | **0.031** | 0.105 | 0.296 | 0.083 | 12x RT |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | --- | --- | --- |
| 3s | **1.3x** | **1.3x** | **3.0x** |
| 7s | **1.4x** | **1.9x** | **5.2x** |
| 15s | 0.7x (A wins) | **1.6x** | **4.8x** |
| 30s | 0.4x (A wins) | **1.3x** | **3.6x** |

### Config F stage breakdown (warm median, ms)

| Stage | 3s | 7s | 15s | 30s |
| --- | --- | --- | --- | --- |
| Duration CoreML | 12 ms | 20 ms | 51 ms | 105 ms |
| Matrix ops | 21 ms | 40 ms | 88 ms | 4 ms |
| F0Ntrain CoreML | 4 ms | 26 ms | 25 ms | 26 ms |
| Padding | 0.1 ms | 0.1 ms | 0.3 ms | 0.7 ms |
| DecoderPre CoreML | 3 ms | 7 ms | 13 ms | 31 ms |
| hn-nsf Swift | 16 ms | 37 ms | 75 ms | 161 ms |
| GeneratorFromHar CoreML | 204 ms | 261 ms | 681 ms | 1971 ms |

### Cross-machine comparison (Config F)

| Input | M2 Ultra | M2 Air | Air/Ultra ratio |
| --- | --- | --- | --- |
| 3s | 65 ms | 256 ms | 3.9x slower |
| 7s | 142 ms | 371 ms | 2.6x slower |
| 15s | 254 ms | 833 ms | 3.3x slower |
| 30s | 349 ms | 2280 ms | 6.5x slower |

### Cross-machine comparison (Config A)

| Input | M2 Ultra | M2 Air | Air/Ultra ratio |
| --- | --- | --- | --- |
| 3s | 161 ms | 322 ms | 2.0x slower |
| 7s | 219 ms | 507 ms | 2.3x slower |
| 15s | 276 ms | 554 ms | 2.0x slower |
| 30s | 435 ms | 854 ms | 2.0x slower |

### The GeneratorFromHar anomaly

The headline finding is that Config F's GeneratorFromHar `predict()` is dramatically slower on M2 Air than Config A's call to the *same* model at the same bucket size:

| Bucket | Config A (Python coremltools) | Config F (Swift MLModel) | Ratio |
| --- | --- | --- | --- |
| 30s | 259 ms | 1971 ms | **7.6x** |

Both call the same `kokoro_decoder_har_post_30s.mlpackage`. The difference is the API layer:
- Config A: Python `coremltools.models.MLModel.predict()` 
- Config F: Swift `MLModel.prediction()` via subprocess

Possible explanations:
1. **Compute unit routing:** The Swift binary may not be requesting `computeUnits: .all` (ANE), forcing CPU/GPU fallback.
2. **Model compilation caching:** Each Swift subprocess may not benefit from the macOS model compilation cache the way a long-lived Python process does.
3. **Memory pressure:** The M2 Air's 24GB vs M2 Ultra's 64GB may cause memory-constrained scheduling differences.

This anomaly does **not** appear on M2 Ultra (Config F is 1.2x faster than A at 30s), suggesting it's specific to constrained hardware.

### Interpretation

1. **Config F wins at short durations (3s-7s) but loses at long durations (15s-30s) on M2 Air.** The crossover point is around 10s. This is the opposite of M2 Ultra where Config F wins everywhere.

2. **Config A scales consistently: 2.0x slower on Air vs Ultra** across all durations. This is expected given the ~2x difference in memory bandwidth and core count between M2 and M2 Ultra.

3. **Config F scales inconsistently: 2.6-6.5x slower on Air vs Ultra.** The GeneratorFromHar anomaly causes the gap to widen dramatically at longer durations (6.5x at 30s vs 2.6x at 7s).

4. **Config F still beats MPS (1.3-1.9x) and CPU (3.0-5.2x) everywhere.** The Swift pipeline is never the worst option — it just loses to the shipping Python HAR-post path at longer inputs on constrained hardware.

5. **For M2 Air deployment, use Config A for inputs > 10s.** Config F is only beneficial for short inputs. A hybrid routing strategy would be optimal.

### Provenance

- Machine: Apple M2 MacBook Air, 24 GB, macOS 15.5
- Git: current main branch
- Swift: Apple Swift version 6.1
- Torch: 2.6.0 / coremltools: 8.3.0
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m2_air_v4.json` (A, D, E), `outputs/bakeoff/results_m2_air_v4_f.json` (F)
- Config F timeout: 300s (increased from 120s for M2 Air model compilation time)

---

## Bakeoff v5: Corrected benchmark (3s-30s) on M2 MacBook Air

**First collected:** 2026-04-15
**Status:** Complete

### Summary

Reruns the v4 extended-duration bakeoff on M2 Air after fixing **critical measurement bugs** discovered in the audit:

1. **Config A only had 2 bucket models (3s, 10s)** — for 7s/15s/30s inputs, `_select_bucket_seconds` fell back to the 10s model. Config F had all 5 buckets. The v4 "Config F loses to Config A at long durations" was comparing different workloads.
2. **Swift F0Ntrain tFrames mapping was wrong** — hardcoded `400` for all non-3s buckets instead of `560/1200/2400`. This silently truncated 28-64% of aligned features for 15s/30s inputs.
3. **Duplicate matmul** inflated Config F wall time by one redundant 512×T×N matrix multiply.
4. **ANE plan compilation** could contaminate timed blocks after bucket-switch eviction.

With all bugs fixed, **Config F wins at every duration on M2 Air** — the v4 anomaly is eliminated.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | Bucket | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 3s | 355 ms | 394 ms | 736 ms | **200 ms** |
| 7s | 6.75s | 7s | 544 ms | 812 ms | 1985 ms | **326 ms** |
| 15s | 13.90s | 15s | 1178 ms | 1573 ms | 4002 ms | **783 ms** |
| 30s | 27.38s | 30s | 2443 ms | 3350 ms | 8065 ms | **1829 ms** |

### RTF and realtime factor

| Input | Audio | A RTF | D RTF | E RTF | F RTF | F realtime |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 0.127 | 0.141 | 0.263 | **0.071** | **14x RT** |
| 7s | 6.75s | 0.081 | 0.120 | 0.294 | **0.048** | **21x RT** |
| 15s | 13.90s | 0.085 | 0.113 | 0.288 | **0.056** | **18x RT** |
| 30s | 27.38s | 0.089 | 0.122 | 0.295 | **0.067** | **15x RT** |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | --- | --- | --- |
| 3s | **1.8x** | **2.0x** | **3.7x** |
| 7s | **1.7x** | **2.5x** | **6.1x** |
| 15s | **1.5x** | **2.0x** | **5.1x** |
| 30s | **1.3x** | **1.8x** | **4.4x** |

### Config F stage breakdown (warm median, ms)

| Stage | 3s | 7s | 15s | 30s |
| --- | --- | --- | --- | --- |
| Duration CoreML | 8 ms | 13 ms | 27 ms | 83 ms |
| Matrix ops | 18 ms | 36 ms | 73 ms | 2 ms |
| F0Ntrain CoreML | 3 ms | 32 ms | 70 ms | 140 ms |
| Padding | 0.1 ms | 0.1 ms | 0.3 ms | 0.7 ms |
| DecoderPre CoreML | 3 ms | 5 ms | 11 ms | 28 ms |
| hn-nsf Swift | 9 ms | 22 ms | 47 ms | 100 ms |
| GeneratorFromHar CoreML | 158 ms | 218 ms | 552 ms | 1471 ms |

### Cross-machine comparison (Config F)

| Input | M2 Ultra | M2 Air | Air/Ultra ratio |
| --- | --- | --- | --- |
| 3s | 65 ms | 200 ms | 3.1x slower |
| 7s | 142 ms | 326 ms | 2.3x slower |
| 15s | 254 ms | 783 ms | 3.1x slower |
| 30s | 349 ms | 1829 ms | 5.2x slower |

### Cross-machine comparison (Config A)

| Input | M2 Ultra | M2 Air | Air/Ultra ratio |
| --- | --- | --- | --- |
| 3s | 161 ms | 355 ms | 2.2x slower |
| 7s | 219 ms | 544 ms | 2.5x slower |
| 15s | 276 ms | 1178 ms | 4.3x slower |
| 30s | 435 ms | 2443 ms | 5.6x slower |

### What changed from v4

Config A at 15s went from 554 ms to 1178 ms (+624 ms) and at 30s from 854 ms to 2443 ms (+1589 ms) because it now runs the correct bucket model instead of the 10s fallback. Config F at 30s went from 2280 ms to 1829 ms (-451 ms) thanks to removing the duplicate matmul, fixing F0Ntrain truncation (which corrupted pitch predictions), and eliminating ANE plan compilation from timed blocks.

### The v4 "GeneratorFromHar anomaly" — debunked

The v4 finding that "Config F is slower than Config A at 15s/30s on M2 Air" was entirely an artifact of the bucket mismatch. Config A was running a 10s model for 15s/30s inputs while Config F ran the correctly-sized models. Additionally, Config F's F0Ntrain was fed truncated data (tFrames=400 instead of 1200/2400), producing corrupted pitch predictions that cascaded through the pipeline.

With fair bucket parity, Config F is **1.3-1.8x faster than Config A** at every duration on M2 Air, consistent with the M2 Ultra pattern.

### Interpretation

1. **Config F wins everywhere on M2 Air.** The v4 anomaly was a measurement bug, not a hardware limitation. The Swift+CoreML pipeline is 1.3-1.8x faster than Python HAR-post at all durations.

2. **GeneratorFromHar dominates at all durations.** It accounts for 79% of wall time at 3s and 80% at 30s. This is the primary optimization target.

3. **Config F achieves 14-21x realtime on M2 Air.** Peak throughput is at 7s (21x RT). The 30s input drops to 15x RT because GeneratorFromHar scales superlinearly with duration.

4. **Air/Ultra ratio for Config F is 2.3-5.2x.** This is wider than Config A's 2.2-5.6x ratio, but both show the same scaling pattern — the gap grows with duration because longer models stress the M2 Air's reduced ANE cores and memory bandwidth.

5. **Config D (MPS) OOM'd in the main run** due to memory pressure from loading 5 bucket CoreML models simultaneously. A separate MPS-only pass succeeded; results in table above. Config F beats MPS by 1.8-2.5x.

### Provenance

- Machine: Apple M2 MacBook Air, 24 GB, macOS 15.7.5
- Git: main branch, commit `61f1dc5` (audit fixes)
- Swift: Apple Swift version 6.1
- Torch: 2.6.0 / coremltools: 8.3.0
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m2_air_v5.json` (A, E, F), `outputs/bakeoff/results_m2_air_v5_mps.json` (D)
- Config D ran in a separate pass due to MPS OOM in the combined run

---

## Bakeoff v6: Full-pipeline re-export benchmark (3s-30s) on M2 MacBook Air

**First collected:** 2026-04-17
**Status:** Complete

### Summary

Re-ran the corrected v5 bakeoff on M2 Air after regenerating **every Core ML
package** from scratch (Duration T=32/64/128/256/512 padded + exact T=44/105/219/476,
F0Ntrain T=120/280/400/600/1200, DecoderPre {3,7,10,15,30}s, GeneratorFromHar
{3,7,10,15,30}s) on commit `fa2a24d`. Fixed a regression in
`export_synth/wrappers.py` where `CoreMLFriendlyTextEncoder.__init__` double-wrapped
the shared `kmodel.text_encoder.lstm` when `SynthesizerModel` was constructed after
`DurationModel`, failing GeneratorFromHar export with
`AttributeError: 'MaskedBidirectionalLSTM' object has no attribute 'num_layers'`.

**Config F wins at every duration.** End-to-end wall time is **185 ms for 3s audio
(15x realtime) and 3021 ms for 30s audio (9x realtime).** Config F beats Config A
by **1.3–2.5x** and CPU PyTorch by **2.5–4.6x**. MPS OOM'd on 15s and 30s even in
a solo pass (MPS pool cap ~27 GB vs 32 GB resident on this machine).

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | Bucket | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 3s | 461 ms | 739 ms | 723 ms | **185 ms** |
| 7s | 6.75s | 7s | 771 ms | 907 ms | 1839 ms | **396 ms** |
| 15s | 13.90s | 15s | 1896 ms | OOM | 3737 ms | **1326 ms** |
| 30s | 27.38s | 30s | 3918 ms | OOM | 7567 ms | **3021 ms** |

### RTF and realtime factor

| Input | Audio | A RTF | D RTF | E RTF | F RTF | F realtime |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 0.165 | 0.264 | 0.258 | **0.066** | **15x RT** |
| 7s | 6.75s | 0.114 | 0.134 | 0.272 | **0.059** | **17x RT** |
| 15s | 13.90s | 0.136 | OOM | 0.269 | **0.095** | **10x RT** |
| 30s | 27.38s | 0.143 | OOM | 0.276 | **0.110** | **9x RT** |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | --- | --- | --- |
| 3s | **2.5x** | **4.0x** | **3.9x** |
| 7s | **1.9x** | **2.3x** | **4.6x** |
| 15s | **1.4x** | OOM | **2.8x** |
| 30s | **1.3x** | OOM | **2.5x** |

### Config F stage breakdown (warm median, ms)

| Stage | 3s | 7s | 15s | 30s |
| --- | --- | --- | --- | --- |
| Duration CoreML | 10.6 ms | 12.8 ms | 39.6 ms | 48.5 ms |
| Matrix ops | 0.1 ms | 0.3 ms | 0.5 ms | 1.0 ms |
| F0Ntrain CoreML | 3.3 ms | 7.5 ms | 35.1 ms | 68.7 ms |
| Padding | 0.0 ms | 0.0 ms | 0.0 ms | 0.1 ms |
| DecoderPre CoreML | 2.6 ms | 4.9 ms | 13.0 ms | 28.4 ms |
| hn-nsf Swift | 9.0 ms | 21.3 ms | 46.9 ms | 95.7 ms |
| GeneratorFromHar CoreML | 159.1 ms | 348.9 ms | 1188.4 ms | 2780.1 ms |

### Delta vs v5 (M2 Air)

| Input | v5 F | v6 F | Δ F | v5 A | v6 A | Δ A |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 200 ms | 185 ms | **-8%** | 355 ms | 461 ms | +30% |
| 7s | 326 ms | 396 ms | +22% | 544 ms | 771 ms | +42% |
| 15s | 783 ms | 1326 ms | +69% | 1178 ms | 1896 ms | +61% |
| 30s | 1829 ms | 3021 ms | +65% | 2443 ms | 3918 ms | +60% |

GeneratorFromHar at 15s/30s went from 552/1471 ms (v5) to 1188/2780 ms (v6) —
roughly 2x slower. Both pipelines regressed at 7s+, but Config F regressed more
at longer buckets because GeneratorFromHar is the dominant stage.

Candidate causes (not yet isolated): (1) `torch==2.5.0` in this machine's
`requirements-bakeoff.txt` vs `torch==2.6.0` in the v5 provenance — different
tracing behavior may change MIL op selection; (2) thermal state after the
back-to-back export run that preceded the bakeoff (30s-bucket exports are
CPU-intensive); (3) variance in CoreML ANE plan compilation caching across
fresh `.mlpackage` directories. Re-running with a cold machine and matching
torch version is the next step to isolate this.

### Interpretation

1. **Config F still wins at every duration.** 185 ms for 3s (15x RT) and 3021 ms
   for 30s (9x RT) remain the best numbers across all four configs on this
   machine. Pitch parity and bucket geometry are intact after the full
   re-export.

2. **GeneratorFromHar dominates Config F wall time more than ever.** It accounts
   for 86% of wall time at 3s and 92% at 30s — up from 79–80% in v5. Any further
   optimization must target this package; everything else is already sub-100 ms.

3. **MPS (Config D) is unusable on 24 GB M2 Air for 15s+ inputs.** Even in a solo
   pass with `PYTORCH_ENABLE_MPS_FALLBACK=1` and no other configs loaded, the
   MPS pool cap (27 GB) conflicts with the 32 GB already allocated for this
   process. The production app should never route to MPS on this hardware.

4. **15s/30s regression vs v5 is real and worth investigating.** The +60-70%
   delta on GeneratorFromHar is large enough that it cannot be attributed to
   run-to-run variance (5 iterations, medians). Likely attribution: torch 2.5
   vs 2.6 export, thermal state, or both.

5. **Config D partial data is retained rather than dropped.** 3s/7s still
   provide an apples-to-apples MPS baseline; 15s/30s MPS numbers are marked OOM.

### Provenance

- Machine: Apple M2 MacBook Air, 24 GB, macOS 15.7.5
- Git: main branch, commit `fa2a24d`
- Swift: Apple Swift version 6.2.4
- Torch: 2.5.0 / coremltools: 8.3.0
- Order seed: 0, iterations: 5
- Machine id: `m2_air_v6` (A, E, F combined); `m2_air_v6_mps` (D solo pass)
- Results: `outputs/bakeoff/results_m2_air_v6.json` (A, E, F),
  `outputs/bakeoff/results_m2_air_v6_mps.json` (D partial: 3s/7s ok, 15s/30s OOM)
- Export fix: `export_synth/wrappers.py` — `CoreMLFriendlyTextEncoder.__init__`
  and `CoreMLFriendlyDurationEncoder.__init__` made idempotent on already-masked
  LSTM blocks so `SynthesizerModel(kmodel)` after `DurationModel(kmodel)` no
  longer re-wraps `MaskedBidirectionalLSTM`.

---

## Bakeoff v2: Controlled benchmark on M1 Mini

**First collected:** 2026-04-15
**Status:** Complete

### Summary

Same bakeoff harness and frozen inputs as the M2 Ultra and M2 Air runs, now on a base M1 Mac Mini (8-core CPU, 8-core GPU, 16-core ANE, 16 GB). Config A (shipping HAR-post) is **2.3–4.8x faster than PyTorch CPU** and **6–14x realtime**. CoreML `predict()` consumes 50–51% of wall time on the 10s bucket and 40–42% on the 3s bucket — a balanced split between CPU-side overhead and CoreML inference, similar to the M2 Air pattern.

### What was measured

- **Config A:** Shipping hybrid HAR-post path (`coreml/kokoro_decoder_har_post_{3,10}s.mlpackage`)
- **Config B:** Naive decoder-only 10s artifact, `compute_units=ALL`
- **Config C:** Naive decoder-only 10s artifact, `compute_units=CPU_AND_GPU`
- **Config D:** PyTorch end-to-end on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- **Config E:** PyTorch end-to-end on CPU

All configs used identical frozen inputs, `voice=af_heart`, `speed=1.0`, `torch.manual_seed(0)`.

### Method

Same as M2 Ultra run: `scripts/bakeoff_harness.py` with `run --configs a,b,c,d,e --iterations 5 --order-seed 0`. Counterbalanced ordering, models preloaded and warmed. `PYTORCH_ENABLE_MPS_FALLBACK=1` set for the full run (Config D).

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (HAR-post) | B (.all) | C (.cpuAndGPU) | D (MPS) | E (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `1.55s` | `259 ms` | `1589 ms` | `1556 ms` | `360 ms` | `601 ms` |
| `short` | `2.80s` | `240 ms` | `1561 ms` | `1577 ms` | `750 ms` | `896 ms` |
| `medium` | `6.58s` | `603 ms` | `1624 ms` | `1598 ms` | `1025 ms` | `2223 ms` |
| `long` | `8.35s` | `600 ms` | `1628 ms` | `1625 ms` | `1257 ms` | `2849 ms` |

### RTF (canonical audio duration / wall time)

| Input | A (HAR-post) | B (.all) | C (.cpuAndGPU) | D (MPS) | E (CPU) |
| --- | --- | --- | --- | --- | --- |
| `tiny` | `0.167` (6x RT) | `1.025` | `1.004` | `0.232` | `0.388` |
| `short` | `0.086` (12x RT) | `0.557` | `0.563` | `0.268` | `0.320` |
| `medium` | `0.092` (11x RT) | `0.247` | `0.243` | `0.156` | `0.338` |
| `long` | `0.072` (14x RT) | `0.195` | `0.195` | `0.151` | `0.341` |

### Speedup: Config A vs PyTorch baselines

| Input | Audio | A vs E (CPU) | A vs D (MPS) |
| --- | --- | --- | --- |
| `tiny` | `1.55s` | `2.3x` | `1.4x` |
| `short` | `2.80s` | `3.7x` | `3.1x` |
| `medium` | `6.58s` | `3.7x` | `1.7x` |
| `long` | `8.35s` | `4.8x` | `2.1x` |

### Config A stage breakdown (warm median)

| Input | Bucket | Prefix extract | HAR builder (CPU) | CoreML predict | Orchestration | Total |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `3s` | `67.3 ms` (26%) | `74.2 ms` (29%) | `102.4 ms` (40%) | `2.1 ms` | `259 ms` |
| `short` | `3s` | `73.5 ms` (31%) | `64.3 ms` (27%) | `100.6 ms` (42%) | `1.6 ms` | `240 ms` |
| `medium` | `10s` | `131.4 ms` (22%) | `153.6 ms` (25%) | `310.4 ms` (51%) | `2.1 ms` | `603 ms` |
| `long` | `10s` | `149.2 ms` (25%) | `149.4 ms` (25%) | `298.0 ms` (50%) | `1.6 ms` | `600 ms` |

### Interpretation

1. **Config A is 6–14x realtime on M1 Mini.** The shortest input (`tiny`, 1.55s audio) completes in 259 ms; the longest (`long`, 8.35s audio) in 600 ms. Roughly 1.7–2.2x slower than M2 Ultra, similar to M2 Air.

2. **CoreML predict scales with bucket size.** On the 3s bucket, predict is ~100 ms (40% of wall time). On the 10s bucket, predict jumps to ~300 ms (50% of wall time). CPU-side overhead (prefix + HAR builder) stays proportional to input length regardless of bucket.

3. **Speedup vs CPU scales strongly with duration.** The 2.3x speedup at `tiny` grows to 4.8x at `long` — matching the M2 Air scaling curve and exceeding M2 Ultra's 1.7x → 3.5x range. PyTorch CPU scales linearly with sequence length while CoreML predict is sublinear.

4. **MPS shows high variance.** Config D (MPS) has significant run-to-run variation (std up to 1.8s on `medium`), consistent with `aten::angle` fallback and thermal throttling. Treat as directional only.

5. **Configs B and C remain indistinguishable.** Both hover around 1.55–1.63s regardless of input length, consistent with all other machines. ANE participation under `.all` remains **indeterminate** without powermetrics telemetry.

6. **No OOM on 16 GB.** All five configs loaded and ran successfully on the M1 Mini's 16 GB unified memory. This resolves the Phase 4 deferral in the bakeoff plan — M1 Mini is viable as a benchmark target.

### Cross-machine comparison: M1 Mini vs M2 Air vs M2 Ultra

| Input | M1 Mini A | M2 Air A | M2 Ultra A | M1/Ultra | M1 Mini E | M2 Air E | M2 Ultra E | M1/Ultra |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `259 ms` | `329 ms` | `151 ms` | `1.7x` | `601 ms` | `436 ms` | `258 ms` | `2.3x` |
| `short` | `240 ms` | `323 ms` | `155 ms` | `1.5x` | `896 ms` | `819 ms` | `396 ms` | `2.3x` |
| `medium` | `603 ms` | `521 ms` | `283 ms` | `2.1x` | `2223 ms` | `1929 ms` | `782 ms` | `2.8x` |
| `long` | `600 ms` | `513 ms` | `274 ms` | `2.2x` | `2849 ms` | `2441 ms` | `966 ms` | `2.9x` |

Config A on M1 Mini is 1.5–2.2x slower than M2 Ultra and surprisingly close to M2 Air (within 20% on short inputs, ~15% slower on long inputs). The M1 Mini's CoreML predict is ~100 ms on the 3s bucket (vs M2 Air's 220 ms and M2 Ultra's 19–57 ms), suggesting the M1's 16-core ANE handles the small bucket efficiently. On the 10s bucket, predict jumps to ~300 ms, closer to the M2 Air's 260 ms. PyTorch CPU is 2.3–2.9x slower than M2 Ultra, consistent with the M1 → M2 Ultra compute gap.

### Provenance

- Machine: Apple M1 Mac Mini, 16 GB
- Git: `46c9b7f0517e`
- Torch: `2.5.0` / coremltools: `8.3.0` / numpy: `1.26.4`
- Order seed: `0`, iterations: `5`
- Results: `outputs/bakeoff/results_apple_m1.json`

### Plan reference

Full experiment design: `README/Plans/kokoro-bakeoff-v2.md`

---

## Bakeoff v2: Controlled benchmark on M2 MacBook Air

**First collected:** 2026-04-15
**Status:** Complete

### Summary

Same bakeoff harness and frozen inputs as the M2 Ultra run above, now on a consumer M2 MacBook Air (8-core CPU, 10-core GPU, 16-core ANE, 24 GB). Config A (shipping HAR-post) is **2.5–4.8x faster than PyTorch CPU** on medium-to-long inputs and **5–16x realtime**. CoreML `predict()` is substantially slower than on M2 Ultra (234–262 ms vs 19–84 ms), now consuming 50–71% of wall time — the bottleneck has shifted from CPU-side overhead to CoreML inference on this lower-end chip.

### What was measured

- **Config A:** Shipping hybrid HAR-post path (`coreml/kokoro_decoder_har_post_{3,10}s.mlpackage`)
- **Config B:** Naive decoder-only 10s artifact, `compute_units=ALL`
- **Config C:** Naive decoder-only 10s artifact, `compute_units=CPU_AND_GPU`
- **Config D:** PyTorch end-to-end on MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- **Config E:** PyTorch end-to-end on CPU

All configs used identical frozen inputs, `voice=af_heart`, `speed=1.0`, `torch.manual_seed(0)`.

### Method

Same as M2 Ultra run: `scripts/bakeoff_harness.py` with `run --configs a,b,c,d,e --iterations 5 --order-seed 0`. Counterbalanced ordering, models preloaded and warmed. Config D was run in a separate pass with `PYTORCH_ENABLE_MPS_FALLBACK=1` set.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (HAR-post) | B (.all) | C (.cpuAndGPU) | D (MPS) | E (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `1.55s` | `329 ms` | `1453 ms` | `1437 ms` | `194 ms` | `436 ms` |
| `short` | `2.80s` | `323 ms` | `1431 ms` | `1447 ms` | `329 ms` | `819 ms` |
| `medium` | `6.58s` | `521 ms` | `1494 ms` | `1475 ms` | `682 ms` | `1929 ms` |
| `long` | `8.35s` | `513 ms` | `1531 ms` | `1523 ms` | `860 ms` | `2441 ms` |

### RTF (canonical audio duration / wall time)

| Input | A (HAR-post) | B (.all) | C (.cpuAndGPU) | D (MPS) | E (CPU) |
| --- | --- | --- | --- | --- | --- |
| `tiny` | `0.212` (5x RT) | `0.937` | `0.927` | `0.125` (8x RT) | `0.281` |
| `short` | `0.115` (9x RT) | `0.511` | `0.517` | `0.118` (9x RT) | `0.293` |
| `medium` | `0.079` (13x RT) | `0.227` | `0.224` | `0.104` (10x RT) | `0.293` |
| `long` | `0.061` (16x RT) | `0.183` | `0.182` | `0.103` (10x RT) | `0.292` |

### Speedup: Config A vs PyTorch baselines

| Input | Audio | A vs E (CPU) | A vs D (MPS) |
| --- | --- | --- | --- |
| `tiny` | `1.55s` | `1.3x` | `0.6x` (MPS faster) |
| `short` | `2.80s` | `2.5x` | `1.0x` |
| `medium` | `6.58s` | `3.7x` | `1.3x` |
| `long` | `8.35s` | `4.8x` | `1.7x` |

### Config A stage breakdown (warm median)

| Input | Bucket | Prefix extract | HAR builder (CPU) | CoreML predict | Orchestration | Total |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `3s` | `46.9 ms` (14%) | `45.8 ms` (14%) | `234.0 ms` (71%) | `1.8 ms` | `329 ms` |
| `short` | `3s` | `64.9 ms` (20%) | `46.4 ms` (14%) | `220.5 ms` (68%) | `1.8 ms` | `323 ms` |
| `medium` | `10s` | `107.4 ms` (21%) | `123.5 ms` (24%) | `262.3 ms` (50%) | `1.8 ms` | `521 ms` |
| `long` | `10s` | `137.5 ms` (27%) | `113.7 ms` (22%) | `259.6 ms` (51%) | `1.8 ms` | `513 ms` |

### Interpretation

1. **Config A is 5–16x realtime on M2 Air.** The shortest input (`tiny`, 1.55s audio) completes in 329 ms; the longest (`long`, 8.35s audio) in 513 ms. Roughly 2x slower than M2 Ultra across the board.

2. **CoreML predict is now the bottleneck.** On M2 Ultra, CPU-side overhead dominated (62–86% of wall time). On M2 Air, CoreML `predict()` takes 220–262 ms (50–71% of wall time), while prefix extract + HAR builder are roughly similar to M2 Ultra. The M2 Air's 16-core ANE (vs Ultra's 32-core) and lower memory bandwidth explain the shift.

3. **Speedup vs CPU scales with duration.** The 1.3x speedup at `tiny` grows to 4.8x at `long` — even steeper scaling than M2 Ultra (1.7x → 3.5x) because PyTorch CPU is proportionally slower on M2 Air while CoreML predict stays relatively flat.

4. **MPS is surprisingly competitive on short inputs.** Config D (PyTorch MPS) beats Config A on `tiny` (194 ms vs 329 ms) and ties on `short`. Config A only pulls ahead at `medium` (1.3x) and `long` (1.7x). This is the opposite of M2 Ultra where MPS was consistently slower — suggesting the M2 Air's 10-core GPU handles this workload well, and the CoreML predict overhead (220–260 ms) is the limiter on short inputs.

5. **Configs B and C remain indistinguishable.** Both hover around 1.4–1.5s regardless of input length, consistent with M2 Ultra. ANE participation under `.all` remains **indeterminate** without powermetrics telemetry.

### Cross-machine comparison: M2 Air vs M2 Ultra

| Input | M2 Air A | M2 Ultra A | Air/Ultra | M2 Air D | M2 Ultra D | Air/Ultra | M2 Air E | M2 Ultra E | Air/Ultra |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `329 ms` | `151 ms` | `2.2x` | `194 ms` | `215 ms` | `0.9x` | `436 ms` | `258 ms` | `1.7x` |
| `short` | `323 ms` | `155 ms` | `2.1x` | `329 ms` | `287 ms` | `1.1x` | `819 ms` | `396 ms` | `2.1x` |
| `medium` | `521 ms` | `283 ms` | `1.8x` | `682 ms` | `351 ms` | `1.9x` | `1929 ms` | `782 ms` | `2.5x` |
| `long` | `513 ms` | `274 ms` | `1.9x` | `860 ms` | `436 ms` | `2.0x` | `2441 ms` | `966 ms` | `2.5x` |

Config A scales roughly 2x between Air and Ultra. PyTorch CPU scales 1.7–2.5x. MPS (Config D) is nearly identical on short inputs across both machines but diverges on longer ones — consistent with the Ultra's larger GPU providing more parallelism for longer sequences. The CoreML path degrades more gracefully than CPU because the CPU-side prefix cost is similar on both machines — only the predict portion scales with ANE core count.

### Switching penalty analysis

The counterbalanced ordering shuffles configs between repetitions, so Config A sometimes runs immediately after B/C (decoder-only), potentially paying ANE model-reload costs. Per-iteration predict times, grouped by Config A's position in the execution order:

| Config A position | Mean predict | Median predict | N |
| --- | --- | --- | --- |
| Position 0 (runs first) | `225 ms` | `226 ms` | 4 |
| Position 2+ (after B/C) | `241 ms` | `242 ms` | 15 |

**Switching penalty: ~16 ms (~7%).** The ANE likely reloads the HAR-post model plan after running the decoder-only model, but the cost is small.

One outlier was excluded: `medium` on iteration 0 spiked to `2101 ms` predict (vs typical 237–275 ms). This is a one-time ANE compilation hit for the 10s bucket — the warmup pass may not have fully compiled the 10s model for all input shapes. Every subsequent `medium` run was normal regardless of position.

**Conclusion:** ~93% of the M2 Air vs M2 Ultra gap is real compute (16 vs 32 ANE cores, lower memory bandwidth). The counterbalanced switching penalty adds ~7% noise to predict times but does not explain the cross-machine difference.

### Provenance

- Machine: Apple M2 MacBook Air, 24 GB
- Git: `1426c2182b5d`
- Torch: `2.5.0` / coremltools: `8.3.0` / numpy: `1.26.4`
- Order seed: `0`, iterations: `5`
- Results: `outputs/bakeoff/results_m2_air.json`, `outputs/bakeoff/results_m2_air_mps.json`

### Plan reference

Full experiment design: `README/Plans/kokoro-bakeoff-v2.md`

---

## Bakeoff v3: Swift pipeline on M2 MacBook Air

**First collected:** TBD
**Status:** Pending — run `$bakeoff` on M2 Air

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- |
| tiny | 1.55s | ___ ms | ___ ms | ___ ms | **___ ms** |
| short | 2.80s | ___ ms | ___ ms | ___ ms | **___ ms** |
| medium | 6.58s | ___ ms | ___ ms | ___ ms | **___ ms** |
| long | 8.35s | ___ ms | ___ ms | ___ ms | **___ ms** |

### RTF (wall time / audio duration)

| Input | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- |
| tiny | ___ | ___ | ___ | **___** |
| short | ___ | ___ | ___ | **___** |
| medium | ___ | ___ | ___ | **___** |
| long | ___ | ___ | ___ | **___** |

### Speedup: Config F vs baselines

| Input | F vs A | F vs D | F vs E |
| --- | --- | --- | --- |
| tiny | ___x | ___x | ___x |
| short | ___x | ___x | ___x |
| medium | ___x | ___x | ___x |
| long | ___x | ___x | ___x |

### Interpretation

TBD — fill after running `$bakeoff` on M2 Air.

### Provenance

- Machine: Apple M2 MacBook Air, ___ GB
- Git: ___
- Results: `outputs/bakeoff/results_m2_air_v3.json`

---

## Bakeoff v3: Swift pipeline on M1 Mac Mini

**First collected:** TBD
**Status:** Pending — run `$bakeoff` on M1 Mini

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- |
| tiny | 1.55s | ___ ms | ___ ms | ___ ms | **___ ms** |
| short | 2.80s | ___ ms | ___ ms | ___ ms | **___ ms** |
| medium | 6.58s | ___ ms | ___ ms | ___ ms | **___ ms** |
| long | 8.35s | ___ ms | ___ ms | ___ ms | **___ ms** |

### RTF (wall time / audio duration)

| Input | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- |
| tiny | ___ | ___ | ___ | **___** |
| short | ___ | ___ | ___ | **___** |
| medium | ___ | ___ | ___ | **___** |
| long | ___ | ___ | ___ | **___** |

### Speedup: Config F vs baselines

| Input | F vs A | F vs D | F vs E |
| --- | --- | --- | --- |
| tiny | ___x | ___x | ___x |
| short | ___x | ___x | ___x |
| medium | ___x | ___x | ___x |
| long | ___x | ___x | ___x |

### Interpretation

TBD — fill after running `$bakeoff` on M1 Mini.

### Provenance

- Machine: Apple M1 Mac Mini, ___ GB
- Git: ___
- Results: `outputs/bakeoff/results_m1_mini.json`

---

## Cross-Machine Comparison: Swift Pipeline (Config F) Scaling

**Status:** Pending — requires M1 Mini and M2 Air data

### Config F wall time across machines (warm median, ms)

| Input | Audio | M1 Mini | M2 Air | M2 Ultra |
| --- | --- | --- | --- | --- |
| tiny | 1.55s | ___ ms | ___ ms | 86 ms |
| short | 2.80s | ___ ms | ___ ms | 84 ms |
| medium | 6.58s | ___ ms | ___ ms | 166 ms |
| long | 8.35s | ___ ms | ___ ms | 165 ms |

### Config A (Python HAR-post) across machines

| Input | M1 Mini | M2 Air | M2 Ultra |
| --- | --- | --- | --- |
| tiny | ___ ms | 329 ms | 122 ms |
| short | ___ ms | 323 ms | 136 ms |
| medium | ___ ms | 521 ms | 232 ms |
| long | ___ ms | 513 ms | 286 ms |

### Config D (MPS) across machines

| Input | M1 Mini | M2 Air | M2 Ultra |
| --- | --- | --- | --- |
| tiny | ___ ms | 194 ms | 127 ms |
| short | ___ ms | 329 ms | 190 ms |
| medium | ___ ms | 682 ms | 348 ms |
| long | ___ ms | 860 ms | 449 ms |

### Config F speedup vs Config A per machine

| Input | M1 Mini | M2 Air | M2 Ultra |
| --- | --- | --- | --- |
| tiny | ___x | ___x | 1.4x |
| short | ___x | ___x | 1.6x |
| medium | ___x | ___x | 1.4x |
| long | ___x | ___x | 1.7x |

### Config F speedup vs MPS per machine

| Input | M1 Mini | M2 Air | M2 Ultra |
| --- | --- | --- | --- |
| tiny | ___x | ___x | 1.5x |
| short | ___x | ___x | 2.3x |
| medium | ___x | ___x | 2.1x |
| long | ___x | ___x | 2.7x |

### Scaling: relative to M2 Ultra

| Machine | Config F (tiny) | Config F (long) | Config D MPS (tiny) | Config D MPS (long) |
| --- | --- | --- | --- | --- |
| M2 Ultra | 1.0x (86 ms) | 1.0x (165 ms) | 1.0x (127 ms) | 1.0x (449 ms) |
| M2 Air | ___x | ___x | 1.5x (194 ms) | 1.9x (860 ms) |
| M1 Mini | ___x | ___x | ___x | ___x |

### Interpretation

### M1 Mini data (2026-04-15)

Config A (HAR-post), D (MPS), E (CPU) collected. Config F (Swift) not available — input manifest used legacy keys (`tiny/short/medium/long`) incompatible with Swift benchmark's `3s/7s/15s/30s` inputs. Requires re-running `prepare-inputs` with updated harness to align input keys.

#### End-to-end wall time (warm median, milliseconds)

| Input | Audio | A (HAR-post) | D (MPS) | E (CPU) |
| --- | --- | --- | --- | --- |
| `tiny` | `1.55s` | `232 ms` | `292 ms` | `521 ms` |
| `short` | `2.80s` | `245 ms` | `468 ms` | `843 ms` |
| `medium` | `6.58s` | `573 ms` | `980 ms` | `2122 ms` |
| `long` | `8.35s` | `604 ms` | `1215 ms` | `2736 ms` |

#### RTF (canonical audio duration / wall time)

| Input | A (HAR-post) | D (MPS) | E (CPU) |
| --- | --- | --- | --- |
| `tiny` | `0.149` (7x RT) | `0.188` (5x RT) | `0.336` (3x RT) |
| `short` | `0.087` (11x RT) | `0.167` (6x RT) | `0.301` (3x RT) |
| `medium` | `0.087` (11x RT) | `0.149` (7x RT) | `0.323` (3x RT) |
| `long` | `0.072` (14x RT) | `0.146` (7x RT) | `0.328` (3x RT) |

#### Speedup: Config A vs PyTorch baselines

| Input | Audio | A vs E (CPU) | A vs D (MPS) |
| --- | --- | --- | --- |
| `tiny` | `1.55s` | `2.2x` | `1.3x` |
| `short` | `2.80s` | `3.4x` | `1.9x` |
| `medium` | `6.58s` | `3.7x` | `1.7x` |
| `long` | `8.35s` | `4.5x` | `2.0x` |

#### Config A stage breakdown (warm median)

| Input | Bucket | Prefix extract | HAR builder (CPU) | CoreML predict | Orchestration | Total |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | `3s` | `61.2 ms` (26%) | `64.0 ms` (28%) | `101.3 ms` (44%) | `2.2 ms` | `232 ms` |
| `short` | `3s` | `78.4 ms` (32%) | `63.7 ms` (26%) | `101.6 ms` (41%) | `2.1 ms` | `245 ms` |
| `medium` | `10s` | `121.5 ms` (21%) | `148.3 ms` (26%) | `298.4 ms` (52%) | `2.0 ms` | `573 ms` |
| `long` | `10s` | `148.4 ms` (25%) | `151.1 ms` (25%) | `301.2 ms` (50%) | `2.3 ms` | `604 ms` |

#### Interpretation

1. **Config A is 7–14x realtime on M1 Mini.** Consistent with the earlier bakeoff v2 M1 Mini results (6-14x RT), confirming the M1 Mini as a viable benchmark target.

2. **MPS is consistently slower than Config A.** Unlike the M2 Air where MPS beat Config A on `tiny`, the M1 Mini's 8-core GPU never outperforms the hybrid CoreML path. Config A's advantage is 1.3–2.0x across all inputs.

3. **Speedup vs CPU scales with duration.** The 2.2x speedup at `tiny` grows to 4.5x at `long` — consistent with M2 Ultra and M2 Air scaling patterns.

4. **CoreML predict is ~100 ms on the 3s bucket and ~300 ms on the 10s bucket.** Same pattern as the earlier v2 run, confirming stable predict-time behavior on this machine.

#### Provenance

- Machine: Apple M1 Mac Mini, 16 GB
- Git: `97c394526f69`
- Torch: `2.6.0` / coremltools: `9.0` / numpy: `1.26.4`
- Order seed: `0`, iterations: `5`
- Results: `outputs/bakeoff/results_m1_mini.json`
- Note: Config F (Swift) unavailable due to input key mismatch between manifest and Swift benchmark. Non-essential models stashed during run to fit in 16 GB.

Key questions this section will answer when Config F data is collected:
1. Does the Swift pipeline speedup vs MPS hold on lower-end hardware, or does MPS become competitive as the GPU handles more of the work?
2. On the M2 Air where MPS was competitive with Config A on short inputs (bakeoff v2), does Config F still win?
3. How does the M1 Mini's 8-core GPU (MPS) compare to its 16-core ANE (Swift+CoreML)?

---

## Bakeoff v5: Corrected benchmark (3s-30s) on M1 Mac Mini

**First collected:** 2026-04-15
**Status:** Complete

### Summary

Same corrected v5 harness as the M2 Air run (all audit bugs fixed: bucket parity, tFrames mapping, duplicate matmul, ANE plan compilation), now on the lowest-spec Apple Silicon tested: M1 Mac Mini (8-core CPU, 8-core GPU, 16-core ANE, 16 GB). Configs D, E, F were run together; Config A was run separately (it OOMed when loaded alongside D+E due to 5 HAR-post CoreML models + 2 PyTorch models in 16 GB).

**Config F wins at every duration on M1 Mini**, achieving **18–22x realtime**. Config F is **1.1–1.5x faster than Config A** — a tighter margin than M2 Air (1.3–1.8x) but consistent across all durations.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | Bucket | A (HAR-post) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 3s | 238 ms | 492 ms | 894 ms | **157 ms** |
| 7s | 6.75s | 7s | 577 ms | 1038 ms | 2233 ms | **511 ms** |
| 15s | 13.90s | 15s | 837 ms | 1958 ms | 4458 ms | **691 ms** |
| 30s | 27.38s | 30s | 1637 ms | 4167 ms | 8934 ms | **1229 ms** |

### RTF and realtime factor

| Input | Audio | A RTF | D RTF | E RTF | F RTF | F realtime |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 0.085 (12x RT) | 0.176 | 0.319 | **0.056** | **18x RT** |
| 7s | 6.75s | 0.085 (12x RT) | 0.154 | 0.331 | **0.076** | **13x RT** |
| 15s | 13.90s | 0.060 (17x RT) | 0.141 | 0.321 | **0.050** | **20x RT** |
| 30s | 27.38s | 0.060 (17x RT) | 0.152 | 0.326 | **0.045** | **22x RT** |

### Speedup: Config F vs baselines

| Input | F vs A (HAR-post) | F vs D (MPS) | F vs E (CPU) |
| --- | --- | --- | --- |
| 3s | **1.5x** | **3.1x** | **5.7x** |
| 7s | **1.1x** | **2.0x** | **4.4x** |
| 15s | **1.2x** | **2.8x** | **6.4x** |
| 30s | **1.3x** | **3.4x** | **7.3x** |

### Config A stage breakdown (warm median)

| Input | Bucket | Prefix extract | HAR builder (CPU) | CoreML predict | Total |
| --- | --- | --- | --- | --- | --- |
| 3s | 3s | 86 ms (36%) | 57 ms (24%) | 96 ms (40%) | 238 ms |
| 7s | 7s | 130 ms (23%) | 97 ms (17%) | 346 ms (60%) | 577 ms |
| 15s | 15s | 198 ms (24%) | 187 ms (22%) | 434 ms (52%) | 837 ms |
| 30s | 30s | 409 ms (25%) | 364 ms (22%) | 858 ms (52%) | 1637 ms |

### Interpretation

1. **Config F is 18–22x realtime on M1 Mini** — the lowest-spec Apple Silicon we've tested. Even 30s of audio completes in 1.2 seconds. This confirms the Swift+CoreML pipeline is viable on all shipping Apple Silicon Macs.

2. **Config F beats Config A at every duration**, 1.1–1.5x faster. The margin is tighter than M2 Air (1.3–1.8x), consistent with the M1's smaller ANE/CPU gap — both pipelines are more ANE-bound here.

3. **Config F is 4.4–7.3x faster than PyTorch CPU and 2.0–3.4x faster than MPS.** The CPU speedup is the largest we've seen on any machine, because the M1's CPU is the weakest tested while the ANE remains competitive.

4. **CoreML predict dominates Config A at longer durations.** At 30s, predict is 858 ms (52% of wall time) — the CPU-side prefix (409 ms) and HAR builder (364 ms) are also substantial. Config F avoids both Python-side costs.

5. **The Swift pipeline's model eviction strategy is essential for 16 GB.** Config A could only run in isolation (not alongside D+E). The production app should follow Config F's pattern of loading one bucket at a time.

### Provenance

- Machine: Apple M1 Mac Mini, 16 GB, macOS 15.7.5
- Git: main branch, commit `5a8e7a3`
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m1_mini_def.json` (D/E/F), `outputs/bakeoff/results_m1_mini_a.json` (A)
- Note: Config A run separately due to OOM when loaded alongside D+E

### Plan reference

Bakeoff plan Phase 7: `README/Plans/kokoro-bakeoff-v2.md`

---

## Bakeoff v5: Corrected benchmark (3s-30s) on M2 Ultra

**First collected:** 2026-04-15
**Status:** Complete

### Summary

Reran the corrected v5 bakeoff on M2 Ultra after the same audit fixes used for
the M2 Air and M1 Mini runs: bucket parity, corrected Swift F0Ntrain frame
mapping, duplicate matmul removal, and ANE plan compilation kept out of timed
blocks. All four configs completed successfully across all four inputs.

**Config F wins at every duration on M2 Ultra**, achieving **48-70x realtime**.
The margin over Config A is largest at 3s (2.0x) and tightest at 15s (1.1x),
while Config F remains **2.2-3.0x faster than MPS** and **3.5-4.4x faster than
CPU** across the corrected duration range.

### End-to-end wall time (warm median, milliseconds)

| Input | Audio | Bucket | A (Python HAR) | D (MPS) | E (CPU) | F (Swift) |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 3s | 117 ms | 176 ms | 255 ms | **59 ms** |
| 7s | 6.75s | 7s | 179 ms | 319 ms | 501 ms | **136 ms** |
| 15s | 13.90s | 15s | 309 ms | 602 ms | 975 ms | **278 ms** |
| 30s | 27.38s | 30s | 555 ms | 1233 ms | 1870 ms | **422 ms** |

### RTF and realtime factor

| Input | Audio | A RTF | D RTF | E RTF | F RTF | F realtime |
| --- | --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 0.042 (24x RT) | 0.063 | 0.091 | **0.021** | **48x RT** |
| 7s | 6.75s | 0.027 (38x RT) | 0.047 | 0.074 | **0.020** | **50x RT** |
| 15s | 13.90s | 0.022 (45x RT) | 0.043 | 0.070 | **0.020** | **50x RT** |
| 30s | 27.38s | 0.020 (49x RT) | 0.045 | 0.068 | **0.015** | **70x RT** |

### Speedup: Config F vs baselines

| Input | F vs A (Python HAR) | F vs D (MPS) | F vs E (CPU) |
| --- | --- | --- | --- |
| 3s | **2.0x** | **3.0x** | **4.3x** |
| 7s | **1.3x** | **2.3x** | **3.7x** |
| 15s | **1.1x** | **2.2x** | **3.5x** |
| 30s | **1.3x** | **2.9x** | **4.4x** |

### Interpretation

1. **Config F wins everywhere on M2 Ultra.** The corrected Swift+CoreML path is
   faster than Python HAR-post, MPS, and CPU for all measured durations.

2. **The MPS gap remains large.** Config F is 2.2-3.0x faster than the PyTorch
   MPS baseline, preserving the core bakeoff result that the Swift+CoreML path
   beats the "just use the GPU" path.

3. **The Config A margin is now more conservative than v4.** With corrected
   bucket parity and v5 fixes, Config F is 1.1-2.0x faster than Config A. The
   15s/30s results are close enough that future optimization work should focus
   on stage-level costs rather than assuming large end-to-end headroom.

4. **Config F remains extremely realtime.** Even the 30s input completes in
   422 ms, roughly 70x realtime on M2 Ultra.

### Provenance

- Machine: Apple M2 Ultra, 64 GB, macOS 26.4.1
- Git: main branch, commit `f9276800`
- Python: 3.12.13
- Torch: 2.6.0 / coremltools: 8.3.0
- Order seed: 0, iterations: 5
- Results: `outputs/bakeoff/results_m2_ultra_v5.json`
- Note: Config F batch harness required the stdout sentinel parser fix in
  `scripts/bakeoff_harness.py` because Core ML may emit ANE diagnostics on the
  same stdout line as `DONE`.

### Plan reference

Bakeoff plan Phase 7: `README/Plans/kokoro-bakeoff-v2.md`
