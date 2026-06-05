# External Bakeoff Phase 2 Run Log

**Date:** 2026-06-05
**Plan:** `README/Plans/kokoro-external-bakeoff-v1.md`
**Status:** M2 Studio local collection rerun with durable spot-check WAVs;
long-bucket Core ML backup collected; fleet-wide Phase 2 remains incomplete.

## M2 Studio Precheck

Before local collection, botnet health was green:

```json
{
  "ok": true,
  "queueDepth": 0,
  "claimedFresh": 0,
  "freshWorkerCount": 2,
  "canaryStatus": "passing",
  "canaryWorkerId": "operator-prove-live"
}
```

After the spot-check rerun, botnet health was still green:

```json
{
  "ok": true,
  "queueDepth": 0,
  "claimedFresh": 0,
  "freshWorkerCount": 3,
  "canaryStatus": "passing",
  "canaryWorkerId": "operator-prove-live"
}
```

## M2 Studio Result Files

Generated, uncommitted result files:

- `outputs/external_bakeoff/results_config_f_reference_m2-studio.json`
- `outputs/external_bakeoff/results_mlx_audio_m2-studio.json`
- `outputs/external_bakeoff/results_soniqo_speech_swift_kokoro_m2-studio.json`
- `outputs/external_bakeoff/results_laishere_kokoro_coreml_m2-studio.json`

Each file validates against `scripts/external_bakeoff/schema.py`.

## Config F Same-Window Result

Config F used the main checkout Core ML artifacts at
`/Users/mm/Documents/GitHub/kokoro-coreml/coreml` and the persistent
`kokoro-bench --batch` adapter.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Bucket |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 0.125504 | 0.131485 | 5 | 2.800 | 3s |
| 7s | ok | 0.309493 | 0.284174 | 5 | 6.750 | 7s |
| 10s | ok | 0.570868 | 0.548194 | 5 | 9.625 | 10s |
| 15s | ok | 0.647346 | 0.632877 | 5 | 13.900 | 15s |
| 30s | ok | 1.389132 | 1.191795 | 5 | 27.400 | 30s |

The 30s first compile spent roughly 20 minutes in Core ML on-device AOT
compilation before timed synthesis. A sampled stack showed Core ML / Espresso
inside program-library preparation and shortest-path segmentation, so the long
silence was compiler work, not a harness deadlock.

## MLX Result

`Blaizzy/mlx-audio` was run from the pinned current clone
`862dfbe5338e91df6f74ac986b4df8bede7961a6` with `mlx-audio 0.4.3` and
`mlx 0.31.2`.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | error | - | - | 0 | - | deterministic broadcast-shape failure |
| 7s | ok | 0.195928 | 0.223944 | 5 | 6.750 | - |
| 10s | ok | 4.737087 | 0.288822 | 5 | 9.600 | - |
| 15s | ok | 0.438077 | 0.376303 | 5 | 13.900 | - |
| 30s | ok | 0.930204 | 0.762699 | 5 | 27.375 | - |

The 3s cell failed on the initial run and a one-input retry with:

```text
ValueError: [broadcast_shapes] Shapes (1,67200,1) and (1,67500,9) cannot be broadcast.
```

Because the adapter is using the public current clone and the shared manifest
text, this is recorded as competitor behavior rather than patched locally.

## Soniqo Speech Swift Result

`soniqo/speech-swift` was run through the generated macOS Swift CLI at pinned
SHA `0d09a2ed5464c7c94cf4545be59043c21f8775ea` with
`KokoroTTSModel.fromPretrained(computeUnits: .all)`.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 0.615211 | 0.071711 | 5 | 2.700 | duration mismatch |
| 7s | ok | 0.432999 | 0.069311 | 5 | 5.000 | truncated versus manifest |
| 10s | ok | 0.397993 | 0.071024 | 5 | 5.000 | truncated versus manifest |
| 15s | ok | 0.411708 | 0.068065 | 5 | 5.000 | truncated versus manifest |
| 30s | ok | 0.414261 | 0.069504 | 5 | 5.000 | truncated versus manifest |

These Soniqo cells are not quality-parity evidence yet. The timing is useful
for implementation behavior, but the emitted audio duration must be resolved or
clearly caveated before using the cells in the paper table.

Source and artifact check:

- `KokoroTTSModel.fromPretrained(...)` downloads `kokoro_5s.mlmodelc/**`.
- `KokoroNetwork` would load `kokoro_10s` or `kokoro_15s` if present, but the
  upstream `aufklarer/Kokoro-82M-CoreML` file listing only contains
  `kokoro_5s.mlmodelc`.
- Local caches under both Hugging Face and `qwen3-speech` contain only
  `kokoro_5s.mlmodelc`.

This makes the 5.0s cap public-comparator behavior for the selected Soniqo
model artifact, not an adapter timing-boundary bug.

## Laishere Core ML Backup Result

`laishere/kokoro-coreml` was probed as the long-bucket Core ML backup at pinned
SHA `484907db6a8347a6afb6e7b86850ea2878c6a3fb`. The repo does not ship
prebuilt `.mlpackage` artifacts, so the seven public Core ML packages were
converted under `/tmp/kokoro-external-bakeoff/laishere-kokoro-coreml/output`
with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python convert.py --max-frames 2000
```

`convert.py` emitted all seven packages, then its final chained validation hit
a Core ML/Espresso dynamic-shape error in the PostAlbert path:

```text
RuntimeError: Unable to compute the prediction using ML Program
Tile: Shape deduction failed as reps[0]=-1317260229 < 0
```

The repo's standalone `benchmark.py --n-runs 1` still ran successfully against
the generated packages. It rendered six built-in passages from 1.50s to 28.12s
audio with chain times from 62.9ms to 606.7ms on this M2 Studio. The normalized
adapter then ran the shared runtime manifest with N=5 warm calls:

| Input | Status | Cold s | Warm median s | Warm N | Observed s | T_a |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3s | ok | 0.236952 | 0.212307 | 5 | 2.775 | 111 |
| 7s | ok | 0.359018 | 0.403259 | 5 | 6.800 | 272 |
| 10s | ok | 0.839707 | 0.626281 | 5 | 9.625 | 385 |
| 15s | ok | 0.676127 | 0.429827 | 5 | 13.975 | 559 |
| 30s | ok | 1.955135 | 0.925116 | 5 | 27.375 | 1095 |

These cells are the current long-bucket Core ML parity backup. The adapter
times the seven-stage Core ML chain only; G2P and feed preparation are outside
the timed calls, matching laishere's public benchmark boundary. That boundary
must be stated if these numbers are used beside Soniqo or MLX in the paper.

## Spot-Check WAV Support

The M2 Studio collection was rerun after adapters were updated to write durable
spot-check WAV files:

- Config F keeps the last warm `kokoro-bench` WAV per input.
- MLX writes the last warm PCM array as mono 16-bit WAV.
- Soniqo writes the last warm Swift `[Float]` audio as mono 16-bit WAV.

One-input smokes verified valid mono 24 kHz WAV files for Config F, MLX, and
Soniqo. The full M2 Studio rerun then produced durable WAVs for every successful
result cell. MLX has no 3s WAV because that cell errors before audio is
materialized.

## Remaining Phase 2 Work

- Decide whether the MLX 3s public-implementation failure is a paper caveat or
  requires an alternate, predeclared 3s input.
- Decide how the paper table presents Soniqo's high-adoption 5s-only result
  beside laishere's lower-adoption long-bucket Core ML backup.
- Collect the same matrix on `irvine-m1` and `m2-air`.
- Capture hardware-placement evidence for MLX GPU and Core ML / ANE paths.
