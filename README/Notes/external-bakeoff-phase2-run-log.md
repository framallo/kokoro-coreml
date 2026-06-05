# External Bakeoff Phase 2 Run Log

**Date:** 2026-06-05
**Plan:** `README/Plans/kokoro-external-bakeoff-v1.md`
**Status:** M2 Studio local collection started; fleet-wide Phase 2 remains
incomplete.

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

After local collection, botnet health was still green:

```json
{
  "ok": true,
  "queueDepth": 0,
  "claimedFresh": 2,
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

Each file validates against `scripts/external_bakeoff/schema.py`.

## Config F Same-Window Result

Config F used the main checkout Core ML artifacts at
`/Users/mm/Documents/GitHub/kokoro-coreml/coreml` and the persistent
`kokoro-bench --batch` adapter.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Bucket |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 0.135342 | 0.139271 | 5 | 2.800 | 3s |
| 7s | ok | 0.314786 | 0.332596 | 5 | 6.750 | 7s |
| 10s | ok | 0.551341 | 0.511796 | 5 | 9.625 | 10s |
| 15s | ok | 0.733556 | 0.581774 | 5 | 13.900 | 15s |
| 30s | ok | 1.241910 | 1.195325 | 5 | 27.400 | 30s |

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
| 7s | ok | 0.202146 | 0.201625 | 5 | 6.750 | - |
| 10s | ok | 4.587298 | 0.277975 | 5 | 9.600 | - |
| 15s | ok | 0.568535 | 0.369075 | 5 | 13.900 | - |
| 30s | ok | 0.888687 | 0.749096 | 5 | 27.375 | - |

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
| 3s | ok | 0.481583 | 0.069233 | 5 | 2.700 | duration mismatch |
| 7s | ok | 0.475177 | 0.068821 | 5 | 5.000 | truncated versus manifest |
| 10s | ok | 0.542702 | 0.073327 | 5 | 5.000 | truncated versus manifest |
| 15s | ok | 0.495827 | 0.071984 | 5 | 5.000 | truncated versus manifest |
| 30s | ok | 0.477556 | 0.069727 | 5 | 5.000 | truncated versus manifest |

These Soniqo cells are not quality-parity evidence yet. The timing is useful
for implementation behavior, but the emitted audio duration must be resolved or
clearly caveated before using the cells in the paper table.

## Spot-Check WAV Support

After the first M2 Studio collection, adapters were updated to write durable
spot-check WAV files:

- Config F keeps the last warm `kokoro-bench` WAV per input.
- MLX writes the last warm PCM array as mono 16-bit WAV.
- Soniqo writes the last warm Swift `[Float]` audio as mono 16-bit WAV.

One-input smokes verified valid mono 24 kHz WAV files for Config F, MLX, and
Soniqo. The M2 Studio full collection should be rerun with this support before
Phase 3 listening checks.

## Remaining Phase 2 Work

- Rerun M2 Studio full collection with durable spot-check WAV output.
- Decide whether the MLX 3s public-implementation failure is a paper caveat or
  requires an alternate, predeclared 3s input.
- Resolve Soniqo duration truncation before treating its speed numbers as
  parity-comparable.
- Collect the same matrix on `irvine-m1` and `m2-air`.
- Capture hardware-placement evidence for MLX GPU and Core ML / ANE paths.
