# External Bakeoff Phase 2 Run Log

**Date:** 2026-06-05
**Plan:** `README/Plans/kokoro-external-bakeoff-v1.md`
**Status:** M2 Studio local collection rerun with durable spot-check WAVs;
long-bucket Core ML backup collected; m2-air Config F, MLX, and Soniqo
collected; fleet-wide Phase 2 remains incomplete.

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

## M2 Air Partial Collection

`m2-air` was prepared from a disposable checkout at
`/tmp/kokoro-coreml-bakeoff-run`, using that host's existing Core ML artifacts
from `/Users/mattmireles/Documents/GitHub/kokoro-coreml/coreml`. Before the
run, `pmset -g therm` reported no thermal or performance warning. Config F,
MLX, and Soniqo completed and were copied back to:

- `outputs/external_bakeoff/results_config_f_reference_m2-air.json`
- `outputs/external_bakeoff/results_mlx_audio_m2-air.json`
- `outputs/external_bakeoff/results_soniqo_speech_swift_kokoro_m2-air.json`

All three files validate against `scripts/external_bakeoff/schema.py`, and every
successful cell has a mono 24 kHz spot-check WAV.

### M2 Air Config F

| Input | Status | Cold s | Warm median s | Warm N | Observed s |
| --- | --- | ---: | ---: | ---: | ---: |
| 3s | ok | 0.313816 | 0.317402 | 5 | 2.800 |
| 7s | ok | 0.683149 | 0.808074 | 5 | 6.750 |
| 10s | ok | 1.054014 | 1.373335 | 5 | 9.625 |
| 15s | ok | 2.134623 | 2.052364 | 5 | 13.900 |
| 30s | ok | 9.447099 | 9.559135 | 5 | 27.400 |

The 30s bucket compiled successfully after the long Core ML AOT window. The
fanless host stayed thermally nominal during the run.

### M2 Air MLX

`Blaizzy/mlx-audio` ran from pinned SHA
`862dfbe5338e91df6f74ac986b4df8bede7961a6` in a disposable Python 3.12 venv
with `mlx-audio 0.4.3`, `mlx 0.31.2`, and MLX default device `gpu`.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | error | - | - | 0 | - | deterministic broadcast-shape failure |
| 7s | ok | 0.670390 | 0.685626 | 5 | 6.750 | - |
| 10s | ok | 20.802833 | 0.835810 | 5 | 9.600 | model/cache cold start |
| 15s | ok | 1.636798 | 1.520953 | 5 | 13.900 | - |
| 30s | ok | 2.851399 | 2.600340 | 5 | 27.375 | - |

The 3s error matched the M2 Studio failure:

```text
ValueError: [broadcast_shapes] Shapes (1,67200,1) and (1,67500,9) cannot be broadcast.
```

### M2 Air Soniqo

`soniqo/speech-swift` ran from pinned SHA
`0d09a2ed5464c7c94cf4545be59043c21f8775ea`. The first retry was required
because SwiftPM attempted to use the remote keychain credential helper while
downloading Soniqo's public `SpeechCore.xcframework` artifact; rerunning with
the Git credential helper disabled forced anonymous access and completed.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 1.189790 | 1.097402 | 5 | 2.700 | - |
| 7s | ok | 1.210679 | 1.135835 | 5 | 5.000 | public 5s artifact |
| 10s | ok | 1.237192 | 1.122956 | 5 | 5.000 | public 5s artifact |
| 15s | ok | 1.273711 | 1.125471 | 5 | 5.000 | public 5s artifact |
| 30s | ok | 1.233093 | 1.123540 | 5 | 5.000 | public 5s artifact |

laishere was not started on `m2-air` in this pass because it requires a heavier
Core ML conversion/build path, and the no-disruption rule still requires a
fresh botnet health check before occupying the production host.

## Irvine M1 Aborted Collection

`irvine-m1` was prepared from the same disposable checkout and built the current
`kokoro-bench` successfully. The first Config F attempt reached the 30s bucket,
and stack sampling showed Core ML / Espresso / ANE on-device AOT compilation.
Because the original SSH session had been lost, the process later sat idle
after writing `30s.wav` without producing a result JSON; this was treated as a
detached-pipe stall, not a completed benchmark.

A redirected rerun was started with logs at `/tmp/kokoro-configf-irvine.log`.
It progressed through 10s and 15s and re-entered 30s compilation. During that
rerun, botnet health showed production pressure:

```json
{
  "ok": true,
  "queueDepth": 9,
  "claimedFresh": 11,
  "freshWorkerCount": 3,
  "canaryStatus": "passing"
}
```

Per the plan's no-disruption guardrail, the `irvine-m1` benchmark was killed
before a result JSON was produced. After stopping it, health improved to
`queueDepth=1` with canary still passing. The partial `irvine-m1` WAVs under
the remote `/tmp` directory are not publication data because there is no
schema-valid JSON for that host.

## Continuation Gate: Production Pressure

On 2026-06-05, a follow-up attempt checked for a low-traffic window before
resuming `m2-air` Soniqo/laishere or restarting `irvine-m1`. No remote bakeoff
processes were running on `irvine-m1` or `m2-air`, but botnet health showed
active production claims throughout the polling window:

| Time | Queue depth | Claimed fresh | Fresh workers | Canary |
| --- | ---: | ---: | ---: | --- |
| 00:14:29 | 1 | 14 | 3 | passing |
| 00:15:30 | 2 | 13 | 3 | passing |
| 00:16:31 | 2 | 13 | 3 | passing |
| 00:17:33 | 0 | 15 | 3 | passing |
| 00:18:34 | 0 | 15 | 3 | passing |

Because `claimedFresh` stayed high, no additional benchmark runs were started.
Resume Phase 2 only after both queue depth and active fresh claims are low
enough that a benchmark host will not compete with production TTS work.

## Phase 3 Partial Quality Sanity

While remote collection was paused, the existing spot-check WAVs were run
through `scripts/audio_quality_probe.py`. Config F WAVs were used as
machine-local references, and competitor WAVs were classified against derived
thresholds for RMS, activity, zero-crossing rate, speech-band energy, clipping,
sample rate, and channel count.

Generated, uncommitted reports:

- `outputs/external_bakeoff/quality/m2-studio/audio_quality_report.json`
- `outputs/external_bakeoff/quality/m2-studio/audio_quality_summary.md`
- `outputs/external_bakeoff/quality/m2-air/audio_quality_report.json`
- `outputs/external_bakeoff/quality/m2-air/audio_quality_summary.md`

No collected competitor WAV was rejected by the waveform sanity gate. Every
candidate requires human listening before quality parity is claimed. Soniqo's
longer-bucket files remain duration caveats: they look healthy by waveform
metrics but are 5.0s clips for 7s, 10s, 15s, and 30s manifest inputs.

### M2 Studio Quality Probe

| Impl | Input | Decision | Duration s | RMS |
| --- | --- | --- | ---: | ---: |
| Config F reference | 3s | reference_pass | 2.800 | 4433.4 |
| Config F reference | 7s | reference_pass | 6.750 | 4696.9 |
| Config F reference | 10s | reference_pass | 9.625 | 4261.0 |
| Config F reference | 15s | reference_pass | 13.900 | 5226.7 |
| Config F reference | 30s | reference_pass | 27.400 | 4590.8 |
| MLX | 7s | needs_listening | 6.750 | 3471.8 |
| MLX | 10s | needs_listening | 9.600 | 2355.2 |
| MLX | 15s | needs_listening | 13.900 | 4836.9 |
| MLX | 30s | needs_listening | 27.375 | 3267.7 |
| Soniqo | 3s | needs_listening | 2.700 | 4504.1 |
| Soniqo | 7s | needs_listening | 5.000 | 5136.5 |
| Soniqo | 10s | needs_listening | 5.000 | 4235.7 |
| Soniqo | 15s | needs_listening | 5.000 | 4937.7 |
| Soniqo | 30s | needs_listening | 5.000 | 5243.8 |
| laishere | 3s | needs_listening | 2.775 | 4592.0 |
| laishere | 7s | needs_listening | 6.800 | 3864.8 |
| laishere | 10s | needs_listening | 9.625 | 3279.0 |
| laishere | 15s | needs_listening | 13.975 | 4906.4 |
| laishere | 30s | needs_listening | 27.375 | 3602.0 |

### M2 Air Quality Probe

| Impl | Input | Decision | Duration s | RMS |
| --- | --- | --- | ---: | ---: |
| Config F reference | 3s | reference_pass | 2.800 | 920.8 |
| Config F reference | 7s | reference_pass | 6.750 | 4603.5 |
| Config F reference | 10s | reference_pass | 9.625 | 4204.8 |
| Config F reference | 15s | reference_pass | 13.900 | 6087.4 |
| Config F reference | 30s | reference_pass | 27.400 | 4592.0 |
| MLX | 7s | needs_listening | 6.750 | 3424.5 |
| MLX | 10s | needs_listening | 9.600 | 2346.2 |
| MLX | 15s | needs_listening | 13.900 | 4788.8 |
| MLX | 30s | needs_listening | 27.375 | 3114.0 |
| Soniqo | 3s | needs_listening | 2.700 | 4483.9 |
| Soniqo | 7s | needs_listening | 5.000 | 5140.1 |
| Soniqo | 10s | needs_listening | 5.000 | 4244.0 |
| Soniqo | 15s | needs_listening | 5.000 | 4911.8 |
| Soniqo | 30s | needs_listening | 5.000 | 5331.4 |

## Remaining Phase 2 Work

- Decide whether the MLX 3s public-implementation failure is a paper caveat or
  requires an alternate, predeclared 3s input.
- Decide how the paper table presents Soniqo's high-adoption 5s-only result
  beside laishere's lower-adoption long-bucket Core ML backup.
- Finish laishere collection on `m2-air` after production pressure clears.
- Re-run `irvine-m1` during a lower-traffic window with stdout/stderr redirected
  from the start.
- Listen to all `needs_listening` WAVs before making quality-parity claims.
- Capture hardware-placement evidence for MLX GPU and Core ML / ANE paths.
