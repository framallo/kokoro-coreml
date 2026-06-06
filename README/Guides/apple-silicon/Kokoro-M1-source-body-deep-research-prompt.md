# Kokoro M1 Source/Body Deep Research Prompt

June 6, 2026

Use this as the handoff prompt for the next optimization pass. The goal is not
to find another plausible Core ML tweak. The goal is to close the remaining
real warmed-inference loss against laishere on Irvine M1 while preserving the
paper-facing comparison rules.

## Objective

Make first-party Config F the fastest warmed-inference Kokoro implementation on
Apple Silicon for the runtime buckets `3s`, `7s`, `10s`, `15s`, and `30s`.
Prioritize Irvine M1 losses in this order: `3s`, `7s`, `10s`, `15s`, then
`30s`.

Current real Irvine M1 losses:

| Bucket | Config F | laishere | Gap |
| --- | ---: | ---: | ---: |
| `3s` | `233.6 ms` | `195.0 ms` | `38.5 ms / 19.75%` |
| `7s` | `492.7 ms` | `444.2 ms` | `48.4 ms / 10.90%` |
| `10s` | `685.5 ms` | `644.9 ms` | `40.6 ms / 6.29%` |
| `15s` | `1014.9 ms` | `990.6 ms` | `24.3 ms / 2.46%` |

Do not use cold compile/cache timings. Every claim must use warmed inference.

## Research Question

Design an M1 MLProgram source/STFT/vocoder body that preserves the current
Swift HAR/source contract, keeps strict waveform parity or earns explicit
no-ASR listening acceptance, and shifts the expensive convolution/add/mul/
instance-norm body work into a laishere-like mixed CPU/Neural Engine plan
without the existing split-boundary synchronization penalty or a warmed `3s`
regression.

## Evidence To Respect

Authoritative frontier and target files:

- `outputs/external_bakeoff/goal_frontier_status.md`
- `outputs/external_bakeoff/irvine_3s_placement_target.md`
- `outputs/external_bakeoff/irvine_next_targets.md`
- `README/Guides/apple-silicon/Kokoro-M1-vocoder-boundary-research-brief.md`
- `README/Notes/performance-notes.md`

The apparent MLX win was a comparison bug: cold compile/cache behavior,
padding, `.all` behavior, and HAR overhead were mixed into non-equivalent
timings. Under warmed inference, Config F beats or ties MLX on validated Mac
cells. The remaining strict competitor is laishere on Irvine M1.

The current strict CPU+NE body split is the main warning sign. It already gets
laishere-like Neural Engine preferred-op counts, but it is slower. Therefore
the target is not "more NE placement." The target is a runtime-positive graph
boundary and synchronization pattern.

## Do Not Repeat

Treat these as measured rejections unless the proposal changes the boundary or
runtime synchronization mechanism:

- `.all` toggles.
- Palette-only changes.
- Final-waveform int4/int8 quantization.
- fp16 input-only changes for the current static body.
- iOS17/spec8 metadata-only rebuilds.
- More exact decoder+vocoder multi-package splits.
- fp16 body inputs for the exact decoder+vocoder split.
- Plain or native-InstanceNorm style specialization.
- Native-InstanceNorm style specialization plus HAR trim.
- Broad generator noise/stage splits that add Core ML calls.
- Standalone strict HAR-source fused paths that preserve padded geometry but
  lose the speed edge.

## Promising Hypotheses

### 1. Single-Package Body Reshaping

Keep the current `GeneratorFromHar` runtime call boundary but reshape the graph
surface inside the package. The candidate should change operator fusion,
layout, or partitioning enough that M1 avoids repeated CPU/GPU/NE sync while
keeping the Swift input/output contract stable.

Acceptance:

- strict waveform gate passes;
- Irvine M1 `3s` improves by at least `20 ms` warmed median, or combines with
  measured upstream savings to close the full `38.5 ms` gap;
- M2 Studio does not materially regress.

### 2. In-Package HAR Source Consumption

Find a smaller Swift-produced tensor that preserves source quality but avoids
the HAR-post input cost, then consume it inside one body+tail package. Do not
add an extra Core ML call on the `3s` hot path unless the measured savings are
larger than the call/sync overhead.

Acceptance:

- strict parity or no-ASR listening acceptance;
- same five bucket contract;
- warmed Irvine M1 win against laishere on at least one real-loss bucket.

### 3. Fast F0/Source Quality Recovery

The speed-positive candidates are quality-failing F0/source simplifications.
They are useful only if quality can be recovered without losing the speed
signal, or if human listening explicitly accepts the exact generated WAVs.

Acceptance:

- strict waveform parity, or accepted rows in
  `outputs/f0_source_listening/irvine_exact_speed_branch/f0_source_listening_decisions.csv`;
- no Whisper/ASR proxy metrics;
- exact candidate WAVs and warmed timing reports are linked from the review
  dashboard.

## Required Output For A Candidate

Every candidate must leave a durable report containing:

- exact git SHA and command line;
- exported or reused model package paths;
- deployment target, precision, input dtypes, and compute units;
- warmup count, warm iteration count, cold latency, and warm median;
- waveform metrics against the same reference tensor dump or emitted WAV;
- `MLComputePlan` preferred-device counts and cost weights;
- local M2 Studio timing before any Irvine timing;
- Irvine timing only when the host is quiet enough for publishable data.

## Current External Blocks

The iPhone 12 Pro is paired and the Config F manual runner is installed, but
launch is denied while the physical phone is locked. Irvine M1 is not quiet
enough for publishable timing while `mediaanalysisd` or Spotlight consumes CPU.
