# Kokoro TTS Bakeoff — Implementation Plan

**Date:** 2026-04-14
**Status:** Planned

> Research design lives in `README/Plans/kokoro-bakeoff-experiment-v1.md`. This
> plan covers what to build and run to produce reproducible benchmark data for
> the current repo state.

## Executive Summary

Build one benchmark harness around the current shipping HAR-post hybrid runtime,
plus one fixed-shape decoder-only control artifact loaded under multiple Core ML
compute-unit policies. The harness must produce publication-grade M2 Ultra data,
conditional M1 Mini data, and a run manifest with git provenance, artifact
hashes, and telemetry strong enough to answer two separate questions:

1. How much faster is the shipping hybrid ANE path than PyTorch CPU and MPS?
2. Does a naive fixed-shape Core ML decoder actually use ANE when loaded with
   `.all`, or does it fall back to GPU or CPU?

## Problem Statement

- **Symptom:** The repo has multiple timing anecdotes, but no controlled,
  reproducible benchmark tied to exact code and model artifacts.
- **Root Cause:** Existing numbers were collected from different runtime paths,
  different inputs, and different levels of instrumentation.
- **Impact:** Paper-level claims about ANE speedup and silent fallback are too
  weak without a unified harness and stronger evidence.

## Mode Definitions

The harness should support four explicit modes. These are implementation
requirements for `scripts/bakeoff_harness.py`.

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| `prepare-inputs` | Measure canonical audio durations once with PyTorch CPU and write an input manifest. | Freezes the benchmark dataset and denominator for canonical RTF. |
| `run` | Run timed benchmark iterations for selected configs using preloaded models. | Produces the main results JSON. |
| `telemetry-loop` | Run one config repeatedly for a sustained interval. | Gives `powermetrics` enough time to observe ANE activity or the lack of it. |
| `summarize` | Read one or more results files and emit tables plus gate answers. | Produces the publication-ready outputs. |

## Goals and Non-Goals

### Goals

- [ ] One harness runs five headline configs with identical text, voice, and
      speed under a single results schema.
- [ ] M2 Ultra benchmark data includes per-iteration timings, run provenance,
      and telemetry evidence for the naive Core ML control artifact.
- [ ] M1 Mini benchmark data is collected when the required models fit in 16 GB
      using explicit-path loading; otherwise the skip reason is documented.
- [ ] The benchmark answers whether the shipping hybrid ANE path is faster than
      PyTorch CPU/MPS and whether the naive fixed-shape decoder participates on
      ANE under `.all`.

### Non-Goals

- Fixing the MPS `aten::angle` fallback.
- Benchmarking the deprecated `Decoder_HAR` 5s/15s/30s path from older notes.
- Benchmarking utterances longer than the current shipping HAR-post 10s bucket.
- Full-pipeline Core ML export with dynamic alignment.
- Audio-quality comparison, power-efficiency analysis beyond ANE participation,
  or Instruments screenshots for this first benchmark pass.

## Scope and Constraints

- **Scope:** Benchmark harness, one decoder-only control export, M2 Ultra data,
  and conditional M1 Mini data.
- **Constraints:** The current shipping HAR-post path in this repo has only
  `coreml/kokoro_decoder_har_post_3s.mlpackage` and
  `coreml/kokoro_decoder_har_post_10s.mlpackage`. Benchmark inputs must fit
  within that envelope so Config A never truncates.
- **Constraints:** The M1 Mini has 16 GB unified memory. The harness must load
  only the exact artifact paths it needs rather than relying on
  `HybridTTSPipeline` bucket auto-discovery.
- **Guardrails:** Reuse canonical repo contracts:
  `HybridTTSPipeline.extract_vocoder_inputs()` is the shared Core ML prefix,
  and `python -m export_synth.main --mode decoder --buckets ...` is the export
  path for the naive decoder control. Do not invent a second exporter or a new
  production `_run_shared_prefix`.
- **Guardrails:** Benchmark code lives in `scripts/`. Generated models, logs,
  manifests, and result files live in `outputs/bakeoff/` and remain gitignored.

## Ground Truth Contracts (Do Not Violate)

- **Same benchmark inputs:** Every headline config gets the same exact text,
  voice preset, and speed value from the frozen input manifest.
- **Current Config A only:** Config A is the current shipping HAR-post path:
  `HybridTTSPipeline.extract_vocoder_inputs()` plus HAR-post Core ML buckets
  from `coreml/kokoro_decoder_har_post_{3,10}s.mlpackage`.
- **Frozen inputs must stay below the 10s ceiling with margin:** `prepare-inputs`
  must fail if any canonical audio duration exceeds `9.0s`. Do not accept an
  input set that merely routes to `10s`; keep headroom so Config A cannot
  silently truncate from minor duration drift.
- **Single naive artifact for Configs B/C:** Configs B and C use the same
  decoder-only 10s Core ML artifact. The only variable is load-time
  `compute_units`:
  `ct.ComputeUnit.ALL` for B and `ct.ComputeUnit.CPU_AND_GPU` for C.
- **Diagnostic CPU-only control is not a headline config:** A temporary
  `.cpuOnly` load of the same decoder-only artifact may be used inside
  `telemetry-loop` and Gate 1 analysis, but it is not part of the main five-row
  benchmark matrix.
- **Model initialization is out of band:** All model loading, compilation, and
  warmup happen before timed iterations. Timed runs measure steady-state wall
  time only.
- **Timer discipline:** Wall time starts before text processing and ends after
  the final waveform buffer is ready. For MPS, call `torch.mps.synchronize()`
  immediately before stopping the timer.
- **MPS fallback is explicit:** Config D requires
  `PYTORCH_ENABLE_MPS_FALLBACK=1` in the benchmark environment. If that env var
  is absent, do not treat the run as the repo’s intended MPS baseline.
- **Canonical and observed durations are both recorded:** Canonical duration is
  measured once from PyTorch CPU during `prepare-inputs`. Each run also records
  observed audio duration so duration drift remains visible.
- **Deterministic seeding:** Call `torch.manual_seed(0)` before each warmup and
  timed iteration.
- **Counterbalanced order:** The harness must not run configs or inputs in a
  single fixed order. Use a deterministic order seed and counterbalance across
  repetitions.
- **Telemetry proof is loop-based:** Gate 1 cannot rely on a single short
  inference call plus one `powermetrics -i 1000` sample. Use a sustained loop.
- **Run manifest is mandatory:** Every results file records git commit, dirty
  tree state, Python executable, package versions, exact artifact paths, input
  shapes, and SHA256 hashes.

## Already Shipped (Do Not Re-Solve)

- **Canonical shared prefix:** `HybridTTSPipeline.extract_vocoder_inputs()` in
  [kokoro/coreml_pipeline.py](/Users/mm/Documents/GitHub/kokoro-coreml/kokoro/coreml_pipeline.py:205)
  is the shared DurationModel + alignment + hn-nsf entry point.
- **Current shipping ANE backend:** `decoder_har_post_bucket_impl()` in
  [kokoro/synthesis_backends.py](/Users/mm/Documents/GitHub/kokoro-coreml/kokoro/synthesis_backends.py:84)
  defines the current HAR-post runtime contract.
- **Canonical decoder-only export path:** `python -m export_synth.main --mode decoder --buckets ...`
  routes through [export_synth/main.py](/Users/mm/Documents/GitHub/kokoro-coreml/export_synth/main.py:10)
  and [export_synth/convert.py](/Users/mm/Documents/GitHub/kokoro-coreml/export_synth/convert.py:391).
- **Integration coverage:** [tests/test_mlpackage_exports.py](/Users/mm/Documents/GitHub/kokoro-coreml/tests/test_mlpackage_exports.py:83)
  already validates the decoder-only and HAR-post package I/O contracts.

## Fresh Baseline (Current State)

- **Shipping benchmark target:** HAR-post buckets at 3s and 10s only.
- **Historical long-form reference:** The older `Decoder_HAR` 5s/15s/30s path
  reached RTF `~0.057` on a `~23.7s` utterance, but that is not the current
  shipping path and must not be reused as Config A’s baseline.
- **Current HAR-post smoke reference:** The repo notes one short HAR-post smoke
  run at `time_sec≈0.374`, `audio_sec≈1.36`, `RTF≈0.27`, but the paired PyTorch
  comparison used a different output duration and is explicitly not a clean A/B.
- **Known gaps:** No reproducible dataset, no run manifest with artifact hashes,
  no sustained telemetry loop for naive Core ML, and no counterbalanced timing.

## Solution Overview

```text
scripts/bakeoff_harness.py   <- one harness with prepare-inputs / run / telemetry-loop / summarize
requirements-bakeoff.txt     <- pinned benchmark environment
outputs/bakeoff/             <- manifests, results, telemetry logs, summaries
outputs/bakeoff/models/      <- freshly exported naive decoder-only control artifact
```

Headline configs:

```text
Config A: Shipping hybrid HAR-post path (3s/10s buckets, explicit path load)
Config B: Same naive decoder-only 10s artifact, loaded with compute_units=ALL
Config C: Same naive decoder-only 10s artifact, loaded with compute_units=CPU_AND_GPU
Config D: PyTorch end-to-end on MPS (known fallback-heavy)
Config E: PyTorch end-to-end on CPU
```

Diagnostic-only control:

```text
Config Bcpu: Same naive decoder-only 10s artifact, loaded with compute_units=CPU_ONLY
```

Why this split is correct for the current repo:

- Config A measures the current production path.
- Configs B/C isolate Core ML scheduling policy on the same exact decoder-only
  graph instead of changing both graph and compute-unit policy at once.
- Bcpu gives a classification aid for Gate 1 without bloating the public matrix.

## Output Contracts

### Input Manifest

`outputs/bakeoff/input_manifest.json`

- exact text for each input key: `tiny`, `short`, `medium`, `long`
- voice preset and speed
- canonical audio duration from Config E
- expected Config A bucket (`3s` or `10s`)
- `sha256` of each text string

### Results Manifest

`outputs/bakeoff/results_{machine_id}.json`

Top-level fields:

- `run_id`
- `order_seed`
- `git_commit`
- `git_dirty`
- `python_executable`
- `machine`
- `package_versions`
- `artifacts`
- `inputs`
- `results`

`artifacts` must record path, SHA256, load-time compute units, and input shapes
for every Core ML load instance used in the run. Key artifacts by logical load
instance, not by raw file path, because the same decoder-only package may be
loaded multiple times under different compute-unit policies. Example keys:

- `config_a_har_post_3s`
- `config_a_har_post_10s`
- `config_b_decoder_all`
- `config_c_decoder_cpu_and_gpu`
- `config_bcpu_decoder_cpu_only`

### Per-Iteration Record

Fields that do not apply to a config are `null`.

```json
{
  "config": "a",
  "input_key": "medium",
  "iteration": 0,
  "wall_time_s": 0.45,
  "canonical_audio_duration_s": 6.12,
  "observed_audio_duration_s": 6.08,
  "rtf_canonical": 0.074,
  "rtf_observed": 0.074,
  "speed_vs_realtime_canonical": 13.5,
  "bucket_used": "10s",
  "t_prefix_extract_s": 0.11,
  "t_decoder_pre_cpu_s": 0.05,
  "t_har_builder_cpu_s": 0.04,
  "t_coreml_predict_s": 0.09,
  "t_trim_s": 0.001,
  "t_orchestration_s": 0.159,
  "status": "ok",
  "error": null
}
```

For Configs B/C/D/E, the stage timing fields are `null` unless the harness adds
additional config-specific timers later.

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Freeze Inputs Inside the Current 10s Envelope

**Goal:** Define a benchmark dataset that the current shipping Config A can run
without truncation.

**Tasks:**

- [ ] Create `BAKEOFF_INPUTS` in `scripts/bakeoff_harness.py` with exactly four
      named inputs:
  - `tiny` targeting `~1s`
  - `short` targeting `~3s`
  - `medium` targeting `~6s`
  - `long` targeting `~9s`
      Do not exceed the current 10s HAR-post ceiling.
- [ ] Hardcode `VOICE = "af_heart"` and `SPEED = 1.0`.
- [ ] Implement `prepare-inputs` mode:
  - instantiate a CPU benchmark context once
  - run each input once under Config E
  - write `outputs/bakeoff/input_manifest.json`
  - record canonical duration and expected Config A bucket
- [ ] Hard-fail `prepare-inputs` if any canonical duration exceeds `9.0s`.
      Shorten the offending text and rerun instead of accepting an input set
      that sits on the `10s` edge.
- [ ] Add a smoke assertion that Config A would choose only `3s` or `10s` for
      the frozen inputs.

**Verification:**

- `python scripts/bakeoff_harness.py prepare-inputs`
- The generated manifest shows four inputs, canonical durations, and expected
  buckets with no input routed beyond `10s`.
- The command exits non-zero if any canonical duration exceeds `9.0s`.

---

### Phase 1: Build the Harness Around Existing Contracts

**Goal:** Create one benchmark harness that reuses repo contracts instead of
forking them.

**Tasks:**

- [ ] Create `scripts/bakeoff_harness.py`.
- [ ] Implement benchmark contexts that preload everything before timing:
  - `ConfigAContext`: `HybridTTSPipeline(force_engine="pytorch")` plus explicit
    `MLModel` loads for HAR-post `3s` and `10s` artifacts by path.
  - `DecoderOnlyContext(compute_units)`: same shared prefix pipeline plus one
    explicit decoder-only artifact path loaded with the requested compute units.
  - `PyTorchContext(device)`: `KPipeline(lang_code="a", model=False)` plus one
    preloaded `KModel().to(device).eval()`.
- [ ] Make `PyTorchContext(device="mps")` validate that
      `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before the benchmark starts.
- [ ] Reuse `HybridTTSPipeline.extract_vocoder_inputs()` as the canonical shared
      prefix for Configs A/B/C. Do not add a second shared-prefix helper to the
      production code.
- [ ] Implement a benchmark-local timed replica of Config A that follows the
      exact geometry in `decoder_har_post_bucket_impl()` but exposes stage
      timings:
  - `t_prefix_extract_s`
  - `t_decoder_pre_cpu_s`
  - `t_har_builder_cpu_s`
  - `t_coreml_predict_s`
  - `t_trim_s`
  - `t_orchestration_s`
- [ ] Add one smoke check proving the timed Config A runner still produces a
      finite waveform and the same bucket choice as the canonical backend on a
      representative input.
- [ ] Add counterbalanced order:
  - warm each context once
  - use a deterministic `--order-seed`
  - shuffle config order per repetition
  - shuffle input order per repetition
  - call `gc.collect()` between configs
  - call `torch.mps.empty_cache()` after MPS runs when available
- [ ] Write results with failure sentinels instead of aborting the whole run.
- [ ] Record top-level provenance:
  - `order_seed`
  - `git rev-parse HEAD`
  - dirty-tree state
  - Python executable
  - `sw_vers`
  - `sysctl -n machdep.cpu.brand_string`
  - `sysctl -n hw.memsize`
  - package versions
  - SHA256 and input shapes for each Core ML artifact

**Verification:**

- `python scripts/bakeoff_harness.py run --configs a,e --iterations 2 --order-seed 0`
- Results JSON exists, records provenance, and shows two iterations for each
  input/config pair.
- Config A results include bucket name and stage timings.

---

### Phase 2: Prepare the Naive Decoder-Only Control Artifact

**Goal:** Export one fresh decoder-only 10s Core ML package using the canonical
  exporter, then load it under multiple compute-unit policies.

**Tasks:**

- [ ] Create `requirements-bakeoff.txt` with concrete pins aligned to the repo’s
      working export stack:
  - `torch==2.5.0`
  - `coremltools==8.3.0`
  - `numpy==1.26.4`
  - `transformers==4.44.2`
  - `huggingface_hub==0.28.1`
  - `loguru==0.7.3`
  - `misaki[en]==0.9.4`
- [ ] Export one fresh decoder-only 10s artifact with the canonical exporter:

```bash
python -m export_synth.main --mode decoder --buckets 10s -o outputs/bakeoff/models
```

- [ ] Capture stdout/stderr to
      `outputs/bakeoff/conversion_decoder_only_10s.log`.
- [ ] Reload the same saved package three ways:
  - `.all`
  - `.cpuAndGPU`
  - `.cpuOnly`
- [ ] Validate those loads on one realistic decoder-only input bundle derived
      from `HybridTTSPipeline.extract_vocoder_inputs()` and padded to the 10s
      decoder-only contract. Do not use zero-only or synthetic smoke tensors for
      the control artifact acceptance gate.
- [ ] Record artifact hash, load-time compute units, and input shapes in the
      run manifest.
- [ ] If export fails, write the exact failure to the conversion log and mark
      Configs B/C unavailable. This is a valid experiment outcome and must not
      block Configs A/D/E.

**Verification:**

- The decoder-only 10s artifact exists under `outputs/bakeoff/models/`.
- `.all` and `.cpuAndGPU` both return finite waveforms on the same realistic
  DurationModel-derived input bundle; otherwise do not claim B-vs-C
  comparability.
- `.cpuOnly` either returns a finite waveform on that same realistic input
  bundle or is
  explicitly marked unavailable in the manifest and excluded from Gate 1
  classification.
- If export fails, the log clearly explains why and the harness skips B/C
  cleanly.

---

### Phase 3: Collect the Primary M2 Ultra Dataset

**Goal:** Produce the benchmark data and telemetry needed for the paper’s main
claims on the primary machine.

**Tasks:**

- [ ] Close non-essential apps.
- [ ] Pre-cache sudo credentials with `sudo -v`.
- [ ] Export `PYTORCH_ENABLE_MPS_FALLBACK=1` in the shell before any Config D
      run.
- [ ] Run `prepare-inputs` if the input manifest does not already exist.
- [ ] Run the main benchmark:

```bash
python scripts/bakeoff_harness.py run --configs a,b,c,d,e --iterations 5 --order-seed 0
```

- [ ] Run telemetry loops for Gate 1 on the frozen `long` input:
  - terminal A, using the smallest supported interval on the host
    (`100 ms` preferred; `1000 ms` acceptable if `powermetrics` rejects
    smaller):

```bash
INTERVAL_MS=100
SAMPLES=$((60000 / INTERVAL_MS + 5))
sudo powermetrics -i "${INTERVAL_MS}" --samplers ane -n "${SAMPLES}" > outputs/bakeoff/powermetrics_config_b_all.txt
```

  - terminal B:

```bash
python scripts/bakeoff_harness.py telemetry-loop --config b --input long --seconds 60
```

- [ ] Repeat the same pair for Config C (`.cpuAndGPU`) with a distinct capture
      path:

```bash
INTERVAL_MS=100
SAMPLES=$((60000 / INTERVAL_MS + 5))
sudo powermetrics -i "${INTERVAL_MS}" --samplers ane -n "${SAMPLES}" > outputs/bakeoff/powermetrics_config_c_cpu_and_gpu.txt
```
- [ ] Run Config Bcpu telemetry only if Gate 1 classification is still
      ambiguous after comparing B and C, again with its own capture path.
- [ ] Sanity-check the results:
  - no negative times
  - no NaN RTF values
  - canonical durations match the input manifest
  - Config A never uses a bucket beyond `10s`

**Verification:**

- `outputs/bakeoff/results_m2_ultra.json` exists.
- Telemetry loop logs exist for Configs B and C.
- Gate 1 has enough evidence to classify ANE participation as yes, no, or
  indeterminate.

---

### Phase 4: Collect Conditional M1 Mini Data

**Goal:** Gather cross-machine data when the required models fit in 16 GB using
explicit-path loading.

**Tasks:**

- [ ] Install the same benchmark environment from `requirements-bakeoff.txt`.
- [ ] Export `PYTORCH_ENABLE_MPS_FALLBACK=1` before any Config D run.
- [ ] Copy only the required artifacts:
  - `coreml/kokoro_duration.mlpackage`
  - `coreml/kokoro_decoder_har_post_3s.mlpackage`
  - `coreml/kokoro_decoder_har_post_10s.mlpackage`
  - `outputs/bakeoff/models/kokoro_decoder_only_10s.mlpackage`
- [ ] Run:

```bash
python scripts/bakeoff_harness.py run --configs a,d,e --iterations 5 --order-seed 0
```

- [ ] Attempt Configs B/C only if the decoder-only 10s control artifact loads
      successfully under 16 GB.
- [ ] If Config A or B/C still OOM on M1, write the failure sentinel and stop
      treating M1 as a hard requirement for this plan revision.

**Verification:**

- Either `outputs/bakeoff/results_m1_mini.json` exists, or
  `outputs/bakeoff/results_m1_mini_skipped.json` records the exact reason the
  run could not complete.

---

### Phase 5: Summarize Results and Answer the Gates

**Goal:** Produce the tables and written conclusions that replace the current
timing anecdotes.

**Tasks:**

- [ ] Implement `summarize` mode in `scripts/bakeoff_harness.py`.
- [ ] Produce per-machine tables for:
  - mean / median / std / min / max of `wall_time_s`
  - mean / median / std / min / max of `rtf_canonical`
- [ ] Produce `outputs/bakeoff/summary.md` with five gate answers:
  - **Gate 1 — Does the naive decoder-only Core ML artifact use ANE under `.all`?**
    Primary evidence is the telemetry loop plus B-vs-C latency on the same
    artifact. Use Bcpu only if classification between CPU and GPU fallback is
    still ambiguous.
  - **Gate 2 — How large is the shipping hybrid speedup versus PyTorch CPU and MPS?**
    Compare Config A against Config E and Config D using `wall_time_s` and
    `rtf_canonical`. Treat Config D as the real observed MPS baseline in this
    repo, not as an ideal GPU ceiling.
  - **Gate 3 — How does the advantage scale with sequence length?**
    Analyze the four frozen inputs from `~1s` through `~9s`.
  - **Gate 4 — How much CPU-side overhead remains in Config A?**
    Use Config A stage timings.
  - **Gate 5 — How does scaling change on weaker hardware?**
    Answer only if M1 Mini data exists.
- [ ] If Config B export fails or telemetry is inconclusive, say so explicitly
      in the gate answer instead of improvising certainty.

**Verification:**

- `python scripts/bakeoff_harness.py summarize --results outputs/bakeoff/results_m2_ultra.json`
- `outputs/bakeoff/summary.md` exists and every gate has a written answer with
  supporting numbers or an explicit limitation.

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] The frozen input set fits within the current shipping HAR-post `3s` / `10s`
      bucket set with no truncation and at least `1.0s` of headroom below the
      `10s` ceiling.
- [ ] Configs A, D, and E run end-to-end on M2 Ultra with identical inputs.
- [ ] The naive decoder-only 10s export is attempted and its outcome is
      documented.
- [ ] Results JSON includes git provenance and artifact hashes.
- [ ] Gate 1 and Gate 2 are answered from M2 Ultra data without relying on the
      old anecdotal numbers.

### Definition of Done

- [ ] `scripts/bakeoff_harness.py` committed
- [ ] `requirements-bakeoff.txt` committed
- [ ] `outputs/bakeoff/results_m2_ultra.json` saved locally
- [ ] `outputs/bakeoff/summary.md` saved locally
- [ ] M1 Mini results saved or an explicit skip file saved
- [ ] No production runtime behavior changed as part of the benchmark harness

## Open Questions

### Resolved

- **Q:** What exactly is Config A?
- **A:** The current shipping HAR-post path only, using `3s` and `10s` buckets.

- **Q:** What exactly are Configs B and C?
- **A:** The same freshly exported decoder-only 10s artifact loaded with
  `.all` and `.cpuAndGPU`.

- **Q:** Should Gate 1 compare Config B directly to PyTorch CPU for short inputs?
- **A:** No. Gate 1 must use same-graph controls plus telemetry because the
  naive decoder-only artifact is intentionally fixed-shape and padded.

- **Q:** Should the harness add a new shared-prefix helper to production code?
- **A:** No. `extract_vocoder_inputs()` already exists and is the canonical
  prefix.

### Unresolved

- None. If a new unresolved technical decision appears during implementation,
  stop and update this plan before proceeding.

## References

### Internal

- [Bakeoff Experiment Design](kokoro-bakeoff-experiment-v1.md)
- [Debug Notes](../Notes/debug-notes.md)
- [Learnings](../learnings.md)
- [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [PyTorch MPS Field Guide](../Guides/apple-silicon/pytorch-mps.md)
- [Plan Workflow Skills Guide](../Skills/plan-workflow-skills-guide.md)
- [Phase Audit Rubric](../Skills/phase-audit-rubric.md)

### External

- [Kokoro-82M on Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M)
- [Core ML Tools Docs](https://apple.github.io/coremltools/docs-guides/)

## Risks and Mitigations

- **The naive decoder-only export fails:** This is a valid result. Capture the
  conversion log, skip B/C, and still complete A/D/E.
- **M1 Mini still OOMs with explicit-path loading:** Save a skip file and treat
  cross-machine data as deferred rather than silently dropping the machine.
- **MPS results remain poor because of fallback-heavy ops:** That is still worth
  reporting as the path-of-least-resistance baseline, not as the GPU ceiling.
- **Telemetry interval support differs by macOS version:** Use the smallest
  supported `powermetrics` interval on the host and record the exact command in
  the output directory. Prefer `100 ms`; fall back to `1000 ms` only when the
  host rejects smaller intervals.
- **Stale or wrong artifacts get benchmarked:** Always record exact paths,
  input shapes, and SHA256 hashes. Avoid app-bundle `.mlmodelc` assumptions.

## Rollback and Cleanup

- Delete `outputs/bakeoff/` to remove generated models, telemetry logs, and
  results.
- Revert `scripts/bakeoff_harness.py` and `requirements-bakeoff.txt` if the
  benchmark branch is abandoned.
- No production runtime changes are expected. If a minimal helper extraction
  becomes necessary during implementation, keep it isolated in its own commit so
  it can be reverted independently.

## Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `README/Plans/kokoro-bakeoff-v2.md` | Update | This revised implementation-ready plan |
| `scripts/bakeoff_harness.py` | Create | Unified benchmark harness with four modes |
| `requirements-bakeoff.txt` | Create | Pinned benchmark environment |
