# Performance Notes

This note captures one thing only: **how the current repo `coreml/` HAR-post packages compare to the baseline packages downloaded from [mattmireles/kokoro-coreml on Hugging Face](https://huggingface.co/mattmireles/kokoro-coreml)**.

## Baseline

- **Candidate:** local repo packages in `coreml/`
- **Baseline:** downloaded HF packages in `outputs/hf_baseline/coreml/`
- **Artifacts compared:** `kokoro_decoder_har_post_3s.mlpackage` and `kokoro_decoder_har_post_10s.mlpackage`

## Method

- Measured **Core ML `predict()` median wall time only**
- Used the same input generation path for both sides:
  `HybridTTSPipeline` + `build_decoder_har_post_inputs_np`
- Verified that input/output **names and shapes** match between local and HF HAR-post packages
- Benchmark settings:
  - `warmup = 3`
  - `iterations = 21`
  - `torch.manual_seed(0)`
  - voice `af_heart`
  - speed `1.0`

This is **not** an end-to-end RTF comparison. It does **not** include prefix extraction, CPU orchestration, trimming, or full pipeline wall clock.

## Results

| Preset | Bucket | Approx F0 span | Local repo median | HF median | Result |
| --- | --- | --- | --- | --- | --- |
| `tiny` | 3s | `~1.55s` | `18.56 ms` | `17.11 ms` | Repo is `~8.5%` slower |
| `long` | 10s | `~9.48s` | `43.00 ms` | `41.01 ms` | Repo is `~4.9%` slower |

## Takeaway

On this run, the **downloaded Hugging Face baseline was slightly faster** than the new local repo HAR-post packages on raw `predict()` time.

The gap is small. Treat this as a **single-run micro-benchmark**, not a final end-to-end performance verdict.

## Artifact

- Results JSON: `outputs/bakeoff/local_vs_hf_har_post_predict.json`
# Performance notes (bakeoff-aligned)

Institutional memory for **benchmark targets**, **reported timings**, and **how to interpret** RTF / stage splits. Canonical experiment design lives in the bakeoff plan; this file extracts what matters for performance work without duplicating the full procedure.

**Source of truth:** [kokoro-bakeoff-v2.md](../Plans/kokoro-bakeoff-v2.md) (implementation plan, schemas, gates).

**Related:** [learnings.md](../learnings.md) (older CPU/MPS RTF tables from ad hoc runs), [debug-notes.md](debug-notes.md) (export/quality; not primary perf baseline).

**Pre-built packages (distribution):** [mattmireles/kokoro-coreml on Hugging Face](https://huggingface.co/mattmireles/kokoro-coreml) — same family of `.mlpackage` files; the model card’s RTF table (5s/15s/30s buckets, ~0.057 RTF on ~23.7s audio, stage breakdown) is **not** the same thing as Config A HAR-post-only timing below.

---

## Shipping path and scope

- **Config A (production hybrid):** `HybridTTSPipeline.extract_vocoder_inputs()` plus HAR-post Core ML only: `coreml/kokoro_decoder_har_post_3s.mlpackage` and `coreml/kokoro_decoder_har_post_10s.mlpackage`. No longer-valid baseline: legacy `Decoder_HAR` 5s/15s/30s RTF numbers.
- **Bakeoff non-goals:** Do not benchmark utterances **longer** than the shipping **10s** HAR-post bucket; do not treat deprecated `Decoder_HAR` multi-bucket paths as Config A.

---

## Fresh baseline (from bakeoff plan)

These are **anecdotes and placeholders** until `scripts/bakeoff_harness.py` exists and writes manifests under `outputs/bakeoff/`.

| Item | Value / note |
| --- | --- |
| Historical long-form (non-shipping) | Older `Decoder_HAR` path: RTF **~0.057** on **~23.7s** utterance — **not** Config A baseline. |
| HAR-post smoke (repo note) | **~0.374s** wall, **~1.36s** audio, RTF **~0.27** — paired PyTorch comparison used **different** duration; **not** a clean A/B. |
| Example results JSON sketch | `t_coreml_predict_s`: **~0.09** (illustrative per-iteration field in plan). |
| Known gaps | No frozen input manifest, no run manifest with artifact hashes, no sustained telemetry loop for naive Core ML, no counterbalanced ordering in one harness yet. |

---

## Headline configs (what each row measures)

| Config | Meaning |
| --- | --- |
| **A** | Shipping hybrid HAR-post (explicit 3s/10s loads). |
| **B** | Same naive decoder-only **10s** artifact, `ComputeUnit.ALL`. |
| **C** | Same artifact as B, `CPU_AND_GPU`. |
| **D** | PyTorch E2E on **MPS** (requires `PYTORCH_ENABLE_MPS_FALLBACK=1` for intended baseline). |
| **E** | PyTorch E2E on **CPU** (canonical duration for inputs comes from this path in `prepare-inputs`). |

Diagnostic **Bcpu** (`CPU_ONLY`) may appear in telemetry / Gate 1; it is not a headline matrix row.

---

## Inputs and guardrails

- **Named keys:** `tiny`, `short`, `medium`, `long` — target audio **~1s / ~3s / ~6s / ~9s** (see plan Phase 0).
- **Voice / speed (planned):** `af_heart`, `1.0`.
- **Hard rule:** `prepare-inputs` must **fail** if any canonical duration **> 9.0s** (headroom under 10s ceiling; avoid silent edge truncation).
- **Manifest:** `outputs/bakeoff/input_manifest.json` — text, canonical duration, expected Config A bucket (`3s` or `10s`), text `sha256`.

---

## Bucket selection (Config A)

Implementation: `HybridTTSPipeline._select_bucket_seconds()` — **smallest loaded bucket ≥ `ceil(total_seconds)`**, where `total_seconds` is derived from the vocoder F0 span (e.g. `T_f0 / 80` in the bench script).

With only **{3s, 10s}** HAR-post buckets:

- Utterances with duration **> 3s** (e.g. **~3.25s**) require the **10s** package, not 3s.
- Presets that mirror bakeoff **short** (~3s target) may still **route to 10s** if measured duration rounds up past 3s.

---

## Results schema (performance fields)

Per-iteration records (when harness exists) should include at least:

- **`wall_time_s`**, **`canonical_audio_duration_s`**, **`observed_audio_duration_s`**
- **`rtf_canonical`**, **`rtf_observed`**, **`speed_vs_realtime_canonical`**
- **`bucket_used`** (`3s` / `10s`)
- **Config A stage splits:** `t_prefix_extract_s`, `t_decoder_pre_cpu_s`, `t_har_builder_cpu_s`, **`t_coreml_predict_s`**, `t_trim_s`, `t_orchestration_s`

**Timer discipline (plan):** Load/compile/warmup **out of band**; wall clock for full pipeline includes prefix through final waveform; MPS stops timer after `torch.mps.synchronize()`.

---

## Decoder HAR-post `predict()` micro-bench (today)

Pending the full harness, median Core ML `predict()` time for HAR-post packages can be measured with:

```bash
uv run python scripts/bench_decoder_har_post_predict.py --preset long --warmup 1 --iterations 11
uv run python scripts/bench_decoder_har_post_predict.py --all-presets
```

That script maps to the bakeoff field **`t_coreml_predict_s`** (it prints milliseconds; divide by 1000 for seconds). It uses bakeoff-style presets defined in-script; **bucket and package path are inferred** from the pipeline.

A/B against the HF drop (same script, `--baseline` pointing at downloaded packages):

```bash
uv run python scripts/bench_decoder_har_post_predict.py \
  --package coreml/kokoro_decoder_har_post_3s.mlpackage \
  --baseline outputs/hf_baseline/coreml/kokoro_decoder_har_post_3s.mlpackage \
  --preset tiny
```

---

## Measured: repo `coreml/` vs Hugging Face HAR-post `predict()`

**What was compared:** Median Core ML **`predict()`** wall time only — not full hybrid RTF, not the model card’s end-to-end story.

**Method (one local run):** Identical inputs from `HybridTTSPipeline` / `build_decoder_har_post_inputs_np`; Core ML **input/output names and shapes** for `kokoro_decoder_har_post_{3,10}s` matched between [repo `coreml/`](../../coreml) and a `snapshot_download` of [mattmireles/kokoro-coreml](https://huggingface.co/mattmireles/kokoro-coreml) under `outputs/hf_baseline/coreml/` (weights/graph differ by export). **Candidate** = repo tree; **baseline** = HF files. Warmup **3**, iterations **21**, `torch.manual_seed(0)`, voice `af_heart`, speed `1.0`.

| Preset | Bucket | ~F0 span (s) | Repo median (ms) | HF median (ms) | Repo vs HF |
| --- | --- | --- | --- | --- | --- |
| `tiny` | 3s | ~1.55 | ~18.6 | ~17.1 | Repo **~8.5% slower** |
| `long` | 10s | ~9.48 | ~43.0 | ~41.0 | Repo **~4.9% slower** |

**Artifact:** `outputs/bakeoff/local_vs_hf_har_post_predict.json` (regenerate after re-download; typically gitignored).

**Interpretation:** Differences are small; another machine, OS build, or thermal state can flip them. For a stronger claim, repeat runs, pin compute units if needed, and add full-pipeline / bakeoff harness timings.

---

## Gates (summary)

Bakeoff **summarize** mode is expected to enforce (among others): no NaN RTF; Config A never uses a bucket beyond `10s`; frozen inputs stay under the duration guardrail with margin. See the plan for full Gate 1 / telemetry requirements (`powermetrics`, sustained loops).
