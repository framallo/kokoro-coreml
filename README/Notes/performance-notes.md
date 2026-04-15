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
**Status:** Complete (M1 Mini and powermetrics telemetry deferred)

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
