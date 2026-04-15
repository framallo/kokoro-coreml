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

Cross-referenced the [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) against the production `GeneratorFromHar` ANE path. The Orion reverse-engineering project (Constraint #17) claims matmul executes 3x slower than 1x1 convolution on the ANE. We replaced `nn.Linear` with `nn.Conv1d(kernel_size=1)` in `AdaIN1d.fc` — dozens of instances in the hot ANE path. **Result: no improvement.** Core ML `predict()` time was unchanged or marginally worse. CoreML's MIL compiler already lowers `linear` ops to conv internally, making the source-level change redundant. The real latency gap (vs HF baseline packages) lives in `extract_vocoder_inputs()`, the PyTorch CPU prefix path — not the ANE decoder at all.

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

**Conclusion:** CoreML's MIL compiler already optimizes `linear` → conv internally during the `.mlpackage` compilation/specialization step. The source-level Conv1d change is redundant — the compiled ANE graph is the same either way. Orion Constraint #17 applies to **direct ANE programming** (bypassing CoreML), not to the CoreML conversion pipeline.

**The real regression is in the PyTorch CPU prefix:**

| Input | Stage | Repo | HF | Delta |
| --- | --- | --- | --- | --- |
| tiny | extract_vocoder_inputs | 49.9 ms | 41.5 ms | +8.4 ms |
| long | extract_vocoder_inputs | 128.6 ms | 119.4 ms | +9.2 ms |

This +8-9ms penalty is consistent across inputs and dwarfs the Core ML predict delta. It lives in the shared PyTorch path (duration model + alignment + hn-nsf), which the Conv1d change does not touch. The cause is not yet identified — may be a codebase drift between the HF-published packages and the current repo state.

### Key takeaways for future work

1. **Don't optimize what CoreML already optimizes.** The MIL compiler's internal lowering passes handle Linear → Conv conversion. Source-level changes to match Orion constraints only matter when programming the ANE directly (bypassing CoreML).
2. **Profile before optimizing.** The stage breakdown showed the bottleneck was in the PyTorch prefix, not the ANE decoder. Without the breakdown, we'd have spent more time on the wrong problem.
3. **The dead `torch.cat` removal was valid.** Even though it was dead code, removing it prevents future accidental activation and eliminates an ANE-banned op from the source.
4. **The `extract_vocoder_inputs()` regression is the real target.** A +8-9ms consistent penalty across inputs points to something in the duration model, alignment, or hn-nsf CPU path that changed between the HF baseline and the current repo.

### Related Guides

- [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) — Orion Constraints #1 (concat) and #17 (matmul vs conv); verification techniques
- [Apple: Deploying Transformers on the ANE](https://machinelearning.apple.com/research/neural-engine-transformers) — Linear-to-Conv2d recommendation (applies to direct ANE, not CoreML pipeline)
- [Orion paper](https://arxiv.org/abs/2603.06728) — reverse-engineered ANE constraints

### Files changed (then reverted)

- `kokoro/istftnet.py:129` — `AdaIN1d.fc`: Linear → Conv1d (reverted)
- `kokoro/istftnet.py:131-152` — `AdaIN1d.forward`: input reshape for Conv1d (reverted)
- `kokoro/istftnet.py:146-155` — dead `torch.cat` padding branch removed (kept)

### Plan reference

Full experiment design: `README/Plans/ane-optimization-v1.md`
