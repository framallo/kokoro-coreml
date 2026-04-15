# Swift Prefix Rewrite Plan

**Date:** 2026-04-15
**Status:** Planned

## Executive Summary

The bakeoff v2 stage breakdown shows CPU-side Python consumes 62–86% of end-to-end wall time in the shipping HAR-post path. Two functions account for this: `extract_vocoder_inputs()` (52–127 ms) and `build_decoder_har_post_inputs_np()` (40–86 ms). CoreML `predict()` is only 19–57 ms. This plan replaces the Python prefix with a chain of CoreML models + native Swift DSP so the full text-to-waveform path can run without Python. The highest-risk piece (DecoderPre CoreML export with AdaIN) is deferred until after a working end-to-end Swift pipeline is benchmarked — most of the savings come from eliminating Python orchestration overhead and replacing PyTorch CPU inference with existing CoreML models + Swift DSP.

## Problem Statement

- **Symptom:** 150–280 ms end-to-end latency for the HAR-post path on M2 Ultra; 62–86% of that is CPU-side Python (PyTorch neural networks + numpy).
- **Root Cause:** `extract_vocoder_inputs()` runs 7 PyTorch sub-models (BERT, LSTM, duration predictor, F0/N prediction, text encoder) in Python. `build_decoder_har_post_inputs_np()` runs the full PyTorch decoder stack (Conv1d + AdainResBlk1d encode/decode blocks) plus hn-nsf harmonic source + STFT — this is ~80 ms of neural network + DSP computation, not tensor formatting.
- **Impact:** Latency floor of ~100 ms even on fast hardware; blocks real-time streaming and iOS deployment where Python is unavailable.

## Critical Scope Clarification

**What these functions actually compute (not just "format arrays"):**

| Function | What it really does | Time |
| --- | --- | --- |
| `extract_vocoder_inputs()` | Runs BERT → bert_encoder → predictor.text_encoder (LSTM+CNN) → predictor.lstm → duration_proj → F0Ntrain → text_encoder → alignment matrix → matrix multiplies | 52–127 ms |
| `build_decoder_har_post_inputs_np()` | Runs F0_conv + N_conv (Conv1d) → decoder.encode (AdainResBlk1d) → decoder.decode (multiple blocks) → f0_upsamp → m_source (SineGen/hn-nsf) → STFT transform → padding | 40–86 ms |

**What already exists as CoreML:**

| Model | Covers | Status |
| --- | --- | --- |
| `kokoro_duration.mlpackage` | BERT + predictor + duration_proj + text_encoder → outputs pred_dur, d, t_en, s, ref_s | In production (Swift app) |
| `kokoro_decoder_har_post_{3,10}s.mlpackage` (GeneratorFromHar) | Post-har conv stack + iSTFT → waveform from (x_pre, ref_s, har) | In production (Swift app) |

**What still needs to be built:**

| Component | Type | Priority | Notes |
| --- | --- | --- | --- |
| F0Ntrain prediction | New CoreML export | P0 | Small model: (en, s) → (F0_pred, N_pred). Not in duration model. |
| hn-nsf harmonic source | Swift + Accelerate | P0 | SineGen + STFT. Known to fail in CoreML (correlation ~0.00; see debug-notes.md). Must stay on CPU. |
| Alignment matrix builder | Swift | P0 | Pure tensor ops. Trivial. Already specified in `_build_alignment_matrix()`. |
| Swift pipeline orchestration | Swift Package | P0 | Chain models + DSP, stage timing. |
| Decoder pre-processing | New CoreML export | P1 (deferred) | F0_conv + N_conv + encode + decode blocks → x_pre. Has AdaIN (same concern as decoder-only). **Highest technical risk. Defer until pipeline works end-to-end with PyTorch bridge for this stage.** |
| Phoneme extraction + voice embedding | Swift | P0 (bench only) | Use pre-tokenized JSON inputs for benchmarking. Full Swift tokenizer is separate scope. |

## Goals and Non-Goals

### Goals

- [ ] Export `F0Ntrain` as a CoreML `.mlpackage` with numeric validation
- [ ] Implement hn-nsf (SineGen + STFT) in Swift using Accelerate/vDSP
- [ ] Create a Swift Package (`KokoroPipeline`) that chains: Duration CoreML → alignment (Swift) → F0Ntrain CoreML → padding (Swift) → DecoderPre (PyTorch bridge initially) → hn-nsf (Swift) → GeneratorFromHar CoreML → trim
- [ ] Benchmark the Swift chain with the same 4 bakeoff inputs (tiny/short/medium/long) and report wall time, stage breakdown, and RTF
- [ ] Run a full five-config bakeoff with Swift pipeline as Config F for publication-ready comparison
- [ ] Record all results in `README/Notes/performance-notes.md`

### Non-Goals

- Modifying existing CoreML models (duration, GeneratorFromHar)
- Fixing the decoder-only audio quality issue (SourceModuleHnNSF/SineGen CoreML parity — separate concern)
- Porting phoneme text normalization to Swift (use pre-tokenized inputs for benchmarking)
- iOS/macOS app integration (that's the TalkToMe repo)
- Changing the Python pipeline (it continues to work as-is for development/export)

## Ground Truth Contracts (Do Not Violate)

- **Duration model interface:** input_ids `[1,128]` int32, ref_s `[1,256]` float32, speed `[1]` float32, attention_mask `[1,128]` int32 → pred_dur, d, t_en, s, ref_s_out
- **GeneratorFromHar interface:** x_pre `[1,C,T_asr]`, ref_s `[1,256]`, har `[1,2*n_fft,T_har]` → waveform — shapes are bucket-specific, read from model spec
- **Audio sample rate:** 24,000 Hz
- **F0 frame rate:** 80 Hz (F0 frames / 80.0 = seconds)
- **Trim contract:** `target_len = round((T_f0 / 80.0) * 24000.0)`, trim waveform to this length
- **Bucket selection:** smallest bucket ≥ ceil(total_seconds)
- **Voice embedding split:** ref_s[:, :128] = baseline, ref_s[:, 128:] = style
- **hn-nsf phase accumulation:** MUST use Float64 (Double) for the cumulative phase integrator in SineGen. This is **load-bearing, not optional.** The `long` input at 8.35 s means ~200,000 samples of phase accumulation. Float32 drift compounds over the utterance and will corrupt the harmonic spectrum. Downcast to Float32 only at the final output stage (har tensor for CoreML input).

## Already Shipped (Do Not Re-Solve)

- **Duration CoreML model:** `coreml/kokoro_duration.mlpackage` — exported at `export_duration.py`, validated, in production Swift app
- **GeneratorFromHar CoreML models:** `coreml/kokoro_decoder_har_post_{3,10}s.mlpackage` — exported at `export_synth/convert.py`, HAR-post path in production
- **Alignment matrix algorithm:** `coreml_pipeline.py:336-362` `_build_alignment_matrix()` — proven correct, just needs Swift port
- **Bucket geometry calculation:** `synthesis_backends.py:53-57` — `conv1d_output_length_from_module()` for frame_count from bucket seconds
- **Bakeoff harness + baseline numbers:** `scripts/bakeoff_harness.py`, results in `outputs/bakeoff/results_m2_ultra.json`
- **Production Swift app architecture:** Duration + alignment + HAR-post already working in TalkToMe (learnings.md §10: "17x faster than real-time")

## Fresh Baseline (Current State)

**Architecture:** Python orchestration with PyTorch CPU prefix → CoreML ANE decoder tail.

**Bakeoff v2 Config A stage breakdown (warm median, M2 Ultra):**

| Input | Audio | Prefix extract | HAR builder (CPU) | CoreML predict | Total |
| --- | --- | --- | --- | --- | --- |
| tiny | 1.55 s | 52.7 ms (35%) | 40.9 ms (27%) | 57.0 ms (38%) | 151 ms |
| short | 2.80 s | 92.4 ms (60%) | 39.9 ms (26%) | 19.1 ms (12%) | 155 ms |
| medium | 6.58 s | 109.5 ms (39%) | 80.4 ms (28%) | 84.0 ms (30%) | 283 ms |
| long | 8.35 s | 127.0 ms (46%) | 85.7 ms (31%) | 47.5 ms (17%) | 274 ms |

**Measured CoreML predict times (GeneratorFromHar, from bakeoff v2):** 19–84 ms depending on input and bucket. These are real numbers — the latency budget must not claim less than this.

**Missing data:** No per-stage timing from the production Swift app. The Duration CoreML model predict time on M2 Ultra is unknown — the "17x faster than real-time" number from learnings.md §10 is end-to-end, not per-stage. **Phase 0 must measure this before setting a latency target.**

**Known gaps:**
- No Swift code in this repo (Swift app is separate TalkToMe repo)
- F0Ntrain not exported to CoreML
- Decoder pre-processing not exported to CoreML
- hn-nsf has no Swift implementation
- No Swift-side per-stage timing data

## Solution Overview

**Phase ordering rationale:** The DecoderPre CoreML export (AdaIN risk) is deferred. The plan builds a working Swift pipeline first with DecoderPre still calling PyTorch via a lightweight bridge, benchmarks that, then decides whether the AdaIN fight is worth it.

```
Text → Phonemes (pre-tokenized JSON for benchmark)
    ↓
Voice embedding lookup (Swift: load .bin, average)
    ↓
Duration CoreML [EXISTING] → pred_dur, d, t_en, s, ref_s
    ↓
Alignment matrix (Swift: repeat_interleave → one-hot)
    ↓
Matrix ops (Swift/Accelerate: en = d × alignment, asr = t_en × alignment)
    ↓
F0Ntrain CoreML [NEW] → F0_pred, N_pred
    ↓
Pad to bucket geometry (Swift: zero-pad asr, F0, N)
    ↓
DecoderPre [PyTorch bridge initially; CoreML in Phase 4 if AdaIN exports] → x_pre
    ↓ (parallel with:)
hn-nsf (Swift/Accelerate, Double precision phase) [NEW] → har
    ↓
GeneratorFromHar CoreML [EXISTING] → waveform
    ↓
Trim (Swift)
```

## Implementation Phases

### Phase 0: Export F0Ntrain to CoreML + Measure Duration Model

**Goal:** Export `predictor.F0Ntrain` as a standalone CoreML `.mlpackage`. Also measure the Duration CoreML model's actual predict latency to ground the latency budget in real data.

**Context:** F0Ntrain takes aligned duration features `en` and style embedding `s`, predicts pitch (F0) and noise (N) curves. It is the only neural-network call between the duration model output and the decoder pre-processing input. Relatively small model.

**Tasks:**

- [x] Inspect `predictor.F0Ntrain` architecture: `kokoro/modules.py:260-365` — ProsodyPredictor.F0Ntrain method. Shared LSTM (640→512) + parallel F0/N branches (3× AdainResBlk1d + Conv1d proj). ~1.8M params. All ops CoreML-compatible.
- [x] Create `export_f0ntrain.py`:
  - F0NtrainWrapper extracts shared/F0/N/proj sub-modules from predictor
  - Input shapes: `en [1, 640, T]`, `s [1, 128]` — exported for T=120 (3s bucket) and T=400 (10s bucket)
  - `torch.jit.trace` → `ct.convert` with `FLOAT16`, `compute_units=ALL`
- [x] Numeric validation: F0 corr=0.999995, N corr=0.999999 (T=120). F0 corr=0.999997, N corr=0.999999 (T=400). Both PASS > 0.99.
- [x] Saved to `coreml/kokoro_f0ntrain_t120.mlpackage` and `coreml/kokoro_f0ntrain_t400.mlpackage`
- [x] **Duration CoreML predict latency measured:** 13.30 ms median (20 calls, M2 Ultra). Compare to ~50 ms for PyTorch equivalent. Benchmark built into `export_f0ntrain.py --bench-duration`.

**Verification:** `uv run python export_f0ntrain.py` succeeds, numeric validation passes, `pytest` still green. Duration model timing recorded.

---

### Phase 1: Implement hn-nsf in Swift with Accelerate

**Goal:** Native Swift implementation of the harmonic source generation path (f0 upsample → SineGen → STFT transform).

**Context:** hn-nsf (`SourceModuleHnNSF` / `SineGen`) generates harmonic waveforms from the F0 pitch curve, then STFT-transforms them into spectral features. This path has correlation ~0.00 in CoreML (see `README/Notes/debug-notes.md`) and MUST stay on CPU. The computation is pure DSP: sine wave generation at harmonic frequencies + FFT — ideal for Accelerate/vDSP.

**Tasks:**

- [x] Create `swift/Sources/KokoroPipeline/HarmonicSource.swift`:
  - `f0Upsample`: nearest-neighbor, scale_factor=300
  - `sineGen`: 9 harmonics, Double precision phase accumulator (load-bearing), downsample-cumsum-upsample matching PyTorch. Includes learned Linear(9→1) merge + Tanh.
  - `stftTransform`: n_fft=20, hop=5, Hann window (periodic), center=True with replicate padding, DFT basis computation matching custom_stft.py
  - `buildHar`: top-level function, returns (har, nFrames) with shape (22, nFrames)
- [x] **Numeric validation script** (`scripts/validate_hnsf_swift.py`): generates PyTorch reference outputs + learned l_linear weights for 4 test cases. Cross-validation with Swift deferred to Phase 2 (needs Swift CLI to read .npy).
- [x] Match PyTorch STFT: Hann window (periodic), `center=True` padding (replicate-pad input by n_fft/2=10 on each side)
- [x] Swift Package (`swift/Package.swift`) created with macOS 13+ / iOS 16+ targets
- [x] 7 unit tests pass: upsample shape/values, STFT shape/DC bin, buildHar shape, interpolation correctness
- [x] Release build: 7 tests in 0.060s (vs 0.406s debug). `buildHar` for 80 F0 frames ~= 60ms release.

**Verification:** Swift hn-nsf output matches PyTorch `gen.f0_upsamp → gen.m_source → gen.stft.transform` to correlation > 0.99 on all 4 bakeoff inputs. Phase accumulation verified at Double precision.

---

### Phase 2: Swift Package and Pipeline Orchestration

**Goal:** Create `KokoroPipeline` Swift Package that chains all models + Swift DSP into a single `synthesize()` call. DecoderPre stays as a PyTorch bridge call for now.

**Tasks:**

- [x] Create `swift/Package.swift` (done in Phase 1)
- [x] `swift/Sources/KokoroPipeline/KokoroPipeline.swift`:
  - `KokoroPipeline.init(modelsDirectory:, buckets:, linearWeights:, linearBias:)` — loads Duration, F0Ntrain, GeneratorFromHar CoreML models
  - `synthesize(inputIds:, attentionMask:, refS:, speed:, decoderPreKey:) -> SynthesisResult` — full 9-stage pipeline with stage timing
  - `StageTimings` struct with per-stage and total/preDecoder computed properties
  - `SynthesisResult` with audio, timings, bucket, audio duration
  - DecoderPre bridge: `precomputedDecoderPre` dictionary loaded from disk
- [x] `swift/Sources/KokoroPipeline/AlignmentBuilder.swift`:
  - `buildAlignmentMatrix(predDur:, traceLength:, frameCount:) -> [Float]` — flat row-major one-hot matrix
- [x] `swift/Sources/KokoroPipeline/MLMultiArrayHelpers.swift`:
  - `matmul3D(a:, b:, M:, K:, N:)` using `cblas_sgemm` via Accelerate
  - `zeroPad3D`, `zeroPad1D` for bucket geometry padding
  - `makeZeroArray3D`, `makeZeroArray2D`, `copyInto` for MLMultiArray construction
  - `inputShapes(from:)` for model spec introspection
- [x] `swift/Sources/KokoroPipeline/BucketSelector.swift`:
  - `selectBucket(totalSeconds:, availableBuckets:) -> Int?`
- [x] **DecoderPre bridge:** `scripts/decoder_pre_bridge.py` pre-computes x_pre for all 4 bakeoff inputs. Outputs saved to `outputs/decoder_pre_bridge/{key}/x_pre.npy`. Bridge times: tiny=44.7ms, short=38.4ms, medium=91.7ms, long=103.8ms.
- [x] Integration: KokoroPipeline.synthesize() chains all 9 stages. `swift build` clean (zero warnings). `swift test` 7 tests pass.

**Verification:** `swift build` succeeds. `swift test` runs unit tests for alignment builder, hn-nsf, padding. Full chain produces audio matching Python pipeline (correlation > 0.95 on bakeoff inputs).

---

### Phase 3: Benchmark Swift Pipeline and Record Results

**Goal:** Benchmark the Swift pipeline with the same 4 bakeoff inputs. Measure every stage. Ground the latency budget in reality.

**Tasks:**

- [ ] Create `swift/Sources/KokoroBenchmark/main.swift` (or a test target):
  - Same 4 inputs: tiny, short, medium, long (pre-tokenized, saved as JSON with pre-computed voice embeddings)
  - Same voice: af_heart, speed 1.0
  - 1 warmup + 5 timed iterations, report median
  - Stage timing (ContinuousClock): t_duration_coreml, t_alignment, t_f0ntrain_coreml, t_padding, t_decoder_pre_bridge, t_hnsf_swift, t_generator_coreml, t_trim, t_total
  - Report: wall time, RTF, stage breakdown percentages
- [ ] Compare against bakeoff v2 Config A numbers (Python baseline)
- [ ] **Identify where the savings actually land:** The three stages that should be dramatically faster are:
  - Duration: CoreML vs PyTorch CPU (expecting large speedup, but measure — don't guess)
  - hn-nsf: Accelerate vs PyTorch CPU (expecting ~3-5x speedup)
  - Orchestration overhead: zero Python interpreter overhead
- [ ] **Identify the new bottleneck:** With the bridge, DecoderPre is still Python. How much of total time does it now represent? This data decides whether Phase 4 is worth it.
- [ ] Update `README/Notes/performance-notes.md` with new section: "Swift prefix rewrite v1 (with DecoderPre bridge)"
- [ ] Record provenance: machine, git commit, model SHAs, Swift version, Xcode version

**Verification:** Benchmark completes on M2 Ultra. Results recorded in performance-notes.md with honest numbers — not targets, actuals.

---

### Phase 4: (Conditional) Export DecoderPre to CoreML

**Goal:** If Phase 3 benchmark shows DecoderPre bridge is a significant bottleneck AND the remaining savings justify the risk, export the decoder stack (F0_conv + N_conv + encode + decode → x_pre) as a CoreML `.mlpackage`.

**Gate:** Only proceed if Phase 3 data shows DecoderPre bridge time > 20% of total pipeline time AND total pipeline exceeds the target latency.

**Context:** The decode blocks use real AdaIN (style-conditioned normalization). The decoder-only export replaced AdaIN with IdentityAdaIN to work around MIL broadcast failures — we need to determine if real AdaIN can be preserved here. This is the highest technical risk in the plan. The ANE optimization v1 experiment showed CoreML's MIL compiler may already handle some of these ops internally.

**Tasks:**

- [ ] Create `export_decoder_pre.py`:
  - Wrap decoder pre-processing as a module: `DecoderPre(decoder)`
  - `forward(asr_pad, f0_pad, n_pad, ref_s) → x_pre`
  - Internal: F0_conv, N_conv, cat, encode, asr_res, decode loop (with AdaIN)
  - Bucket-specific static shapes (3s and 10s)
- [ ] **AdaIN export gate:** Trace with real AdaIN first. If MIL broadcast fails:
  - Option A: Use `IdentityAdaIN` (known quality degradation — document and measure)
  - Option B: Rewrite AdaIN as explicit MIL-friendly ops (scale+shift without broadcast)
  - Option C: Accept the bridge and close this phase as "deferred — AdaIN blocks export"
- [ ] Numeric validation: correlation > 0.99 with real AdaIN, or > 0.95 with IdentityAdaIN (document the gap)
- [ ] Save to `coreml/kokoro_decoder_pre_{3,10}s.mlpackage`
- [ ] Update Swift Package to use CoreML DecoderPre instead of bridge
- [ ] Re-benchmark and update performance-notes.md

**Verification:** Export succeeds for 3s and 10s buckets; numeric validation passes; full chain (Duration → F0Ntrain → DecoderPre CoreML → hn-nsf Swift → GeneratorFromHar) produces intelligible audio. Re-benchmark shows measurable improvement over bridge.

**Fallback:** If AdaIN export blocks this phase, document the blocker, keep the PyTorch bridge, and close the plan. The bridge version is still a significant improvement over the all-Python pipeline.

---

### Phase 5: Full Five-Config Bakeoff with Swift Pipeline (Config F)

**Goal:** Run a controlled comparison of all synthesis paths — including the Swift pipeline — for publication-ready results.

**Context:** Phase 3 benchmarks Swift in isolation. This phase puts it side-by-side with the existing five configs using the same methodology, inputs, counterbalancing, and machine. This produces the table where Swift+ANE is directly comparable to Python HAR-post, naive decoder-only, PyTorch MPS, and PyTorch CPU.

**Tasks:**

- [ ] Add Config F to `scripts/bakeoff_harness.py`:
  - Config F: Swift pipeline (calls the compiled Swift benchmark binary as a subprocess, or integrates via shared library)
  - Same 4 frozen inputs, same voice, same speed
  - Same counterbalanced ordering with Config F added to the config shuffle
- [ ] Run full bakeoff: `--configs a,b,c,d,e,f --iterations 5 --order-seed 0`
- [ ] Update `scripts/bakeoff_summarize.py` to include Config F in tables
- [ ] Update `README/Notes/performance-notes.md` with new section: "Bakeoff v3: Swift pipeline comparison"
  - Wall time table (6 configs × 4 inputs)
  - RTF table
  - Speedup: F vs A (Python HAR-post), F vs E (CPU), F vs D (MPS)
  - Config F stage breakdown
  - Provenance

**Verification:** Full bakeoff completes. Config F numbers are consistent with Phase 3 standalone measurements. Publication-ready tables in performance-notes.md.

---

### Phase 6: Validation and Cleanup

**Goal:** End-to-end validation and documentation.

**Tasks:**

- [ ] Full-chain audio quality check: listen to Swift-generated audio for all 4 inputs, compare with Python reference
- [ ] Ensure `uv run python -m pytest tests/ -x` still passes (Python pipeline unchanged)
- [ ] `swift build && swift test` passes
- [ ] Update README.md Swift integration section with new model list and pipeline architecture
- [ ] Clean up any temporary validation scripts and bridge code (if Phase 4 succeeded)

**Verification:** All tests pass. Audio is intelligible and matches Python reference. Performance numbers recorded.

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] Swift-generated audio correlation > 0.95 vs Python pipeline output on all 4 inputs
- [ ] No Python required at inference time for the CoreML model chain (excluding DecoderPre bridge if Phase 4 is deferred)
- [ ] Existing Python pipeline and tests unmodified and passing
- [ ] Full five-config bakeoff (Phase 5) completed with publication-ready tables

### Stretch Goals (Measure, Don't Mandate)

- [ ] Pre-decoder overhead under 30 ms warm median on M2 Ultra — **measure after Phase 3 and report actuals, do not force-fit**
- [ ] Full pipeline under 80 ms for `long` input — **depends on measured Duration CoreML and GeneratorFromHar times which are 19–84 ms alone**

### Definition of Done

- [ ] F0Ntrain CoreML model exported and validated
- [ ] hn-nsf Swift implementation validated against PyTorch (correlation > 0.99)
- [ ] Swift Package builds and tests pass
- [ ] Benchmark results recorded in `README/Notes/performance-notes.md` (both standalone and bakeoff v3)
- [ ] Code committed and pushed

## Open Questions

### Unresolved

- **Q:** What is the Duration CoreML model's actual predict latency on M2 Ultra?
- **Context:** The production Swift app reports "17x faster than real-time" end-to-end, but no per-stage breakdown exists. The Python pipeline takes ~50 ms for the equivalent PyTorch computation. CoreML will be faster, but how much? **Phase 0 must measure this.** The latency budget depends on this number.

- **Q:** What is the F0Ntrain architecture? Need to inspect `kmodel.predictor.F0Ntrain` to confirm it's a small, traceable model.
- **Options:** Likely a small CNN or MLP. If it's simple enough, could even implement directly in Swift with Accelerate instead of CoreML.

- **Q:** Can the decoder pre-processing (encode + decode blocks with real AdaIN) export to CoreML without the MIL broadcast failures that forced IdentityAdaIN in the decoder-only path?
- **Options:** (A) Real AdaIN exports fine for this subgraph since it's smaller, (B) Same MIL failure → use IdentityAdaIN and measure quality, (C) Rewrite AdaIN as explicit scale+shift. **Current lean:** Defer this question until Phase 4; the bridge buys time.

- **Q:** Where should the Swift Package live — repo root or `swift/` subdirectory?
- **Options:** (A) `swift/` subdirectory to keep Python and Swift separate, (B) repo root Package.swift. **Current lean:** A (`swift/`).

## Risks and Mitigations

- **hn-nsf phase drift (HIGH):** SineGen accumulates phase over the entire utterance. Float32 has ~7 decimal digits of precision. At 24 kHz over 8.35 s, that's ~200,000 samples. Phase wraps at 2π ≈ 6.28, so after ~1M increments the accumulated error is significant. **Mitigation: Double (Float64) precision for the phase accumulator is mandatory, not optional. This is load-bearing.** Final conversion to Float32 happens only when writing the har tensor.
- **AdaIN export blocks Phase 4:** The bridge version from Phase 2/3 is the fallback. Plan delivers value even without Phase 4.
- **MLMultiArray overhead:** CoreML model chaining may have per-call overhead that adds up across 3–4 models. Mitigation: measure each call individually in Phase 3. If overhead is significant, consider fusing F0Ntrain into the Duration model export.
- **Duration CoreML latency is higher than hoped:** If Duration predict is >15 ms (unknown until measured), the "< 30 ms pre-decoder" target may be unreachable without fusing models. Accept the measured number and adjust targets.

## Performance and Latency Budget

**Important: This budget uses measured data where available and explicitly marks unknowns. Do not treat unknowns as targets.**

| Stage | Estimate | Basis | Notes |
| --- | --- | --- | --- |
| Duration CoreML | **13.3 ms** | **Measured (Phase 0)** | 20 calls warm median on M2 Ultra, `compute_units=ALL`. 3.8x faster than PyTorch (~50 ms). |
| Alignment + matrix ops | < 1 ms | Trivial computation | Pure Swift, O(T) loop + two matrix multiplies on small tensors |
| F0Ntrain CoreML | **UNKNOWN** | Small model, needs measurement | Export in Phase 0, measure in Phase 3 |
| Padding | < 0.5 ms | Trivial memset/copy | Same as Python, just without numpy overhead |
| DecoderPre | **Bridge: ~40-86 ms** | Bakeoff v2 measured | Stays PyTorch until Phase 4. Phase 4 CoreML time is unknown. |
| hn-nsf Swift/Accelerate | **UNKNOWN, expect 5-15 ms** | Accelerate FFT is fast, but need measurement | vDSP FFT + vectorized sine. Double precision phase adds cost. |
| GeneratorFromHar CoreML | **19–84 ms** | **Bakeoff v2 measured** | Varies by input/bucket. Tiny=57ms, short=19ms, medium=84ms, long=47ms. |
| **Total (with bridge)** | **Measured stages + unknowns** | | Phase 3 will fill in all unknowns |
| **Total (with DecoderPre CoreML)** | **TBD after Phase 4** | | Only if Phase 4 proceeds |

**What we know will be faster (Python overhead elimination):**
- No Python interpreter startup/GIL overhead per call
- No numpy ↔ torch ↔ numpy conversion overhead
- No PyTorch eager-mode dispatch overhead for duration model (CoreML instead)
- Accelerate-native DSP instead of PyTorch CPU for hn-nsf

**What we don't know yet:**
- Duration CoreML predict time in isolation
- F0Ntrain CoreML predict time
- hn-nsf Swift/Accelerate actual latency
- MLMultiArray construction and model-chaining overhead

## References

### Internal

- [Performance Notes (baseline)](../Notes/performance-notes.md) — bakeoff v2 stage breakdown, measured CoreML predict times
- [Debug Notes (decoder-only quality)](../Notes/debug-notes.md) — hn-nsf CoreML failure (correlation ~0.00), AdaIN/IdentityAdaIN issues
- [ANE Optimization Plan](ane-optimization-v1.md) — prior AdaIN experiment, MIL compiler behavior
- [Bakeoff v2 Plan](kokoro-bakeoff-v2.md) — benchmark methodology, counterbalanced design
- [Learnings](../learnings.md) — §10 production status (17x RT), §3 decoder-only architecture
- [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)

### External

- [Apple Accelerate vDSP documentation](https://developer.apple.com/documentation/accelerate/vdsp)
- [CoreML MLModel prediction API](https://developer.apple.com/documentation/coreml/mlmodel)

## Files Likely to Change

| File | Change Type | Phase | Notes |
| --- | --- | --- | --- |
| `export_f0ntrain.py` | Create | 0 | F0Ntrain CoreML export script |
| `scripts/bench_duration_coreml.py` | Create | 0 | Duration model latency micro-benchmark |
| `swift/Package.swift` | Create | 2 | Swift Package manifest |
| `swift/Sources/KokoroPipeline/KokoroPipeline.swift` | Create | 2 | Main pipeline orchestration |
| `swift/Sources/KokoroPipeline/AlignmentBuilder.swift` | Create | 2 | Alignment matrix construction |
| `swift/Sources/KokoroPipeline/HarmonicSource.swift` | Create | 1 | hn-nsf in Swift/Accelerate (Double precision phase) |
| `swift/Sources/KokoroPipeline/MLMultiArrayHelpers.swift` | Create | 2 | MLMultiArray utilities |
| `swift/Sources/KokoroPipeline/BucketSelector.swift` | Create | 2 | Bucket selection logic |
| `swift/Sources/KokoroBenchmark/main.swift` | Create | 3 | Swift benchmark harness |
| `scripts/validate_hnsf_swift.py` | Create | 1 | hn-nsf cross-validation |
| `scripts/decoder_pre_bridge.py` | Create | 2 | PyTorch bridge for DecoderPre |
| `scripts/bakeoff_harness.py` | Modify | 5 | Add Config F (Swift pipeline) |
| `scripts/bakeoff_summarize.py` | Modify | 5 | Add Config F to tables |
| `export_decoder_pre.py` | Create | 4 | DecoderPre CoreML export (conditional) |
| `README/Notes/performance-notes.md` | Modify | 3, 5 | Swift results + bakeoff v3 |

> SIMPLER IS BETTER. Build the pipeline end-to-end first, benchmark with real numbers, then decide where to invest further. The bridge buys time on the hardest problem (AdaIN export) while delivering most of the value.
