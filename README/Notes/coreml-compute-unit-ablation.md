# Core ML Compute-Unit Ablation Notes

Institutional memory for isolating Core ML `.all`, `.cpuAndGPU`,
`.cpuAndNeuralEngine`, and `.cpuOnly` behavior in the Swift Kokoro pipeline.

**Quick filter:** `grep -n "— Active" README/Notes/coreml-compute-unit-ablation.md`

---

## Issue: Swift Core ML `.all` Is Slower Than `.cpuAndGPU` On M2 Air-Class Hardware — Active

**First spotted:** 2026-04-17
**Status:** Active

### Summary

The first F/G ablation shows that allowing ANE via Core ML `.all` does not help
on the local Apple M2 24 GB machine. Config G (`.cpuAndGPU`) beats Config F
(`.all`) at every benchmark length, especially 15s and 30s, where the slowdown
is concentrated in `GeneratorFromHar`.

Latency alone does not prove `.all` actually used ANE. It proves only that
allowing ANE did not help. The next ablations must distinguish a bad ANE path
from a bad `.all` mixed execution plan and from GPU doing the useful work.

### Current Evidence

Completed run:

```bash
BAKEOFF_SKIP_SMOKE=1 uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs f,g \
  --iterations 5 \
  --order-seed 0 \
  --machine-id ane_ablation_fg_local
```

Result file:

- `outputs/bakeoff/results_ane_ablation_fg_local.json`

Machine:

- Apple M2, 24 GB unified memory
- macOS 15.7.5
- Git commit `9738030122ac00fe5fbe25930c094c467e70552a`
- Dirty tree: true

Median wall time:

| Input | F `.all` | G `.cpuAndGPU` | G/F |
| --- | ---: | ---: | ---: |
| 3s | 227.5 ms | 175.7 ms | 0.77 |
| 7s | 478.8 ms | 384.2 ms | 0.80 |
| 15s | 2051.9 ms | 802.3 ms | 0.39 |
| 30s | 4803.5 ms | 1592.2 ms | 0.33 |

Median `GeneratorFromHar` time:

| Input | F `.all` | G `.cpuAndGPU` |
| --- | ---: | ---: |
| 3s | 195.3 ms | 131.7 ms |
| 7s | 421.6 ms | 302.3 ms |
| 15s | 1911.9 ms | 638.8 ms |
| 30s | 4512.6 ms | 1273.7 ms |

### Interpretation Boundary

This result does **not** prove ANE execution was bad, because `.all` is a
scheduler request, not proof of active ANE placement. It establishes a narrower
but important claim:

> On this machine and artifact set, Core ML `.all` is slower than excluding ANE
> with `.cpuAndGPU` for the Swift decomposed pipeline.

Three mechanisms remain possible:

1. **ANE path hurts:** the ANE placement itself is slow for these bucket shapes.
2. **Mixed `.all` plan hurts:** `.all` creates a bad CPU/GPU/ANE partition or
   synchronization pattern, while ANE-only plus CPU might be fine.
3. **ANE fallback hurts:** `.all` attempts ANE, then spills or falls back into a
   worse execution path than explicitly excluding ANE.

### Required Ablations

Add two Swift Core ML compute-unit controls to the harness:

| Human label | Suggested harness id | Core ML compute units | Purpose |
| --- | --- | --- | --- |
| G-prime | `gne` | `.cpuAndNeuralEngine` | Force CPU + ANE only; exclude GPU. |
| G-double-prime | `gcpu` | `.cpuOnly` | Exclude both GPU and ANE. |

The full compute-unit matrix:

| Config | Core ML compute units | Meaning |
| --- | --- | --- |
| F | `.all` | Let Core ML choose CPU, GPU, and ANE. |
| G | `.cpuAndGPU` | Exclude ANE. |
| G-prime | `.cpuAndNeuralEngine` | Exclude GPU. |
| G-double-prime | `.cpuOnly` | CPU baseline for Swift Core ML packages. |

Decision logic:

| Result pattern | Interpretation |
| --- | --- |
| G-prime is catastrophic like F at 15s/30s | ANE path is likely the problem, or `.all` and CPU+ANE share the same bad fallback. |
| G-prime is fast while F is slow | The problem is specifically the `.all` mixed plan. |
| G-double-prime is about 2x slower than G | GPU is doing useful work in G. |
| G-double-prime is near G | The win is mostly "not ANE"; GPU is not carrying much useful work. |

### Cross-Machine Gates

Before rewriting the paper framing, run F/G at minimum on Ultra and Mini with
the same artifacts and harness shape.

| Scenario | Paper implication |
| --- | --- |
| G beats F on Ultra, Air, and Mini | Thesis is inverted: Swift + decomposed Core ML on CPU+GPU beats PyTorch-on-MPS, and ANE does not help this workload. |
| G beats F on Air, but F beats G on Ultra | Most interesting result: ANE behavior is hardware-dependent for generative audio, and consumer-tier chips can be penalized by `.all`. |
| F beats G on Ultra and Mini, but G beats F only on Air | Air may be an outlier due to thermal state, memory pressure, or a chip-specific Core ML plan. Paper can survive with a caveat. |

### Related Guides

- [Core ML compute-unit scheduling](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md) - explains `.all`, `.cpuAndGPU`, `.cpuAndNeuralEngine`, silent fallback, and powermetrics verification.
- [Bakeoff results v2](bakeoff-results-v2.md) - current cross-machine A/D/E/F benchmark context before this F/G ablation.
- [Performance notes](performance-notes.md) - historical timing and Core ML performance observations.

### Verification Gap

`powermetrics` was not captured during the first F/G run because non-interactive
sudo was unavailable:

```log
sudo: a password is required
```

For publication claims about ANE participation, latency comparisons must be
paired with telemetry:

```bash
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000
```

If the sampler names differ on the target macOS version, use the ANE-only form
from the scheduling guide:

```bash
sudo powermetrics -i 1000 --samplers ane
```

### Next Steps

- [ ] Add Swift Core ML `G-prime` (`.cpuAndNeuralEngine`) to the harness.
- [ ] Add Swift Core ML `G-double-prime` (`.cpuOnly`) to the harness.
- [ ] Run F/G/G-prime/G-double-prime on the local M2 24 GB machine.
- [ ] Run F/G on M2 Ultra.
- [ ] Run F/G on M1 Mini.
- [ ] Capture powermetrics during steady-state F and G loops.
- [ ] Update [bakeoff results v2](bakeoff-results-v2.md) only after cross-machine data is collected.

### Investigation Log

**2026-06-05**

- **Generator-isolation harness:** `kokoro-bench --generator-input-dump` now
  accepts `--warmup` and `--iterations`, so the generator can be timed against
  a dumped Swift tensor boundary (`x_pre_padded`, `ref_s`, `har_padded`) without
  rerunning duration, F0, decoder-pre, HnSF, trimming, or JSON orchestration.
- **Generator compute-unit result:** CPU+GPU remains the fastest policy for the
  GPU-preferred `GeneratorFromHar` graph on the losing machines. N=10 medians
  after three discarded warmups:

  | Machine | Input | `cpuAndGPU` | `.all` | `cpuAndNeuralEngine` | `cpuOnly` |
  | --- | --- | ---: | ---: | ---: | ---: |
  | m2-studio | 3s | 27.2 ms | 27.0 ms | 1535.5 ms | 100.9 ms |
  | m2-studio | 7s | 59.5 ms | 60.3 ms | not rerun | not rerun |
  | m2-air | 3s | 120.1 ms | 155.4 ms | not rerun | not rerun |
  | m2-air | 7s | 277.6 ms | 426.2 ms | not rerun | not rerun |
  | irvine-m1 | 3s | 168.9 ms | 172.8 ms | not rerun | not rerun |
  | irvine-m1 | 7s | 384.7 ms | 394.2 ms | not rerun | not rerun |

- **Decision:** Do not spend more time trying to make the existing generator
  package faster by changing `MLComputeUnits`. The path to beating laishere on
  M2 Air/M1 short and medium rows is generator graph work: splitting the graph
  into noise/vocoder/tail stages, or rewriting the ANE-hostile operator surface
  (`conv_transpose`, long harmonic temporal axes, and Snake activations lowered
  to `sin`).
- **Exact-generator-only geometry rejected:** `scripts/probe_generator_exact_geometry.py`
  exported exact 2.8s and 6.75s `GeneratorFromHar` packages from the dumped
  Swift tensor boundary. They ran, but failed parity against the current
  trimmed bucket reference: 2.8s corr `0.927`, SNR `8.87 dB`, max abs `0.240`,
  median `27.1 ms`; 6.75s corr `0.952`, SNR `10.73 dB`, max abs `0.389`,
  median `55.7 ms`. A shorter HAR-post package fed by cropped bucket tensors is
  not a production-safe optimization. If exact-duration generator packages are
  revisited, they need an end-to-end exact graph plus listening/quality gates.
- **Final-tail split rejected:** `scripts/probe_generator_split.py` exported a
  body package ending at pre-tail logits plus a tiny tail package for
  `exp`/`sin`/iSTFT. Parity passed, but the split was slower on local M2 Studio:
  3s fused `28.3 ms` vs split `29.1 ms`; 7s fused `57.5 ms` vs split `58.4 ms`.
  The tail costs only `0.8-1.2 ms`, so separating it does not address the body
  bottleneck.
- **HAR-noise split rejected as a direct static-HAR optimization:** `scripts/probe_generator_noise_split.py`
  exported `ref_s + har -> x_source_0/x_source_1` and a separate body package.
  Output matched the fused package exactly on 3s and 7s, proving the boundary is
  semantically safe, but total latency got worse: 3s fused `28.9 ms` vs split
  `32.5 ms`; 7s fused `57.6 ms` vs split `63.4 ms`. The body package shrank
  materially (`19.5 ms` at 3s and `37.7 ms` at 7s), but the noise package
  (`13.1 ms` / `25.8 ms`) plus dispatch overhead outweighed it. Forcing the
  noise-split body to CPU+ANE was catastrophic (`~237-248 ms` body at 3s), so
  the direct split does not recreate laishere's ANE residency.
- **Actual laishere lesson:** its own README says tail split alone failed. Its
  working path combines a separate fp32 noise stage, a dual-output vocoder with
  a discarded anchor to influence scheduling, a separate fp32 tail, cos-form
  Snake, and palettization where the audio output is discarded. The next graph
  experiment should port that scheduler trick or rewrite the body operator
  surface; more naive split boundaries are not supported by the data.
- **Dual-output anchor trick rejected for the current HAR-post graph:** `scripts/probe_generator_dual_anchor_split.py`
  ports the remaining visible laishere ingredients onto the static Swift dump
  boundary. On local M2 Studio `3s`, mean-anchor + cos-Snake + fp32 tail on
  CPU+ANE still spent `236.7 ms` in the vocoder and `249.8 ms` total versus
  `29.3 ms` fused. The audio-anchor variant was also slow (`242.3 ms` total,
  N=3 rejection run), and int8 palettizing the vocoder did not help
  (`252.3 ms` total, N=3 rejection run, with lower parity margin). The CPU+GPU
  variants passed parity but stayed slower than fused (`32.0-33.0 ms` split
  versus `28.1-28.9 ms` fused). This closes the "maybe laishere's dual output
  alone fixes scheduling" hypothesis for our current generator boundary.
- **Revised decision:** stop adding generator split boundaries unless a new
  hypothesis changes the operator surface. The remaining path to beat
  laishere's chain-only M2 Air/M1 short buckets is to reduce generator math or
  rewrite the GPU/ANE-hostile operators themselves (`conv_transpose`, AdaIN
  reductions/broadcasts, and Snake lowering), or to evaluate a larger
  end-to-end graph reshape rather than the already-rejected static HAR-post
  splits.
- **Visible laishere graph-surface rewrites rejected:** `scripts/probe_generator_cos_snake.py`
  now patches the actual dynamically loaded export module
  (`export_synth.wrappers.kokoro_istftnet`) and can test iOS17/CoreML7 target
  packages, broadcast AdaIN, native `nn.InstanceNorm1d` AdaIN, and cos-form
  Snake. `scripts/compare_coreml_graph_surface.py` records the MIL histogram.
  The strongest combined 3s candidate drops the fused graph from `2207` ops to
  `1635`, removing explicit `tile`, explicit reduction-based normalization,
  and nearly all Snake `sin` ops. It still does not speed prediction:
  M2 Studio `30.08 ms` fused vs `30.68 ms` candidate, M2 Air `120.51 ms` vs
  `120.65 ms`, and Irvine M1 `167.83 ms` vs `167.60 ms`. Parity passed
  (`corr >= 0.999994`, `SNR >= 49.7 dB`). This rejects the simple hypothesis
  that laishere wins because of visible MIL op cleanup alone.
- **Palettized fused generator rejected:** the same probe can now apply 8-bit
  k-means Core ML palettization. The strongest visible-surface candidate
  (native InstanceNorm + broadcast AdaIN + cos Snake + pal8) reproduced
  laishere's visible LUT surface (`101` `constexpr_lut_to_dense` ops) and cut
  the 3s package from `38M` to `19M`, but local M2 Studio predict-only latency
  worsened from `29.64 ms` fused to `31.31 ms` and the row missed the existing
  max-abs gate (`0.01007` vs threshold `0.01`). This rejects "palettize the
  fused final-waveform package" as the explanation for laishere's M1 lead.
  Laishere's palettized vocoder output is discarded before its separate tail,
  so its quantization error does not map directly onto our fused final
  waveform output.
- **coremltools 9 conversion-only path rejected:** the probe now records
  conversion toolchain versions. A plain iOS17 fused-generator export with
  `coremltools==9.0` preserved the same visible MIL surface as CT8 (`2207`
  ops; `51` conv, `4` conv_transpose, `88` reduce_mean, `96` tile, `50` sin).
  Same-process local 3s timing was a tie: shipping fused `30.07 ms`, CT8 iOS17
  `29.76 ms`, CT9 iOS17 `29.81 ms`. Remote 3s timing also tied on the losing
  machines: M2 Air shipping `120.803 ms` vs CT9 `120.816 ms`; Irvine M1
  shipping `167.900 ms` vs CT9 `167.947 ms`. The 7s local CT9 candidate was
  slower (`60.49 ms` vs `59.87 ms`). Do not pursue a CT9-only migration for the
  current fused final-waveform package without a new compute-plan signal.
- **RangeDim input contract rejected for the fused generator:** laishere's
  vocoder uses flexible ranges, so `scripts/probe_generator_cos_snake.py` now
  supports `--input-shape-mode range` for `x_pre` and `har`. Plain RangeDim
  converted but ran catastrophically (`1561.07 ms` vs `49.73 ms` fused) and
  emitted E5RT `tile` shape-propagation failures. The tile-free
  native+broadcast+cos graph still failed quality and speed on both CT8 and
  CT9 (`343.61-343.96 ms` candidate vs `31.91-33.05 ms` fused) with E5RT
  dynamic `add` broadcast failures. The flexible-shape advantage, if any, is
  tied to laishere's different split boundary; it is not a safe drop-in for the
  current fused final-waveform package.
- **Style-specialized generator rejected:** `scripts/probe_generator_style_specialization.py`
  bakes the dump's `ref_s` into the generator and replaces all `AdaIN1d`
  projections with fixed gamma/beta constants. The 3s MIL graph shrank from
  `2207` to `1625` ops and removed all `linear`/`reshape`/`split`/`tile` ops,
  but latency got worse on every tested Mac: M2 Studio `31.3 ms` fused vs
  `31.9 ms` specialized, M2 Air `120.8 ms` vs `123.0 ms`, and Irvine M1
  `167.8 ms` vs `170.8 ms`. It also bloats artifacts (`315 MB` for 3s,
  `690 MB` for 7s). Per-voice generator packages are not a speed path.
- **Per-stage generator split rejected as a production split and useful as a
  profiler:** `scripts/probe_generator_stage_split.py` splits the current
  static HAR-post generator into noise, first upsample/resblock stage, and
  second upsample/resblock stage plus tail. CPU+GPU parity passed, but the split
  is slower than fused because of extra package dispatch: `3s` fused `28.6 ms`
  vs split `33.0 ms`; `7s` fused `58.5 ms` vs split `66.5 ms`. The 3s stage
  medians are noise `12.6 ms`, stage0 `9.0 ms`, and stage1+tail `11.3 ms`.
  CPU+ANE is not viable for any substage: noise `67.0 ms`, stage0 `37.8 ms`,
  and stage1+tail `93.1 ms` when each is isolated with the other stages on
  CPU+GPU. The 3s MIL operation distribution is broad: fused `2207` ops, noise
  `562`, stage0 `807`, stage1+tail `856`. There is no hidden ANE island inside
  the current generator; the next real optimization must remove/rewrite work
  across the generator regions instead of only repartitioning them.
- **Remote stage-placement check:** the same 3s stage packages and tensor dump
  were copied to `m2-air` and `irvine-m1` and run predict-only. CPU+GPU parity
  passed, with M2 Air stage medians noise `51.2 ms`, stage0 `31.2 ms`,
  stage1+tail `44.1 ms`, and Irvine M1 medians noise `74.4 ms`, stage0
  `44.9 ms`, stage1+tail `64.6 ms`. CPU+ANE substage placement is invalid on
  both losing machines: M2 Air stage0 CPU+ANE corr `0.403806`, stage1+tail
  CPU+ANE corr `0.120825`; Irvine M1 stage0 CPU+ANE corr `0.403829`,
  stage1+tail CPU+ANE corr `0.121443`. Do not use ANE for any current generator
  substage on these hosts.
- **Laishere stage profile narrowed the remaining loss:** `scripts/external_bakeoff/profile_laishere_stages.py`
  times laishere's seven-package chain by stage while preserving its public
  timing boundary. On M2 Air, laishere's noise+vocoder+tail portion is
  effectively tied with our isolated generator (`3s` `123.7 ms` vs `120.1 ms`;
  `7s` `279.0 ms` vs `277.6 ms`). Source audit later showed this is not a
  pure same-boundary comparison: laishere's `KokoroVocoder` includes F0/N conv,
  decoder encode/decode, and the generator body, then emits `x_pre` for a
  separate fp32 tail. On Irvine M1, laishere remains faster at that broader
  boundary (`3s` `145.1 ms` vs our generator `168.9 ms`; `7s` `340.4 ms` vs
  `384.7 ms`) and also has faster upstream stages. The next useful comparison
  is an exact laishere-style decoder-plus-generator boundary probe against the
  Swift dumps, not more `MLComputeUnits` experiments or static HAR-post
  boundary splitting.
- **Laishere-style decoder+vocoder boundary rejected on local 3s:** `scripts/probe_decoder_vocoder_split.py`
  ports the broader laishere boundary onto the Swift tensor dumps: checked-in
  `decoder_pre` + fused HAR-post generator as the baseline versus a candidate
  chain of HAR-noise, decoder encode/decode plus generator body with a discarded
  anchor, and an fp32 tail. Parity passed against the fused baseline, but the
  boundary did not win. With body on CPU+ANE, Core ML emitted an ANE compiler
  failure and the warmed median was `119.5 ms` versus `32.9 ms` baseline
  (`noise 12.0 ms`, `body 105.6 ms`, `tail 1.5 ms`; corr `0.999917`, SNR
  `38.19 dB`). Reusing the same packages with body on CPU+GPU removed the ANE
  catastrophe but still lost: `38.0 ms` versus `33.2 ms` baseline
  (`noise 11.9 ms`, `body 24.7 ms`, `tail 1.4 ms`; corr `0.999991`, SNR
  `47.76 dB`). This rejects a simple "move our split to laishere's
  decoder+generator boundary" explanation. The remaining laishere gap is either
  from its full chain/runtime details, a hardware-specific Core ML compile plan,
  or work reduction outside this boundary; do not ship this split.

**2026-05-17**

- **Powermetrics result:** Config F/reference single-stream was not ANE-bound
  on Studio (`ANE Power: 0 mW`) even when the 3s path stayed fast. Irvine
  single-stream showed only tiny ANE readings while GPU power dominated.
- **Production-shaped result:** Irvine `.all` could stall in
  `ANECompilerService` before emitting workload traces. The same shape completed
  when forced to `.cpuAndGPU`.
- **Compute-plan result:** `kokoro_decoder_pre_3s.mlpackage` is
  NeuralEngine-preferred, but `kokoro_decoder_har_post_3s.mlpackage`
  (`GeneratorFromHar`) is GPU-preferred. The generator contains ANE-hostile
  structure: exported `conv_transpose`, >16k temporal axes in the harmonic
  branch, and Snake activations lowered to `sin` ops.
- **Prototype tried:** Rewriting `conv_transpose1d` as zero-insertion plus
  `conv1d` removed MIL `conv_transpose`, but the compute plan remained
  GPU-preferred. Cropping the harmonic branch before residual convs reduced the
  >16k conv surface but changed output because AdaIN normalizes over the full
  time axis. That path was not shipped.
- **Runtime decision:** The Swift pipeline now uses explicit stage placement:
  duration/F0 on `.cpuAndGPU`, `decoder_pre` on `.cpuAndNeuralEngine`, and
  `GeneratorFromHar` on `.cpuAndGPU`. This is a manual partition in the sense of
  the compute-unit scheduling guide: run the ANE-eligible static conv island on
  ANE, and keep the generator away from the ANE compiler until the model
  architecture changes.
- **Verification after commit `cdc4f86`:** Studio and Irvine both built the
  Swift package. Irvine's staged full workload completed without the earlier
  `.all` `ANECompilerService` stall and powermetrics showed nonzero ANE samples
  (p95 47 mW, max 82 mW). Studio still reported 0 mW ANE even when an isolated
  `kokoro_decoder_pre_3s` loop was loaded with `CPU_AND_NE`; the compute plan
  for that exact model remains 100% NeuralEngine-preferred, so Studio
  powermetrics is non-confirming rather than proof of a different plan.
- **Decoder-pre latency control:** Isolated `kokoro_decoder_pre_3s` was faster
  with `CPU_AND_NE` than `CPU_ONLY`, and `CPU_AND_NE` matched `.all`: Studio
  p50 3.231ms (`CPU_ONLY`) vs 2.476ms (`CPU_AND_NE`) vs 2.469ms (`.all`);
  Irvine p50 5.024ms vs 2.734ms vs 2.741ms. This is the strongest confirmation
  that the staged runtime is preserving the ANE-eligible decoder-pre island
  while deliberately keeping the ANE-hostile generator on CPU+GPU.

**2026-04-17**

- **Hypothesis:** Config F's speedup over Config A may not isolate ANE, because
  F changes host language, orchestration, and the number of Core ML models.
- **Tried:** Added Config G as Swift + Core ML with `.cpuAndGPU`, preserving the
  same Swift pipeline and model packages as Config F.
- **Outcome:** Config G was faster than F on the local Apple M2 24 GB machine.
  The result invalidates an ANE-latency-win claim for this machine, but does not
  yet identify whether `.all` used ANE, mixed a bad plan, or fell back poorly.
