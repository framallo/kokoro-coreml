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
