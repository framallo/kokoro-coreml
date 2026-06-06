# Kokoro M1 Graph Surface Target

June 6, 2026

This note turns the laishere-vs-first-party Core ML graph comparison into an
implementation target. It is not a general Core ML guide. It is the current
strict-parity frontier for making Config F faster than laishere on Irvine M1.

## Current Surface Delta

Command used for the latest local refresh:

```bash
uv run --no-sync python scripts/compare_coreml_graph_surface.py \
  --model ours3=coreml/kokoro_decoder_har_post_3s.mlpackage \
  --model laishere_vocoder=outputs/external_bakeoff/laishere_packages/KokoroVocoder.mlpackage \
  --model laishere_noise=outputs/external_bakeoff/laishere_packages/KokoroNoise.mlpackage \
  --model laishere_tail=outputs/external_bakeoff/laishere_packages/KokoroTail.mlpackage \
  --output outputs/graph_surface/laishere_vs_local_generator_refresh.json
```

Summary:

| Model | Spec | Size | Ops | Conv | ConvT | InstNorm | ReduceMean | Tile | Sin | Cos | LUT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| first-party `GeneratorFromHar 3s` | 7 | `39.7 MB` | `2207` | `51` | `4` | `0` | `88` | `96` | `50` | `1` | `0` |
| laishere `KokoroVocoder` | 8 | `49.1 MB` | `1534` | `54` | `3` | `42` | `1` | `0` | `0` | `36` | `101` |
| laishere `KokoroNoise` | 8 | `4.7 MB` | `529` | `16` | `0` | `12` | `0` | `0` | `13` | `0` | `26` |
| laishere `KokoroTail` | 8 | `0.1 MB` | `41` | `1` | `2` | `0` | `0` | `0` | `2` | `1` | `0` |

The important point is not only operation count. The first-party strict fused
generator has manual AdaIN lowering with `88` reductions and `96` tiles, while
laishere's vocoder has native `instance_norm`, no tiles, and LUT-backed weight
decompression. Existing first-party native-InstanceNorm and cos-Snake probes
matched small pieces of this surface but did not reproduce laishere's runtime
benefit.

## Target

Find a strict single-package surface that eliminates the manual AdaIN
`reduce_mean`/`tile` footprint without creating the split-boundary sync penalty
seen in decoder+vocoder and generator-stage splits.

The next useful candidate must do at least one of these:

- preserve the `GeneratorFromHar` call boundary while replacing manual AdaIN
  lowering with native `instance_norm` and no materialized time-axis tiles;
- combine native `instance_norm` with a weight-compression surface that remains
  strict and avoids the previous palettization/linear-quantization failures;
- change the tensor layout enough that Core ML can select a laishere-like
  mixed CPU/Neural Engine partition without increasing call count;
- prove that a smaller Swift-produced source/HAR tensor can be consumed inside
  one package without the existing source/STFT strict-path regression.

## Non-Targets

Do not spend more time on these as standalone probes:

- native InstanceNorm alone;
- cos-Snake alone;
- iOS17/spec8 alone;
- broadcast AdaIN alone;
- style specialization;
- HAR trim alone;
- fused `GeneratorFromHar` fp16 input dtype alone;
- fused native `instance_norm` plus fp16 input dtype without eliminating tiles;
- fused native `instance_norm` plus broadcast AdaIN plus fp16 input dtype as a
  standalone surface-only change;
- fused native `instance_norm` plus broadcast AdaIN plus fp16 input dtype plus
  8-bit palettization as a standalone surface-only change;
- adding more package boundaries.

Each one has already been measured as slower, noise-sized, or quality-failing.

## Latest Rejection

Fused `GeneratorFromHar` fp16 input dtype was tested directly at the existing
single-package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --input-dtype fp16 \
  --label 3s_fused_input_dtype \
  --report-name report_fp16_inputs_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict pass: corr `1.0` vs fused trimmed, SNR `142.94 dB`, max abs `0`;
- warmed local M2 Studio CPU+GPU: fused `26.434 ms`, fp16-input candidate
  `26.453 ms`, speedup `-0.07%`;
- graph surface: `2207 -> 2201` ops, but still `88` reductions and `96` tiles.

Decision: reject as standalone. It does not remove the manual AdaIN/tile
surface and is not worth Irvine timing.

## Latest Partial Surface Repair

Fused native `instance_norm` plus fp16 input dtype was also tested inside the
same `GeneratorFromHar` package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --native-instance-norm-adain \
  --input-dtype fp16 \
  --deployment-target ios17 \
  --label 3s_native_in_fp16_inputs \
  --report-name report_ios17_native_in_fp16_inputs_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict pass: corr `0.9999942588867583` vs fused trimmed, SNR `49.84 dB`,
  max abs `0.00256348`;
- warmed local M2 Studio CPU+GPU: fused `26.349 ms`, candidate `26.316 ms`,
  speedup `0.12%`;
- graph surface: `2207 -> 1725` ops, `88 -> 0` reductions,
  `0 -> 44` native `instance_norm`, but still `96` tiles and no LUT
  decompression.

Decision: partial surface repair, not material. Do not promote to Irvine by
itself. A useful next candidate must also eliminate the tile footprint or
otherwise create a measurable local win before remote timing.

## Latest Near-Surface Match

Fused native `instance_norm`, broadcast AdaIN, and fp16 input dtype was tested
inside the same `GeneratorFromHar` package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --native-instance-norm-adain \
  --broadcast-adain \
  --input-dtype fp16 \
  --deployment-target ios17 \
  --label 3s_native_in_broadcast_fp16_inputs \
  --report-name report_ios17_native_broadcast_fp16_inputs_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict pass: corr `0.9999942588867583` vs fused trimmed, SNR `49.84 dB`,
  max abs `0.00256348`;
- warmed local M2 Studio CPU+GPU: fused `26.424 ms`, candidate `26.402 ms`,
  speedup `0.08%`;
- graph surface: `2207 -> 1533` ops, `88 -> 0` reductions, `96 -> 0` tiles,
  `0 -> 44` native `instance_norm`, but still no LUT decompression and no
  material local runtime gain.

Decision: this is the closest first-party fused graph surface to laishere so
far, but it proves that removing reductions and tiles is not sufficient by
itself. Do not promote to Irvine unless combined with a change that creates a
larger local win, changes placement, or adds laishere-like weight
decompression without quality/runtime failure.

## Latest Full Surface Match

Fused native `instance_norm`, broadcast AdaIN, fp16 input dtype, and 8-bit
palettization was tested inside the same `GeneratorFromHar` package boundary:

```bash
uv run --no-sync python scripts/probe_generator_cos_snake.py \
  --no-cos-snake \
  --native-instance-norm-adain \
  --broadcast-adain \
  --input-dtype fp16 \
  --deployment-target ios17 \
  --palettize \
  --label 3s_native_broadcast_fp16_pal8 \
  --report-name report_ios17_native_broadcast_fp16_pal8_cpu_gpu.json \
  --warmup 3 \
  --iterations 10
```

Result:

- strict threshold pass but thinner margin: corr `0.9998795313806623` vs fused
  trimmed, SNR `36.55 dB`, max abs `0.00868225`;
- warmed local M2 Studio CPU+GPU: fused `26.333 ms`, candidate `27.077 ms`,
  speedup `-2.83%`;
- graph surface: `2207 -> 1533` ops, `88 -> 0` reductions, `96 -> 0` tiles,
  `0 -> 44` native `instance_norm`, and `0 -> 101` LUT decompression ops.

Decision: reject as standalone. This is a strong negative result because it
matches the visible laishere-like surface most closely so far, yet loses local
warmed runtime and reduces quality margin. The remaining win is not explained
by surface counts alone; it likely depends on placement, layout, external
runtime scheduling, or a boundary effect outside this fused package.

## Latest Placement Check

`MLComputePlan` confirms that the visible graph-surface match still does not
produce laishere-like ANE residency. Under `cpuAndNeuralEngine`, laishere's
vocoder has `597` Neural-Engine-preferred ops and about `0.56` NE cost weight.
The first-party near-surface fused generator has the same order of operations
(`1533` vs laishere `1534`) and hundreds of NE-supported ops, but Core ML still
prefers `0` ops on the Neural Engine and assigns `0` NE cost weight:

| Package | Units | Ops | Preferred CPU | Preferred GPU | Preferred NE | Unknown | CPU cost | GPU cost | NE cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| laishere vocoder | CPU+NE | 1534 | 58 | 0 | 597 | 879 | 0.409 | 0 | 0.560 |
| first-party HAR-post baseline | CPU+NE | 2207 | 1038 | 0 | 0 | 1169 | 0 | 0 | 0 |
| first-party native-IN+broadcast+fp16 | CPU+NE | 1533 | 677 | 0 | 0 | 856 | 0 | 0 | 0 |
| first-party native-IN+broadcast+fp16+pal8 | CPU+NE | 1533 | 677 | 0 | 0 | 856 | 0 | 0 | 0 |
| first-party native-IN+broadcast+fp16 | CPU+GPU | 1533 | 0 | 678 | 0 | 855 | 0 | 1.000 | 0 |

Reports:

- `outputs/graph_surface/compute_plan_laishere_vocoder_cpu_ne.json`
- `outputs/external_bakeoff/compute_plan/ours_har_post_3s_cpu_ne.json`
- `outputs/graph_surface/compute_plan_generator_native_broadcast_fp16_3s_cpu_ne.json`
- `outputs/graph_surface/compute_plan_generator_native_broadcast_fp16_pal8_3s_cpu_ne.json`
- `outputs/graph_surface/compute_plan_generator_native_broadcast_fp16_3s_cpu_gpu.json`

Decision: do not keep optimizing by only matching op counts, AdaIN lowering, or
palettization. The current first-party fused package is GPU-friendly but not
ANE-preferred on M1. The next useful target is whatever changes Core ML's
placement decision: tensor layout, operation dialect, package boundary,
conversion target/toolchain, or a narrower laishere-style vocoder contract.

## Deep Research Request

A useful external deep-research guide would be narrower than "Core ML
optimization." Ask for:

> How can an MLProgram for a 1D vocoder on M1 replace manual AdaIN
> reduce/tile lowering with native instance normalization or an equivalent
> ANE-friendly pattern, while preserving fixed-shape Core ML input/output
> contracts and avoiding CPU/NE synchronization regressions? Include concrete
> PyTorch/coremltools rewrite patterns, expected MIL ops, and ways to verify
> residency with `MLComputePlan`, Instruments, or Core ML performance reports.

The guide should specifically address why a graph with visible
Neural-Engine-preferred ops can still lose warmed runtime, because our strict
decoder+vocoder split already demonstrates that placement alone is not enough.
