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
