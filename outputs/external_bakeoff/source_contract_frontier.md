# Source Contract Frontier

Warmed-inference evidence for the remaining Irvine M1 source/body gap.

## Summary

- Source equation solved: `true`.
- Recomputed HAR/STFT solved: `false`.
- Source-variant buckets scanned: `5`.
- Swift-like source minimum SNR: `138.15 dB`.
- Dumped source recomputed-HAR maximum SNR: `8.23 dB`.
- Irvine real loss rows: `4`.
- Irvine source/body loss rows: `4`.
- Saved strict candidates closing Irvine rows: `0`.
- Quality-fail source candidates that beat warmed laishere profile: `3`.

## Post Overlap + Rewrite Budget

Additional strict speed still required after the current runtime overlap
and HAR-post upsample rewrite projection:

| Target | Bucket | Extra save needed | Extra generator speedup needed |
| --- | --- | ---: | ---: |
| warmed profile | `3s` | 27.0 ms | 16.88% |
| warmed profile | `7s` | 28.0 ms | 7.52% |
| warmed profile | `10s` | 12.8 ms | 2.40% |
| paper frontier | `3s` | 45.7 ms | 28.57% |
| paper frontier | `7s` | 77.6 ms | 20.88% |
| paper frontier | `10s` | 63.7 ms | 12.00% |
| paper frontier | `15s` | 64.4 ms | 8.06% |

## Implementation Queue

| Priority | Track | Experiment | Target buckets | Promotion gate |
| ---: | --- | --- | --- | --- |
| 1 | strict | algebraically fold STFT/HAR into first noise convolutions | `3s`, `7s`, `10s` | same x_source_* or early activation parity first, then waveform parity and warmed lower-end timing |
| 2 | strict | distill a different strict boundary, not compact direct x_source tensors | `3s`, `7s`, `10s` | activation or waveform parity first, then warmed end-to-end win on quiet Irvine M1; no new hot Core ML boundary unless it removes more cost than it adds |
| 3 | quality-changing | no-ASR human listening review for saved source/body speed branches | `7s`, `10s`, `15s` | filled no-ASR listening decisions plus waveform-health review; keep separate from strict paper claims unless the methodology explicitly accepts listening-equivalent quality |
| 4 | stop | do not repeat exact HAR-post splits, sine-source variants, compact direct x_source or pre-noise adapters, or no-side-input phase+rewrite packages | n/a | n/a |

## Source-Side Feasibility Smoke

- Tensor dump: `outputs/generator_isolation/dumps/3s`.
- Decision: `promising_adapter`.
- Target mode: `pre_noise_conv`.
- Model: `ridge`.
- Feature set: `har_conv_geometry`.
- Radius: `8`.
- Holdout stride: `5`.
- Hidden: `128`.
- Steps: `300`.
- Conv kernel: `9`.
- Conv depth: `3`.

| Target | Features | Validation SNR | Validation corr | Validation max abs |
| --- | ---: | ---: | ---: | ---: |
| `pre_noise_conv_0` | 265 | 96.64 dB | 1.000000 | 0.000142 |
| `pre_noise_conv_1` | 23 | 107.06 dB | 1.000000 | 0.000017 |

## Pre-Noise Folding Surface

- Decision: `fold_for_memory_locality_not_frame_skipping`.

| Bucket | HAR fp16 | Touched HAR frames | Pre-noise fp16 | Pre-noise/HAR values |
| --- | ---: | ---: | ---: | ---: |
| `3s` | 1.21 MiB | 100.00% | 9.38 MiB | 7.76x |
| `7s` | 2.82 MiB | 100.00% | 21.88 MiB | 7.76x |
| `10s` | 4.03 MiB | 100.00% | 31.25 MiB | 7.76x |
| `15s` | 6.04 MiB | 100.00% | 46.88 MiB | 7.76x |
| `30s` | 12.09 MiB | 100.00% | 93.75 MiB | 7.76x |

## Strict Folding Ceiling

- Decision: `do_not_build_materialized_pre_noise_boundary; only pursue fused runtime/kernel if it also reduces generator body scheduling or synchronization cost`.
- Profile rows closed by STFT-only removal: `0`.
- Paper rows closed by STFT-only removal: `0`.

| Bucket | Removable STFT | Profile coverage | Paper coverage |
| --- | ---: | ---: | ---: |
| `3s` | 0.5 ms | 1.92% | 1.13% |
| `7s` | 1.3 ms | 4.63% | 1.67% |
| `10s` | 1.7 ms | 13.61% | 2.73% |
| `15s` | 0.0 ms | n/a | 0.00% |

## Body Counterfactual

| Machine | Fused | Body only | Source/noise | Full split | Body-only save | Full split delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m2-studio | 26.4 ms | 17.6 ms | 11.3 ms | 28.9 ms | 8.8 ms | -2.4 ms |
| irvine-m1 | 168.3 ms | 105.9 ms | 74.0 ms | 179.8 ms | 62.4 ms | -11.5 ms |

## Quality-Fail Closers

Quality-fail buckets that would beat the warmed laishere profile if accepted:
`7s`, `10s`, `15s`.

## Decision

The Swift-like source equation is solved, but recomputed HAR/STFT is not. The body package is fast if x_source tensors are free, and quality-fail F0/source branches would close several warmed Irvine profile rows. The next useful work is a cheaper strict source/HAR contract or listening-accepted source replacement, not another exact HAR-post split.
