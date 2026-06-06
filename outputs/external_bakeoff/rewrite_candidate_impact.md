# Rewrite Candidate Impact

The HAR-post upsample rewrite is a measured local win, but this report keeps
it separate from the paper-facing frontier until quiet lower-end-device
timing proves the projection.

## Local End-to-End Proof

| Bucket | Baseline | Rewrite | Delta | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `3s` | 50.7 ms | 49.7 ms | -1.0 ms | 1.97% |
| `7s` | 95.5 ms | 93.8 ms | -1.7 ms | 1.79% |
| `10s` | 125.7 ms | 123.7 ms | -2.0 ms | 1.62% |
| `15s` | 185.8 ms | 183.5 ms | -2.3 ms | 1.22% |
| `30s` | 383.9 ms | 374.0 ms | -9.9 ms | 2.58% |

## Lower-End Projection

Projection uses measured local package-level generator speedup applied only
to each machine's current Config F generator stage. It does not assume
non-generator stages improve.

| Machine | Bucket | Generator | Package speedup | Projected save | Projected Config F | laishere | Gap after rewrite | Closes profile gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `irvine-m1` | `3s` | 167.1 ms | 4.28% | 7.2 ms | 226.4 ms | 195.0 ms | 31.4 ms | `no` |
| `irvine-m1` | `7s` | 383.7 ms | 3.15% | 12.1 ms | 480.6 ms | 444.2 ms | 36.4 ms | `no` |
| `irvine-m1` | `10s` | 548.6 ms | 3.17% | 17.4 ms | 668.0 ms | 644.9 ms | 23.2 ms | `no` |
| `irvine-m1` | `15s` | 820.8 ms | 2.60% | 21.4 ms | 993.6 ms | 990.6 ms | 3.0 ms | `no` |
| `m2-air` | `3s` | 120.6 ms | 4.28% | 5.2 ms | 142.9 ms | 153.0 ms | -10.1 ms | `yes` |
| `m2-air` | `7s` | 278.3 ms | 3.15% | 8.8 ms | 321.9 ms | 334.7 ms | -12.8 ms | `yes` |
| `m2-air` | `10s` | 396.0 ms | 3.17% | 12.6 ms | 453.4 ms | 467.3 ms | -13.9 ms | `yes` |
| `m2-air` | `15s` | 591.5 ms | 2.60% | 15.4 ms | 678.2 ms | 691.5 ms | -13.4 ms | `yes` |

## Decision

- Local end-to-end positive buckets: `5`.
- Projected lower-end profile gaps closed: `4`.
- Projected Irvine profile gaps closed: `0`.
- Decision: keep the rewrite candidate, but it is not sufficient alone to
  prove absolute fastest on Irvine M1. The next strict win still needs
  either quiet Irvine timing plus another source/body improvement, or a
  stronger single-package graph change.

