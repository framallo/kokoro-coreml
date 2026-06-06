# Strict Win Budget After Rewrite

This table starts from the measured HAR-post upsample rewrite candidate and
asks what additional strict speed is still required on Irvine M1. It uses
warmed profile medians only. The rewrite itself is still a projection for
Irvine until the host is quiet enough for publishable timing.

## Profile Target

Profile target means beating the newer warmed laishere stage-profile row.

| Bucket | Projected Config F | laishere profile | Extra save needed | Extra generator speedup needed |
| --- | ---: | ---: | ---: | ---: |
| `3s` | 226.4 ms | 195.0 ms | 31.4 ms | 19.61% |
| `7s` | 480.6 ms | 444.2 ms | 36.4 ms | 9.79% |
| `10s` | 668.0 ms | 644.9 ms | 23.2 ms | 4.36% |
| `15s` | 993.6 ms | 990.6 ms | 3.0 ms | 0.37% |

## Paper Frontier Target

Paper frontier target means beating the current strict paper-facing row,
which may be stricter than the newer warmed profile row.

| Bucket | Paper frontier best | Extra save needed | Extra generator speedup needed |
| --- | ---: | ---: | ---: |
| `3s` | 176.3 ms | 50.1 ms | 31.30% |
| `7s` | 394.6 ms | 86.0 ms | 23.15% |
| `10s` | 593.9 ms | 74.2 ms | 13.96% |
| `15s` | 912.0 ms | 81.6 ms | 10.20% |

## Decision

- Irvine profile rows remaining after rewrite projection: `4`.
- Irvine paper rows remaining after rewrite projection: `4`.
- The next strict candidate must be much larger than another 1-3% local
  generator tweak unless it targets only the nearly closed `15s` row.
- For `3s/7s/10s`, the remaining profile target needs roughly
  `4-20%` additional generator-stage improvement after the rewrite.

