# Irvine Paper Frontier Path

Warmed inference only. This report combines the best saved Irvine
source/body candidate per bucket with the measured HAR-post rewrite
projection. It uses the stricter paper-facing `competitive_frontier`
rows, not only newer stage-profile rows.

Paper rows closed by independent source+rewrite projection: `1`.
Paper rows still open after independent projection: `3`.
Paper rows closed by direct measured source/body rewrite: `0`.
Rows with direct source/body rewrite measurement: `4`.

| Bucket | Candidate | Human | Paper frontier | Source projected | Independent rewrite save | Independent projected | Independent margin | Direct source rewrite | Direct margin | Quality |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `10s` | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | pass | laishere: 593.9 ms | 608.7 ms | 17.4 ms | 591.2 ms | +2.6 ms | 4.5 ms local save | -10.3 ms | corr 0.867, SNR 6.55 dB |
| `15s` | `15s_padded_cos_resblock_cos_rsqrt` | pass | laishere: 912.0 ms | 936.1 ms | 21.4 ms | 914.7 ms | -2.7 ms | 3.6 ms local save | -20.5 ms | corr 0.957, SNR 11.00 dB |
| `3s` | `3s_natural_asr_cos_rsqrt` | pass | laishere: 176.3 ms | 214.8 ms | 7.2 ms | 207.7 ms | -31.4 ms | 3.3 ms local save | -35.2 ms | corr 0.814, SNR 5.08 dB |
| `7s` | `7s_natural_asr_cos_rsqrt` | pass | laishere: 394.6 ms | 444.1 ms | 12.1 ms | 432.0 ms | -37.4 ms | 1.7 ms local save | -47.8 ms | corr 0.797, SNR 4.77 dB |

## Decision

The independent source/body plus production-rewrite projection is optimistic and must not be treated as a direct combined measurement. Direct local source/body+upsample-rewrite probes on all accepted buckets are speed-positive, but much smaller than the production-rewrite projection; Irvine paper rows still require another implementation win.

## Immediate Work

- Do not publish the independent source+rewrite projection as a win.
- Do not promote direct source/body+upsample-rewrite for Irvine; local direct saves are too small on all accepted buckets.
- Find another source/body implementation win of about `10 ms` for `10s`, `20 ms` for `15s`, and much larger wins for `3s`/`7s`, or prove a better direct stack on quiet Irvine.
- Treat Irvine `3s` and `7s` as unsolved implementation work; current saved source candidates remain far short of the paper frontier.
