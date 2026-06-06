# Irvine Paper Frontier Path

Warmed inference only. This report combines the best saved Irvine
source/body candidate per bucket with the measured HAR-post rewrite
projection. It uses the stricter paper-facing `competitive_frontier`
rows, not only newer stage-profile rows.

Paper rows closed by source+rewrite: `1`.
Paper rows still open after source+rewrite: `3`.

| Bucket | Candidate | Paper frontier | Source projected | Rewrite save | Combined projected | Combined margin | Extra save needed | Quality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `10s` | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | laishere: 593.9 ms | 608.7 ms | 17.4 ms | 591.2 ms | +2.6 ms | 0.0 ms | corr 0.867, SNR 6.55 dB |
| `15s` | `15s_padded_cos_resblock_cos_rsqrt` | laishere: 912.0 ms | 936.1 ms | 21.4 ms | 914.7 ms | -2.7 ms | 2.7 ms | corr 0.957, SNR 11.00 dB |
| `3s` | `3s_natural_asr_cos_rsqrt` | laishere: 176.3 ms | 214.8 ms | 7.2 ms | 207.7 ms | -31.4 ms | 31.4 ms | corr 0.814, SNR 5.08 dB |
| `7s` | `7s_natural_asr_cos_rsqrt` | laishere: 394.6 ms | 444.1 ms | 12.1 ms | 432.0 ms | -37.4 ms | 37.4 ms | corr 0.797, SNR 4.77 dB |

## Decision

On Irvine M1, source/body plus the HAR-post rewrite is enough for the 10s paper row if listening accepts the source candidate. The 15s row is close, while 3s and 7s still need larger implementation wins than the current saved probes provide.

## Immediate Work

- Promote the HAR-post rewrite into any publishable Irvine source/body rerun.
- If listening accepts `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt`, rerun Irvine `10s` on a quiet host because source+rewrite should beat the paper row.
- Find at least another `2.7 ms` for Irvine `15s` after source+rewrite, or prove a quieter rerun changes that margin.
- Treat Irvine `3s` and `7s` as unsolved implementation work; current source+rewrite projections remain far short of the paper frontier.
