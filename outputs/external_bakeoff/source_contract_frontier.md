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
