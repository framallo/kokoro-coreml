# F0 Source Variant Summary

Cheap PyTorch-only source formulation probe across runtime buckets.
This summary exists to prevent re-exporting Core ML packages for source
equations that already fail before conversion.

Rows: `5`.
Source equation solved: `true`.
Recomputed HAR/STFT solved: `false`.

| Bucket | Swift-like source corr | Swift-like source SNR | Best simplified source | Simplified corr | Simplified SNR | Dump source -> HAR corr | Dump source -> HAR SNR | HAR max abs |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `10s` | 1.000000 | 139.89 | `probe_avg_pool_noise_0` | 0.956636 | 12.06 | 0.924214 | 8.23 | 6.28 |
| `15s` | 1.000000 | 140.04 | `linear_interp_noise_0` | 0.965101 | 12.98 | 0.924284 | 8.23 | 6.28 |
| `30s` | 1.000000 | 140.33 | `linear_interp_noise_0` | 0.963368 | 12.77 | 0.923606 | 8.19 | 6.28 |
| `3s` | 1.000000 | 138.15 | `linear_interp_noise_0` | 0.939782 | 10.83 | 0.922796 | 8.15 | 6.28 |
| `7s` | 1.000000 | 139.65 | `linear_interp_noise_0` | 0.967318 | 13.26 | 0.922073 | 8.11 | 6.28 |

## Interpretation

Swift-like seeded source generation matches dumped har_source across buckets, but recomputing HAR/STFT from even the dumped source stays near 8 dB SNR with 2*pi phase-wrap max errors. The remaining quality blocker is the HAR/STFT representation contract, not the sine source equation.
