# Irvine 3s Residual

Warmed profile denominator. Additive rows are estimates from saved
sub-stack probes, not full-path promotion proof.

Config F: `233.6 ms`.
laishere warmed profile: `195.0 ms`.
Profile gap: `38.5 ms`.
Matched source/body gap: `22.0 ms`.
Matched upstream/runtime gap: `12.9 ms`.

## Residual Budget

| Scenario | Remaining gap vs warmed laishere |
| --- | ---: |
| Best quality-fail signal `3s_natural_asr_cos_rsqrt` | 19.8 ms |
| Best quality-fail + best strict signal `3s_har28561` | 19.1 ms |
| Best quality-fail + eliminate matched upstream/runtime gap | 6.9 ms |
| Best quality-fail + best strict + eliminate upstream/runtime gap | 6.2 ms |

## Positive Quality-Fail Signals

| Candidate | Delta | Corr | SNR dB | Report |
| --- | ---: | ---: | ---: | --- |
| `3s_natural_asr_cos_rsqrt` | 18.7 ms | 0.813995 | 5.08 | `outputs/f0_noise_exact_shape/remote_reports/report_f0_noise_exact_3s_irvine.json` |
| `3s_padded_native_in_nopal_cos_rsqrt_native_in` | 12.8 ms | 0.931801 | 9.19 | `outputs/f0_noise_exact_shape/3s_padded_native_in_nopal_cos_rsqrt_native_in/report_native_in_nopal_irvine.json` |
| `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | 10.9 ms | 0.931840 | 9.19 | `outputs/f0_noise_exact_shape/remote_reports/report_native_in_ios17_nopal_irvine.json` |
| `3s_padded_native_in_fp16_pal_v2_cos_rsqrt_body_pal_native_in` | 4.2 ms | 0.931617 | 9.17 | `outputs/f0_noise_exact_shape/3s_padded_native_in_fp16_pal_v2_cos_rsqrt_body_pal_native_in/report_native_in_fp16_pal_v2_irvine.json` |

## Interpretation

Saved 3s signals do not close warmed laishere. Even after the best quality-fail F0/source branch, the estimated residual is material; removing the matched upstream gap still leaves a positive residual.
