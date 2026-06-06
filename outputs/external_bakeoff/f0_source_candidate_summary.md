# F0 Source Candidate Ranking

Sorted by warm median speedup versus the strict-equivalent baseline stack.
Strict waveform failures are not production approvals; rows with blank
`Human` still need no-ASR listening decisions.

| Rank | Machine | Bucket | Candidate | Speedup | Baseline ms | Candidate ms | Corr | SNR dB | Strict | Human | Notes |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1 | m2-studio | 30s | `30s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | 28.9% | 268.9 | 191.3 | 0.794801 | 4.78 | fail | blank |  |
| 2 | m2-studio | 30s | `30s_padded_cos_resblock_phase_acos_cos_rsqrt` | 25.0% | 256.7 | 192.6 | 0.964458 | 11.99 | fail | blank |  |
| 3 | m2-studio | 15s | `15s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | 21.9% | 129.9 | 101.4 | 0.838603 | 5.73 | fail | blank |  |
| 4 | m2-studio | 30s | `30s_padded_cos_resblock_cos_rsqrt` | 21.7% | 269.9 | 211.4 | 0.949790 | 10.40 | fail | blank |  |
| 5 | m2-studio | 15s | `15s_padded_cos_resblock_phase_acos_cos_rsqrt` | 16.9% | 122.7 | 102.0 | 0.969391 | 12.60 | fail | blank |  |
| 6 | m2-studio | 15s | `15s_padded_cos_resblock_cos_rsqrt` | 16.0% | 130.0 | 109.2 | 0.956701 | 10.99 | fail | blank |  |
| 7 | m2-air | 3s | `3s_natural_asr_cos_rsqrt` | 13.9% | 123.9 | 106.7 | 0.813986 | 5.08 | fail | blank |  |
| 8 | irvine-m1 | 10s | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | 13.6% | 563.9 | 487.1 | 0.867049 | 6.55 | fail | blank |  |
| 9 | irvine-m1 | 7s | `7s_natural_asr_cos_rsqrt` | 12.2% | 398.4 | 349.8 | 0.796785 | 4.77 | fail | blank |  |
| 10 | m2-studio | 10s | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | 11.2% | 86.0 | 76.4 | 0.866976 | 6.55 | fail | blank |  |
| 11 | irvine-m1 | 3s | `3s_natural_asr_cos_rsqrt` | 10.9% | 172.0 | 153.3 | 0.813995 | 5.08 | fail | blank |  |
| 12 | m2-studio | 7s | `7s_natural_asr_cos_rsqrt` | 10.4% | 63.1 | 56.5 | 0.796791 | 4.77 | fail | blank |  |
| 13 | irvine-m1 | 10s | `10s_padded_cos_resblock_cos_rsqrt` | 10.0% | 565.7 | 509.0 | 0.955223 | 10.87 | fail | blank |  |
| 14 | irvine-m1 | 10s | `10s_padded_cos_resblock_phase_acos_cos_rsqrt` | 9.9% | 562.9 | 506.9 | 0.968961 | 12.58 | fail | blank |  |
| 15 | m2-studio | 10s | `10s_padded_cos_resblock_phase_acos_cos_rsqrt` | 9.8% | 81.0 | 73.1 | 0.968904 | 12.57 | fail | blank |  |
| 16 | m2-studio | 10s | `10s_padded_cos_resblock_cos_rsqrt` | 9.7% | 87.6 | 79.0 | 0.955085 | 10.86 | fail | blank |  |
| 17 | irvine-m1 | 15s | `15s_padded_cos_resblock_cos_rsqrt` | 9.4% | 837.7 | 758.8 | 0.956796 | 11.00 | fail | blank |  |
| 18 | m2-air | 3s | `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | 8.2% | 123.5 | 113.4 | 0.931815 | 9.19 | fail | blank |  |
| 19 | irvine-m1 | 7s | `7s_cos_rsqrt` | 8.2% | 390.8 | 358.9 | 0.962306 | 11.52 | fail | blank |  |
| 20 | irvine-m1 | 3s | `3s_padded_native_in_nopal_cos_rsqrt_native_in` | 7.3% | 174.5 | 161.7 | 0.931801 | 9.19 | fail | blank |  |
| 21 | m2-studio | 7s | `7s_padded_cos_resblock_cos_rsqrt` | 6.9% | 61.3 | 57.0 | 0.962251 | 11.51 | fail | blank |  |
| 22 | m2-studio | 3s | `3s_padded_swift_like_atan2_cos_resblock_cos_rsqrt_swift_like` | 6.7% | 31.2 | 29.1 | 0.215760 | 0.34 | fail | blank |  |
| 23 | m2-studio | 7s | `7s_cos_rsqrt` | 6.7% | 63.0 | 58.8 | 0.962251 | 11.51 | fail | blank |  |
| 24 | irvine-m1 | 3s | `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | 6.3% | 173.0 | 162.1 | 0.931840 | 9.19 | fail | blank |  |
| 25 | m2-studio | 7s | `7s_padded_cos_resblock_phase_acos_cos_rsqrt` | 6.2% | 63.8 | 59.9 | 0.972150 | 13.04 | fail | blank |  |
| 26 | irvine-m1 | 3s | `3s_padded_native_in_fp16_pal_v2_cos_rsqrt_body_pal_native_in` | 2.4% | 174.2 | 169.9 | 0.931617 | 9.17 | fail | blank |  |
| 27 | m2-studio | 3s | `3s_natural_asr_cos_rsqrt` | 2.2% | 33.4 | 32.7 | 0.814034 | 5.08 | fail | blank |  |
| 28 | m2-studio | 3s | `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | 1.8% | 33.0 | 32.5 | 0.931854 | 9.19 | fail | blank |  |
| 29 | m2-studio | 3s | `3s_padded_cos_resblock_phase_swift_cos_rsqrt` | 1.5% | 31.4 | 30.9 | 0.915815 | 7.44 | fail | blank |  |
| 30 | m2-studio | 3s | `3s_padded_cos_resblock_cos_rsqrt` | 0.8% | 30.9 | 30.6 | 0.931895 | 9.19 | fail | blank |  |
| 31 | m2-studio | 3s | `3s_padded_cos_resblock_phase_acos_cos_rsqrt` | 0.8% | 32.8 | 32.5 | 0.949566 | 10.34 | fail | blank |  |
| 32 | m2-studio | 3s | `3s_padded_swift_like_phase_swift_cos_resblock_cos_rsqrt_swift_like` | 0.3% | 32.1 | 32.0 | 0.186514 | 0.32 | fail | blank |  |
| 33 | m2-studio | 3s | `3s_cos_rsqrt` | -0.5% | 33.5 | 33.7 | 0.931895 | 9.19 | fail | blank |  |
| 34 | m2-studio | 3s | `3s_padded_cos_resblock_phase_manual_cos_rsqrt` | -1.1% | 30.9 | 31.3 | 0.938613 | 9.47 | fail | blank |  |
| 35 | m2-studio | 3s | `3s_padded_cos_resblock_torchref_cos_rsqrt` | -3.5% | 31.0 | 32.1 | 0.931895 | 9.19 | fail | blank |  |
| 36 | m2-studio | 3s | `3s_padded_native_in_nopal_cos_rsqrt_native_in` | -5.7% | 33.1 | 35.0 | 0.931816 | 9.19 | fail | blank |  |
| 37 | m2-studio | 3s | `3s_padded_swift_like_native_in_cos_rsqrt_native_in_swift_like` | -9.3% | 31.8 | 34.7 | 0.215731 | 0.34 | fail | blank |  |
| 38 | m2-studio | 3s | `3s_padded_native_in_fp16_pal_v2_cos_rsqrt_body_pal_native_in` | -15.8% | 31.9 | 36.9 | 0.931531 | 9.17 | fail | blank |  |
| 39 | m2-studio | 3s | `3s_padded_native_in_fp16_pal_cos_rsqrt_body_pal_native_in` | -18.2% | 32.4 | 38.3 | 0.931636 | 9.17 | fail | blank |  |
| 40 | m2-studio | 3s | `3s_padded_body_fp16_inputs` | -549.2% | 34.4 | 223.1 | 0.931413 | 9.17 | fail | blank |  |
| 41 | m2-studio | 3s | `3s_padded_body_fp16_inputs_pal_body_pal` | -582.4% | 32.7 | 223.1 | 0.930939 | 9.14 | fail | blank |  |
| 42 | m2-studio | 3s | `3s_ios17_body_fp16_inputs_pal_body_pal` | -674.3% | 34.3 | 265.9 | 0.930895 | 9.13 | fail | blank |  |
