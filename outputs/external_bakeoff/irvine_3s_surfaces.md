# Irvine 3s Graph Surfaces

Saved Irvine 3s reports only. This scanner understands split/fused
benchmark keys that the generic optimization scanner does not classify.

Rows classified: `19`.
Strict-pass positive rows: `2`.
Quality-fail positive rows: `4`.
Best strict-pass row: `3s_har28561`.
Best quality-fail row: `3s_natural_asr_cos_rsqrt`.

| Quality | Family | Candidate | Speedup | Delta | Baseline | Candidate | Corr | SNR dB | Report |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| quality-fail | f0_noise_exact_shape | `3s_natural_asr_cos_rsqrt` | 10.88% | 18.7 ms | 172.0 ms | 153.3 ms | 0.813995 | 5.08 | `outputs/f0_noise_exact_shape/remote_reports/report_f0_noise_exact_3s_irvine.json` |
| quality-fail | f0_noise_exact_shape | `3s_padded_native_in_nopal_cos_rsqrt_native_in` | 7.35% | 12.8 ms | 174.5 ms | 161.7 ms | 0.931801 | 9.19 | `outputs/f0_noise_exact_shape/3s_padded_native_in_nopal_cos_rsqrt_native_in/report_native_in_nopal_irvine.json` |
| quality-fail | f0_noise_exact_shape | `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | 6.29% | 10.9 ms | 173.0 ms | 162.1 ms | 0.931840 | 9.19 | `outputs/f0_noise_exact_shape/remote_reports/report_native_in_ios17_nopal_irvine.json` |
| quality-fail | f0_noise_exact_shape | `3s_padded_native_in_fp16_pal_v2_cos_rsqrt_body_pal_native_in` | 2.44% | 4.2 ms | 174.2 ms | 169.9 ms | 0.931617 | 9.17 | `outputs/f0_noise_exact_shape/3s_padded_native_in_fp16_pal_v2_cos_rsqrt_body_pal_native_in/report_native_in_fp16_pal_v2_irvine.json` |
| strict-pass | generator_har_input_trim | `3s_har28561` | 0.43% | 0.7 ms | 168.4 ms | 167.6 ms | 0.999984 | 45.28 | `outputs/generator_har_input_trim/remote_reports/report_har28561_3s_irvine.json` |
| strict-pass | generator_cos_snake | `3s_broadcast_adain_native_in_ios17` | 0.14% | 0.2 ms | 167.8 ms | 167.6 ms | 0.999994 | 50.00 | `outputs/generator_cos_snake/3s_broadcast_adain_native_in_ios17/report_irvine_ios17_native_broadcast_cos.json` |
| strict-pass | generator_style_specialization | `3s` | -1.79% | -3.0 ms | 167.8 ms | 170.8 ms | n/a | n/a | `outputs/generator_style_specialization/3s/report_irvine.json` |
| strict-pass | generator_noise_split | `3s_native_in_broadcast_ios17` | -6.85% | -11.5 ms | 168.3 ms | 179.8 ms | 0.999995 | 50.29 | `outputs/generator_noise_split/3s_native_in_broadcast_ios17/irvine/report_native_in_ios17_cpu_gpu_irvine.json` |
| strict-pass | generator_stage_split | `3s` | -9.23% | -15.5 ms | 168.4 ms | 184.0 ms | 0.999997 | 52.62 | `outputs/generator_stage_split/3s/report_irvine_cpugpu.json` |
| strict-pass | har_source_fused | `3s_atan_manual_fp32_nyquist_padded_dual_anchor` | -13.59% | -22.9 ms | 168.2 ms | 191.0 ms | 0.999995 | 50.00 | `outputs/har_source_fused/3s_atan_manual_fp32_nyquist_padded_dual_anchor/irvine/report_irvine_dual_anchor_cpu_gpu.json` |
| strict-pass | decoder_vocoder_split | `3s_cos_rsqrt` | -14.19% | -24.8 ms | 174.6 ms | 199.3 ms | 0.999991 | 47.72 | `outputs/decoder_vocoder_split/remote_reports/report_laishere_boundary_3s_irvine_body_cpu_gpu.json` |
| strict-pass | decoder_vocoder_split | `3s_har_cos_rsqrt_native_in_broadcast_ios17` | -16.69% | -29.3 ms | 175.4 ms | 204.7 ms | 0.999991 | 47.76 | `outputs/decoder_vocoder_split/3s_har_cos_rsqrt_native_in_broadcast_ios17/irvine/report_ios17_native_broadcast_cpu_gpu_irvine.json` |
| quality-fail | generator_stage_split | `3s` | -17.82% | -29.9 ms | 167.5 ms | 197.4 ms | 0.403829 | 0.47 | `outputs/generator_stage_split/3s/report_irvine_stage0_cpune.json` |
| strict-pass | decoder_vocoder_split | `3s_har_cos_rsqrt_native_in_broadcast_ios17` | -66.42% | -116.3 ms | 175.1 ms | 291.3 ms | 0.999910 | 37.82 | `outputs/decoder_vocoder_split/3s_har_cos_rsqrt_native_in_broadcast_ios17/irvine/report_ios17_native_broadcast_cpu_ne_irvine.json` |
| quality-fail | generator_noise_split | `3s_native_in_broadcast_ios17` | -70.76% | -121.9 ms | 172.2 ms | 294.1 ms | 0.077158 | 0.01 | `outputs/generator_noise_split/3s_native_in_broadcast_ios17/irvine/report_native_in_ios17_body_cpu_ne_irvine.json` |
| strict-pass | decoder_vocoder_split | `3s_cos_rsqrt` | -78.45% | -138.3 ms | 176.3 ms | 314.6 ms | 0.999907 | 37.69 | `outputs/decoder_vocoder_split/remote_reports/report_laishere_boundary_3s_irvine_body_cpu_ne.json` |
| strict-pass | generator_cos_snake | `3s_broadcast_adain_native_in_ios17` | -79.73% | -141.2 ms | 177.1 ms | 318.2 ms | 0.999965 | 41.91 | `outputs/generator_cos_snake/3s_broadcast_adain_native_in_ios17/report_irvine_all.json` |
| quality-fail | generator_stage_split | `3s` | -82.94% | -143.2 ms | 172.7 ms | 315.8 ms | 0.121443 | 0.05 | `outputs/generator_stage_split/3s/report_irvine_stage1_cpune.json` |
| strict-pass | har_source_fused | `3s_atan_manual_fp32_nyquist_padded_dual_anchor` | -95.42% | -163.8 ms | 171.7 ms | 335.5 ms | 0.999995 | 50.00 | `outputs/har_source_fused/3s_atan_manual_fp32_nyquist_padded_dual_anchor/irvine/report_irvine_dual_anchor_body_cpu_ne.json` |
