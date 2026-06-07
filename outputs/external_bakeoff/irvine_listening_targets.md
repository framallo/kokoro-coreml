# Irvine Listening Targets

No ASR/Whisper gate is used. Remote Irvine speed rows are mapped to same-label local WAV artifacts when exact remote-report WAVs are not present.

Rows: `4`.
Rows with a no-ASR listening artifact: `4`.
Rows where the listening artifact uses the exact Irvine timing report: `4`.

| Bucket | Candidate | Irvine speedup | Waveform | Listening artifact | Exact timing report? | Human |
| --- | --- | ---: | --- | --- | --- | --- |
| 3s | `3s_natural_asr_cos_rsqrt` | 10.9% (172.0 -> 153.3 ms) | corr 0.813995, SNR 5.08 dB; gate `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/3s_natural_asr_cos_rsqrt/wav/3s_natural_asr_cos_rsqrt_candidate.wav` | yes | pass |
| 7s | `7s_natural_asr_cos_rsqrt` | 12.2% (398.4 -> 349.8 ms) | corr 0.796785, SNR 4.77 dB; gate `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/7s_natural_asr_cos_rsqrt/wav/7s_natural_asr_cos_rsqrt_candidate.wav` | yes | pass |
| 10s | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | 13.6% (563.9 -> 487.1 ms) | corr 0.867049, SNR 6.55 dB; gate `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/wav/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt_candidate.wav` | yes | pass |
| 15s | `15s_padded_cos_resblock_cos_rsqrt` | 9.4% (837.7 -> 758.8 ms) | corr 0.956796, SNR 11.00 dB; gate `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/15s_padded_cos_resblock_cos_rsqrt/wav/15s_padded_cos_resblock_cos_rsqrt_candidate.wav` | yes | pass |
