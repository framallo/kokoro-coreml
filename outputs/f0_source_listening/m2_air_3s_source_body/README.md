# F0 Source Listening Packs

No ASR/Whisper gate is used here. These artifacts support human listening after objective waveform-health checks.

Fill decisions in `outputs/f0_source_listening/m2_air_3s_source_body/f0_source_listening_decisions.csv` with `pass`, `caveat`, or `fail`.
`caveat` requires notes. Validate with `python scripts/validate_f0_source_listening_decisions.py --decisions <csv>`.

| Label | Waveform gate | Candidate WAV | Review |
| --- | --- | --- | --- |
| `3s_natural_asr_cos_rsqrt` | `needs_listening` | `outputs/f0_source_listening/m2_air_3s_source_body/3s_natural_asr_cos_rsqrt/wav/3s_natural_asr_cos_rsqrt_candidate.wav` | `outputs/f0_source_listening/m2_air_3s_source_body/3s_natural_asr_cos_rsqrt/listening_review.md` |
| `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | `needs_listening` | `outputs/f0_source_listening/m2_air_3s_source_body/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/wav/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in_candidate.wav` | `outputs/f0_source_listening/m2_air_3s_source_body/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/listening_review.md` |
