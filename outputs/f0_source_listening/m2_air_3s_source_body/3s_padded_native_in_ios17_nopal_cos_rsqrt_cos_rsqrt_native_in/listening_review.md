# F0 Source Listening Review: 3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in

This review intentionally does not use ASR or Whisper. It is a waveform-health and human-listening artifact.

## Files

- Swift dump reference: `outputs/f0_source_listening/m2_air_3s_source_body/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/wav/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in_swift_dump.wav`
- Baseline Core ML path: `outputs/f0_source_listening/m2_air_3s_source_body/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/wav/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in_baseline.wav`
- F0-source candidate: `outputs/f0_source_listening/m2_air_3s_source_body/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/wav/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in_candidate.wav`
- Objective quality report: `outputs/f0_source_listening/m2_air_3s_source_body/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/quality/audio_quality_report.json`

## Waveform Metrics

| Comparison | Metrics |
| --- | --- |
| Baseline vs Swift dump | corr 0.999996, SNR 51.60 dB, max 0.01279 |
| Candidate vs Swift dump | corr 0.931855, SNR 9.19 dB, max 0.23769 |
| Candidate vs baseline | corr 0.931854, SNR 9.19 dB, max 0.23769 |

## Machine Gate

- Decision: `needs_listening`
- Reject reasons: `-`

## Listening Decision

- [ ] Candidate sounds acceptable versus the baseline.
- [ ] Candidate has unacceptable artifacts.
- [ ] Unsure; needs more samples.

Notes:

