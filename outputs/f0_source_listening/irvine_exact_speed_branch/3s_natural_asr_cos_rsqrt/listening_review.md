# F0 Source Listening Review: 3s_natural_asr_cos_rsqrt

This review intentionally does not use ASR or Whisper. It is a waveform-health and human-listening artifact.

## Files

- Swift dump reference: `outputs/f0_source_listening/irvine_exact_speed_branch/3s_natural_asr_cos_rsqrt/wav/3s_natural_asr_cos_rsqrt_swift_dump.wav`
- Baseline Core ML path: `outputs/f0_source_listening/irvine_exact_speed_branch/3s_natural_asr_cos_rsqrt/wav/3s_natural_asr_cos_rsqrt_baseline.wav`
- F0-source candidate: `outputs/f0_source_listening/irvine_exact_speed_branch/3s_natural_asr_cos_rsqrt/wav/3s_natural_asr_cos_rsqrt_candidate.wav`
- Objective quality report: `outputs/f0_source_listening/irvine_exact_speed_branch/3s_natural_asr_cos_rsqrt/quality/audio_quality_report.json`

## Waveform Metrics

| Comparison | Metrics |
| --- | --- |
| Baseline vs Swift dump | corr 0.999996, SNR 51.60 dB, max 0.01279 |
| Candidate vs Swift dump | corr 0.814046, SNR 5.08 dB, max 0.43998 |
| Candidate vs baseline | corr 0.814034, SNR 5.08 dB, max 0.43998 |

## Machine Gate

- Decision: `needs_listening`
- Reject reasons: `-`

## Listening Decision

- [ ] Candidate sounds acceptable versus the baseline.
- [ ] Candidate has unacceptable artifacts.
- [ ] Unsure; needs more samples.

Notes:

