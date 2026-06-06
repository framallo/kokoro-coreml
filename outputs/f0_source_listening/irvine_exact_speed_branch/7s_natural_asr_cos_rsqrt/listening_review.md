# F0 Source Listening Review: 7s_natural_asr_cos_rsqrt

This review intentionally does not use ASR or Whisper. It is a waveform-health and human-listening artifact.

## Files

- Swift dump reference: `outputs/f0_source_listening/irvine_exact_speed_branch/7s_natural_asr_cos_rsqrt/wav/7s_natural_asr_cos_rsqrt_swift_dump.wav`
- Baseline Core ML path: `outputs/f0_source_listening/irvine_exact_speed_branch/7s_natural_asr_cos_rsqrt/wav/7s_natural_asr_cos_rsqrt_baseline.wav`
- F0-source candidate: `outputs/f0_source_listening/irvine_exact_speed_branch/7s_natural_asr_cos_rsqrt/wav/7s_natural_asr_cos_rsqrt_candidate.wav`
- Objective quality report: `outputs/f0_source_listening/irvine_exact_speed_branch/7s_natural_asr_cos_rsqrt/quality/audio_quality_report.json`

## Waveform Metrics

| Comparison | Metrics |
| --- | --- |
| Baseline vs Swift dump | corr 1.000000, SNR 80.90 dB, max 0.00060 |
| Candidate vs Swift dump | corr 0.796791, SNR 4.77 dB, max 0.36303 |
| Candidate vs baseline | corr 0.796791, SNR 4.77 dB, max 0.36303 |

## Machine Gate

- Decision: `needs_listening`
- Reject reasons: `-`

## Listening Decision

- [ ] Candidate sounds acceptable versus the baseline.
- [ ] Candidate has unacceptable artifacts.
- [ ] Unsure; needs more samples.

Notes:

