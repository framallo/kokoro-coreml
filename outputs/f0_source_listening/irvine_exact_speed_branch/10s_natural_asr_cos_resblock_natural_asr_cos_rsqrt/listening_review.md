# F0 Source Listening Review: 10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt

This review intentionally does not use ASR or Whisper. It is a waveform-health and human-listening artifact.

## Files

- Swift dump reference: `outputs/f0_source_listening/irvine_exact_speed_branch/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/wav/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt_swift_dump.wav`
- Baseline Core ML path: `outputs/f0_source_listening/irvine_exact_speed_branch/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/wav/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt_baseline.wav`
- F0-source candidate: `outputs/f0_source_listening/irvine_exact_speed_branch/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/wav/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt_candidate.wav`
- Objective quality report: `outputs/f0_source_listening/irvine_exact_speed_branch/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/quality/audio_quality_report.json`

## Waveform Metrics

| Comparison | Metrics |
| --- | --- |
| Baseline vs Swift dump | corr 0.978630, SNR 14.10 dB, max 0.25293 |
| Candidate vs Swift dump | corr 0.849124, SNR 6.00 dB, max 0.40681 |
| Candidate vs baseline | corr 0.866976, SNR 6.55 dB, max 0.40681 |

## Machine Gate

- Decision: `needs_listening`
- Reject reasons: `-`

## Listening Decision

- [ ] Candidate sounds acceptable versus the baseline.
- [ ] Candidate has unacceptable artifacts.
- [ ] Unsure; needs more samples.

Notes:

