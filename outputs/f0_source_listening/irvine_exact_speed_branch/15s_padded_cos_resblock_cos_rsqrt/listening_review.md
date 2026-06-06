# F0 Source Listening Review: 15s_padded_cos_resblock_cos_rsqrt

This review intentionally does not use ASR or Whisper. It is a waveform-health and human-listening artifact.

## Files

- Swift dump reference: `outputs/f0_source_listening/irvine_exact_speed_branch/15s_padded_cos_resblock_cos_rsqrt/wav/15s_padded_cos_resblock_cos_rsqrt_swift_dump.wav`
- Baseline Core ML path: `outputs/f0_source_listening/irvine_exact_speed_branch/15s_padded_cos_resblock_cos_rsqrt/wav/15s_padded_cos_resblock_cos_rsqrt_baseline.wav`
- F0-source candidate: `outputs/f0_source_listening/irvine_exact_speed_branch/15s_padded_cos_resblock_cos_rsqrt/wav/15s_padded_cos_resblock_cos_rsqrt_candidate.wav`
- Objective quality report: `outputs/f0_source_listening/irvine_exact_speed_branch/15s_padded_cos_resblock_cos_rsqrt/quality/audio_quality_report.json`

## Waveform Metrics

| Comparison | Metrics |
| --- | --- |
| Baseline vs Swift dump | corr 0.983647, SNR 15.28 dB, max 0.35791 |
| Candidate vs Swift dump | corr 0.939319, SNR 9.75 dB, max 0.31872 |
| Candidate vs baseline | corr 0.956701, SNR 10.99 dB, max 0.31872 |

## Machine Gate

- Decision: `needs_listening`
- Reject reasons: `-`

## Listening Decision

- [ ] Candidate sounds acceptable versus the baseline.
- [ ] Candidate has unacceptable artifacts.
- [ ] Unsure; needs more samples.

Notes:

