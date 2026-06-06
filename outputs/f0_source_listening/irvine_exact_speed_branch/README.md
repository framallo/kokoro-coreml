# F0 Source Listening Packs

No ASR/Whisper gate is used here. These artifacts support human listening after objective waveform-health checks.

Fill decisions in `outputs/f0_source_listening/irvine_exact_speed_branch/f0_source_listening_decisions.csv` with `pass`, `caveat`, or `fail`.
`caveat` requires notes. Validate with `python scripts/validate_f0_source_listening_decisions.py --decisions <csv>`.

Open `outputs/f0_source_listening/irvine_exact_speed_branch/review.html` in a
browser for baseline/candidate audio controls in one table.

| Label | Waveform gate | Candidate WAV | Review |
| --- | --- | --- | --- |
| `3s_natural_asr_cos_rsqrt` | `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/3s_natural_asr_cos_rsqrt/wav/3s_natural_asr_cos_rsqrt_candidate.wav` | `outputs/f0_source_listening/irvine_exact_speed_branch/3s_natural_asr_cos_rsqrt/listening_review.md` |
| `7s_natural_asr_cos_rsqrt` | `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/7s_natural_asr_cos_rsqrt/wav/7s_natural_asr_cos_rsqrt_candidate.wav` | `outputs/f0_source_listening/irvine_exact_speed_branch/7s_natural_asr_cos_rsqrt/listening_review.md` |
| `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/wav/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt_candidate.wav` | `outputs/f0_source_listening/irvine_exact_speed_branch/10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt/listening_review.md` |
| `15s_padded_cos_resblock_cos_rsqrt` | `needs_listening` | `outputs/f0_source_listening/irvine_exact_speed_branch/15s_padded_cos_resblock_cos_rsqrt/wav/15s_padded_cos_resblock_cos_rsqrt_candidate.wav` | `outputs/f0_source_listening/irvine_exact_speed_branch/15s_padded_cos_resblock_cos_rsqrt/listening_review.md` |
