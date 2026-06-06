# Lower-End Mac Win Attempts

This note records lower-end Mac promotion attempts that are useful evidence but
not publishable frontier rows unless the quiet gate passes. All timing is warmed
inside the runner; rows marked non-publishable were collected while the remote
host failed `scripts/external_bakeoff/check_remote_host_quiet.py`.

## 2026-06-06 M2 Air HAR-Post Rewrite Overlay Smoke

Remote overlay setup:

- Synced rewritten `kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage`
  packages to `m2-air`.
- Created `outputs/export_rewrite_smoke/coreml_overlay` on `m2-air` with
  symlinks to the remote `coreml/` tree and rewritten HAR-post packages.

Quiet gate:

- Latest M2 Air-only quiet check:
  `outputs/external_bakeoff/remote_host_quiet_m2_air_latest.md`.
- Result: not publishable; load1 remained above `1.00`.

Smoke results:

| Bucket | Rewrite overlay smoke | Current Config F | laishere paper row | Status |
| --- | ---: | ---: | ---: | --- |
| `3s` | `145.8 ms` | `148.0 ms` | `142.0 ms` | improved Config F, still short |
| `7s` | `358.9 ms` | `330.7 ms` | `316.9 ms` | polluted / do not promote |
| `10s` | `714.8 ms` | `466.0 ms` | `450.2 ms` | polluted / do not promote |
| `15s` | `1011.3 ms` | `693.6 ms` | `657.3 ms` | polluted / do not promote |

Raw artifacts:

- `outputs/external_bakeoff/results_config_f_reference_m2-air_rewrite_overlay_smoke.json`
- `outputs/external_bakeoff/results_config_f_reference_m2-air_rewrite_overlay_smoke_shortmid.json`

Decision:

- The overlay is valid and remotely runnable.
- `3s` moved in the expected direction even under load, but still needs another
  small strict save to beat the paper-facing `142.0 ms` laishere row.
- `7s/10s/15s` were collected during active host load and are failure evidence
  for the timing environment, not candidate performance.
- Do not update `competitive_frontier.md` from these smoke rows.

## 2026-06-06 HnSF Per-Harmonic Merge Smoke

Candidate:

- Replaced the frame-based HnSF path's full noisy harmonic matrix +
  `vDSP_mmul` merge with per-harmonic mask/noise/weight accumulation.
- Goal was to reduce host-side memory traffic on lower-end Macs.
- The edit preserved harmonic-source test parity at `2e-6`.

Smoke result:

| Machine | Bucket | Candidate warmed median | Current rewrite row | Status |
| --- | --- | ---: | ---: | --- |
| `m2-studio` | `3s` | `53.5 ms` | `49.7 ms` | rejected |

Raw artifact:

- `outputs/external_bakeoff/results_config_f_reference_m2-studio_hnsf_merge_smoke.json`

Decision:

- Rejected. The candidate regressed the fast local row, so it is not worth
  promoting to lower-end Macs.
- Keep the existing `vDSP_mmul` merge path; the BLAS kernel beats the lower
  scratch-memory version for this contract.

## 2026-06-06 M2 Air Source/Body 3s Diagnostic

Setup:

- Synced existing 3s source/body package artifacts to `m2-air`.
- Ran `scripts/probe_f0_noise_exact_shape.py --skip-export` on M2 Air with
  warmed inference only (`--warmup 5 --iterations 9`).
- Host quiet gate still failed at collection time:
  `outputs/external_bakeoff/remote_host_quiet_m2_air_latest.md`.

Diagnostic results:

| Candidate | Baseline stack | Candidate stack | Save | Corr | SNR | Projected full 3s | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3s_natural_asr_cos_rsqrt` | `123.9 ms` | `106.7 ms` | `17.2 ms` | `0.813986` | `5.08 dB` | `~128.5 ms` | fastest, listening-gated |
| `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | `123.5 ms` | `113.4 ms` | `10.1 ms` | `0.931815` | `9.19 dB` | `~135.6 ms` | better-quality win candidate, listening-gated |

Reference:

- M2 Air HAR-post rewrite overlay diagnostic full 3s median:
  `145.7 ms`.
- Current paper-facing laishere 3s row:
  `142.0 ms`.

Raw artifacts:

- `outputs/f0_noise_exact_shape/3s_natural_asr_cos_rsqrt/report_f0_noise_exact_3s_m2_air.json`
- `outputs/f0_noise_exact_shape/3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in/report_native_in_ios17_nopal_m2_air.json`
- `outputs/external_bakeoff/results_config_f_reference_m2-air_rewrite_overlay_diagnostic_3s.json`
- `outputs/external_bakeoff/f0_source_candidate_summary.md`

Decision:

- HAR-post rewrite alone is not enough on M2 Air 3s (`145.7 ms` vs `142.0 ms`).
- Source/body is the first measured lower-end path with enough M2 Air 3s margin.
- This is not a strict production/paper approval yet: both source/body rows fail
  strict waveform parity and still require no-ASR human listening acceptance.
- Do not update `competitive_frontier.md` until the host quiet gate and listening
  decision are both satisfied.
