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
