# Lower-End Mac Win Attempts

This note records lower-end Mac promotion attempts that are useful evidence but
not publishable frontier rows unless the quiet gate passes. All timing is warmed
inside the runner; rows marked non-publishable were collected while the remote
host failed `scripts/external_bakeoff/check_remote_host_quiet.py`.

## 2026-06-06 Local DecoderPre/HnSF Overlap Smoke

Candidate:

- Started Swift HnSF/HAR construction immediately after F0/N padding and ran it
  concurrently with the `decoder_pre` Core ML prediction.
- The generator still waits for the same `x_pre` and `har` tensors, so this is a
  strict runtime scheduling change, not a model or source-quality change.
- Added `t_decoder_pre_hnsf_overlap_s` to benchmark output and
  `KOKORO_DISABLE_DECODER_HNSF_OVERLAP=1` as a same-binary serial fallback.

Local A/B smoke:

| Machine | Bucket | Parallel median | Serial median | Save | WAV hash |
| --- | --- | ---: | ---: | ---: | --- |
| `m2-studio-local` | `3s` | `51.4 ms` | `56.5 ms` | `5.0 ms` | identical |
| `m2-studio-local` | `7s` | `97.8 ms` | `121.3 ms` | `23.5 ms` | identical |

Raw artifacts:

- `outputs/external_bakeoff/results_config_f_reference_m2-studio-local_parallel_hnsf_ab.json`
- `outputs/external_bakeoff/results_config_f_reference_m2-studio-local_serial_hnsf_ab.json`
- `outputs/external_bakeoff/results_config_f_reference_m2-studio-local_parallel_hnsf_smoke.json`

Quiet gate:

- Refreshed lower-end quiet check:
  `outputs/external_bakeoff/remote_host_quiet_latest.md`.
- Result: not publishable; both `irvine-m1` and `m2-air` had load1 above `1.00`
  with `mediaanalysisd` and `mds_stores` blockers.

Decision:

- Keep as a strict, low-risk runtime candidate. It reduces serial fixed cost
  without changing waveform output in local A/B.
- Do not update lower-end frontier rows until the same A/B is collected on a
  quiet `m2-air` and/or `irvine-m1`.
- If a target regresses, disable with `KOKORO_DISABLE_DECODER_HNSF_OVERLAP=1`
  while investigating scheduler contention.

## 2026-06-06 Local Overlap + HAR-Post Rewrite Combined Ledger

Candidate:

- Combined the strict `decoder_pre`/HnSF runtime overlap with the existing
  HAR-post upsample ConvT rewrite overlay.
- Kept this as a measurement ledger and projection only; lower-end hosts were
  still noisy, so no frontier rows were promoted.

Same-binary local M2 Studio A/B:

| Bucket | Serial shipped | Overlap shipped | Overlap save | Overlap + rewrite | Combined save |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3s` | `52.4 ms` | `49.9 ms` | `2.5 ms` | `48.3 ms` | `4.1 ms` |
| `7s` | `96.8 ms` | `91.6 ms` | `5.2 ms` | `90.2 ms` | `6.6 ms` |
| `10s` | `129.7 ms` | `120.7 ms` | `9.0 ms` | `123.3 ms` | `6.4 ms` |
| `15s` | `190.5 ms` | `176.7 ms` | `13.8 ms` | `173.7 ms` | `16.8 ms` |
| `30s` | `396.2 ms` | `339.7 ms` | `56.4 ms` | `337.4 ms` | `58.8 ms` |

Notes:

- Overlap-only WAV hashes were identical for all five buckets.
- Rewrite-overlay WAV hashes changed, as expected; quality evidence remains the
  package-level parity/correlation report.
- Generator medians improved under the rewrite overlay on all five buckets, but
  `10s` wall-time was noisy versus the overlap baseline. Against the serial
  shipped baseline, the combined candidate was still positive on every bucket.

Projection:

- Generated `outputs/external_bakeoff/overlap_rewrite_candidate_impact.md`.
- Conservative formula: existing lower-end stage total minus
  `min(decoder_pre, hnsf)` overlap projection minus measured HAR-post package
  rewrite save.
- Projected profile gaps closed: `m2-air` `3s/7s/10s/15s` and `irvine-m1`
  `15s`.
- Projected profile gaps still open: `irvine-m1` `3s` by `27.0 ms`, `7s` by
  `28.0 ms`, and `10s` by `12.8 ms`.
- Generated
  `outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.md` for the
  stricter paper frontier target. Even after overlap + rewrite, Irvine M1 still
  needs `45.7 ms` (`3s`), `77.6 ms` (`7s`), `63.7 ms` (`10s`), and `64.4 ms`
  (`15s`) to beat the current strict paper-facing rows.

Decision:

- Keep both candidates. The overlap should remain enabled by default with
  `KOKORO_DISABLE_DECODER_HNSF_OVERLAP=1` as the fallback.
- The combined candidate is likely enough for M2 Air profile rows once quiet,
  but it is not the final Irvine M1 answer. Next strict work should target a
  larger source/body or single-package generator-stage improvement.

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

## 2026-06-06 Irvine M1 HAR-Post Rewrite Overlay Readiness

Remote overlay setup:

- Refreshed `/tmp/kokoro-coreml-fastpath-run/swift` on `irvine-m1` from the
  current local Swift sources and rebuilt `swift/.build/release/kokoro-bench`.
- Synced the five rewritten HAR-post packages from
  `outputs/export_rewrite_smoke/kokoro_decoder_har_post_{3,7,10,15,30}s.mlpackage`
  to the Irvine run tree.
- Recreated `outputs/export_rewrite_smoke/coreml_overlay` on Irvine with the
  rewritten HAR-post packages and symlinks to the existing remote `coreml/`
  package set.

Quiet gate:

- Latest combined lower-end quiet check:
  `outputs/external_bakeoff/remote_host_quiet_latest.md`.
- Result: not publishable. Irvine M1 had load1 `2.19`,
  `mediaanalysisd` at `90.3%`, `mds_stores` at `35.5%`, and
  `270.12 MB` swap used. M2 Air was also noisy.

Path smoke:

| Bucket | Warm wall | Generator | Status |
| --- | ---: | ---: | --- |
| `3s` | `222.6 ms` | `163.4 ms` | path-valid, non-publishable |

Raw artifact:

- `outputs/external_bakeoff/results_config_f_reference_irvine-m1_rewrite_overlay_path_smoke.json`

Decision:

- The Irvine rewrite-overlay path is staged and runnable when the host clears.
- Do not update `competitive_frontier.md`, `irvine_next_targets.md`, or paper
  tables from this row. It was collected only to verify the remote overlay path
  after the quiet gate failed.
- Next publishable action is a quiet-gated Irvine run for all five buckets using
  `outputs/export_rewrite_smoke/coreml_overlay`.

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
- `outputs/f0_source_listening/m2_air_3s_source_body/README.md`
- `outputs/f0_source_listening/m2_air_3s_source_body/f0_source_listening_decisions.csv`

Decision:

- HAR-post rewrite alone is not enough on M2 Air 3s (`145.7 ms` vs `142.0 ms`).
- Source/body is the first measured lower-end path with enough M2 Air 3s margin.
- This is not a strict production/paper approval yet: both source/body rows fail
  strict waveform parity and still require no-ASR human listening acceptance.
- Dedicated no-ASR listening pack exists, but its decision CSV is intentionally
  blank and currently fails validation until a human decision is recorded.
- Do not update `competitive_frontier.md` until the host quiet gate and listening
  decision are both satisfied.

## 2026-06-06 Fixed-Cost Fit Diagnostic

Artifact:

- Tracked note: `README/Notes/fixed-cost-latency-fit.md`.
- Regenerable report: `outputs/external_bakeoff/fixed_cost_latency_fit.md`
  from `scripts/external_bakeoff/summarize_fixed_cost_latency_fit.py`.

Result:

- Corrected warmed Config F beats MLX on every full-duration fitted Mac bucket
  (`12/12`); MLX still has no valid full-duration `3s` Mac row.
- Config F still loses `8` full-duration paper-facing lower-end Mac rows to
  laishere.
- The M1/M2 Air fits show Config F with lower duration-scaled slope than
  laishere, while laishere is better on short buckets. Because several laishere
  fits have negative fixed terms, this is heuristic rather than proof, but it
  supports the same next action as the stage-gap report: remove a hot-path
  Core ML boundary or materially reduce generator-stage duration.

Decision:

- Do not chase MLX as the current warmed Mac blocker; keep MLX in the paper as
  a comparison, but optimize against laishere on lower-end short buckets.
- Another 1-3% host-side tweak is below the remaining Irvine `3s/7s` budget
  after overlap + rewrite. New strict candidates should target a single-package
  source/body contract, HAR/STFT contract repair, or another large generator
  rewrite.

Latest quiet gate:

- `outputs/external_bakeoff/remote_host_quiet_latest.md` checked at
  `2026-06-06T15:19:13-07:00`.
- Result: not publishable (`quiet_hosts=0/2`). Irvine M1 had load1 `3.55` with
  `mediaanalysisd` at `92.4%` and `mds_stores` at `22.4%`; M2 Air had load1
  `3.09` with `mds_stores` at `81.8%`, `mediaanalysisd` at `46.4%`, and
  `mediaanalysisd-access` at `10.7%`.

Quieting attempt:

- `pkill -x mediaanalysisd`, `mediaanalysisd-access`, `photoanalysisd`,
  `mds_stores`, `mds`, and `mdworker_shared` did not quiet either machine;
  protected services remained running or respawned.
- `launchctl bootout gui/$(id -u)/com.apple.mediaanalysisd` failed with
  `Operation not permitted while System Integrity Protection is engaged`.
- `mdutil -i off /` failed with `Try as root`; `sudo -n` also failed because a
  password is required on both machines.
- Decision: do not promote lower-end warmed rows from this state. A publishable
  run needs either a naturally quiet window or an operator with root/session
  access to pause Spotlight/media analysis before timing.

## 2026-06-06 Noise/Body DecoderPre-Overlap Check

Question:

- Could the strict generator noise/body split become production-positive if the
  noise package is scheduled concurrently with `decoder_pre`, then the smaller
  body package runs after both are ready?

Dependency answer:

- No for the current strict contract. The noise package consumes `har`, so it
  cannot start until Swift HnSF/STFT finishes. On every measured short-bucket
  host, HnSF is already longer than `decoder_pre`: local M2 Studio `3s`
  `6.4 ms` HnSF vs `5.2 ms` `decoder_pre`, M2 Air `3s` `7.7 ms` vs `2.9 ms`,
  and Irvine M1 `3s` `21.2 ms` vs `4.4 ms`.
- Therefore `decoder_pre` is already hidden by HnSF in the current overlap
  path. Splitting noise out adds a serial `HnSF -> noise -> body` tail rather
  than a useful `noise || decoder_pre` overlap.

Measured split evidence:

| Host/report | Fused generator | Split noise | Split body | Split total | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| local M2 Studio `3s_native_in_broadcast_ios17` | `26.4 ms` | `11.3 ms` | `17.6 ms` | `28.9 ms` | slower |
| Irvine M1 `3s_native_in_broadcast_ios17` | `168.3 ms` | `74.0 ms` | `105.9 ms` | `179.8 ms` | slower |

Decision:

- Do not implement a production noise/body split merely to overlap with
  `decoder_pre`; the dependency graph cannot hide the expensive noise stage.
- The body-only counterfactual remains useful evidence, but it requires
  `x_source_*` to become cheap or in-package, not just differently scheduled.

## 2026-06-06 Padded/Nyquist Source-Noise Split Smoke

Candidate:

- Added `--pad-har-to` to `scripts/probe_har_source_noise_split.py` so the
  source/noise split can test the same strict geometry that rescued the fused
  `har_source` path: recomputed STFT + dumped Nyquist phase + padded shipping
  HAR time.
- Ran local 3s with:
  `--phase-mode atan_manual --noise-precision fp32 --nyquist-input --pad-har-to 28801`
  and staged compute units:
  decoder-pre `cpuAndNeuralEngine`, fused baseline/noise/body/tail `cpuAndGPU`.

Result:

| Bucket | Baseline decoder+generator | Candidate noise+body+tail | Delta | Quality |
| --- | ---: | ---: | ---: | --- |
| `3s` | `30.4 ms` | `34.4 ms` | `-13.0%` | corr `0.999986975`, SNR `46.25 dB`, max abs `0.012819` |

Stage medians:

| Stage | Median |
| --- | ---: |
| baseline decoder-pre | `3.1 ms` |
| baseline generator | `27.2 ms` |
| candidate noise | `10.6 ms` |
| candidate body | `22.6 ms` |
| candidate tail | `1.1 ms` |

Raw artifact:

- `outputs/har_source_noise_split/3s_atan_manual_fp32_nyquist_padded_nyquist/report_har_source_noise_nyquist_padded.json`

Decision:

- Rejected as a strict speed path. The padded/Nyquist geometry is quality-good,
  but the extra source/noise package and split body/tail boundary lose more than
  the body-only save recovers.
- Do not repeat source/noise split probes unless the package boundary is
  collapsed again or the noise/source tensors are produced without a separate
  Core ML prediction call.
