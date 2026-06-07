# Candidate Frontier Matrix

This matrix records the current measured optimization frontier for Config F.
It uses warmed-inference evidence only and separates strict production
candidates from non-strict or quality-changing branches.

## Summary

- Candidates recorded: `22`.
- Production-ready strict candidates: `2`.
- Strict rejected or too-small candidates: `16`.
- Non-strict or quality-changing candidates: `4`.
- Irvine profile rows remaining after current projection: `3`.
- iPhone Config F launch blocker: `device_locked`.

## Matrix

| Family | Scope | Best signal | Strict | Production-ready | Decision | Next gate |
| --- | --- | --- | --- | --- | --- | --- |
| DecoderPre/HnSF runtime overlap | Swift runtime scheduling inside current Config F boundary | local M2 Studio hash-identical all-bucket save: +4.80% 3s, +5.36% 7s, +6.92% 10s, +7.24% 15s, +14.25% 30s versus same-binary serial shipped path | `yes` | `yes` | keep enabled by default; fallback env KOKORO_DISABLE_DECODER_HNSF_OVERLAP=1 | quiet lower-end A/B; if a host regresses, use fallback and inspect decoder-pre CPU contention |
| HAR-post upsample ConvT rewrite | single-package GeneratorFromHar | M2 Studio package +4.28% 3s, +3.15% 7s, +3.17% 10s, +2.60% 15s, +2.20% 30s; local E2E +1.22-2.58% | `yes` | `yes` | keep; promote to quiet Irvine timing before replacing checked-in packages | quiet Irvine M1 warmed run; quiet M2 Air rerun for fresh lower-end rows |
| CT8/CT9/iOS17 toolchain-only rebuild | single-package GeneratorFromHar rebuild with newer conversion target | initial local 3s CT9 +2.14%, but 10s -0.16% and 15s -0.27%; later same-process rows tied | `yes` | `no` | reject; toolchain metadata alone is below material threshold | only revisit if paired with a real graph rewrite and same-process baseline |
| DecoderPre + Generator merged package | single-package DecoderPre plus GeneratorFromHar while preserving Swift HAR/STFT boundary | local M2 Studio 3s merged CPU+GPU is slower than two predictions: 31.929 -> 33.589 ms (-5.20%); .all is also slower (-4.70%); CPU+NE hard-rejects at 1568.424 ms with ANE compiler failure | `yes` | `no` | reject; removing this boundary makes the MLProgram larger and less schedulable than the cheap separate DecoderPre call plus GPU generator | do not promote; only revisit if a graph rewrite removes the handoff without absorbing DecoderPre into the generator scheduling surface |
| Embedded DC-branch + exact Nyquist fused Core ML graph | single-package har_source -> waveform graph with explicit DC branch and exact Swift Nyquist atan2 inside MLProgram | corrected graph is strict but slower: local 3s 27.53 ms vs 26.98 ms (-2.05%); local 7s 58.36 ms vs 56.40 ms (-3.47%) | `yes` | `no` | keep as correctness repair for no-side-input source graphs; reject as current speed win | only revisit if this source graph is fused with another removed boundary or if MIL/profile evidence shows the added STFT ops can be scheduled cheaper |
| Exact Nyquist padding trim | PyTorch-only source/HAR padding sweep | first strict exact-Nyquist padding points are 3s har_time=28561 and 7s har_time=66601, saving only 240/28801 frames (0.83%) and 600/67201 frames (0.89%) versus full padded HAR | `yes` | `no` | reject as standalone win; the strict padding context is still almost the full shipping HAR axis | only revisit if a graph rewrite makes HAR length reductions superlinear or combines trim with a removed Core ML call boundary |
| Exact decoder+vocoder split | multi-package exact Swift HAR contract | Irvine 3s CPU+GPU -24.8 ms; CPU+NE -138.3 ms | `yes` | `no` | reject; split boundary/sync overhead exceeds body savings | do not repeat broad exact split; only try if a Core ML call boundary is removed |
| Exact generator noise/body split | multi-package exact HAR-post generator split | body-only is faster if x_source tensors are free (M2 17.6 vs 26.4 ms; Irvine 105.9 vs 168.3 ms), but full strict split loses once noise/source is included; decoderPre overlap cannot hide noise because noise waits for HAR and HnSF already exceeds decoderPre on measured 3s hosts; padded/Nyquist 3s source-noise split is quality-good but slower (34.4 vs 30.4 ms, -13.0%) | `yes` | `no` | reject; x_source body package is promising only with a cheaper strict source contract | only revisit if x_source is produced without a separate Core ML noise call or with a new source representation |
| Full visible surface rewrite | single-package GeneratorFromHar | local E2E only wins 3s (49.532 ms vs production rewrite 49.669 ms) and noise-ties 10s | `yes` | `no` | reject as production replacement; simpler rewrite wins more buckets | none unless a new operator rewrite changes runtime behavior beyond surface matching |
| Generator outputBackings | Swift generator-isolation harness using MLPredictionOptions.outputBackings | local M2 Studio CPU+GPU generator-only: 3s -0.077 ms, 7s +0.414 ms versus plain prediction(from:) | `yes` | `no` | reject as current production win; keep --generator-output-backing as a device-check harness flag | only revisit if a lower-end Mac or iPhone shows a material >1 ms warmed median gain |
| HAR input trim | single-package GeneratorFromHar with shorter strict HAR axis | Irvine 3s +0.43%; M2 Studio strict point is slower (-1.15%) | `yes` | `no` | reject; less than one millisecond on M1 cannot close laishere gap | do not repeat tail trim unless a new source/HAR representation removes much more padding |
| HAR-source fused strict path | source/STFT/HAR fused path | natural har_source is speed-positive but quality-failing; strict padded/Nyquist fused path has replacement-quality versus the current generator but no net win after Swift STFT credit (+0.051 ms 3s, +1.326 ms 7s, +2.231 ms 10s, +14.977 ms 30s); exact Swift Float Nyquist atan2 matches the dumped-Nyquist oracle but keeps the same direct speed envelope | `yes` | `no` | reject; preserving strict source contract loses the speed edge | new representation only: phase reparameterization, weight folding, or a no-extra-boundary Nyquist side input |
| LUT-palettized full surface plus upsample rewrite | single-package GeneratorFromHar with native-IN, broadcast AdaIN, fp16 inputs, pal8 weights, and zero-insert upsample rewrite | local 3s -2.78% versus production upsample rewrite; CPU+NE still CPU-preferred after ANE compile failure | `yes` | `no` | reject; reproduces laishere-like LUT surface but is slower and does not fix placement | do not repeat palettized final-waveform packages unless compression is moved behind a separate strict tail or changes placement |
| Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite | single-package GeneratorFromHar | local 3s roughly +0.08-0.26%; .all/CPU+NE remains harmful | `yes` | `no` | reject as too small; graph-surface parity alone is not enough | do not repeat without a new layout, fusion, or partitioning mechanism |
| Reusable Swift input MLMultiArrays | Swift benchmark runtime input-buffer reuse for DecoderPre and Generator features | local M2 Studio staged A/B is hash-identical but slower: 3s 49.412 -> 50.011 ms (-1.21%), 7s 93.877 -> 97.355 ms (-3.71%) | `yes` | `no` | reject as production win; keep --reuse-input-arrays as a diagnostic flag only | only revisit if a lower-end Mac shows a contradictory material win; do not enable by default |
| Style-specialized fused generator | single-package GeneratorFromHar with fixed af_heart projections | Irvine 3s -3.0 ms; M2 Air 3s -2.2 ms; local native-IN variant only +0.07 ms | `yes` | `no` | reject; freezing style is not a speed path | none unless combined with a material new operator rewrite |
| Style-specialized generator plus upsample rewrite | single-package fixed-voice GeneratorFromHar with native-IN and zero-insert upsample rewrite | local 3s +4.54% vs shipped fused, only +0.17% versus production upsample rewrite at N=30; CPU+NE still CPU-preferred after ANE compile failure | `yes` | `no` | reject; noise-sized over the simpler rewrite and does not fix partitioning | do not promote unless multi-bucket local evidence beats production rewrite by a material margin |
| Swift exact Nyquist phase repair | PyTorch-only source/HAR sensitivity probe | exact Swift Float real/imag dot products followed by atan2 match the dumped-Nyquist oracle on padded buckets: 50.06/49.14/49.87/49.21/48.42 dB SNR; branch-only Swift basis remains only 25.59/25.84/26.55/25.80/25.41 dB | `yes` | `no` | keep as source-contract unlock; reject as standalone speed candidate because padded strict source path still has no net warmed win | put the exact formula inside a single no-extra-boundary graph, fold it into first noise-conv weights, or use it to design a strict source representation with less padding |
| Fast F0/source simplification | laishere-like source/body branch | Irvine 3s +10.9 to +18.7 ms depending branch | `no` | `no` | not paper-strict; only useful with source recovery or no-ASR listening acceptance | human listening decisions or source/STFT representation repair |
| Linear weight quantization | single-package final-waveform GeneratorFromHar compression | int8 CPU-only +4.27% but CPU+GPU crashes; int4 iOS18 is slower | `no` | `no` | reject for final-waveform generator; compression is not the missing speed path | only revisit on discarded-output intermediate stages with separate final-quality tail |
| Per-stage prefix compute-unit overrides | Swift runtime policy for duration/F0Ntrain/decoder-pre | local 3s duration+F0Ntrain CPU+ANE N=5 looked +1.858 ms, but N=20 shrank to +0.213 ms and 7s/10s regressed by 11.241/18.366 ms | `no` | `no` | reject as production candidate; keep per-stage override harness for diagnostics | none unless a future export makes duration/F0Ntrain CPU+ANE numerically stable and materially faster |
| RangeDim/flexible input generator | single-package GeneratorFromHar with bounded dynamic time axes | local 3s 343-1561 ms candidate latency versus 31-50 ms fused baseline | `no` | `no` | reject; dynamic broadcast/shape propagation is both slower and not strict | do not use RangeDim for the fused generator hot path; keep fixed buckets |

## Evidence Links

- DecoderPre/HnSF runtime overlap: `outputs/external_bakeoff/lower_end_mac_win_attempts.md`.
- HAR-post upsample ConvT rewrite: `outputs/external_bakeoff/rewrite_candidate_impact.md`.
- CT8/CT9/iOS17 toolchain-only rebuild: `README/Notes/performance-notes.md`.
- DecoderPre + Generator merged package: `README/Notes/performance-notes.md; outputs/decoder_pre_generator_merge/3s/report.json; outputs/decoder_pre_generator_merge/3s/report_all.json; outputs/decoder_pre_generator_merge/3s/report_cpune.json`.
- Embedded DC-branch + exact Nyquist fused Core ML graph: `README/Notes/har-stft-phase-contract.md`.
- Exact Nyquist padding trim: `README/Notes/har-stft-phase-contract.md`.
- Exact decoder+vocoder split: `README/Kokoro-M1-vocoder-boundary-research-brief.md`.
- Exact generator noise/body split: `outputs/external_bakeoff/lower_end_mac_win_attempts.md`.
- Full visible surface rewrite: `outputs/external_bakeoff/results_config_f_reference_m2-studio-local_full_surface_ups_as_conv.json`.
- Generator outputBackings: `README/Notes/kokoro-restarted-guide-triage-2026-06-06.md; outputs/generator_output_backing/`.
- HAR input trim: `README/Notes/performance-notes.md`.
- HAR-source fused strict path: `README/Notes/har-stft-phase-contract.md; scripts/external_bakeoff/summarize_hnsf_source_boundary.py`.
- LUT-palettized full surface plus upsample rewrite: `outputs/generator_cos_snake/3s_native_broadcast_fp16_pal8_ups_as_conv_vs_rewrite_plain_broadcast_adain_native_in_pal8_fp16_inputs_ups_as_conv_ios17/report_cpu_gpu_vs_rewrite.json`.
- Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite: `README/Notes/performance-notes.md`.
- Reusable Swift input MLMultiArrays: `outputs/external_bakeoff/results_config_f_reference_m2-studio-local_reuse_arrays_baseline_v3_short.json; outputs/external_bakeoff/results_config_f_reference_m2-studio-local_reuse_arrays_candidate_v3_short.json`.
- Style-specialized fused generator: `README/Notes/performance-notes.md`.
- Style-specialized generator plus upsample rewrite: `outputs/generator_style_specialization/3s_style_native_in_ups_as_conv_ios17/report_cpu_gpu_vs_rewrite_n30.json`.
- Swift exact Nyquist phase repair: `outputs/nyquist_phase_contribution/summary.md`.
- Fast F0/source simplification: `outputs/f0_source_listening/cos_resblock_speed_branch/README.md`.
- Linear weight quantization: `README/Notes/performance-notes.md`.
- Per-stage prefix compute-unit overrides: `README/Notes/stage-compute-policy-ablation.md`.
- RangeDim/flexible input generator: `README/Notes/performance-notes.md`.

## Next Actions

- Run scripts/external_bakeoff/check_remote_host_quiet.py before any lower-end Mac promotion run.
- Retest the HAR-post upsample rewrite on Irvine M1 and M2 Air only when outputs/external_bakeoff/remote_host_quiet_latest.md reports quiet=yes.
- Use scripts/external_bakeoff/run_rewrite_promotion_when_quiet.py for lower-end rewrite promotion runs; it wraps the quiet gate and passes --generator-models-dir outputs/export_rewrite_smoke.
- Do not use cold compile/cache timings; every frontier update must use warmed medians.
- Use README/Notes/fixed-cost-latency-fit.md to separate fixed-boundary overhead from duration-scaled generator cost before promoting a new optimization family.
- Use README/Notes/har-stft-phase-contract.md and scripts/external_bakeoff/summarize_hnsf_source_boundary.py before revisiting any har_source boundary; exact Swift Nyquist atan2 solves parity, but tail padding trim saves <1% of HAR frames and Swift STFT credit alone does not make the strict padded/Nyquist path win.
- Use README/Kokoro-M1-HAR-STFT-contract-deep-research-prompt.md for the next external research pass; the source equation is solved, the HAR/STFT contract is not.
- For a new strict candidate, require a single-package graph or a removed Core ML call boundary before lower-end promotion.
- Run the installed Config F iPhone runner only after the physical iPhone is unlocked; current launch blocker is device_locked.
- Keep fast F0/source branches separate from strict paper claims unless no-ASR human listening accepts the exact WAVs.
