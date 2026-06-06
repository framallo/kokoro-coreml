# Candidate Frontier Matrix

This matrix records the current measured optimization frontier for Config F.
It uses warmed-inference evidence only and separates strict production
candidates from non-strict or quality-changing branches.

## Summary

- Candidates recorded: `14`.
- Production-ready strict candidates: `1`.
- Strict rejected or too-small candidates: `10`.
- Non-strict or quality-changing candidates: `3`.
- Irvine profile rows remaining after rewrite projection: `4`.
- iPhone Config F launch blocker: `device_locked`.

## Matrix

| Family | Scope | Best signal | Strict | Production-ready | Decision | Next gate |
| --- | --- | --- | --- | --- | --- | --- |
| HAR-post upsample ConvT rewrite | single-package GeneratorFromHar | M2 Studio package +4.28% 3s, +3.15% 7s, +3.17% 10s, +2.60% 15s, +2.20% 30s; local E2E +1.22-2.58% | `yes` | `yes` | keep; promote to quiet Irvine timing before replacing checked-in packages | quiet Irvine M1 warmed run; quiet M2 Air rerun for fresh lower-end rows |
| CT8/CT9/iOS17 toolchain-only rebuild | single-package GeneratorFromHar rebuild with newer conversion target | initial local 3s CT9 +2.14%, but 10s -0.16% and 15s -0.27%; later same-process rows tied | `yes` | `no` | reject; toolchain metadata alone is below material threshold | only revisit if paired with a real graph rewrite and same-process baseline |
| Exact decoder+vocoder split | multi-package exact Swift HAR contract | Irvine 3s CPU+GPU -24.8 ms; CPU+NE -138.3 ms | `yes` | `no` | reject; split boundary/sync overhead exceeds body savings | do not repeat broad exact split; only try if a Core ML call boundary is removed |
| Exact generator noise/body split | multi-package exact HAR-post generator split | body-only is faster if x_source tensors are free (M2 17.6 vs 26.4 ms; Irvine 105.9 vs 168.3 ms), but full strict split loses once noise/source is included | `yes` | `no` | reject; x_source body package is promising only with a cheaper strict source contract | only revisit if x_source is produced without a separate Core ML noise call or with a new source representation |
| Full visible surface rewrite | single-package GeneratorFromHar | local E2E only wins 3s (49.532 ms vs production rewrite 49.669 ms) and noise-ties 10s | `yes` | `no` | reject as production replacement; simpler rewrite wins more buckets | none unless a new operator rewrite changes runtime behavior beyond surface matching |
| HAR input trim | single-package GeneratorFromHar with shorter strict HAR axis | Irvine 3s +0.43%; M2 Studio strict point is slower (-1.15%) | `yes` | `no` | reject; less than one millisecond on M1 cannot close laishere gap | do not repeat tail trim unless a new source/HAR representation removes much more padding |
| HAR-source fused strict path | source/STFT/HAR fused path | Irvine 3s CPU+GPU -22.9 ms; CPU+NE -163.8 ms | `yes` | `no` | reject; preserving strict source contract loses the speed edge | new representation only; do not promote padded strict path |
| LUT-palettized full surface plus upsample rewrite | single-package GeneratorFromHar with native-IN, broadcast AdaIN, fp16 inputs, pal8 weights, and zero-insert upsample rewrite | local 3s -2.78% versus production upsample rewrite; CPU+NE still CPU-preferred after ANE compile failure | `yes` | `no` | reject; reproduces laishere-like LUT surface but is slower and does not fix placement | do not repeat palettized final-waveform packages unless compression is moved behind a separate strict tail or changes placement |
| Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite | single-package GeneratorFromHar | local 3s roughly +0.08-0.26%; .all/CPU+NE remains harmful | `yes` | `no` | reject as too small; graph-surface parity alone is not enough | do not repeat without a new layout, fusion, or partitioning mechanism |
| Style-specialized fused generator | single-package GeneratorFromHar with fixed af_heart projections | Irvine 3s -3.0 ms; M2 Air 3s -2.2 ms; local native-IN variant only +0.07 ms | `yes` | `no` | reject; freezing style is not a speed path | none unless combined with a material new operator rewrite |
| Style-specialized generator plus upsample rewrite | single-package fixed-voice GeneratorFromHar with native-IN and zero-insert upsample rewrite | local 3s +4.54% vs shipped fused, only +0.17% versus production upsample rewrite at N=30; CPU+NE still CPU-preferred after ANE compile failure | `yes` | `no` | reject; noise-sized over the simpler rewrite and does not fix partitioning | do not promote unless multi-bucket local evidence beats production rewrite by a material margin |
| Fast F0/source simplification | laishere-like source/body branch | Irvine 3s +10.9 to +18.7 ms depending branch | `no` | `no` | not paper-strict; only useful with source recovery or no-ASR listening acceptance | human listening decisions or source/STFT representation repair |
| Linear weight quantization | single-package final-waveform GeneratorFromHar compression | int8 CPU-only +4.27% but CPU+GPU crashes; int4 iOS18 is slower | `no` | `no` | reject for final-waveform generator; compression is not the missing speed path | only revisit on discarded-output intermediate stages with separate final-quality tail |
| RangeDim/flexible input generator | single-package GeneratorFromHar with bounded dynamic time axes | local 3s 343-1561 ms candidate latency versus 31-50 ms fused baseline | `no` | `no` | reject; dynamic broadcast/shape propagation is both slower and not strict | do not use RangeDim for the fused generator hot path; keep fixed buckets |

## Evidence Links

- HAR-post upsample ConvT rewrite: `outputs/external_bakeoff/rewrite_candidate_impact.md`.
- CT8/CT9/iOS17 toolchain-only rebuild: `README/Notes/performance-notes.md`.
- Exact decoder+vocoder split: `README/Guides/apple-silicon/Kokoro-M1-vocoder-boundary-research-brief.md`.
- Exact generator noise/body split: `README/Notes/performance-notes.md`.
- Full visible surface rewrite: `outputs/external_bakeoff/results_config_f_reference_m2-studio-local_full_surface_ups_as_conv.json`.
- HAR input trim: `README/Notes/performance-notes.md`.
- HAR-source fused strict path: `outputs/nyquist_phase_contribution/summary.md`.
- LUT-palettized full surface plus upsample rewrite: `outputs/generator_cos_snake/3s_native_broadcast_fp16_pal8_ups_as_conv_vs_rewrite_plain_broadcast_adain_native_in_pal8_fp16_inputs_ups_as_conv_ios17/report_cpu_gpu_vs_rewrite.json`.
- Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite: `README/Notes/performance-notes.md`.
- Style-specialized fused generator: `README/Notes/performance-notes.md`.
- Style-specialized generator plus upsample rewrite: `outputs/generator_style_specialization/3s_style_native_in_ups_as_conv_ios17/report_cpu_gpu_vs_rewrite_n30.json`.
- Fast F0/source simplification: `outputs/f0_source_listening/cos_resblock_speed_branch/README.md`.
- Linear weight quantization: `README/Notes/performance-notes.md`.
- RangeDim/flexible input generator: `README/Notes/performance-notes.md`.

## Next Actions

- Run scripts/external_bakeoff/check_remote_host_quiet.py before any lower-end Mac promotion run.
- Retest the HAR-post upsample rewrite on Irvine M1 and M2 Air only when outputs/external_bakeoff/remote_host_quiet_latest.md reports quiet=yes.
- Do not use cold compile/cache timings; every frontier update must use warmed medians.
- For a new strict candidate, require a single-package graph or a removed Core ML call boundary before lower-end promotion.
- Run the installed Config F iPhone runner only after the physical iPhone is unlocked; current launch blocker is device_locked.
- Keep fast F0/source branches separate from strict paper claims unless no-ASR human listening accepts the exact WAVs.
