# Candidate Frontier Matrix

This matrix records the current measured optimization frontier for Config F.
It uses warmed-inference evidence only and separates strict production
candidates from quality-changing speed branches.

## Summary

- Candidates recorded: `8`.
- Production-ready strict candidates: `1`.
- Strict rejected or too-small candidates: `6`.
- Quality-fail speed candidates: `1`.
- Irvine profile rows remaining after rewrite projection: `4`.
- iPhone Config F launch blocker: `device_locked`.

## Matrix

| Family | Scope | Best signal | Strict | Production-ready | Decision | Next gate |
| --- | --- | --- | --- | --- | --- | --- |
| HAR-post upsample ConvT rewrite | single-package GeneratorFromHar | M2 Studio package +4.28% 3s, +3.15% 7s, +3.17% 10s, +2.60% 15s, +2.20% 30s; local E2E +1.22-2.58% | `yes` | `yes` | keep; promote to quiet Irvine timing before replacing checked-in packages | quiet Irvine M1 warmed run; quiet M2 Air rerun for fresh lower-end rows |
| Exact decoder+vocoder split | multi-package exact Swift HAR contract | Irvine 3s CPU+GPU -24.8 ms; CPU+NE -138.3 ms | `yes` | `no` | reject; split boundary/sync overhead exceeds body savings | do not repeat broad exact split; only try if a Core ML call boundary is removed |
| Exact generator noise/body split | multi-package exact HAR-post generator split | Irvine 3s CPU+GPU -11.5 ms; CPU+NE quality/speed failure | `yes` | `no` | reject; tightest strict split still loses | do not repeat unless packaging/synchronization changes |
| Full visible surface rewrite | single-package GeneratorFromHar | local E2E only wins 3s (49.532 ms vs production rewrite 49.669 ms) and noise-ties 10s | `yes` | `no` | reject as production replacement; simpler rewrite wins more buckets | none unless a new operator rewrite changes runtime behavior beyond surface matching |
| HAR-source fused strict path | source/STFT/HAR fused path | Irvine 3s CPU+GPU -22.9 ms; CPU+NE -163.8 ms | `yes` | `no` | reject; preserving strict source contract loses the speed edge | new representation only; do not promote padded strict path |
| Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite | single-package GeneratorFromHar | local 3s roughly +0.08-0.26%; .all/CPU+NE remains harmful | `yes` | `no` | reject as too small; graph-surface parity alone is not enough | do not repeat without a new layout, fusion, or partitioning mechanism |
| Style-specialized fused generator | single-package GeneratorFromHar with fixed af_heart projections | Irvine 3s -3.0 ms; M2 Air 3s -2.2 ms; local native-IN variant only +0.07 ms | `yes` | `no` | reject; freezing style is not a speed path | none unless combined with a material new operator rewrite |
| Fast F0/source simplification | laishere-like source/body branch | Irvine 3s +10.9 to +18.7 ms depending branch | `no` | `no` | not paper-strict; only useful with source recovery or no-ASR listening acceptance | human listening decisions or source/STFT representation repair |

## Evidence Links

- HAR-post upsample ConvT rewrite: `outputs/external_bakeoff/rewrite_candidate_impact.md`.
- Exact decoder+vocoder split: `README/Guides/apple-silicon/Kokoro-M1-vocoder-boundary-research-brief.md`.
- Exact generator noise/body split: `README/Notes/performance-notes.md`.
- Full visible surface rewrite: `outputs/external_bakeoff/results_config_f_reference_m2-studio-local_full_surface_ups_as_conv.json`.
- HAR-source fused strict path: `outputs/nyquist_phase_contribution/summary.md`.
- Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite: `README/Notes/performance-notes.md`.
- Style-specialized fused generator: `README/Notes/performance-notes.md`.
- Fast F0/source simplification: `outputs/f0_source_listening/cos_resblock_speed_branch/README.md`.

## Next Actions

- Retest the HAR-post upsample rewrite on Irvine M1 and M2 Air only when Spotlight/mediaanalysis load is quiet.
- Do not use cold compile/cache timings; every frontier update must use warmed medians.
- For a new strict candidate, require a single-package graph or a removed Core ML call boundary before lower-end promotion.
- Run the installed Config F iPhone runner only after the physical iPhone is unlocked; current launch blocker is device_locked.
- Keep fast F0/source branches separate from strict paper claims unless no-ASR human listening accepts the exact WAVs.
