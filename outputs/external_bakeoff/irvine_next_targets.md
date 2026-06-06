# Irvine Next Targets

Warmed inference only. This narrows the current frontier to real Irvine M1
losses after filtering stale/tie paper-facing rows.

Real Irvine loss rows: `4`.
Saved strict-pass candidates that close these losses: `0`.
Saved quality-fail candidates that close these losses: `0`.
Quality-fail candidates that would beat warmed laishere profile: `3`.

| Bucket | Config F | laishere | Gap | Source/body gap | Upstream/runtime gap | Target class | Best saved strict pass | Best quality-fail speed signal | Quality-fail vs warmed profile |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 3s | 233.6 ms | 195.0 ms | 38.5 ms / 19.75% | 22.0 ms | 12.9 ms | source/body primary; upstream/runtime material | `3s_har28561` (0.7 ms saved; still 56.5 ms short) | `3s_natural_asr_cos_rsqrt` (18.7 ms saved; still 38.5 ms short) | loss: projected 214.8 ms, margin -19.8 ms |
| 7s | 492.7 ms | 444.2 ms | 48.4 ms / 10.90% | 43.3 ms | 3.6 ms | source/body dominates | `7s_har_cos_resblock_cos_rsqrt` (48.8 ms slower; still 146.9 ms short) | `7s_natural_asr_cos_rsqrt` (48.6 ms saved; still 49.5 ms short) | win: projected 444.1 ms, margin 0.1 ms |
| 10s | 685.5 ms | 644.9 ms | 40.6 ms / 6.29% | 56.0 ms | -9.8 ms | source/body dominates | `10s_har_cos_resblock_cos_rsqrt` (68.8 ms slower; still 160.4 ms short) | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` (76.8 ms saved; still 14.8 ms short) | win: projected 608.7 ms, margin 36.2 ms |
| 15s | 1014.9 ms | 990.6 ms | 24.3 ms / 2.46% | 40.9 ms | -23.6 ms | source/body dominates | none | `15s_padded_cos_resblock_cos_rsqrt` (78.9 ms saved; still 24.1 ms short) | win: projected 936.1 ms, margin 54.5 ms |

## Next Actions

- Do not promote fresh Irvine timing until background indexing/media analysis is idle.
- Strict path: find a single-package or narrower exact-HAR source/body graph surface; saved strict-pass probes do not close any current Irvine loss.
- Non-strict path: F0/source simplification has speed signal but remains quality-fail and needs human listening acceptance or reformulation.
- iPhone path: once the device is unlocked, launch the installed Config F runner, wait for compile/warm inference, then pull and ingest the app Documents result.
