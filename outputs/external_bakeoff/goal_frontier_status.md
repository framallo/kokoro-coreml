# Fastest Goal Frontier Status

Absolute fastest verified: `false`.
Config F loss cells: `8`.
Real Irvine loss cells: `4`.
Stale/tie loss cells: `4`.
Strict-pass closers: `0`.
Quality-fail warmed-profile closers: `3`.

## Real Irvine Losses

| Bucket | Config F | laishere | Gap |
| --- | ---: | ---: | ---: |
| `3s` | 233.6 ms | 195.0 ms | 38.5 ms / 19.75% |
| `7s` | 492.7 ms | 444.2 ms | 48.4 ms / 10.90% |
| `10s` | 685.5 ms | 644.9 ms | 40.6 ms / 6.29% |
| `15s` | 1014.9 ms | 990.6 ms | 24.3 ms / 2.46% |

## No-ASR Listening

Targets: `4`; mapped artifacts: `4`; exact timing artifacts: `4`.
Decision counts: `{"blank": 4}`.

## iPhone

Install OK: `true`.
Launch OK: `false`.
Launch blocker: `device_locked`.
Bundle: `com.kokoro.externalbakeoff.ConfigFIOSRunnerManual`.

## Blockers

- absolute_fastest_verified is false
- no saved strict-pass candidate closes a real Irvine loss
- 4 Irvine no-ASR listening decisions are blank
- iPhone Config F launch blocker: device_locked
- 2026-06-06 06:51 local: irvine-m1 load averages 2.95/2.99/3.01; mediaanalysisd 66.0% CPU and mds_stores 40.3% CPU, so skip publishable Irvine timing for the upsample rewrite candidate.

## Next Actions

- Collect no-ASR human decisions for Irvine F0/source speed candidates.
- Retry iPhone Config F runner only after the physical device is unlocked.
- Run publishable Irvine timings only after mediaanalysisd/Spotlight load is idle.
- Create a new strict single-package or source/body formulation; existing strict probes do not close losses.
