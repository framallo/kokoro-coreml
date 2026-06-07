# Remote Host Quiet Status

Checked at local time: `2026-06-06T20:01:35-07:00`.

| Machine | Quiet | Load 1/5/15 | Swap | Mem free | Power | Thermal | Blockers |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `irvine-m1` | `no` | 1.50/2.07/2.50 | 270.12 MB | 74% | AC | ok | /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 9.6% CPU; swap used 270.12 MB exceeds 0.0 MB |
| `m2-air` | `no` | 2.35/3.35/3.70 | 266.94 MB | 58% | battery/unknown | ok | load1 2.35 exceeds 1.50; swap used 266.94 MB exceeds 0.0 MB; not drawing from AC power |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
