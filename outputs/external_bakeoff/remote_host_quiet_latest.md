# Remote Host Quiet Status

Checked at local time: `2026-06-06T07:12:50-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `irvine-m1` | `no` | 2.08/2.56/2.80 | load1 2.08 exceeds 1.00; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 98.7% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 16.4% CPU |
| `m2-air` | `no` | 3.87/3.83/3.77 | load1 3.87 exceeds 1.00; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 94.1% CPU; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 45.9% CPU |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
