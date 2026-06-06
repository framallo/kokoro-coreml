# Remote Host Quiet Status

Checked at local time: `2026-06-06T07:20:41-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `irvine-m1` | `no` | 3.02/2.78/2.79 | load1 3.02 exceeds 1.00; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 88.5% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 14.0% CPU |
| `m2-air` | `no` | 4.08/3.94/3.81 | load1 4.08 exceeds 1.00; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 115.7% CPU; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 46.9% CPU |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
