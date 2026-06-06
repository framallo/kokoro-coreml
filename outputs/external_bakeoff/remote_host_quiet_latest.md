# Remote Host Quiet Status

Checked at local time: `2026-06-06T12:47:53-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `irvine-m1` | `no` | 1.85/2.15/2.88 | load1 1.85 exceeds 1.00; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 93.4% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 28.1% CPU |
| `m2-air` | `no` | 3.04/3.19/3.32 | load1 3.04 exceeds 1.00; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 81.3% CPU; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 69.8% CPU |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
