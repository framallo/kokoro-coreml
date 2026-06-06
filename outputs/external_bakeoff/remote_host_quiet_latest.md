# Remote Host Quiet Status

Checked at local time: `2026-06-06T15:19:13-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `irvine-m1` | `no` | 3.55/3.16/2.90 | load1 3.55 exceeds 1.00; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 92.4% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 22.4% CPU |
| `m2-air` | `no` | 3.09/2.85/2.95 | load1 3.09 exceeds 1.00; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 81.8% CPU; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 46.4% CPU; /System/Library/PrivateFrameworks/MediaAnalysisAccess.framework/Versions/A/XPCServices/mediaanalysisd-access.xpc/Contents/MacOS/mediaanalysisd-access at 10.7% CPU |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
