# Remote Host Quiet Status

Checked at local time: `2026-06-06T18:07:11-07:00`.

| Machine | Quiet | Load 1/5/15 | Swap | Mem free | Power | Thermal | Blockers |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `irvine-m1` | `no` | 3.27/2.70/2.59 | 270.12 MB | 75% | AC | ok | load1 3.27 exceeds 1.50; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 117.6% CPU; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 67.9% CPU; /System/Library/PrivateFrameworks/MediaAnalysisAccess.framework/Versions/A/XPCServices/mediaanalysisd-access.xpc/Contents/MacOS/mediaanalysisd-access at 7.7% CPU; swap used 270.12 MB exceeds 0.0 MB |
| `m2-air` | `no` | 2.53/2.80/2.82 | 266.94 MB | 76% | AC | ok | load1 2.53 exceeds 1.50; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 96.5% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 28.2% CPU; swap used 266.94 MB exceeds 0.0 MB |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
