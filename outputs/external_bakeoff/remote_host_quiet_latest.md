# Remote Host Quiet Status

Checked at local time: `2026-06-06T17:23:20-07:00`.

| Machine | Quiet | Load 1/5/15 | Swap | Mem free | Power | Thermal | Blockers |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `irvine-m1` | `no` | 2.62/2.51/2.46 | 270.12 MB | 74% | AC | ok | load1 2.62 exceeds 1.50; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 81.0% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 34.5% CPU; swap used 270.12 MB exceeds 0.0 MB |
| `m2-air` | `no` | 3.07/2.89/2.92 | 266.94 MB | 78% | AC | ok | load1 3.07 exceeds 1.50; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 81.1% CPU; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 54.6% CPU; /System/Library/PrivateFrameworks/MediaAnalysisAccess.framework/Versions/A/XPCServices/mediaanalysisd-access.xpc/Contents/MacOS/mediaanalysisd-access at 7.8% CPU; swap used 266.94 MB exceeds 0.0 MB |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
