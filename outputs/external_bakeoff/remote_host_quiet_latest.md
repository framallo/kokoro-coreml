# Remote Host Quiet Status

Checked at local time: `2026-06-06T18:28:40-07:00`.

| Machine | Quiet | Load 1/5/15 | Swap | Mem free | Power | Thermal | Blockers |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `irvine-m1` | `no` | 2.04/2.45/2.53 | 270.12 MB | 75% | AC | ok | load1 2.04 exceeds 1.50; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 83.4% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 18.7% CPU; swap used 270.12 MB exceeds 0.0 MB |
| `m2-air` | `no` | 3.03/2.98/2.92 | 266.94 MB | 76% | AC | ok | load1 3.03 exceeds 1.50; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 63.5% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 48.5% CPU; /System/Library/PrivateFrameworks/MediaAnalysisAccess.framework/Versions/A/XPCServices/mediaanalysisd-access.xpc/Contents/MacOS/mediaanalysisd-access at 8.3% CPU; swap used 266.94 MB exceeds 0.0 MB |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
