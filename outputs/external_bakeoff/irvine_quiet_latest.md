# Remote Host Quiet Status

Checked at local time: `2026-06-06T13:31:58-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `irvine-m1` | `no` | 3.15/2.59/2.27 | load1 3.15 exceeds 1.00; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mds_stores at 60.5% CPU; /System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd at 40.1% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Support/mds at 27.4% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mdworker_shared at 18.6% CPU; /System/Library/Frameworks/CoreServices.framework/Frameworks/Metadata.framework/Versions/A/Support/mdworker_shared at 10.0% CPU |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
