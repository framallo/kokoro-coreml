# Remote Host Quiet Status

Checked at local time: `2026-06-06T12:59:22-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `m2-air` | `no` | 1.51/2.01/2.59 | load1 1.51 exceeds 1.00 |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
