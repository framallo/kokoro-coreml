# Remote Host Quiet Status

Checked at local time: `2026-06-06T13:17:39-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `m2-air` | `no` | 1.55/1.75/2.09 | load1 1.55 exceeds 1.00 |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
