# Remote Host Quiet Status

Checked at local time: `2026-06-06T13:13:35-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `m2-air` | `no` | 2.19/1.92/2.23 | load1 2.19 exceeds 1.00 |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
