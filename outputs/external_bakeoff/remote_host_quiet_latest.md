# Remote Host Quiet Status

Checked at local time: `2026-06-06T12:06:42-07:00`.

| Machine | Quiet | Load 1/5/15 | Blockers |
| --- | --- | ---: | --- |
| `irvine-m1` | `no` | 2.67/2.46/2.43 | load1 2.67 exceeds 1.00 |
| `m2-air` | `no` | 3.78/3.65/3.49 | load1 3.78 exceeds 1.00 |

Publishable lower-end Mac timing is allowed only when every target row is
`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record
the blocker instead.
