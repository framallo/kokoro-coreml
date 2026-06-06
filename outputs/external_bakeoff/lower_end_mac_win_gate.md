# Lower-End Mac Win Gate

Warmed inference only. This report tracks the remaining `m2-air` and
`irvine-m1` gates needed before claiming we beat the external Apple
Silicon implementations on lower-end Macs.

Pending paper-frontier listening wins: `2`.
Accepted paper-frontier wins: `0`.
Pending profile-only listening wins: `5`.
Paper-frontier blocked rows: `4`.

## Candidate Gates

| Machine | Bucket | Candidate | Gate | Human | Current Config F | Projected | Paper margin | Profile margin | Quality |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| m2-air | `3s` | `3s_natural_asr_cos_rsqrt` | human-listening | blank | 148.0 ms | 130.8 ms | +11.2 ms vs laishere: 142.0 ms | +22.2 ms vs laishere: 153.0 ms | corr 0.814, SNR 5.08 dB |
| m2-air | `3s` | `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | human-listening | blank | 148.0 ms | 137.9 ms | +4.1 ms vs laishere: 142.0 ms | +15.1 ms vs laishere: 153.0 ms | corr 0.932, SNR 9.19 dB |
| irvine-m1 | `3s` | `3s_natural_asr_cos_rsqrt` | implementation-gap | blank | 233.6 ms | 214.8 ms | -38.5 ms vs laishere: 176.3 ms | -19.8 ms vs laishere: 195.0 ms | corr 0.814, SNR 5.08 dB |
| irvine-m1 | `7s` | `7s_natural_asr_cos_rsqrt` | paper-frontier-gap | blank | 492.7 ms | 444.1 ms | -49.5 ms vs laishere: 394.6 ms | +0.1 ms vs laishere: 444.2 ms | corr 0.797, SNR 4.77 dB |
| irvine-m1 | `10s` | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | paper-frontier-gap | blank | 685.5 ms | 608.7 ms | -14.8 ms vs laishere: 593.9 ms | +36.2 ms vs laishere: 644.9 ms | corr 0.867, SNR 6.55 dB |
| irvine-m1 | `15s` | `15s_padded_cos_resblock_cos_rsqrt` | paper-frontier-gap | blank | 1014.9 ms | 936.1 ms | -24.1 ms vs laishere: 912.0 ms | +54.5 ms vs laishere: 990.6 ms | corr 0.957, SNR 11.00 dB |

## Decision

M2 Air 3s has paper-frontier wins gated only by human listening. Irvine source/body candidates can beat several newer warmed profile rows, but none beats the stricter paper-facing frontier yet. Irvine still needs a combined implementation win, not just listening acceptance.

## Immediate Work

- Get no-ASR human listening decisions for the M2 Air `3s` candidates.
- Get no-ASR human listening decisions for Irvine `7s`, `10s`, and `15s` source/body candidates, but treat them as profile-only until the paper frontier is beaten.
- Keep Irvine `3s`, `7s`, `10s`, and `15s` as implementation targets for the stricter paper-facing frontier.
- After acceptance, rerun warmed publishable lower-end rows on quiet hosts and refresh `competitive_frontier`.
