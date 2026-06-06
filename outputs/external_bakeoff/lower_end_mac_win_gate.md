# Lower-End Mac Win Gate

Warmed inference only. This report tracks the remaining `m2-air` and
`irvine-m1` gates needed before claiming we beat the external Apple
Silicon implementations on lower-end Macs.

Pending listening wins: `5`.
Accepted wins: `0`.
Blocked rows: `1`.

## Candidate Gates

| Machine | Bucket | Candidate | Gate | Human | Current Config F | Competitor | Projected | Margin | Quality |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| m2-air | `3s` | `3s_natural_asr_cos_rsqrt` | human-listening | blank | 148.0 ms | laishere: 142.0 ms | 130.8 ms | +11.2 ms | corr 0.814, SNR 5.08 dB |
| m2-air | `3s` | `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in` | human-listening | blank | 148.0 ms | laishere: 142.0 ms | 137.9 ms | +4.1 ms | corr 0.932, SNR 9.19 dB |
| irvine-m1 | `3s` | `3s_natural_asr_cos_rsqrt` | implementation-gap | blank | 233.6 ms | laishere: 195.0 ms | 214.8 ms | -19.8 ms | corr 0.814, SNR 5.08 dB |
| irvine-m1 | `7s` | `7s_natural_asr_cos_rsqrt` | human-listening | blank | 492.7 ms | laishere: 444.2 ms | 444.1 ms | +0.1 ms | corr 0.797, SNR 4.77 dB |
| irvine-m1 | `10s` | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | human-listening | blank | 685.5 ms | laishere: 644.9 ms | 608.7 ms | +36.2 ms | corr 0.867, SNR 6.55 dB |
| irvine-m1 | `15s` | `15s_padded_cos_resblock_cos_rsqrt` | human-listening | blank | 1014.9 ms | laishere: 990.6 ms | 936.1 ms | +54.5 ms | corr 0.957, SNR 11.00 dB |

## Decision

Lower-end Mac wins are available through warmed source/body candidates, but they are not paper-claimable until human listening accepts the rows. Irvine 3s still needs a new implementation change; the best saved source/body candidate does not beat laishere there.

## Immediate Work

- Get no-ASR human listening decisions for the M2 Air `3s` candidates.
- Get no-ASR human listening decisions for Irvine `7s`, `10s`, and `15s` source/body candidates.
- Keep Irvine `3s` as an implementation target; current saved candidates do not beat laishere.
- After acceptance, rerun warmed publishable lower-end rows on quiet hosts and refresh `competitive_frontier`.
