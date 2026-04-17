# Bakeoff Results v1

April 15, 2026

## Scope

This archived note consolidates the pre-v9 corrected bakeoff results for the
`3s` / `7s` / `15s` / `30s` inputs across:

- Apple M2 Ultra, 64 GB
- Apple M2 MacBook Air, 24 GB
- Apple M1 Mac Mini, 16 GB

The `60s` input was skipped and is not included. Earlier v2-v4 benchmark
sections in [performance-notes](performance-notes.md) are useful history, but
the v5 sections are the canonical corrected data for this archived comparison.
For the current Config F host-materialization fix, use
[Bakeoff Results v2](bakeoff-results-v2.md).

## Configs

| Config | Plain meaning |
| --- | --- |
| A | Existing Python HAR-post hybrid: Python/PyTorch prefix plus Core ML decoder |
| D | PyTorch end-to-end on MPS, with CPU fallback enabled |
| E | PyTorch end-to-end on CPU |
| F | Swift + Core ML pipeline with Swift hn-nsf DSP |

## Headline Findings

1. **Config F wins on every machine and every measured duration.** It beats the
   existing Python HAR-post path (Config A), MPS (Config D), and CPU (Config E)
   wherever final corrected data exists.

2. **The MPS gap is the strongest result.** Config F is 1.8-3.4x faster than
   MPS across the final machine matrix. This is the cleanest comparison against
   the "just use the GPU" baseline.

3. **M1 Mini is viable.** The lowest-spec machine tested completes 30s of audio
   in 1.229s with Config F, about 22x realtime.

4. **M2 Ultra is fastest in absolute terms.** Config F completes 30s of audio
   in 422 ms, about 70x realtime.

5. **Config A remains competitive on high-end and ANE-bound cases.** Config F
   beats Config A everywhere, but the margin narrows to 1.1x at 15s on M2 Ultra
   and 1.1x at 7s on M1 Mini.

## Config F Across Machines

Warm median end-to-end wall time:

| Input | Audio | M2 Ultra | M2 Air | M1 Mini |
| --- | --- | --- | --- | --- |
| 3s | 2.80s | **59 ms** | 200 ms | 157 ms |
| 7s | 6.75s | **136 ms** | 326 ms | 511 ms |
| 15s | 13.90s | **278 ms** | 783 ms | 691 ms |
| 30s | 27.38s | **422 ms** | 1829 ms | 1229 ms |

Realtime factor for Config F:

| Input | M2 Ultra | M2 Air | M1 Mini |
| --- | --- | --- | --- |
| 3s | 48x RT | 14x RT | 18x RT |
| 7s | 50x RT | 21x RT | 13x RT |
| 15s | 50x RT | 18x RT | 20x RT |
| 30s | 70x RT | 15x RT | 22x RT |

## Full Wall-Time Matrix

Warm median end-to-end wall time, milliseconds.

### M2 Ultra

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 117 ms | 176 ms | 255 ms | **59 ms** |
| 7s | 6.75s | 179 ms | 319 ms | 501 ms | **136 ms** |
| 15s | 13.90s | 309 ms | 602 ms | 975 ms | **278 ms** |
| 30s | 27.38s | 555 ms | 1233 ms | 1870 ms | **422 ms** |

### M2 Air

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 355 ms | 394 ms | 736 ms | **200 ms** |
| 7s | 6.75s | 544 ms | 812 ms | 1985 ms | **326 ms** |
| 15s | 13.90s | 1178 ms | 1573 ms | 4002 ms | **783 ms** |
| 30s | 27.38s | 2443 ms | 3350 ms | 8065 ms | **1829 ms** |

### M1 Mini

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | --- | --- | --- | --- | --- |
| 3s | 2.80s | 238 ms | 492 ms | 894 ms | **157 ms** |
| 7s | 6.75s | 577 ms | 1038 ms | 2233 ms | **511 ms** |
| 15s | 13.90s | 837 ms | 1958 ms | 4458 ms | **691 ms** |
| 30s | 27.38s | 1637 ms | 4167 ms | 8934 ms | **1229 ms** |

## Config F Speedups

### F vs A: Swift vs Existing Python HAR-Post

| Input | M2 Ultra | M2 Air | M1 Mini |
| --- | --- | --- | --- |
| 3s | 2.0x | 1.8x | 1.5x |
| 7s | 1.3x | 1.7x | 1.1x |
| 15s | 1.1x | 1.5x | 1.2x |
| 30s | 1.3x | 1.3x | 1.3x |

### F vs D: Swift vs PyTorch MPS

| Input | M2 Ultra | M2 Air | M1 Mini |
| --- | --- | --- | --- |
| 3s | 3.0x | 2.0x | 3.1x |
| 7s | 2.3x | 2.5x | 2.0x |
| 15s | 2.2x | 2.0x | 2.8x |
| 30s | 2.9x | 1.8x | 3.4x |

### F vs E: Swift vs PyTorch CPU

| Input | M2 Ultra | M2 Air | M1 Mini |
| --- | --- | --- | --- |
| 3s | 4.3x | 3.7x | 5.7x |
| 7s | 3.7x | 6.1x | 4.4x |
| 15s | 3.5x | 5.1x | 6.4x |
| 30s | 4.4x | 4.4x | 7.3x |

## Cross-Machine Scaling

Config F relative to M2 Ultra wall time:

| Input | M2 Ultra | M2 Air | M1 Mini |
| --- | --- | --- | --- |
| 3s | 1.0x | 3.4x slower | 2.7x slower |
| 7s | 1.0x | 2.4x slower | 3.8x slower |
| 15s | 1.0x | 2.8x slower | 2.5x slower |
| 30s | 1.0x | 4.3x slower | 2.9x slower |

The M1 Mini is slower than M2 Ultra, but it does not collapse. Its Config F
30s result is faster than the M2 Air 30s result in this run, despite having
less memory. Treat that as a measured result, not a general hardware ranking:
thermal state, macOS version, model cache state, and memory pressure can all
affect Core ML scheduling.

## Caveats

- The final comparison uses corrected v5 data only.
- The `60s` input was skipped.
- M1 Mini Config A ran separately because loading A alongside D/E exceeded the
  16 GB harness memory budget. The production app should load one bucket at a
  time.
- M2 Air Config D ran separately because the combined run hit MPS memory
  pressure.
- Result JSON files live under `outputs/bakeoff/`, which is ignored by git.

## Provenance

| Machine | Result files |
| --- | --- |
| M2 Ultra | `outputs/bakeoff/results_m2_ultra_v5.json` |
| M2 Air | `outputs/bakeoff/results_m2_air_v5.json`, `outputs/bakeoff/results_m2_air_v5_mps.json` |
| M1 Mini | `outputs/bakeoff/results_m1_mini_def.json`, `outputs/bakeoff/results_m1_mini_a.json` |

Source note: [performance-notes](performance-notes.md).
