# MLX Speed Explanation

## Verdict

MLX is not faster than corrected warmed Config F on any full-duration Mac row. Apparent MLX wins come from stale/raw Config F files that include Core ML compile/cache behavior or older unpromoted runtime artifacts.

- Corrected warmed full-duration MLX wins: `0`.
- Corrected warmed full-duration Config F wins over MLX: `12`.
- MLX error rows: `3`.
- Raw/stale apparent MLX wins: `11`.

## Corrected Warmed Frontier

These rows use the same paper-facing source set as `competitive_frontier`: corrected warmed Config F rows, full-duration MLX rows only, and no Core ML compile/cache timing.

| Machine | Bucket | Config F | MLX | MLX duration ratio | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| irvine-m1 | 3s | 233.6 ms | n/a | n/a | mlx-error |
| irvine-m1 | 7s | 492.7 ms | 824.0 ms | 1.000 | config-faster |
| irvine-m1 | 10s | 685.5 ms | 1124.3 ms | 0.997 | config-faster |
| irvine-m1 | 15s | 1014.9 ms | 1589.5 ms | 1.000 | config-faster |
| irvine-m1 | 30s | 1959.4 ms | 3077.9 ms | 1.000 | config-faster |
| m2-air | 3s | 148.0 ms | n/a | n/a | mlx-error |
| m2-air | 7s | 330.7 ms | 685.6 ms | 1.000 | config-faster |
| m2-air | 10s | 466.0 ms | 835.8 ms | 0.997 | config-faster |
| m2-air | 15s | 693.6 ms | 1521.0 ms | 1.000 | config-faster |
| m2-air | 30s | 1404.8 ms | 2600.3 ms | 1.000 | config-faster |
| m2-studio | 3s | 50.7 ms | n/a | n/a | mlx-error |
| m2-studio | 7s | 95.5 ms | 223.9 ms | 1.000 | config-faster |
| m2-studio | 10s | 125.7 ms | 288.8 ms | 0.997 | config-faster |
| m2-studio | 15s | 185.8 ms | 376.3 ms | 1.000 | config-faster |
| m2-studio | 30s | 383.9 ms | 762.7 ms | 1.000 | config-faster |

## Apparent MLX Wins Against Raw Config F

These rows explain the confusing view: MLX can beat early/raw Config F JSON files, but those files are not the corrected warmed frontier.

| Machine | Bucket | Raw Config F | MLX | Apparent MLX save | Config duration ratio | MLX duration ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m2-studio | 7s | 284.2 ms | 223.9 ms | 60.2 ms | 1.000 | 1.000 |
| m2-studio | 10s | 548.2 ms | 288.8 ms | 259.4 ms | 1.000 | 0.997 |
| m2-studio | 15s | 632.9 ms | 376.3 ms | 256.6 ms | 1.000 | 1.000 |
| m2-studio | 30s | 1191.8 ms | 762.7 ms | 429.1 ms | 1.001 | 1.000 |
| m2-air | 7s | 808.1 ms | 685.6 ms | 122.4 ms | 1.000 | 1.000 |
| m2-air | 10s | 1373.3 ms | 835.8 ms | 537.5 ms | 1.000 | 0.997 |
| m2-air | 15s | 2052.4 ms | 1521.0 ms | 531.4 ms | 1.000 | 1.000 |
| m2-air | 30s | 9559.1 ms | 2600.3 ms | 6958.8 ms | 1.001 | 1.000 |
| irvine-m1 | 10s | 1348.5 ms | 1124.3 ms | 224.2 ms | 1.000 | 0.997 |
| irvine-m1 | 15s | 1672.7 ms | 1589.5 ms | 83.2 ms | 1.000 | 1.000 |
| irvine-m1 | 30s | 16119.6 ms | 3077.9 ms | 13041.7 ms | 1.001 | 1.000 |

## Why This Happens

- MLX uses a dynamic GPU/Metal runtime and avoids Core ML's expensive `.mlpackage` compile/specialization path during warm calls.
- Early Config F result files captured stale artifacts and/or Core ML compile/cache behavior; the corrected frontier uses targeted warmed reruns such as `*_vector_noise_batch.json`.
- MLX has no valid 3s full-duration row: all Mac 3s MLX cells fail with the same broadcast-shape error.
- On the corrected warmed rows where MLX produces full-duration audio, Config F is faster than MLX on every Mac bucket.
- The real current Core ML competitor on lower-end Macs is laishere, not MLX; laishere's source/body boundary is the architecture clue to reuse.

## Reusable Lesson

Never compare a dynamic runtime's warm loop to a Core ML row unless the Core ML row has explicitly discarded compile/cache behavior and uses the same output-duration eligibility gate.
