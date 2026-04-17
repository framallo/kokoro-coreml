# Bakeoff Results v2

April 17, 2026

## Scope

This note records the current corrected bakeoff series after the Config F
host-materialization fix and supersedes the archived v1 comparison for current
performance claims. It is intentionally conservative: only the M2 Ultra numbers
are populated here because they come from the completed controlled run on the
working tree used during this session. The result JSON records `git_dirty: true`
because it was collected before the final cleanup commit; the M2 Air and M1 Mini
sections are placeholders until those machines rerun the same setup and harness.

The useful rule from the latest debugging pass is simple: measure the deployed
pipeline boundary, not an attractive subgraph. Config F wins on M2 Ultra only
after the Swift hot path stopped doing accidental host work around the Core ML
models.

## Configs

| Config | Plain meaning |
| --- | --- |
| A | Existing Python HAR-post hybrid: PyTorch prefix plus Core ML HAR-post decoder |
| D | PyTorch end-to-end on MPS, with CPU fallback enabled |
| E | PyTorch end-to-end on CPU |
| F | Swift + Core ML pipeline with Swift hn-nsf DSP |

## Headline Result

On M2 Ultra, Config F is now the fastest measured path at every canonical input
length. It beats the fixed Config A HAR-post path by `1.8-5.9x`, PyTorch MPS by
`2.8-4.0x`, and PyTorch CPU by `5.7-7.2x`.

The reason is not a new model. The graph was already good enough. The final
performance fix removed two host-side mistakes:

- sparse one-hot alignment materialization plus dense matmul through zeros
- boxed `MLMultiArray` reads from strided `Float16` waveform output during trim

The current rule for future work is: keep the model split simple, keep dynamic
setup on the host only when it is cheap, and prove the full wall-clock path
with audio-quality gates before ranking configurations.

## M2 Ultra

**Machine:** Apple M2 Ultra Mac Studio, 64 GB
**Status:** Complete
**Result file:** `outputs/bakeoff/results_m2_ultra_parity_final_20260417.json`

### Wall Time

Warm median end-to-end wall time, milliseconds.

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | 333 ms | 225 ms | 409 ms | **57 ms** |
| 7s | 6.75s | 329 ms | 412 ms | 811 ms | **124 ms** |
| 15s | 13.90s | 486 ms | 673 ms | 1467 ms | **239 ms** |
| 30s | 27.38s | 870 ms | 1602 ms | 2714 ms | **476 ms** |

### Realtime Factor

Lower is better.

| Input | A RTF | D RTF | E RTF | F RTF |
| --- | ---: | ---: | ---: | ---: |
| 3s | 0.119 | 0.080 | 0.146 | **0.020** |
| 7s | 0.049 | 0.061 | 0.120 | **0.018** |
| 15s | 0.035 | 0.048 | 0.106 | **0.017** |
| 30s | 0.032 | 0.059 | 0.099 | **0.017** |

### Config F Speedups

| Input | F vs A | F vs D | F vs E |
| --- | ---: | ---: | ---: |
| 3s | 5.9x | 4.0x | 7.2x |
| 7s | 2.7x | 3.3x | 6.5x |
| 15s | 2.0x | 2.8x | 6.1x |
| 30s | 1.8x | 3.4x | 5.7x |

### Config F Stage Medians

| Input | Duration | F0Ntrain | DecoderPre | Matrix | hn-sf | Trim | Core ML total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 10.0 ms | 4.4 ms | 2.8 ms | 0.1 ms | 9.3 ms | 0.2 ms | 28.5 ms |
| 7s | 14.3 ms | 18.9 ms | 8.3 ms | 0.3 ms | 23.1 ms | 0.4 ms | 56.6 ms |
| 15s | 28.8 ms | 38.5 ms | 9.7 ms | 0.7 ms | 46.9 ms | 0.7 ms | 111.6 ms |
| 30s | 52.1 ms | 76.8 ms | 16.6 ms | 1.4 ms | 99.6 ms | 1.6 ms | 224.7 ms |

### Audio Gate

Config F listen samples passed the waveform health gate and remain available at:

- `outputs/bakeoff/listen/config_f_3s.wav`
- `outputs/bakeoff/listen/config_f_7s.wav`
- `outputs/bakeoff/listen/config_f_15s.wav`
- `outputs/bakeoff/listen/config_f_30s.wav`

## M2 Air

**Machine:** Apple M2 MacBook Air
**Status:** Pending rerun
**Result file:** TBD

### Wall Time

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | TBD | TBD | TBD | TBD |
| 7s | 6.75s | TBD | TBD | TBD | TBD |
| 15s | 13.90s | TBD | TBD | TBD | TBD |
| 30s | 27.38s | TBD | TBD | TBD | TBD |

## M1 Mini

**Machine:** Apple M1 Mac Mini
**Status:** Pending rerun
**Result file:** TBD

### Wall Time

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | TBD | TBD | TBD | TBD |
| 7s | 6.75s | TBD | TBD | TBD | TBD |
| 15s | 13.90s | TBD | TBD | TBD | TBD |
| 30s | 27.38s | TBD | TBD | TBD | TBD |

## Cross-Machine Comparison

Pending. Do not reuse older M2 Air or M1 Mini tables in this v2 ledger. Those
machines must run the current setup script and current Swift binary because the
result depends on exact Duration packages, direct alignment expansion, and
stride-aware `Float16` waveform extraction.

## Provenance

M2 Ultra command used for the recorded run:

```bash
BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs a,d,e,f \
  --iterations 5 \
  --order-seed 0 \
  --machine-id m2_ultra_parity_final_20260417
```

Recommended command after `scripts/setup_bakeoff.sh` on a fresh machine:

```bash
BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run --no-sync python scripts/bakeoff_harness.py run \
  --configs a,d,e,f \
  --iterations 5 \
  --order-seed 0 \
  --machine-id <machine_id>
```

Related notes:

- [performance-notes.md](performance-notes.md)
- [debug-notes.md](debug-notes.md)
