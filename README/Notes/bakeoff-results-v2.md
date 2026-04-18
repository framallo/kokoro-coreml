# Bakeoff Results v2

April 17, 2026

## Scope

This note records the current corrected bakeoff series after the Config F
host-materialization fix and supersedes the archived v1 comparison for current
performance claims. It is intentionally conservative: M2 Ultra, M2 Air, and M1
Mini numbers are populated here because each set comes from a completed
controlled run on this branch. The result JSON records `git_dirty: true` because
it was collected before the final cleanup commit.
A later audit refactor moved the timed Swift synthesis orchestration into the
shared pipeline library. That refactor was verified with the F-only smoke result
`outputs/bakeoff/results_shared_executor_smoke_20260417.json`; it does not
replace the full A/D/E/F medians below.

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
**Shared-executor smoke:** `outputs/bakeoff/results_shared_executor_smoke_20260417.json`

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

**Machine:** Apple M2 MacBook Air, 24 GB, macOS 15.7.5
**Status:** Complete (Config D partial; OOM at 15s and 30s)
**Result files:**
- `outputs/bakeoff/results_m2_air_v6.json` (A, E, F; 20/20 ok each)
- `outputs/bakeoff/results_m2_air_v6_mps.json` (D solo pass; 3s/7s ok,
  15s/30s MPS OOM)

Collected on commit `fa2a24d` (plus the `export_synth/wrappers.py`
idempotent-wrap fix landed in the same commit series) after a full
`setup_bakeoff.sh --skip-download` re-export of every Duration,
F0Ntrain, DecoderPre, and GeneratorFromHar package. Swift binary
rebuilt from current sources; same harness, same 5 iterations, same
order seed 0.

### Wall Time

Warm median end-to-end wall time, milliseconds.

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | 461 ms | 739 ms | 723 ms | **185 ms** |
| 7s | 6.75s | 771 ms | 907 ms | 1839 ms | **396 ms** |
| 15s | 13.90s | 1896 ms | OOM | 3737 ms | **1326 ms** |
| 30s | 27.38s | 3918 ms | OOM | 7567 ms | **3021 ms** |

### Realtime Factor

Lower is better.

| Input | A RTF | D RTF | E RTF | F RTF |
| --- | ---: | ---: | ---: | ---: |
| 3s | 0.165 | 0.264 | 0.258 | **0.066** |
| 7s | 0.114 | 0.134 | 0.272 | **0.059** |
| 15s | 0.136 | OOM | 0.269 | **0.095** |
| 30s | 0.143 | OOM | 0.276 | **0.110** |

### Config F Speedups

| Input | F vs A | F vs D | F vs E |
| --- | ---: | ---: | ---: |
| 3s | 2.5x | 4.0x | 3.9x |
| 7s | 1.9x | 2.3x | 4.6x |
| 15s | 1.4x | OOM | 2.8x |
| 30s | 1.3x | OOM | 2.5x |

### Config F Stage Medians

| Input | Duration | F0Ntrain | DecoderPre | Matrix | hn-sf | Trim | Core ML total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s | 10.6 ms | 3.3 ms | 2.6 ms | 0.1 ms | 9.0 ms | 0.2 ms | 175.6 ms |
| 7s | 12.8 ms | 7.5 ms | 4.9 ms | 0.3 ms | 21.3 ms | 0.4 ms | 374.1 ms |
| 15s | 39.6 ms | 35.1 ms | 13.0 ms | 0.5 ms | 46.9 ms | 0.8 ms | 1276.1 ms |
| 30s | 48.5 ms | 68.7 ms | 28.4 ms | 1.0 ms | 95.7 ms | 1.7 ms | 2925.7 ms |

GeneratorFromHar dominates on M2 Air even harder than M2 Ultra (86% at
3s, 92% at 30s of Config F wall time). This is the optimization target
for M2 Air.

### Notes

- Config D OOM at 15s/30s even in a solo pass. MPS pool cap on 24 GB
  M2 Air is ~27 GB, and the kokoro pipeline plus prior MPS
  allocations exceeded that for longer buckets. The production app
  should never route to MPS on this hardware.
- Config F regressed relative to the prior v5 M2 Air numbers (v5: F at
  200/326/783/1829 ms) by roughly `+60-70%` at 15s/30s, driven by a
  `~2x` slowdown in GeneratorFromHar. Candidate causes not yet
  isolated: `torch==2.5.0` in the current
  `requirements-bakeoff.txt` vs `torch==2.6.0` in the v5 provenance;
  thermal state after back-to-back exports; variance in CoreML ANE
  plan compilation across fresh `.mlpackage` directories.
- The `export_synth/wrappers.py` fix was required to produce fresh
  GeneratorFromHar packages at all — without it, the second-stage
  `SynthesizerModel(kmodel)` raised
  `AttributeError: 'MaskedBidirectionalLSTM' object has no attribute
  'num_layers'` when wrapping an already-masked `text_encoder.lstm`
  from `DurationModel(kmodel)`.

## M1 Mini

**Machine:** Apple M1 Mac Mini
**Status:** Complete
**Result files:**
`outputs/bakeoff/results_m1_mini_a.json` and
`outputs/bakeoff/results_m1_mini_def.json`

### Wall Time

| Input | Audio | A Python HAR | D MPS | E CPU | F Swift |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | 2.80s | 237.9 ms | 491.5 ms | 893.9 ms | **156.8 ms** |
| 7s | 6.75s | 577.1 ms | 1038.2 ms | 2232.8 ms | **510.8 ms** |
| 15s | 13.90s | 836.6 ms | 1958.3 ms | 4457.7 ms | **691.5 ms** |
| 30s | 27.38s | 1636.9 ms | 4166.7 ms | 8934.2 ms | **1228.9 ms** |

### Realtime Factor

Lower is better.

| Input | A RTF | D RTF | E RTF | F RTF |
| --- | ---: | ---: | ---: | ---: |
| 3s | 0.0850 | 0.1755 | 0.3192 | **0.0560** |
| 7s | 0.0855 | 0.1538 | 0.3308 | **0.0757** |
| 15s | 0.0602 | 0.1409 | 0.3207 | **0.0497** |
| 30s | 0.0598 | 0.1522 | 0.3264 | **0.0449** |

### Config F Speedups

| Input | F vs A | F vs MPS |
| --- | ---: | ---: |
| 3s | 1.5x | 3.1x |
| 7s | 1.1x | 2.0x |
| 15s | 1.2x | 2.8x |
| 30s | 1.3x | 3.4x |

## Cross-Machine Comparison

Cross-machine takeaway (completed runs):

- M2 Ultra still leads for this suite and remains the top speed point.
- M2 Air has complete A/E/F medians; Config D is partial (MPS OOM at 15s/30s on
  24 GB). Ordering where measured matches M2 Ultra: `F > A > D > E` for the
  shorter inputs.
- M1 Mini now has complete A/D/E/F medians and follows the same ordering: `F > A > D > E`.
- Do not reuse pre-2026 M2 Air or M1 Mini tables, because output depends on exact
  Duration packages, direct alignment expansion, and stride-aware `Float16`
  waveform extraction.

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
