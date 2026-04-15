---
name: bakeoff
description: >-
  Run the controlled bakeoff benchmark on this machine. Prepares inputs,
  builds the Swift pipeline, runs the counterbalanced harness for all
  available configs (A, D, E, F), records results, and updates
  performance-notes.md. Use when the user says "run the bakeoff",
  "benchmark this machine", or invokes $bakeoff.
---

# Bakeoff

## Purpose

Run the full bakeoff benchmark suite on the current machine and record
publication-grade results. This is the one-command path from "I have a
machine" to "results are in performance-notes.md."

## Use When

- The user wants to benchmark the current machine.
- A new machine is available and needs bakeoff data.
- The user says "run the bakeoff", "benchmark", or invokes `$bakeoff`.

## Do Not Use When

- The user only wants to run a single quick test (use the harness directly).
- The user wants to create or modify the bakeoff plan (use `create-plan`).
- The user wants to audit existing results (use `audit`).

## Prerequisites

Everything is handled by the setup script. If this is a fresh machine
(or you're not sure), run it first:

```bash
bash scripts/setup_bakeoff.sh
```

This takes ~10 minutes and handles: Python deps, model downloads,
all model exports (Duration [32-512], F0Ntrain, DecoderPre,
GeneratorFromHar for buckets [3,7,10,15,30]s), Swift binary build,
and benchmark input preparation. Use `--skip-download` if models
are already local.

If setup has already been run, verify quickly:

```bash
ls swift/.build/release/kokoro-bench && \
ls outputs/bakeoff/input_manifest.json && \
ls outputs/swift_bench_inputs/hnsf_weights.json && \
echo "Ready"
```

## Procedure

### 1. Run setup (if needed)

```bash
bash scripts/setup_bakeoff.sh          # full setup (~10 min)
bash scripts/setup_bakeoff.sh --skip-download  # skip HF download
```

### 2. Identify the machine

Record:
- Chip (e.g., M1 Mini, M2 Ultra, M2 MacBook Air)
- RAM
- macOS version
- Any relevant config (plugged in vs battery, thermal state)

Choose a `--machine-id` slug: `m1_mini`, `m2_ultra`, `m2_air`, etc.

### 3. Run the bakeoff

Config D (MPS) requires `PYTORCH_ENABLE_MPS_FALLBACK=1`. Set it for
all runs to keep the command uniform.

```bash
BAKEOFF_SKIP_SMOKE=1 \
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python scripts/bakeoff_harness.py run \
  --configs a,d,e,f \
  --iterations 5 \
  --order-seed 0 \
  --machine-id <machine_id>
```

Expected runtime: 10–20 minutes depending on machine speed.

### 4. Verify results

```bash
python3 -c "
import json, statistics
from collections import defaultdict

data = json.load(open('outputs/bakeoff/results_<machine_id>.json'))
results = [r for r in data['results'] if r.get('status') == 'ok']
groups = defaultdict(list)
for r in results:
    groups[(r['config'], r['input_key'])].append(r['wall_time_s'] * 1000)

for ik in ['tiny', 'short', 'medium', 'long']:
    row = {c: statistics.median(groups.get((c, ik), [0])) for c in ['a','d','e','f']}
    print(f'{ik:8s}  A={row[\"a\"]:.0f}ms  D={row[\"d\"]:.0f}ms  E={row[\"e\"]:.0f}ms  F={row[\"f\"]:.0f}ms')
"
```

All configs should show `status: ok` for all inputs. Config D may show
`config_unavailable` if MPS fallback wasn't set — that's acceptable.

### 5. Update performance-notes.md

Add a new section to `README/Notes/performance-notes.md` following the
existing pattern (see "Bakeoff v2: Controlled benchmark on M2 MacBook Air"
or "Bakeoff v3: Swift pipeline" for the template). Include:

- Machine identification and provenance
- End-to-end wall time table (warm median, ms)
- RTF table
- Speedup: F vs A, F vs E
- Cross-machine comparison table (if prior machine data exists)
- Interpretation (2-4 bullet points)

### 6. Commit and push

Use `git-commit` to stage the results file and performance-notes changes.
Then `git-push` to sync with origin.

## Configs Reference

| Config | What it measures | Runtime |
| --- | --- | --- |
| A | Shipping Python HAR-post hybrid (PyTorch prefix + CoreML decoder) | ~2 min |
| D | PyTorch end-to-end on MPS (GPU with CPU fallback) | ~2 min |
| E | PyTorch end-to-end on CPU | ~3 min |
| F | Swift + CoreML pipeline (5 models + Swift hn-nsf DSP) | ~1 min |

Configs B/C (decoder-only) are omitted by default — they measure ANE
participation, not pipeline speed. Add `b,c` to `--configs` if needed.

## Canonical Docs

- Bakeoff plan: `README/Plans/kokoro-bakeoff-v2.md`
- Swift pipeline plan: `README/Plans/swift-prefix-rewrite-v1.md`
- Harness: `scripts/bakeoff_harness.py`
- Swift CLI: `swift/Sources/KokoroBenchmark/main.swift`
- Performance notes: `README/Notes/performance-notes.md`
