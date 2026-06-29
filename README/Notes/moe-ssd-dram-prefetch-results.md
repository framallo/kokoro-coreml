# MoE SSD/DRAM Prefetch Results

**First spotted:** 2026-06-29
**Status:** Active

## Summary

This note records the staged results for
[MoE SSD/DRAM expert prefetch experiment plan](../Plans/moe-ssd-dram-prefetch-v1.md).
Each stage must end with a written go/kill decision before the next stage
starts. A kill decision is a valid result.

## Related Guides

- [MoE expert offload and prefetch prior art](../Guides/moe-expert-offload-prefetch-prior-art-guide.md)
- [Apple Silicon NVMe and energy measurement](../Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
- [MoE SSD/DRAM guide triage](moe-ssd-dram-prefetch-guide-triage-2026-06-29.md)

## Stage 0: Hardware Envelope

**Status:** Complete.

### Frozen Assumptions

- Model inventory: `outputs/moe_prefetch/stage0/model_inventory.json`
- Gate thresholds: `outputs/moe_prefetch/stage0/thresholds.json`
- Model: `mistralai/Mixtral-8x7B-v0.1`
- Expert bytes: `88080384`
- Active expert bytes/token: `5637144576`
- Target decode rate: `1.0` token/sec

### Measurements

| Pattern | Bandwidth GB/s | p50 latency ns | p95 latency ns | fs_usage proof |
| --- | ---: | ---: | ---: | --- |
| sequential | 6.055 | 13830000 | 17113000 | False |
| random | 6.105 | 13833000 | 16681000 | False |

Oracle bandwidth ceiling: `1.082934` tokens/sec.

### Privileged Capture Status

- `fs_usage`: missing for at least one accepted read cell. This run is not
  valid SSD proof.
- `powermetrics`: captured at `outputs/moe_prefetch/stage0/powermetrics_stage0.plist`

### Decision

KILL: missing fs_usage disk I/O proof for accepted measurements.

### Valid Rerun Path

Run Stage 0 from a terminal that can accept a sudo password prompt:

```bash
sudo -v
python scripts/moe_prefetch/run_stage0_envelope.py \
  --thresholds outputs/moe_prefetch/stage0/thresholds.json \
  --output-dir outputs/moe_prefetch/stage0 \
  --fs-usage-sudo-mode interactive \
  --capture-powermetrics
python scripts/moe_prefetch/summarize.py stage0 \
  --input outputs/moe_prefetch/stage0/results.json \
  --notes README/Notes/moe-ssd-dram-prefetch-results.md
```

Do not proceed to Stage 1 unless the summary reports `fs_usage proof` as true
for every accepted read cell.

## Stage 1: Router Trace and Predictor Replay

**Status:** Not run. Stage 0 killed before router tracing because the session
could not produce required `fs_usage` disk I/O proof.

## Stage 2: Offline Simulator

**Status:** Not run. Stage 1 did not start.

## Stage 3: Runtime Harness

**Status:** Not run. Stage 2 did not start.

## Required Executable Memory

Regression test:

```bash
python3 -m pytest tests/test_moe_prefetch_tools.py
```
