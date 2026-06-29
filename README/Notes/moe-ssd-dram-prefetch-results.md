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
| sequential | 5.929 | 14167000 | 17164000 | True |
| random | 5.962 | 14241000 | 16671000 | True |

Oracle bandwidth ceiling: `1.057558` tokens/sec.
One-layer compute p50: `1.773188` ms (`outputs/moe_prefetch/stage0/compute.json`).
Hideability ratio (max read p95 / compute p50): `9.679741`.

### Privileged Capture Status

- `fs_usage`: present for every accepted read cell.
- `powermetrics`: captured at `outputs/moe_prefetch/stage0/powermetrics_stage0.plist`

### Decision

FLAG: bandwidth passes but one-layer lead time is insufficient.

## Stage 1: Router Trace and Predictor Replay

**Status:** Not run. Stage 0 now has valid `fs_usage` disk I/O proof and clears
the provisional oracle bandwidth floor, but it flags one-layer lead time as
insufficient. Stage 1 must measure whether multi-layer-ahead prediction can
produce hideable recall that beats the trivial baselines.

## Stage 2: Offline Simulator

**Status:** Not run. Stage 1 did not start.

## Stage 3: Runtime Harness

**Status:** Not run. Stage 2 did not start.

## Required Executable Memory

Regression test:

```bash
python3 -m pytest tests/test_moe_prefetch_tools.py
```
