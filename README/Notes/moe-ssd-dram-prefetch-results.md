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

**Status:** Assumptions frozen; hardware envelope not measured yet.

### Frozen Assumptions

- Model inventory: `outputs/moe_prefetch/stage0/model_inventory.json`
- Gate thresholds: `outputs/moe_prefetch/stage0/thresholds.json`

Current inventory was generated on 2026-06-29 with:

```bash
python3 scripts/moe_prefetch/model_inventory.py \
  --model-id mistralai/Mixtral-8x7B-v0.1 \
  --quantization-bits 4 \
  --active-experts-per-token 64 \
  --target-tokens-per-second 1.0 \
  --expert-parameters 176160768 \
  --target-device "Apple Silicon UMA local" \
  --estimate-source "Mixtral per-layer FFN expert parameter estimate: 3 * hidden_size 4096 * intermediate_size 14336" \
  --notes "Initial Stage 0 comparability target from original spec; revise only before running Stage 0, not after seeing measurements." \
  --output outputs/moe_prefetch/stage0/model_inventory.json
```

Recorded values:

| Field | Value |
| --- | ---: |
| Expert parameters | 176,160,768 |
| Quantization | 4-bit |
| Expert bytes | 88,080,384 |
| Active experts/token | 64 |
| Active expert bytes/token | 5,637,144,576 |
| Target decode rate | 1.0 token/sec |
| Speed win threshold | 25% |
| Trivial baseline margin | 10% |
| Energy target | Demand-paging baseline |

The local machine recorded in the inventory is an Apple M2 Ultra Mac Studio
(`Mac14,14`) with 64 GiB RAM, running macOS 26.5.1.

### Required Evidence

- Cold sequential expert-block bandwidth.
- Cold random expert-block bandwidth.
- p50/p95 expert-block latency.
- One-layer compute-time budget.
- Hideability ratio.
- Oracle bandwidth ceiling.
- `fs_usage -f diskio` proof for accepted read measurements.

### Decision

Pending.

## Stage 1: Router Trace and Predictor Replay

**Status:** Blocked until Stage 0 passes.

## Stage 2: Offline Simulator

**Status:** Blocked until Stage 1 passes.

## Stage 3: Runtime Harness

**Status:** Blocked until Stage 2 passes.

## Required Executable Memory

Regression test:

```bash
python3 -m pytest tests/test_moe_prefetch_tools.py
```
