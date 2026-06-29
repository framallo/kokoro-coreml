#!/usr/bin/env python3
"""Summarize MoE prefetch experiment stages and update notes."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.moe_prefetch.schema import load_json, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    stage0 = sub.add_parser("stage0")
    stage0.add_argument("--input", type=Path, required=True)
    stage0.add_argument("--notes", type=Path, required=True)
    stage0.add_argument("--output", type=Path, default=None)
    stage0.add_argument("--one-layer-compute-ms", type=float, default=None)
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _cell_summary(cell: dict[str, Any]) -> dict[str, Any]:
    measurement = cell.get("measurement") or {}
    latencies = [float(v) for v in measurement.get("latencies_ns", [])]
    total_bytes = float(measurement.get("total_bytes_read", 0))
    wall_ns = float(measurement.get("wall_time_ns", 0))
    bandwidth_gbps = (total_bytes / wall_ns) if wall_ns > 0 else 0.0
    # bytes/ns is numerically equal to GB/s when using decimal GB.
    return {
        "pattern": cell.get("pattern"),
        "returncode": cell.get("returncode"),
        "successful_reads": measurement.get("successful_reads", 0),
        "failed_reads": measurement.get("failed_reads", 0),
        "bandwidth_gbps": bandwidth_gbps,
        "latency_p50_ns": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_ns": _percentile(latencies, 0.95),
        "fs_usage_path": cell.get("fs_usage_path", ""),
        "fs_usage_error": cell.get("fs_usage_error", ""),
        "has_fs_usage": bool(cell.get("fs_usage_path")) and not cell.get("fs_usage_error"),
    }


def _replace_stage0_section(note_text: str, stage0_markdown: str) -> str:
    start = note_text.index("## Stage 0: Hardware Envelope")
    end = note_text.index("## Stage 1: Router Trace and Predictor Replay")
    return note_text[:start] + stage0_markdown.rstrip() + "\n\n" + note_text[end:]


def summarize_stage0(args: argparse.Namespace) -> int:
    payload = load_json(args.input)
    inventory = payload["inventory"]
    thresholds = payload["thresholds"]
    cells = [_cell_summary(cell) for cell in payload.get("cells", [])]
    usable = [cell for cell in cells if cell["returncode"] == 0 and cell["failed_reads"] == 0]
    random_cells = [cell for cell in usable if cell["pattern"] == "random"]
    ceiling_cell = random_cells[0] if random_cells else (usable[0] if usable else None)
    active_bytes = float(inventory["active_expert_bytes_per_token"])
    ceiling_tps = 0.0
    if ceiling_cell and ceiling_cell["bandwidth_gbps"] > 0:
        ceiling_tps = (ceiling_cell["bandwidth_gbps"] * 1_000_000_000.0) / active_bytes

    target_tps = float(inventory["target_tokens_per_second"])
    fs_usage_missing = any(not cell["has_fs_usage"] for cell in usable)
    bandwidth_kill = ceiling_tps < target_tps
    if fs_usage_missing:
        decision = "KILL: missing fs_usage disk I/O proof for accepted measurements."
    elif bandwidth_kill:
        decision = "KILL: oracle bandwidth ceiling is below target tokens/sec."
    elif args.one_layer_compute_ms is None:
        decision = "HOLD: bandwidth passes, but one-layer compute time is required for hideability."
    else:
        max_latency_ns = max((cell["latency_p95_ns"] for cell in usable), default=0.0)
        compute_ns = args.one_layer_compute_ms * 1_000_000.0
        hideability = max_latency_ns / compute_ns if compute_ns > 0 else 0.0
        decision = (
            "GO: bandwidth passes and hideability <= 1."
            if hideability <= 1.0
            else "FLAG: bandwidth passes but one-layer lead time is insufficient."
        )

    summary = {
        "inventory": inventory,
        "thresholds": thresholds,
        "config": payload.get("config", {}),
        "cells": cells,
        "oracle_bandwidth_ceiling_tokens_per_second": ceiling_tps,
        "target_tokens_per_second": target_tps,
        "decision": decision,
    }
    output = args.output or args.input.parent / "summary.json"
    write_json(output, summary)

    rows = "\n".join(
        "| {pattern} | {bandwidth_gbps:.3f} | {latency_p50_ns:.0f} | {latency_p95_ns:.0f} | {has_fs_usage} |".format(
            **cell
        )
        for cell in cells
    )
    stage0_markdown = f"""## Stage 0: Hardware Envelope

**Status:** Complete.

### Frozen Assumptions

- Model inventory: `outputs/moe_prefetch/stage0/model_inventory.json`
- Gate thresholds: `outputs/moe_prefetch/stage0/thresholds.json`
- Model: `{inventory["model_id"]}`
- Expert bytes: `{inventory["expert_bytes"]}`
- Active expert bytes/token: `{inventory["active_expert_bytes_per_token"]}`
- Target decode rate: `{target_tps}` token/sec

### Measurements

| Pattern | Bandwidth GB/s | p50 latency ns | p95 latency ns | fs_usage proof |
| --- | ---: | ---: | ---: | --- |
{rows}

Oracle bandwidth ceiling: `{ceiling_tps:.6f}` tokens/sec.

### Privileged Capture Status

- `fs_usage`: missing for at least one accepted read cell. This run is not
  valid SSD proof.
- `powermetrics`: `{payload.get("config", {}).get("powermetrics_error", "") or "not captured or no error recorded"}`

### Decision

{decision}
"""
    args.notes.write_text(_replace_stage0_section(args.notes.read_text(), stage0_markdown))
    print(f"wrote {output}")
    print(f"updated {args.notes}")
    print(decision)
    return 0


def main() -> int:
    args = _parse_args()
    if args.stage == "stage0":
        return summarize_stage0(args)
    raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
