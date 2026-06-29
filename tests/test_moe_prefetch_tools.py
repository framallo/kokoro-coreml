import json
import subprocess
import sys
from pathlib import Path

from scripts.moe_prefetch.schema import (
    estimate_expert_bytes,
    model_inventory_payload,
    thresholds_payload,
    validate_phase0_ready,
)


def test_estimate_expert_bytes_uses_quantization_bits() -> None:
    assert estimate_expert_bytes(176_160_768, 4) == 88_080_384
    assert estimate_expert_bytes(3, 4) == 2


def test_phase0_readiness_requires_inventory_and_threshold_fields() -> None:
    inventory = model_inventory_payload(
        model_id="test/moe",
        quantization_bits=4,
        active_experts_per_token=64,
        target_tokens_per_second=1.0,
        expert_bytes=88_080_384,
        expert_parameters=176_160_768,
        target_device="local",
        estimate_source="unit-test",
        notes="",
        machine={"machine_id": "local"},
    )
    assert validate_phase0_ready(inventory, thresholds_payload()) == []

    broken = dict(inventory)
    broken["expert_bytes"] = 0
    assert "model_inventory.expert_bytes" in validate_phase0_ready(
        broken,
        thresholds_payload(),
    )


def test_model_inventory_cli_writes_inventory_and_thresholds(tmp_path: Path) -> None:
    output = tmp_path / "model_inventory.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/model_inventory.py",
            "--model-id",
            "test/moe",
            "--quantization-bits",
            "4",
            "--active-experts-per-token",
            "64",
            "--target-tokens-per-second",
            "1.0",
            "--expert-parameters",
            "176160768",
            "--target-device",
            "unit-test-mac",
            "--output",
            str(output),
        ],
        check=True,
    )

    inventory = json.loads(output.read_text())
    thresholds = json.loads((tmp_path / "thresholds.json").read_text())
    assert inventory["expert_bytes"] == 88_080_384
    assert inventory["active_expert_bytes_per_token"] == 5_637_144_576
    assert thresholds["speed_win_percent"] == 25.0
    assert thresholds["trivial_margin_percent"] == 10.0


def test_stage0_summarize_kills_on_oracle_bandwidth_ceiling(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    notes = tmp_path / "notes.md"
    notes.write_text(
        "# Results\n\n"
        "## Stage 0: Hardware Envelope\n\n"
        "Pending.\n\n"
        "## Stage 1: Router Trace and Predictor Replay\n\n"
        "Blocked.\n"
    )
    payload = {
        "inventory": {
            "model_id": "test/moe",
            "expert_bytes": 88_080_384,
            "active_expert_bytes_per_token": 5_637_144_576,
            "target_tokens_per_second": 1.0,
        },
        "thresholds": thresholds_payload(),
        "cells": [
            {
                "pattern": "random",
                "returncode": 0,
                "fs_usage_path": str(tmp_path / "fs_usage.txt"),
                "fs_usage_error": "",
                "measurement": {
                    "successful_reads": 1,
                    "failed_reads": 0,
                    "total_bytes_read": 88_080_384,
                    "wall_time_ns": 88_080_384,
                    "latencies_ns": [88_080_384],
                },
            }
        ],
    }
    (tmp_path / "fs_usage.txt").write_text("RdData B=88080384\n")
    results.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            "scripts/moe_prefetch/summarize.py",
            "stage0",
            "--input",
            str(results),
            "--notes",
            str(notes),
        ],
        check=True,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["oracle_bandwidth_ceiling_tokens_per_second"] < 1.0
    assert summary["decision"].startswith("KILL: oracle bandwidth ceiling")
    assert "Oracle bandwidth ceiling" in notes.read_text()
