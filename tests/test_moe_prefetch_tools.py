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
