#!/usr/bin/env python3
"""Run same-window Config F through the existing Swift benchmark CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.external_bakeoff.schema import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    error_record,
    result_file_payload,
    result_record,
    validate_manifest,
    validate_result_payload,
    load_json,
    write_json,
)


def _run_once(
    binary: Path,
    input_key: str,
    compute_units: str,
    models_dir: Path,
    inputs_dir: Path,
    hnsf_weights: Path,
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = Path(f.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = Path(f.name)
    cmd = [
        str(binary),
        "--models-dir", str(models_dir),
        "--inputs-dir", str(inputs_dir),
        "--hnsf-weights", str(hnsf_weights),
        "--input-key", input_key,
        "--compute-units", compute_units,
        "--output", str(out),
        "--wav", str(wav),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "kokoro-bench failed\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    data = json.loads(out.read_text())
    if wav.exists():
        data["output_sha256"] = hashlib.sha256(wav.read_bytes()).hexdigest()
    out.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "runtime_input_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--binary", type=Path, default=Path("swift/.build/release/kokoro-bench"))
    parser.add_argument("--models-dir", type=Path, default=Path("coreml"))
    parser.add_argument("--inputs-dir", type=Path, default=Path("outputs/swift_bench_inputs"))
    parser.add_argument(
        "--hnsf-weights",
        type=Path,
        default=Path("outputs/swift_bench_inputs/hnsf_weights.json"),
    )
    parser.add_argument("--compute-units", default="all")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--input-key", action="append", default=None)
    args = parser.parse_args()

    if not args.binary.exists():
        raise SystemExit(f"missing Swift benchmark binary: {args.binary}")

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    keys = args.input_key or list(manifest["inputs"].keys())

    records = []
    for key in keys:
        item = manifest["inputs"][key]
        try:
            cold_result = _run_once(
                args.binary,
                key,
                args.compute_units,
                args.models_dir,
                args.inputs_dir,
                args.hnsf_weights,
            )
            warm = [
                _run_once(
                    args.binary,
                    key,
                    args.compute_units,
                    args.models_dir,
                    args.inputs_dir,
                    args.hnsf_weights,
                )
                for _ in range(args.iterations)
            ]
            records.append(
                result_record(
                    impl="config-f-reference",
                    framework="Swift + Core ML",
                    hardware_target="ANE/Core ML",
                    version="local",
                    machine_id=args.machine_id,
                    input_key=key,
                    text=item["text"],
                    voice=item["voice"],
                    cold_wall_time_s=float(cold_result["wall_time_s"]),
                    warm_wall_times_s=[float(row["wall_time_s"]) for row in warm],
                    canonical_audio_duration_s=float(item["canonical_duration_s"]),
                    observed_audio_duration_s=float(warm[-1]["observed_audio_duration_s"]),
                    output_sha256=str(warm[-1].get("output_sha256") or ""),
                    provenance={
                        "binary": str(args.binary),
                        "models_dir": str(args.models_dir),
                        "inputs_dir": str(args.inputs_dir),
                        "hnsf_weights": str(args.hnsf_weights),
                        "compute_units": args.compute_units,
                        "bucket_used": warm[-1].get("bucket_used"),
                        "raw_last_result": warm[-1],
                    },
                )
            )
        except Exception as exc:
            records.append(
                error_record(
                    impl="config-f-reference",
                    framework="Swift + Core ML",
                    hardware_target="ANE/Core ML",
                    version="local",
                    machine_id=args.machine_id,
                    input_key=key,
                    text=item["text"],
                    voice=item["voice"],
                    canonical_audio_duration_s=float(item["canonical_duration_s"]),
                    provenance={
                        "binary": str(args.binary),
                        "models_dir": str(args.models_dir),
                        "inputs_dir": str(args.inputs_dir),
                        "hnsf_weights": str(args.hnsf_weights),
                        "compute_units": args.compute_units,
                    },
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    payload = result_file_payload(
        impl="config-f-reference",
        machine_id=args.machine_id,
        records=records,
        provenance={
            "binary": str(args.binary),
            "models_dir": str(args.models_dir),
            "inputs_dir": str(args.inputs_dir),
            "hnsf_weights": str(args.hnsf_weights),
            "compute_units": args.compute_units,
        },
    )
    validate_result_payload(payload)
    output = args.output or (DEFAULT_OUTPUT_DIR / f"results_config_f_reference_{args.machine_id}.json")
    write_json(output, payload)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
