#!/usr/bin/env python3
"""Run the Phase 3 audio parity ladder end to end for one prepared input."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def run(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    python_dir = out_root / f"python_{args.input_key}"
    swift_dir = out_root / f"swift_{args.input_key}"
    compare_json = out_root / f"compare_{args.input_key}.json"
    swift_metrics = swift_dir / "metrics.json"

    python_cmd = [
        sys.executable,
        "scripts/capture_audio_parity_tensors.py",
        "--input-key",
        args.input_key,
        "--inputs-dir",
        args.inputs_dir,
        "--models-dir",
        args.models_dir,
        "--out-dir",
        str(python_dir),
        "--seed",
        str(args.seed),
        "--available-buckets",
        args.available_buckets,
    ]
    if args.bucket_sec:
        python_cmd.extend(["--bucket-sec", str(args.bucket_sec)])
    _run(python_cmd, cwd=_ROOT)

    swift_cmd = [
        "swift",
        "run",
        "--package-path",
        "swift",
        "kokoro-bench",
        "--models-dir",
        args.models_dir,
        "--inputs-dir",
        args.inputs_dir,
        "--hnsf-weights",
        args.hnsf_weights,
        "--input-key",
        args.input_key,
        "--seed",
        str(args.seed),
        "--compute-units",
        args.compute_units,
        "--output",
        str(swift_metrics),
        "--dump-tensors",
        str(swift_dir),
    ]
    _run(swift_cmd, cwd=_ROOT)

    compare_cmd = [
        sys.executable,
        "scripts/compare_audio_parity_tensors.py",
        "--reference",
        str(python_dir),
        "--candidate",
        str(swift_dir),
        "--write-json",
        str(compare_json),
        "--max-abs",
        str(args.max_abs),
        "--min-corr",
        str(args.min_corr),
    ]
    if args.fail_on_difference:
        compare_cmd.append("--fail-on-difference")
    _run(compare_cmd, cwd=_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-key", default="3s")
    parser.add_argument("--inputs-dir", default=str(_ROOT / "outputs" / "swift_bench_inputs"))
    parser.add_argument("--models-dir", default=str(_ROOT / "coreml"))
    parser.add_argument("--hnsf-weights", default=str(_ROOT / "outputs" / "swift_bench_inputs" / "hnsf_weights.json"))
    parser.add_argument("--out-dir", default=str(_ROOT / "outputs" / "audio-parity" / "tensors"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-units", default="cpuOnly")
    parser.add_argument("--available-buckets", default="3,7,10,15,30")
    parser.add_argument("--bucket-sec", type=int, default=None)
    parser.add_argument("--max-abs", type=float, default=1e-3)
    parser.add_argument("--min-corr", type=float, default=0.999)
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
