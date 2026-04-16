#!/usr/bin/env python3
"""Render bakeoff inputs through Config F (Swift + Core ML) and write WAV + metrics JSON.

Uses the same tokenized inputs as the harness (``outputs/swift_bench_inputs``).

Usage::

    uv run python scripts/bakeoff_listen.py

Output: ``outputs/bakeoff/listen/config_f_{3s,7s,15s,30s}.wav`` (and matching ``.json``).

Prereqs: ``swift build -c release --product kokoro-bench`` and
``uv run python scripts/prepare_swift_bench_inputs.py`` (this script runs the
latter if inputs are missing).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KEYS = ("3s", "7s", "15s", "30s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Config F WAV export for bakeoff inputs")
    ap.add_argument(
        "--keys",
        default=",".join(_DEFAULT_KEYS),
        help=f"Comma-separated input keys (default: {','.join(_DEFAULT_KEYS)})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "outputs" / "bakeoff" / "listen",
        help="Directory for WAV and JSON files",
    )
    args = ap.parse_args()
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    bench = _ROOT / "swift" / ".build" / "release" / "kokoro-bench"
    if not bench.exists():
        print(
            "Swift binary not found. Build with:\n"
            "  cd swift && swift build -c release --product kokoro-bench",
            file=sys.stderr,
        )
        return 1

    models = _ROOT / "coreml"
    inputs_dir = _ROOT / "outputs" / "swift_bench_inputs"
    hnsf = inputs_dir / "hnsf_weights.json"

    if not (inputs_dir / "3s.json").exists() or not hnsf.exists():
        print("Preparing Swift bench inputs...", file=sys.stderr)
        subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "prepare_swift_bench_inputs.py")],
            cwd=str(_ROOT),
            check=True,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        wav = args.out_dir / f"config_f_{key}.wav"
        metrics = args.out_dir / f"config_f_{key}.json"
        cmd = [
            str(bench),
            "--models-dir",
            str(models),
            "--inputs-dir",
            str(inputs_dir),
            "--hnsf-weights",
            str(hnsf),
            "--input-key",
            key,
            "--seed",
            "0",
            "--output",
            str(metrics),
            "--wav",
            str(wav),
        ]
        print(" ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, cwd=str(_ROOT), check=True)

    print(f"Wrote WAV + JSON under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
