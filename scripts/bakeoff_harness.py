#!/usr/bin/env python3
"""Kokoro TTS Bakeoff Harness -- unified benchmark with four modes.

Modes
-----
  prepare-inputs   Measure canonical audio durations with PyTorch CPU; write input manifest.
  run              Timed benchmark iterations for selected configs with preloaded models.
  telemetry-loop   Sustained inference loop for powermetrics observation.
  summarize        Read results files and emit tables plus gate answers.

Configs
-------
  A   Shipping hybrid HAR-post path (3s/10s buckets, explicit path load)
  B   Naive decoder-only 10s artifact, compute_units=ALL
  C   Naive decoder-only 10s artifact, compute_units=CPU_AND_GPU
  D   PyTorch end-to-end on MPS (requires PYTORCH_ENABLE_MPS_FALLBACK=1)
  E   PyTorch end-to-end on CPU

Diagnostic-only (telemetry-loop only, not for ``run --configs``):
  bcpu  Naive decoder-only 10s artifact, compute_units=CPU_ONLY

Usage::

    python scripts/bakeoff_harness.py prepare-inputs
    python scripts/bakeoff_harness.py run --configs a,b,c,d,e --iterations 5 --order-seed 0
    python scripts/bakeoff_harness.py telemetry-loop --config b --input long --seconds 60
    python scripts/bakeoff_harness.py summarize --results outputs/bakeoff/results_m2_ultra.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_BAKEOFF_DIR = _REPO_ROOT / "outputs" / "bakeoff"
_BAKEOFF_MODELS_DIR = _BAKEOFF_DIR / "models"
_INPUT_MANIFEST_PATH = _BAKEOFF_DIR / "input_manifest.json"

# The decoder-only control artifact exported by Phase 2.
DECODER_ONLY_ARTIFACT = str(_BAKEOFF_MODELS_DIR / "kokoro_decoder_only_10s.mlpackage")

# Shipping HAR-post bucket paths.
HAR_POST_3S = str(_REPO_ROOT / "coreml" / "kokoro_decoder_har_post_3s.mlpackage")
HAR_POST_10S = str(_REPO_ROOT / "coreml" / "kokoro_decoder_har_post_10s.mlpackage")

# ---------------------------------------------------------------------------
# Benchmark Inputs -- frozen text, voice, and speed for every config.
# Texts chosen to hit duration targets within the 10s HAR-post ceiling.
# Starting point: PRESETS from scripts/bench_pipeline_stages.py:38.
# ---------------------------------------------------------------------------
VOICE = "af_heart"
SPEED = 1.0

BAKEOFF_INPUTS: dict[str, str] = {
    "tiny": "Hello world!",
    "short": "The quick brown fox jumps over the dog.",
    "medium": (
        "This is a longer sentence that will test the performance of our pipeline "
        "running on the Apple GPU."
    ),
    "long": (
        "This is a longer sentence that will test the performance of our pipeline "
        "running on the Apple GPU. A few more words added here."
    ),
}

# Maximum canonical audio duration (seconds).  prepare-inputs hard-fails above this.
MAX_CANONICAL_DURATION_S = 9.0

# Headline config IDs valid for ``run --configs``.
HEADLINE_CONFIGS = {"a", "b", "c", "d", "e"}
# Diagnostic config valid only for ``telemetry-loop --config``.
DIAGNOSTIC_CONFIGS = {"bcpu"}
ALL_CONFIGS = HEADLINE_CONFIGS | DIAGNOSTIC_CONFIGS


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Return hex SHA-256 of a file (or key internal file for .mlpackage dirs)."""
    p = Path(path)
    if p.is_dir():
        # For .mlpackage directories, hash the weights file if present.
        weights = p / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
        if weights.exists():
            return _sha256_file(str(weights))
        spec = p / "Data" / "com.apple.CoreML" / "model.mlmodel"
        if spec.exists():
            return _sha256_file(str(spec))
        return "directory_no_hashable_file"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(_REPO_ROOT), text=True
        ).strip()
        return bool(out)
    except Exception:
        return True


def _machine_info() -> dict:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    try:
        info["sw_vers"] = subprocess.check_output(["sw_vers"], text=True).strip()
    except Exception:
        info["sw_vers"] = "unknown"
    try:
        info["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        info["cpu_brand"] = "unknown"
    try:
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["memory_gb"] = round(mem / (1024 ** 3), 1)
    except Exception:
        info["memory_gb"] = "unknown"
    return info


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ["torch", "torchaudio", "coremltools", "numpy", "transformers"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def _select_bucket_seconds_standalone(
    total_seconds: float, available_buckets: list[int]
) -> int | None:
    """Replicate _select_bucket_seconds logic without a pipeline instance.

    Picks the smallest available bucket >= ceil(total_seconds).
    Matches coreml_pipeline.py:_select_bucket_seconds exactly.
    """
    available = sorted(available_buckets)
    threshold = int(ceil(total_seconds))
    for sec in available:
        if sec >= threshold:
            return sec
    return available[-1] if available else None


# ---------------------------------------------------------------------------
# prepare-inputs mode
# ---------------------------------------------------------------------------

def cmd_prepare_inputs(args: argparse.Namespace) -> None:
    """Measure canonical audio durations with PyTorch CPU and write input manifest."""
    _BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(_REPO_ROOT))
    from kokoro.coreml_pipeline import HybridTTSPipeline
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline

    print("=" * 60)
    print("BAKEOFF: prepare-inputs")
    print("=" * 60)

    print("\n Loading PyTorch model on CPU...")
    torch.manual_seed(0)
    kmodel = KModel().to("cpu")
    kmodel.eval()
    kpipeline = KPipeline(lang_code="a", model=False)

    # Minimal stand-in that only runs extract_vocoder_inputs on CPU.
    # Avoids loading CoreML models, which may not exist yet.
    class _CPUBenchPipeline:
        def __init__(self, km, kp):
            self.pytorch_model = km
            self.pipeline = kp
            self.coreml_synth_buckets = {}
            self.coreml_decoder_har_buckets = {}
            self.coreml_decoder_har_post_buckets = {3: None, 10: None}

        extract_vocoder_inputs = HybridTTSPipeline.extract_vocoder_inputs
        _select_bucket_seconds = HybridTTSPipeline._select_bucket_seconds

    pipe = _CPUBenchPipeline(kmodel, kpipeline)

    entries: dict[str, dict] = {}
    any_failure = False

    for input_key, text in BAKEOFF_INPUTS.items():
        snippet = repr(text)[:60]
        ellipsis = "..." if len(text) > 60 else ""
        print(f"\n--- {input_key}: {snippet}{ellipsis}")
        torch.manual_seed(0)
        vi = pipe.extract_vocoder_inputs(text, VOICE, SPEED)
        if vi is None:
            print(f"  extract_vocoder_inputs returned None for {input_key!r}")
            any_failure = True
            continue

        T_f0 = int(vi["f0_curve"].shape[-1])
        canonical_duration_s = T_f0 / 80.0
        expected_bucket = _select_bucket_seconds_standalone(canonical_duration_s, [3, 10])

        print(f"  T_f0={T_f0}, canonical_duration={canonical_duration_s:.3f}s, bucket={expected_bucket}s")

        # Hard-fail if duration exceeds ceiling.
        if canonical_duration_s > MAX_CANONICAL_DURATION_S:
            print(
                f"  FAIL: canonical duration {canonical_duration_s:.3f}s "
                f"exceeds {MAX_CANONICAL_DURATION_S}s ceiling."
            )
            sys.exit(1)

        # Bucket-boundary assertion for the 'short' input.
        if input_key == "short":
            if ceil(canonical_duration_s) > 3:
                print(
                    f"  FAIL: 'short' input canonical duration {canonical_duration_s:.3f}s "
                    f"routes to {expected_bucket}s bucket (ceil={ceil(canonical_duration_s)}). "
                    f"Must route to 3s. Shorten the text."
                )
                sys.exit(1)

        # Smoke assertion: Config A would only choose 3s or 10s.
        if expected_bucket not in (3, 10):
            print(
                f"  FAIL: expected bucket {expected_bucket}s is not 3s or 10s -- "
                f"Config A cannot serve this input."
            )
            sys.exit(1)

        entries[input_key] = {
            "text": text,
            "voice": VOICE,
            "speed": SPEED,
            "canonical_duration_s": round(canonical_duration_s, 6),
            "expected_bucket_s": expected_bucket,
            "T_f0": T_f0,
            "text_sha256": _sha256_str(text),
        }

    if any_failure:
        print("\nABORT: one or more inputs failed extract_vocoder_inputs.")
        sys.exit(1)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "voice": VOICE,
        "speed": SPEED,
        "git_commit": _git_commit(),
        "python_executable": sys.executable,
        "inputs": entries,
    }

    _INPUT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n Input manifest written: {_INPUT_MANIFEST_PATH}")
    print(f"   {len(entries)} inputs frozen.")
    for k, v in entries.items():
        print(f"   {k}: {v['canonical_duration_s']:.3f}s -> {v['expected_bucket_s']}s bucket")


# ---------------------------------------------------------------------------
# Stub modes (implemented in later phases)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run timed benchmark iterations (Phase 1)."""
    raise NotImplementedError("run mode is implemented in Phase 1")


def cmd_telemetry_loop(args: argparse.Namespace) -> None:
    """Sustained inference loop for powermetrics observation (Phase 1)."""
    raise NotImplementedError("telemetry-loop mode is implemented in Phase 1")


def cmd_summarize(args: argparse.Namespace) -> None:
    """Read results and emit tables + gate answers (Phase 5)."""
    raise NotImplementedError("summarize mode is implemented in Phase 5")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kokoro TTS Bakeoff Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # prepare-inputs
    sub.add_parser(
        "prepare-inputs",
        help="Measure canonical durations and write input manifest",
    )

    # run
    p_run = sub.add_parser("run", help="Run timed benchmark iterations")
    p_run.add_argument(
        "--configs", required=True, help="Comma-separated config IDs (a,b,c,d,e)"
    )
    p_run.add_argument(
        "--iterations", type=int, default=5, help="Iterations per config/input pair"
    )
    p_run.add_argument(
        "--order-seed", type=int, default=0, help="Deterministic order seed"
    )
    p_run.add_argument(
        "--machine-id", default=None, help="Machine identifier for output filename"
    )

    # telemetry-loop
    p_tl = sub.add_parser(
        "telemetry-loop", help="Sustained inference loop for powermetrics"
    )
    p_tl.add_argument(
        "--config", required=True, help="Single config ID (b, c, bcpu)"
    )
    p_tl.add_argument(
        "--input", required=True,
        help="Input key from manifest (tiny, short, medium, long)",
    )
    p_tl.add_argument(
        "--seconds", type=int, default=60, help="Duration of the loop in seconds"
    )

    # summarize
    p_sum = sub.add_parser(
        "summarize", help="Read results files and emit tables + gate answers"
    )
    p_sum.add_argument(
        "--results", required=True, nargs="+", help="Path(s) to results JSON files"
    )

    args = parser.parse_args()

    if args.mode == "prepare-inputs":
        cmd_prepare_inputs(args)
    elif args.mode == "run":
        cmd_run(args)
    elif args.mode == "telemetry-loop":
        cmd_telemetry_loop(args)
    elif args.mode == "summarize":
        cmd_summarize(args)


if __name__ == "__main__":
    main()
