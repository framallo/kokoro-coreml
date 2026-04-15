#!/usr/bin/env python3
"""Median Core ML ``predict()`` wall time for ``kokoro_decoder_har_post_*s`` (Phase 3 fallback).

Uses the same tensor prep as production (:func:`~kokoro.synthesis_backends.build_decoder_har_post_inputs_np`).
Maps to bakeoff schema field ``t_coreml_predict_s`` (this script reports milliseconds per call).

Examples::

    uv run python scripts/bench_decoder_har_post_predict.py \\
      --package coreml/kokoro_decoder_har_post_3s.mlpackage --bucket-sec 3

    uv run python scripts/bench_decoder_har_post_predict.py \\
      --package coreml/kokoro_decoder_har_post_3s.mlpackage \\
      --baseline /path/to/pre_conv1d_3s.mlpackage \\
      --bucket-sec 3 --json-out outputs/bakeoff/ane_optimization_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import coremltools as ct

from kokoro.coreml_pipeline import HybridTTSPipeline
from kokoro.synthesis_backends import build_decoder_har_post_inputs_np


def _inputs_for_package(
    pipe: HybridTTSPipeline,
    mlpackage: Path,
    text: str,
    voice: str,
    speed: float,
    bucket_sec: int,
) -> dict:
    model = ct.models.MLModel(str(mlpackage))
    spec = model.get_spec()
    shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec.description.input}
    asr_len = int(shapes["x_pre"][-1])
    har_t = int(shapes["har"][-1])

    vi = pipe.extract_vocoder_inputs(text, voice, speed)
    if vi is None:
        raise RuntimeError("extract_vocoder_inputs returned None")
    T_f0 = int(vi["f0_curve"].shape[-1])
    total_seconds = T_f0 / 80.0
    selected = pipe._select_bucket_seconds(total_seconds)
    if selected is None or selected not in pipe.coreml_decoder_har_post_buckets:
        raise RuntimeError("no decoder_har_post bucket for pipeline / utterance")
    if selected != bucket_sec:
        raise RuntimeError(
            f"pipeline selects {selected}s for this text; use --bucket-sec {selected}"
        )
    dec = pipe.pytorch_model.decoder
    x_pre, ref_s, har, t_chk, _fc = build_decoder_har_post_inputs_np(
        dec, vi, bucket_sec, asr_len, har_t, warn_geometry=True
    )
    _ = t_chk
    return {"x_pre": x_pre, "ref_s": ref_s, "har": har}


def _median_predict_ms(model: ct.models.MLModel, inputs: dict, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        _ = model.predict(inputs)
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(inputs)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(samples))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True, help="Candidate .mlpackage path")
    p.add_argument("--baseline", type=Path, default=None, help="Optional baseline .mlpackage for A/B median")
    p.add_argument("--bucket-sec", type=int, required=True, choices=(3, 10))
    p.add_argument("--text", type=str, default="Hello from Kokoro.")
    p.add_argument("--voice", type=str, default="af_heart")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=21)
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Merge results into this JSON file (creates parent dirs)",
    )
    args = p.parse_args()

    torch.manual_seed(0)
    pipe = HybridTTSPipeline()
    inputs = _inputs_for_package(
        pipe, args.package, args.text, args.voice, args.speed, args.bucket_sec
    )

    cand = ct.models.MLModel(str(args.package))
    med_c = _median_predict_ms(cand, inputs, args.warmup, args.iterations)
    print(f"candidate_median_ms={med_c:.3f}  package={args.package}")

    doc: dict = {
        "benchmark_mode": "fallback_loop",
        "t_coreml_predict_median_ms": med_c,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "bucket_sec": args.bucket_sec,
        "package": str(args.package.resolve()),
    }

    if args.baseline is not None:
        base = ct.models.MLModel(str(args.baseline))
        med_b = _median_predict_ms(base, inputs, args.warmup, args.iterations)
        doc["baseline_mlpackage"] = str(args.baseline.resolve())
        doc["baseline_median_ms"] = med_b
        if med_b > 0:
            doc["speedup_vs_baseline_pct"] = round(100.0 * (med_b - med_c) / med_b, 2)
        print(f"baseline_median_ms={med_b:.3f}  package={args.baseline}")
        if "speedup_vs_baseline_pct" in doc:
            print(f"speedup_vs_baseline_pct={doc['speedup_vs_baseline_pct']}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        prev = {}
        if args.json_out.is_file():
            try:
                prev = json.loads(args.json_out.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        prev.update(doc)
        args.json_out.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
