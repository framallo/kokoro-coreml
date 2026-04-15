#!/usr/bin/env python3
"""Break the HAR-post TTS pipeline into stage timings to find the real bottleneck.

The Core ML predict() micro-bench (bench_decoder_har_post_predict.py) already
measures GeneratorFromHar alone.  This script instruments the full synthesis
path so you know how much time each CPU stage consumes *before* Core ML even
fires.  Without this data you cannot know whether to optimise the Core ML graph
or the CPU pipeline.

Stages measured
---------------
  duration_model_ms   Core ML DurationModel predict()
  har_builder_ms      CPU: f0_upsamp → m_source (SineGen) → stft.transform
  coreml_predict_ms   Core ML GeneratorFromHar predict()
  trim_ms             Audio trimming / output shaping
  total_synthesis_ms  Wall time from synthesize() entry to waveform

Usage::

    uv run python scripts/bench_pipeline_stages.py \\
        --text "Hello from Kokoro." --voice af_heart \\
        --warmup 2 --iterations 10

    uv run python scripts/bench_pipeline_stages.py --preset long --iterations 7
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent

PRESETS: dict[str, str] = {
    "tiny":   "Hello world!",
    "short":  "The quick brown fox jumps over the lazy dog.",
    "medium": (
        "This is a longer sentence that will test the performance of our pipeline "
        "running on the Apple GPU."
    ),
    "long": (
        "This is a longer sentence that will test the performance of our pipeline "
        "running on the Apple GPU. More text at the end pushes nine seconds."
    ),
}


# ---------------------------------------------------------------------------
# Instrumented synthesis runner
# ---------------------------------------------------------------------------

class _StageClock:
    """Accumulate per-stage wall times across multiple calls."""

    def __init__(self) -> None:
        self._stages: dict[str, list[float]] = {}

    def record(self, stage: str, elapsed_ms: float) -> None:
        self._stages.setdefault(stage, []).append(elapsed_ms)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for stage, samples in self._stages.items():
            out[stage] = {
                "median_ms": round(statistics.median(samples), 2),
                "min_ms":    round(min(samples), 2),
                "max_ms":    round(max(samples), 2),
                "n":         len(samples),
            }
        return out


def _patch_pipeline(pipe: Any, clock: _StageClock) -> None:  # noqa: ANN401
    """Monkey-patch HybridTTSPipeline to record stage timings in *clock*.

    Patches three methods:
      _run_duration_model   → records duration_model_ms
      _build_har            → records har_builder_ms
      _run_coreml_predict   → records coreml_predict_ms

    Falls back gracefully if the pipeline does not expose these as separate
    methods (old layout).  In that case only total_synthesis_ms is recorded.
    """
    import functools

    def _wrap(method_name: str, stage_key: str) -> bool:
        fn = getattr(pipe, method_name, None)
        if fn is None:
            return False

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            clock.record(stage_key, (time.perf_counter() - t0) * 1000.0)
            return result

        setattr(pipe, method_name, wrapper)
        return True

    _wrap("_run_duration_model", "duration_model_ms")
    _wrap("_build_har", "har_builder_ms")
    _wrap("_run_coreml_predict", "coreml_predict_ms")


def _time_synthesis(pipe: Any, text: str, voice: str, speed: float, clock: _StageClock) -> float:
    t0 = time.perf_counter()
    pipe.synthesize(text, voice, speed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    clock.record("total_synthesis_ms", elapsed_ms)
    return elapsed_ms


# ---------------------------------------------------------------------------
# HAR builder direct timing (fallback when pipeline methods aren't patchable)
# ---------------------------------------------------------------------------

def _time_har_builder_direct(pipe: Any, text: str, voice: str, speed: float,
                              n: int, clock: _StageClock) -> None:
    """Extract vocoder inputs then time just the hn-nsf + STFT portion directly."""
    vi = pipe.extract_vocoder_inputs(text, voice, speed)
    if vi is None:
        print("  [warn] extract_vocoder_inputs returned None; skipping HAR builder direct timing")
        return

    gen = pipe.pytorch_model.decoder.generator

    # Pre-compute f0 upsampled (matches decoder_har_post_bucket_impl)
    f0_curve = vi["f0_curve"]
    if not isinstance(f0_curve, torch.Tensor):
        f0_curve = torch.tensor(f0_curve, dtype=torch.float32)
    if f0_curve.dim() == 1:
        f0_curve = f0_curve.unsqueeze(0)  # (1, T_f0)

    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            f0_u = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)   # (1, T_audio, 1)
            har_src, _, _ = gen.m_source(f0_u)
            har_src = har_src.transpose(1, 2).squeeze(1)               # (1, T_audio)
            _, _ = gen.stft.transform(har_src)
        clock.record("har_builder_direct_ms", (time.perf_counter() - t0) * 1000.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(clock: _StageClock, text: str, voice: str, audio_sec: float | None) -> None:
    summary = clock.summary()
    print()
    print("=" * 62)
    print("  Pipeline Stage Timing Report")
    print(f"  voice={voice!r}  audio≈{audio_sec:.2f}s" if audio_sec else "")
    print(f"  text={text[:60]!r}{'…' if len(text) > 60 else ''}")
    print("=" * 62)
    order = [
        "duration_model_ms",
        "har_builder_ms",
        "har_builder_direct_ms",
        "coreml_predict_ms",
        "trim_ms",
        "total_synthesis_ms",
    ]
    found = set(summary.keys())
    for key in order:
        if key in found:
            s = summary[key]
            print(f"  {key:<28s}  median={s['median_ms']:7.1f} ms  "
                  f"[{s['min_ms']:.1f}–{s['max_ms']:.1f}]  n={s['n']}")
            found.discard(key)
    for key in sorted(found):
        s = summary[key]
        print(f"  {key:<28s}  median={s['median_ms']:7.1f} ms  "
              f"[{s['min_ms']:.1f}–{s['max_ms']:.1f}]  n={s['n']}")
    print()

    total = summary.get("total_synthesis_ms")
    har   = summary.get("har_builder_ms") or summary.get("har_builder_direct_ms")
    cml   = summary.get("coreml_predict_ms")
    if total and audio_sec:
        rtf = (total["median_ms"] / 1000.0) / audio_sec
        print(f"  RTF (total/audio)  ≈ {rtf:.3f}  "
              f"({'faster' if rtf < 1 else 'slower'} than real-time)")
    if total and har and cml:
        har_pct  = 100.0 * har["median_ms"]  / total["median_ms"]
        cml_pct  = 100.0 * cml["median_ms"]  / total["median_ms"]
        cpu_pct  = 100.0 * (total["median_ms"] - cml["median_ms"]) / total["median_ms"]
        print(f"  HAR builder share  ≈ {har_pct:.1f}% of total")
        print(f"  Core ML share      ≈ {cml_pct:.1f}% of total")
        print(f"  CPU pipeline share ≈ {cpu_pct:.1f}% of total")
        print()
        if cml_pct < 20:
            print("  ⚡ Core ML predict() is NOT the bottleneck.")
            print("     Optimise the CPU pipeline (HAR builder / hn-nsf) first.")
        elif har_pct > cml_pct:
            print("  ⚡ HAR builder dominates. Consider torch.compile() on hn-nsf")
            print("     or validating whether MPS can run hn-nsf without precision loss.")
        else:
            print("  ⚡ Core ML predict() is significant. ANE profiling warranted.")
            print("     Run scripts/ane_verify.sh while this loop is active.")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text",     default="Hello from Kokoro.")
    p.add_argument("--preset",   choices=sorted(PRESETS), default=None)
    p.add_argument("--voice",    default="af_heart")
    p.add_argument("--speed",    type=float, default=1.0)
    p.add_argument("--warmup",   type=int, default=2)
    p.add_argument("--iterations", type=int, default=7)
    args = p.parse_args()

    text = PRESETS[args.preset] if args.preset else args.text

    print(f"Loading HybridTTSPipeline…")
    from kokoro.coreml_pipeline import HybridTTSPipeline
    pipe = HybridTTSPipeline()

    clock = _StageClock()
    _patch_pipeline(pipe, clock)

    # Warm up (results discarded)
    print(f"Warming up ({args.warmup} iters)…")
    for _ in range(args.warmup):
        pipe.synthesize(text, args.voice, args.speed)

    # Reset after warmup
    clock = _StageClock()
    _patch_pipeline(pipe, clock)

    # Timed iterations
    print(f"Timing {args.iterations} iters…")
    for _ in range(args.iterations):
        _time_synthesis(pipe, text, args.voice, args.speed, clock)

    # Direct HAR builder timing (always available regardless of pipeline layout)
    print(f"Direct HAR builder timing ({args.iterations} iters)…")
    _time_har_builder_direct(pipe, text, args.voice, args.speed,
                             args.iterations, clock)

    # Estimate audio duration from one synthesis run
    audio_sec: float | None = None
    try:
        vi = pipe.extract_vocoder_inputs(text, args.voice, args.speed)
        if vi is not None:
            audio_sec = float(vi["f0_curve"].shape[-1]) / 80.0
    except Exception:
        pass

    _print_report(clock, text, args.voice, audio_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
