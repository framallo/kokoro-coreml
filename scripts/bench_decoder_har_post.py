#!/usr/bin/env python3
"""Smoke-test ``decoder_har_post`` Core ML path with short, medium, long, and ~10s-style text.

Run from repo root::

    uv run python scripts/bench_decoder_har_post.py

Writes WAVs under ``outputs/`` (gitignored) and prints ``time_sec``, ``audio_sec``, ``rtf``.
"""
from __future__ import annotations

import sys
import time
import wave
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kokoro.coreml_pipeline import HybridTTSPipeline  # noqa: E402

_SR = 24000


def _save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio.size == 0:
        data = np.zeros((0,), dtype=np.int16)
    else:
        peak = max(1e-7, float(np.max(np.abs(audio))))
        scaled = np.clip(audio / peak, -1.0, 1.0)
        data = (scaled * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SR)
        wf.writeframes(data.tobytes())


def main() -> int:
    out_dir = _REPO_ROOT / "outputs" / "vocoder_bench"
    cases: list[tuple[str, str, str]] = [
        ("01_short", "Hi.", "Very short"),
        (
            "02_medium",
            "The quick brown fox jumps over the lazy dog twice, just to be sure.",
            "Medium sentence",
        ),
        (
            "03_long",
            "In synthesis benchmarks we care about wall time, audio duration, and real-time factor. "
            "This line is intentionally verbose so phoneme count and F0 length differ from the short case.",
            "Longer explanatory line",
        ),
        (
            "04_ten_sec_style",
            " ".join(
                [
                    "This block is meant to approach several seconds of speech at twenty-four kilohertz.",
                    "We repeat ideas with light variation so duration grows without sounding like pure padding.",
                    "Neural codecs, vocoders, and hybrid pipelines each add their own latency and quality tradeoffs.",
                    "On Apple silicon, Core ML can offload much of the generator while the decoder stays in PyTorch.",
                    "If only a three-second bucket exists, the pipeline falls back to the largest bucket and trims.",
                    "With a ten-second package present, geometry should match longer utterances more faithfully.",
                    "Listen for naturalness at the tail, not just the attack of each phrase.",
                    "Finally, we stop before the text becomes absurdly long for a single synthesis call.",
                ]
            ),
            "Paragraph sized for ~10s audio (depends on speed and voice)",
        ),
    ]

    try:
        pipe = HybridTTSPipeline(force_engine="coreml")
    except Exception as e:
        print(f"FAIL init: {e}")
        return 1

    if not getattr(pipe, "coreml_decoder_har_post_buckets", None):
        print("WARN: no kokoro_decoder_har_post_*s buckets loaded; synthesis may skip decoder_har_post.")
    else:
        print(f"decoder_har_post buckets: {sorted(pipe.coreml_decoder_har_post_buckets.keys())}")

    for stem, text, label in cases:
        t0 = time.time()
        try:
            audio, sr = pipe.synthesize(text, voice="af_heart", speed=1.0)
        except Exception as e:
            print(f"FAIL {stem} ({label}): {e}")
            return 1
        t1 = time.time()
        if audio is None or len(audio) == 0:
            print(f"FAIL {stem}: empty audio")
            return 1
        wav_path = out_dir / f"{stem}.wav"
        _save_wav(wav_path, audio)
        wall = t1 - t0
        sec_audio = len(audio) / float(sr)
        rtf = wall / sec_audio if sec_audio > 0 else float("inf")
        print(
            f"ok {stem} label={label!r} time_sec={wall:.3f} audio_sec={sec_audio:.3f} "
            f"rtf={rtf:.3f} out={wav_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
