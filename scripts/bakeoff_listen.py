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
import array
import hashlib
import json
import math
import subprocess
import sys
import uuid
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_KEYS = ("3s", "7s", "15s", "30s")
_SAMPLE_RATE = 24_000
_DURATION_TOLERANCE_FRACTION = 0.15
_MIN_RMS_PCM = 20.0
_MIN_ACTIVE_FRACTION = 0.001
_MAX_CLIPPED_FRACTION = 0.01


def _load_expected_inputs() -> tuple[dict[str, str], str, float]:
    scripts_dir = str(_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from bakeoff_harness import BAKEOFF_INPUTS, SPEED, VOICE

    return BAKEOFF_INPUTS, VOICE, SPEED


def _hnsf_weights_sha256(hnsf: Path) -> str | None:
    try:
        data = json.loads(hnsf.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if "linear_weights" not in data or "linear_bias" not in data:
        return None
    payload = json.dumps(
        {"linear_weights": data["linear_weights"], "linear_bias": data["linear_bias"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stale_input_reason(
    key: str,
    input_path: Path,
    hnsf: Path,
    expected_text: str,
    expected_voice: str,
    expected_speed: float,
) -> str | None:
    if not input_path.exists():
        return f"{key}: missing {input_path}"
    try:
        data = json.loads(input_path.read_text())
    except json.JSONDecodeError as exc:
        return f"{key}: invalid JSON in {input_path}: {exc}"

    if data.get("text") != expected_text:
        return f"{key}: prepared text does not match bakeoff_harness.BAKEOFF_INPUTS"
    if data.get("voice") != expected_voice:
        return f"{key}: prepared voice={data.get('voice')!r}, expected {expected_voice!r}"
    if abs(float(data.get("speed", -1)) - expected_speed) > 1e-6:
        return f"{key}: prepared speed={data.get('speed')!r}, expected {expected_speed!r}"
    expected_hnsf_hash = data.get("hnsf_weights_sha256")
    actual_hnsf_hash = _hnsf_weights_sha256(hnsf)
    if not expected_hnsf_hash or expected_hnsf_hash != actual_hnsf_hash:
        return f"{key}: prepared input does not match current hn-nsf weights"
    return None


def _ensure_inputs(keys: list[str], inputs_dir: Path, hnsf: Path) -> bool:
    expected_inputs, expected_voice, expected_speed = _load_expected_inputs()
    unknown = [key for key in keys if key not in expected_inputs]
    if unknown:
        print(
            f"Unknown input key(s): {', '.join(unknown)}. "
            f"Available keys: {', '.join(expected_inputs)}",
            file=sys.stderr,
        )
        return False

    stale_reasons = [
        reason
        for key in keys
        if (reason := _stale_input_reason(
            key,
            inputs_dir / f"{key}.json",
            hnsf,
            expected_inputs[key],
            expected_voice,
            expected_speed,
        ))
    ]
    if not stale_reasons and hnsf.exists():
        return True

    print("Preparing Swift bench inputs...", file=sys.stderr)
    subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "prepare_swift_bench_inputs.py")],
        cwd=str(_ROOT),
        check=True,
    )

    stale_reasons = [
        reason
        for key in keys
        if (reason := _stale_input_reason(
            key,
            inputs_dir / f"{key}.json",
            hnsf,
            expected_inputs[key],
            expected_voice,
            expected_speed,
        ))
    ]
    if hnsf.exists() and not stale_reasons:
        return True

    available = sorted(path.stem for path in inputs_dir.glob("*.json") if path.name != "hnsf_weights.json")
    for reason in stale_reasons:
        print(f"Invalid prepared input: {reason}", file=sys.stderr)
    if not hnsf.exists():
        print(f"Missing hn-nsf weights: {hnsf}", file=sys.stderr)
    if stale_reasons:
        print(f"Available prepared keys: {', '.join(available) or '(none)'}", file=sys.stderr)
    return False


def _validate_outputs(key: str, wav: Path, metrics_path: Path, display_wav: Path | None = None) -> None:
    if not wav.exists():
        raise RuntimeError(f"{key}: WAV was not written: {wav}")
    if not metrics_path.exists():
        raise RuntimeError(f"{key}: metrics JSON was not written: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    if metrics.get("status") != "ok":
        raise RuntimeError(f"{key}: benchmark status={metrics.get('status')!r}: {metrics.get('error')}")

    canonical = metrics.get("canonical_audio_duration_s")
    observed = metrics.get("observed_audio_duration_s")
    if isinstance(canonical, (int, float)) and canonical > 0 and isinstance(observed, (int, float)):
        drift = abs(observed - canonical) / canonical
        if drift > _DURATION_TOLERANCE_FRACTION:
            raise RuntimeError(
                f"{key}: observed duration {observed:.3f}s differs from "
                f"canonical {canonical:.3f}s by {drift:.1%}"
            )

    with wave.open(str(wav), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.getnframes()
        width = wf.getsampwidth()

    if channels != 1 or rate != _SAMPLE_RATE or width != 2:
        raise RuntimeError(f"{key}: unexpected WAV format channels={channels}, rate={rate}, width={width}")
    if frames <= 0:
        raise RuntimeError(f"{key}: WAV is empty")

    with wave.open(str(wav), "rb") as wf:
        pcm = array.array("h")
        pcm.frombytes(wf.readframes(frames))
    if pcm and sys.byteorder == "big":
        pcm.byteswap()
    abs_samples = [abs(sample) for sample in pcm]
    peak = max(abs_samples, default=0)
    rms = math.sqrt(sum(sample * sample for sample in pcm) / len(pcm)) if pcm else 0.0
    active_fraction = sum(1 for sample in abs_samples if sample > 32) / len(abs_samples) if abs_samples else 0.0
    clipped_fraction = sum(1 for sample in abs_samples if sample >= 32760) / len(abs_samples) if abs_samples else 0.0
    if rms < _MIN_RMS_PCM or active_fraction < _MIN_ACTIVE_FRACTION:
        raise RuntimeError(
            f"{key}: WAV appears silent/noise-only (rms={rms:.1f}, active={active_fraction:.3%})"
        )
    if clipped_fraction > _MAX_CLIPPED_FRACTION:
        raise RuntimeError(f"{key}: WAV clips too often ({clipped_fraction:.3%})")
    if peak <= 0:
        raise RuntimeError(f"{key}: WAV has zero peak amplitude")

    wav_seconds = frames / rate
    if isinstance(observed, (int, float)) and abs(wav_seconds - observed) > 0.02:
        raise RuntimeError(
            f"{key}: WAV header duration {wav_seconds:.3f}s does not match metrics {observed:.3f}s"
        )

    print(f"  validated {key}: {wav_seconds:.3f}s -> {display_wav or wav}", file=sys.stderr)


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

    if not _ensure_inputs(keys, inputs_dir, hnsf):
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        wav = args.out_dir / f"config_f_{key}.wav"
        metrics = args.out_dir / f"config_f_{key}.json"
        temp_stem = f".config_f_{key}.{uuid.uuid4().hex}"
        temp_wav = args.out_dir / f"{temp_stem}.wav"
        temp_metrics = args.out_dir / f"{temp_stem}.json"
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
            str(temp_metrics),
            "--wav",
            str(temp_wav),
        ]
        print(" ".join(cmd), file=sys.stderr)
        try:
            subprocess.run(cmd, cwd=str(_ROOT), check=True)
            _validate_outputs(key, temp_wav, temp_metrics, display_wav=wav)
            temp_wav.replace(wav)
            temp_metrics.replace(metrics)
        finally:
            temp_wav.unlink(missing_ok=True)
            temp_metrics.unlink(missing_ok=True)

    print(f"Wrote WAV + JSON under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
