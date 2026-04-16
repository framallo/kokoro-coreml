#!/usr/bin/env python3
"""Render bakeoff inputs through Config F (Swift + Core ML) and write WAV + metrics JSON.

Uses the same tokenized inputs as the harness (``outputs/swift_bench_inputs``).

Usage::

    uv run python scripts/bakeoff_listen.py

Output: ``outputs/bakeoff/listen/config_f_{3s,7s,15s,30s}.wav`` (and matching ``.json``).

This script rebuilds the release ``kokoro-bench`` binary when it is missing or
older than the Swift sources, and it prepares Swift bench inputs when needed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from audio_quality_probe import (  # noqa: E402
    Thresholds,
    classify_metrics,
    compute_metrics,
    derive_thresholds,
    sample_record,
    write_quality_report,
)

_DEFAULT_KEYS = ("3s", "7s", "15s", "30s")
_DEFAULT_REFERENCE_WAVS = (
    _ROOT / "outputs" / "audio-parity" / "references" / "pytorch_3s.wav",
    _ROOT / "outputs" / "audio-parity" / "references" / "pytorch_7s.wav",
    _ROOT / "outputs" / "audio-parity" / "references" / "pytorch_15s.wav",
    _ROOT / "outputs" / "audio-parity" / "references" / "pytorch_30s.wav",
    _ROOT / "outputs" / "audio-parity" / "comparators" / "decoder_har_post_demo.wav",
)
_SAMPLE_RATE = 24_000
_DURATION_TOLERANCE_FRACTION = 0.15


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


def _swift_source_mtime() -> float:
    swift_dir = _ROOT / "swift"
    candidates = [swift_dir / "Package.swift"]
    candidates.extend((swift_dir / "Sources").rglob("*.swift"))
    return max(path.stat().st_mtime for path in candidates if path.exists())


def _ensure_bench(bench: Path) -> bool:
    if bench.exists() and bench.stat().st_mtime >= _swift_source_mtime():
        return True

    reason = "missing" if not bench.exists() else "stale"
    print(f"Building release kokoro-bench ({reason})...", file=sys.stderr)
    try:
        subprocess.run(
            ["swift", "build", "-c", "release", "--product", "kokoro-bench"],
            cwd=str(_ROOT / "swift"),
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Release build failed: {exc}", file=sys.stderr)
        return False
    return bench.exists()


def _load_quality_context(reference_wavs: list[Path], plots_dir: Path | None) -> tuple[Thresholds, list[dict]]:
    missing = [path for path in reference_wavs if not path.exists()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise RuntimeError(
            "Missing reference WAV(s) for the audio-quality gate:\n"
            f"  {formatted}\n"
            "Run the audio parity reference phase first or pass --reference-wavs."
        )
    reference_metrics = [compute_metrics(path) for path in reference_wavs]
    thresholds = derive_thresholds(reference_metrics)
    records = [sample_record(path, "reference", thresholds, plots_dir) for path in reference_wavs]
    return thresholds, records


def _write_quality_fields(metrics_path: Path, thresholds: Thresholds, record: dict) -> None:
    metrics = json.loads(metrics_path.read_text())
    quality_pass = record["decision"] != "reject_without_listening"
    metrics["quality_pass"] = quality_pass
    metrics["quality_decision"] = record["decision"]
    metrics["quality_reject_reasons"] = record["reject_reasons"]
    metrics["audio_quality"] = {
        "thresholds": asdict(thresholds),
        "sample": record,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _validate_outputs(
    key: str,
    wav: Path,
    metrics_path: Path,
    thresholds: Thresholds,
    display_wav: Path | None = None,
) -> dict:
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

    audio_metrics = compute_metrics(wav)
    if audio_metrics.channels != 1 or audio_metrics.sample_rate != _SAMPLE_RATE or audio_metrics.sample_width != 2:
        raise RuntimeError(
            f"{key}: unexpected WAV format channels={audio_metrics.channels}, "
            f"rate={audio_metrics.sample_rate}, width={audio_metrics.sample_width}"
        )
    if audio_metrics.frames <= 0:
        raise RuntimeError(f"{key}: WAV is empty")

    wav_seconds = audio_metrics.duration_s
    if isinstance(observed, (int, float)) and abs(wav_seconds - observed) > 0.02:
        raise RuntimeError(
            f"{key}: WAV header duration {wav_seconds:.3f}s does not match metrics {observed:.3f}s"
        )

    decision, reasons = classify_metrics(audio_metrics, thresholds, is_reference=False)
    record = {
        "role": "candidate",
        "decision": decision,
        "reject_reasons": reasons,
        "metrics": asdict(audio_metrics),
    }
    record["metrics"]["path"] = str(display_wav or wav)
    _write_quality_fields(metrics_path, thresholds, record)

    print(
        f"  validated {key}: {wav_seconds:.3f}s quality={decision} "
        f"rms={audio_metrics.rms_pcm:.1f} active32={audio_metrics.active_fraction_gt32:.3%} "
        f"zcr={audio_metrics.zero_crossing_rate:.3%} -> {display_wav or wav}",
        file=sys.stderr,
    )
    return record


def main(argv: list[str] | None = None) -> int:
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
    ap.add_argument(
        "--reference-wavs",
        nargs="+",
        type=Path,
        default=list(_DEFAULT_REFERENCE_WAVS),
        help="Known-good WAVs used to derive the audio-quality gate",
    )
    ap.add_argument(
        "--quality-out-dir",
        type=Path,
        default=None,
        help="Directory for consolidated audio-quality reports (default: OUT_DIR/quality)",
    )
    ap.add_argument("--quality-plots", action="store_true", help="Write waveform and spectrogram PPMs")
    args = ap.parse_args(argv)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    bench = _ROOT / "swift" / ".build" / "release" / "kokoro-bench"
    if not _ensure_bench(bench):
        return 1

    models = _ROOT / "coreml"
    inputs_dir = _ROOT / "outputs" / "swift_bench_inputs"
    hnsf = inputs_dir / "hnsf_weights.json"

    if not _ensure_inputs(keys, inputs_dir, hnsf):
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    quality_out_dir = args.quality_out_dir or args.out_dir / "quality"
    plots_dir = quality_out_dir / "plots" if args.quality_plots else None
    try:
        thresholds, quality_records = _load_quality_context(args.reference_wavs, plots_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    failure: Exception | None = None
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
            record = _validate_outputs(key, temp_wav, temp_metrics, thresholds, display_wav=wav)
            quality_records.append(record)
            if record["decision"] == "reject_without_listening":
                reasons = "; ".join(record["reject_reasons"]) or "unknown quality rejection"
                rejected_wav = args.out_dir / f"config_f_{key}.rejected.wav"
                rejected_metrics = args.out_dir / f"config_f_{key}.rejected.json"
                record["metrics"]["path"] = str(rejected_wav)
                _write_quality_fields(temp_metrics, thresholds, record)
                temp_wav.replace(rejected_wav)
                temp_metrics.replace(rejected_metrics)
                failure = RuntimeError(f"{key}: audio-quality gate rejected sample: {reasons}")
                break
            temp_wav.replace(wav)
            temp_metrics.replace(metrics)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            failure = exc
            break
        finally:
            temp_wav.unlink(missing_ok=True)
            temp_metrics.unlink(missing_ok=True)

    report_path, summary_path = write_quality_report(quality_out_dir, thresholds, quality_records)
    print(f"Wrote audio-quality report: {summary_path}", file=sys.stderr)
    if failure is not None:
        print(f"bakeoff listen generation failed: {failure}", file=sys.stderr)
        return 1

    print(f"Wrote WAV + JSON under: {args.out_dir}")
    print(f"Wrote audio-quality JSON under: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
