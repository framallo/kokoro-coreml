"""Tests for ``scripts/audio_quality_probe.py`` without optional plotting deps."""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audio_quality_probe",
    _ROOT / "scripts" / "audio_quality_probe.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())


def _speech_like_pcm(seconds: float = 1.0) -> np.ndarray:
    sr = 24_000
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    signal = 0.35 * np.sin(2.0 * np.pi * 220.0 * t)
    signal += 0.08 * np.sin(2.0 * np.pi * 880.0 * t)
    signal += 0.02 * np.sin(2.0 * np.pi * 1760.0 * t)
    return np.clip(signal * 32767.0, -32767, 32767).astype(np.int16)


def test_silence_rejects_without_listening(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    silence = tmp_path / "silence.wav"
    _write_wav(ref, _speech_like_pcm())
    _write_wav(silence, np.zeros(24_000, dtype=np.int16))

    thresholds = _mod.derive_thresholds([_mod.compute_metrics(ref)])
    decision, reasons = _mod.classify_metrics(
        _mod.compute_metrics(silence),
        thresholds,
        is_reference=False,
    )

    assert decision == "reject_without_listening"
    assert any("rms" in reason for reason in reasons)
    assert any("active32" in reason for reason in reasons)


def test_reference_classifies_as_reference_pass(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    _write_wav(ref, _speech_like_pcm())

    metrics = _mod.compute_metrics(ref)
    thresholds = _mod.derive_thresholds([metrics])
    decision, reasons = _mod.classify_metrics(metrics, thresholds, is_reference=True)

    assert decision == "reference_pass"
    assert reasons == []


def test_impulse_rejects_on_activity_or_clipping(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    impulse = tmp_path / "impulse.wav"
    pcm = np.zeros(24_000, dtype=np.int16)
    pcm[::2000] = 32767
    _write_wav(ref, _speech_like_pcm())
    _write_wav(impulse, pcm)

    thresholds = _mod.derive_thresholds([_mod.compute_metrics(ref)])
    decision, reasons = _mod.classify_metrics(
        _mod.compute_metrics(impulse),
        thresholds,
        is_reference=False,
    )

    assert decision == "reject_without_listening"
    assert any("active32" in reason or "zcr" in reason for reason in reasons)


def test_clipped_output_rejects_without_listening(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    clipped = tmp_path / "clipped.wav"
    _write_wav(ref, _speech_like_pcm())
    _write_wav(clipped, np.full(24_000, 32767, dtype=np.int16))

    thresholds = _mod.derive_thresholds([_mod.compute_metrics(ref)])
    decision, reasons = _mod.classify_metrics(
        _mod.compute_metrics(clipped),
        thresholds,
        is_reference=False,
    )

    assert decision == "reject_without_listening"
    assert any("clipped" in reason for reason in reasons)
