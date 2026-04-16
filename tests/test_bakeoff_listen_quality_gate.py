"""Tests for bakeoff listen audio-quality annotations."""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bakeoff_listen",
    _ROOT / "scripts" / "bakeoff_listen.py",
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


def _write_metrics(path: Path, duration_s: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "canonical_audio_duration_s": duration_s,
                "observed_audio_duration_s": duration_s,
            }
        )
        + "\n"
    )


def _speech_like_pcm(seconds: float = 1.0) -> np.ndarray:
    sr = 24_000
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    signal = 0.35 * np.sin(2.0 * np.pi * 220.0 * t)
    signal += 0.08 * np.sin(2.0 * np.pi * 880.0 * t)
    signal += 0.02 * np.sin(2.0 * np.pi * 1760.0 * t)
    return np.clip(signal * 32767.0, -32767, 32767).astype(np.int16)


def test_validate_outputs_marks_speech_like_sample_quality_pass(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    candidate = tmp_path / "candidate.wav"
    metrics_path = tmp_path / "candidate.json"
    _write_wav(ref, _speech_like_pcm())
    _write_wav(candidate, _speech_like_pcm())
    _write_metrics(metrics_path)

    thresholds, _records = _mod._load_quality_context([ref], plots_dir=None)
    record = _mod._validate_outputs("3s", candidate, metrics_path, thresholds)
    payload = json.loads(metrics_path.read_text())

    assert record["decision"] == "needs_listening"
    assert payload["quality_pass"] is True
    assert payload["quality_decision"] == "needs_listening"
    assert payload["quality_reject_reasons"] == []
    assert payload["audio_quality"]["sample"]["metrics"]["rms_pcm"] > 0


def test_validate_outputs_marks_silence_as_quality_failure(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    candidate = tmp_path / "silence.wav"
    metrics_path = tmp_path / "silence.json"
    _write_wav(ref, _speech_like_pcm())
    _write_wav(candidate, np.zeros(24_000, dtype=np.int16))
    _write_metrics(metrics_path)

    thresholds, _records = _mod._load_quality_context([ref], plots_dir=None)
    record = _mod._validate_outputs("3s", candidate, metrics_path, thresholds)
    payload = json.loads(metrics_path.read_text())

    assert record["decision"] == "reject_without_listening"
    assert payload["quality_pass"] is False
    assert payload["quality_decision"] == "reject_without_listening"
    assert any("rms" in reason for reason in payload["quality_reject_reasons"])


def test_main_preserves_rejected_sample_without_overwriting_final(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ref = tmp_path / "ref.wav"
    out_dir = tmp_path / "listen"
    _write_wav(ref, _speech_like_pcm())

    def fake_run(cmd: list[str], cwd: str, check: bool) -> None:
        output_path = Path(cmd[cmd.index("--output") + 1])
        wav_path = Path(cmd[cmd.index("--wav") + 1])
        _write_wav(wav_path, np.zeros(24_000, dtype=np.int16))
        _write_metrics(output_path)

    monkeypatch.setattr(_mod, "_ensure_bench", lambda _bench: True)
    monkeypatch.setattr(_mod, "_ensure_inputs", lambda _keys, _inputs_dir, _hnsf: True)
    monkeypatch.setattr(_mod.subprocess, "run", fake_run)

    rc = _mod.main(
        [
            "--keys",
            "3s",
            "--out-dir",
            str(out_dir),
            "--reference-wavs",
            str(ref),
        ]
    )

    assert rc == 1
    assert not (out_dir / "config_f_3s.wav").exists()
    assert (out_dir / "config_f_3s.rejected.wav").exists()
    rejected_metrics = json.loads((out_dir / "config_f_3s.rejected.json").read_text())
    assert rejected_metrics["quality_pass"] is False
    assert rejected_metrics["quality_decision"] == "reject_without_listening"
