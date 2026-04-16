#!/usr/bin/env python3
"""Inspect WAV files for speech-health signals before human listening.

The probe is intentionally dependency-light: stdlib + numpy only. It reports
waveform statistics, coarse spectral features, and optional PPM raster plots so
bad samples can be rejected before asking a human to listen.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SAMPLE_RATE = 24_000
PCM_MAX = 32767.0


@dataclass(frozen=True)
class AudioMetrics:
    path: str
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_s: float
    rms_pcm: float
    peak_pcm: int
    dc_offset_pcm: float
    active_fraction_gt32: float
    active_fraction_gt128: float
    clipped_fraction: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    voiced_band_energy_ratio: float


@dataclass(frozen=True)
class Thresholds:
    min_rms_pcm: float
    min_active_fraction_gt32: float
    min_zero_crossing_rate: float
    min_voiced_band_energy_ratio: float
    max_clipped_fraction: float = 0.01


def read_wav_pcm(path: Path) -> tuple[int, int, int, np.ndarray]:
    """Return ``(sample_rate, channels, sample_width, int16_pcm)`` for a mono WAV."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got sample_width={sample_width}")
    pcm = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return sample_rate, channels, sample_width, pcm


def compute_metrics(path: Path) -> AudioMetrics:
    """Compute waveform and coarse spectral metrics for one WAV file."""
    sample_rate, channels, sample_width, pcm_i16 = read_wav_pcm(path)
    pcm = pcm_i16.astype(np.float64)
    frames = int(pcm.size)
    if frames == 0:
        return AudioMetrics(
            path=str(path),
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frames=0,
            duration_s=0.0,
            rms_pcm=0.0,
            peak_pcm=0,
            dc_offset_pcm=0.0,
            active_fraction_gt32=0.0,
            active_fraction_gt128=0.0,
            clipped_fraction=0.0,
            zero_crossing_rate=0.0,
            spectral_centroid_hz=0.0,
            voiced_band_energy_ratio=0.0,
        )

    abs_pcm = np.abs(pcm)
    signs = np.signbit(pcm)
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1])) if frames > 1 else 0
    y = pcm / PCM_MAX
    spectral_centroid_hz, voiced_band_energy_ratio = _spectral_features(y, sample_rate)
    return AudioMetrics(
        path=str(path),
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frames=frames,
        duration_s=frames / float(sample_rate) if sample_rate else 0.0,
        rms_pcm=float(math.sqrt(float(np.mean(pcm * pcm)))),
        peak_pcm=int(abs_pcm.max()),
        dc_offset_pcm=float(pcm.mean()),
        active_fraction_gt32=float(np.mean(abs_pcm > 32)),
        active_fraction_gt128=float(np.mean(abs_pcm > 128)),
        clipped_fraction=float(np.mean(abs_pcm >= 32760)),
        zero_crossing_rate=float(crossings / frames),
        spectral_centroid_hz=spectral_centroid_hz,
        voiced_band_energy_ratio=voiced_band_energy_ratio,
    )


def _spectral_features(y: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Return coarse spectral centroid and speech-band energy ratio."""
    if y.size == 0 or float(np.max(np.abs(y))) <= 0.0:
        return 0.0, 0.0
    # Limit FFT cost for long clips while keeping deterministic global sampling.
    max_samples = sample_rate * 12
    if y.size > max_samples:
        idx = np.linspace(0, y.size - 1, max_samples).astype(np.int64)
        y = y[idx]
    window = np.hanning(y.size)
    spectrum = np.fft.rfft(y * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(y.size, 1.0 / sample_rate)
    total = float(power[(freqs >= 20.0) & (freqs <= 8000.0)].sum())
    if total <= 1e-18:
        return 0.0, 0.0
    centroid = float((freqs * power).sum() / max(float(power.sum()), 1e-18))
    voiced = float(power[(freqs >= 80.0) & (freqs <= 4000.0)].sum())
    return centroid, voiced / total


def derive_thresholds(reference_metrics: Iterable[AudioMetrics]) -> Thresholds:
    """Derive conservative rejection thresholds from known-good references."""
    refs = list(reference_metrics)
    if not refs:
        raise ValueError("at least one reference WAV is required")
    min_rms = min(m.rms_pcm for m in refs)
    min_active = min(m.active_fraction_gt32 for m in refs)
    min_zcr = min(m.zero_crossing_rate for m in refs)
    min_voiced = min(m.voiced_band_energy_ratio for m in refs)
    return Thresholds(
        min_rms_pcm=max(500.0, min_rms * 0.25),
        min_active_fraction_gt32=max(0.05, min_active * 0.25),
        min_zero_crossing_rate=max(0.005, min_zcr * 0.25),
        min_voiced_band_energy_ratio=max(0.20, min_voiced * 0.50),
    )


def classify_metrics(metrics: AudioMetrics, thresholds: Thresholds, *, is_reference: bool) -> tuple[str, list[str]]:
    """Classify one sample as reference pass, needs listening, or reject."""
    if is_reference:
        return "reference_pass", []
    reasons: list[str] = []
    if metrics.sample_rate != SAMPLE_RATE:
        reasons.append(f"sample_rate {metrics.sample_rate} != {SAMPLE_RATE}")
    if metrics.channels != 1:
        reasons.append(f"channels {metrics.channels} != 1")
    if metrics.frames <= 0:
        reasons.append("empty wav")
    if metrics.peak_pcm <= 0:
        reasons.append("zero peak")
    if metrics.rms_pcm < thresholds.min_rms_pcm:
        reasons.append(f"rms {metrics.rms_pcm:.1f} < {thresholds.min_rms_pcm:.1f}")
    if metrics.active_fraction_gt32 < thresholds.min_active_fraction_gt32:
        reasons.append(
            f"active32 {metrics.active_fraction_gt32:.3%} < "
            f"{thresholds.min_active_fraction_gt32:.3%}"
        )
    if metrics.zero_crossing_rate < thresholds.min_zero_crossing_rate:
        reasons.append(
            f"zcr {metrics.zero_crossing_rate:.3%} < "
            f"{thresholds.min_zero_crossing_rate:.3%}"
        )
    if metrics.voiced_band_energy_ratio < thresholds.min_voiced_band_energy_ratio:
        reasons.append(
            f"voiced_band {metrics.voiced_band_energy_ratio:.3f} < "
            f"{thresholds.min_voiced_band_energy_ratio:.3f}"
        )
    if metrics.clipped_fraction > thresholds.max_clipped_fraction:
        reasons.append(f"clipped {metrics.clipped_fraction:.3%} > {thresholds.max_clipped_fraction:.3%}")
    return ("reject_without_listening" if reasons else "needs_listening"), reasons


def write_waveform_ppm(path: Path, pcm_i16: np.ndarray, width: int = 1200, height: int = 320) -> None:
    """Write a simple waveform raster as PPM."""
    y = pcm_i16.astype(np.float32) / PCM_MAX
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    mid = height // 2
    for gy in np.linspace(0, height - 1, 9).astype(int):
        img[gy : gy + 1, :, :] = 230
    for gx in np.linspace(0, width - 1, 13).astype(int):
        img[:, gx : gx + 1, :] = 235
    if y.size:
        bins = np.linspace(0, y.size, width + 1).astype(int)
        for x in range(width):
            seg = y[bins[x] : bins[x + 1]]
            if not seg.size:
                continue
            lo = int(mid - float(np.max(seg)) * (height * 0.45))
            hi = int(mid - float(np.min(seg)) * (height * 0.45))
            lo = max(0, min(height - 1, lo))
            hi = max(0, min(height - 1, hi))
            if lo > hi:
                lo, hi = hi, lo
            img[lo : hi + 1, x, :] = (20, 78, 140)
    img[mid : mid + 1, :, :] = 180
    _write_ppm(path, img)


def write_spectrogram_ppm(path: Path, pcm_i16: np.ndarray, sample_rate: int, width: int = 1200, height: int = 420) -> None:
    """Write a coarse spectrogram raster as PPM."""
    y = pcm_i16.astype(np.float32) / PCM_MAX
    nfft = 512
    hop = 128
    if y.size < nfft:
        y = np.pad(y, (0, nfft - y.size))
    window = np.hanning(nfft).astype(np.float32)
    frames = []
    for start in range(0, max(1, y.size - nfft + 1), hop):
        frame = y[start : start + nfft]
        if frame.size < nfft:
            frame = np.pad(frame, (0, nfft - frame.size))
        frames.append(np.abs(np.fft.rfft(frame * window)))
    spec = np.stack(frames, axis=1)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    spec = spec[freqs <= 8000.0, :]
    db = 20.0 * np.log10(spec + 1e-6)
    lo, hi = np.percentile(db, [5, 99])
    norm = (db - lo) / max(float(hi - lo), 1e-6)
    yy = np.linspace(norm.shape[0] - 1, 0, height).astype(int)
    xx = np.linspace(0, norm.shape[1] - 1, width).astype(int)
    img = _heatmap(norm[yy][:, xx])
    _write_ppm(path, img)


def _heatmap(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(4.0 * v - 0.5, 0.0, 1.0)
    g = np.clip(4.0 * v - 1.5, 0.0, 1.0)
    b = np.clip(1.5 - 2.5 * v, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _write_ppm(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = img.shape
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(np.asarray(img, dtype=np.uint8).tobytes())


def _sample_record(path: Path, role: str, thresholds: Thresholds, plots_dir: Path | None) -> dict:
    metrics = compute_metrics(path)
    decision, reasons = classify_metrics(metrics, thresholds, is_reference=(role == "reference"))
    record = {
        "role": role,
        "decision": decision,
        "reject_reasons": reasons,
        "metrics": asdict(metrics),
    }
    if plots_dir is not None:
        sample_rate, _channels, _width, pcm = read_wav_pcm(path)
        stem = path.stem
        wave_path = plots_dir / f"{stem}_waveform.ppm"
        spec_path = plots_dir / f"{stem}_spectrogram.ppm"
        write_waveform_ppm(wave_path, pcm)
        write_spectrogram_ppm(spec_path, pcm, sample_rate)
        record["plots"] = {"waveform": str(wave_path), "spectrogram": str(spec_path)}
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="*", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots" if args.plots else None
    refs = [compute_metrics(path) for path in args.reference]
    thresholds = derive_thresholds(refs)

    records = []
    for path in args.reference:
        records.append(_sample_record(path, "reference", thresholds, plots_dir))
    for path in args.candidate:
        records.append(_sample_record(path, "candidate", thresholds, plots_dir))

    report = {
        "thresholds": asdict(thresholds),
        "samples": records,
    }
    report_path = args.out_dir / "audio_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary_path = args.out_dir / "audio_quality_summary.md"
    lines = [
        "# Audio Quality Probe Summary",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in asdict(thresholds).items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Samples", ""]
    for record in records:
        metrics = record["metrics"]
        reasons = "; ".join(record["reject_reasons"]) or "-"
        lines.append(
            f"- `{Path(metrics['path']).name}`: `{record['decision']}` "
            f"duration={metrics['duration_s']:.3f}s rms={metrics['rms_pcm']:.1f} "
            f"active32={metrics['active_fraction_gt32']:.3%} "
            f"zcr={metrics['zero_crossing_rate']:.3%} reasons={reasons}"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(report_path), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
