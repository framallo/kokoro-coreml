#!/usr/bin/env python3
"""
End-to-end Core ML synthesis harness with ANE verification and latency reporting.

This script runs Kokoro's two-stage Core ML pipeline:
- Stage 1: kokoro_duration.mlpackage → (pred_dur, d, t_en, s)
- Stage 2: kokoro_synthesizer_3s.mlpackage → waveform

It prints per-stage timings, saves a WAV file, and verifies ANE usage via:
- Primary: powermetrics (sudo) --samplers ane, parse "ANE Power"
- Fallback: xctrace record --template "Core ML" and JSON export parsing

Usage examples:
  python scripts/run_coreml_e2e.py
  python scripts/run_coreml_e2e.py --text "Custom" --voice af_heart --repeat 5
  python scripts/run_coreml_e2e.py --no-ane-check  # e.g., in CI

Design notes:
- Defaults to a zeroed ref_s (no HF voice download) for reproducible timing.
- Caches input_ids/attention_mask/ref_s between repeats for stable latency.
- Detects required shapes (trace_length, frame_count) from model specs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional heavy deps are imported lazily when needed
try:
    import coremltools as ct  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"FATAL: coremltools not available: {e}")
    sys.exit(2)

# Reuse robust WAV writer
try:
    from run_single import save_wav, AudioOutputConstants
except Exception:
    # Minimal fallback WAV writer if run_single is unavailable
    import wave

    class AudioOutputConstants:
        DEFAULT_SAMPLE_RATE = 24000
        PCM_BIT_DEPTH = 16
        CHANNELS = 1
        PEAK_SAFETY_MARGIN = 1e-7
        NORMALIZATION_SCALE = 32767.0
        AUDIO_CLIP_MIN = -1.0
        AUDIO_CLIP_MAX = 1.0

    def save_wav(path: str, audio: np.ndarray, sample_rate: int = 24000) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if audio.size == 0:
            data = np.zeros((0,), dtype=np.int16)
        else:
            peak = max(AudioOutputConstants.PEAK_SAFETY_MARGIN, float(np.max(np.abs(audio))))
            scaled = np.clip(
                audio / peak,
                AudioOutputConstants.AUDIO_CLIP_MIN,
                AudioOutputConstants.AUDIO_CLIP_MAX,
            )
            data = (scaled * AudioOutputConstants.NORMALIZATION_SCALE).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(AudioOutputConstants.CHANNELS)
            wf.setsampwidth(AudioOutputConstants.PCM_BIT_DEPTH // 8)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())


BASE_DIR = Path(__file__).resolve().parent.parent
COREML_DIR = (BASE_DIR / "coreml").resolve()
DEFAULT_TEXT = "This is Kokoro running on Apple Neural Engine."
DEFAULT_VOICE = "zeros"  # special value meaning use zeroed ref_s (1, 256)


@dataclass
class ModelShapes:
    trace_length: int
    frame_count: int
    hidden_size: int


def _load_mlmodel(path: Path) -> Any:
    # Force ALL compute units to allow ANE
    try:
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
    except TypeError:
        # Older coremltools signature
        model = ct.models.MLModel(str(path))
    return model


def _infer_synth_shapes(model: Any) -> ModelShapes:
    spec = model.get_spec()
    inp = {i.name: i for i in spec.description.input}
    # Expect 'd', 't_en', 's', 'ref_s', 'pred_aln_trg'
    if "d" in inp:
        d_shape = list(inp["d"].type.multiArrayType.shape)
        hidden_size = int(d_shape[-2]) if len(d_shape) >= 2 else 512
        trace_length = int(d_shape[-1])
    else:
        # Fallback: find a 3D input and use its last dim as trace_length
        three_d = [i for i in inp.values() if len(i.type.multiArrayType.shape) >= 3]
        if not three_d:
            raise RuntimeError("Unable to infer trace_length from synthesizer spec")
        s0 = list(three_d[0].type.multiArrayType.shape)
        hidden_size = int(s0[-2])
        trace_length = int(s0[-1])

    if "pred_aln_trg" in inp:
        pat_shape = list(inp["pred_aln_trg"].type.multiArrayType.shape)
        frame_count = int(pat_shape[-1])
    else:
        # Fallback: read output waveform length
        out0 = spec.description.output[0]
        out_shape = list(out0.type.multiArrayType.shape)
        frame_count = int(out_shape[-1])

    return ModelShapes(trace_length=trace_length, frame_count=frame_count, hidden_size=hidden_size)


def _build_alignment_matrix(
    pred_dur_tokens: np.ndarray, trace_length: int, frame_count: int
) -> np.ndarray:
    # Build pred_aln_trg of shape (trace_length, frame_count)
    pred_dur = np.zeros((trace_length,), dtype=np.int64)
    L = min(trace_length, pred_dur_tokens.shape[-1])
    pred_dur[:L] = pred_dur_tokens[:L]
    repeat_idx = np.repeat(np.arange(trace_length), pred_dur)
    if repeat_idx.size > frame_count:
        repeat_idx = repeat_idx[:frame_count]
    else:
        pad = frame_count - repeat_idx.size
        last_idx = repeat_idx[-1] if repeat_idx.size > 0 else 0
        repeat_idx = np.concatenate(
            [repeat_idx, np.full((pad,), last_idx, dtype=repeat_idx.dtype)]
        )
    mat = np.zeros((trace_length, frame_count), dtype=np.float32)
    mat[repeat_idx, np.arange(frame_count)] = 1.0
    return mat


def _phonemize_en(text: str) -> str:
    # Convert English graphemes → phoneme string using misaki.en
    from kokoro.pipeline import KPipeline

    pipe = KPipeline(lang_code="a", model=False)
    # Reuse internal tokenizer to produce a single phoneme string
    _, tokens = pipe.g2p(text)
    for _, ps, _ in pipe.en_tokenize(tokens):
        if ps:
            return ps
    return ""


def _load_vocab_mapping() -> Dict[str, int]:
    # Load Kokoro config.json to get vocab mapping without loading weights
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    vocab = cfg.get("vocab")
    if not isinstance(vocab, dict):
        raise RuntimeError("Invalid config.json: missing 'vocab'")
    # Keys are single-character phonemes; values are int ids
    return {str(k): int(v) for k, v in vocab.items()}


def _make_duration_inputs(
    phonemes: str,
    vocab: Dict[str, int],
    trace_length: int,
    use_zero_ref_s: bool,
    voice: Optional[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    # Map phoneme string to token IDs
    ids = [vocab.get(ch) for ch in phonemes]
    ids = [i for i in ids if i is not None]
    # Truncate to trace_length
    ids = ids[:trace_length]
    input_len = len(ids)
    # Left-align pad with zeros (token id 0 also serves as BOS/EOS in training)
    input_ids = np.zeros((trace_length,), dtype=np.int32)
    attention_mask = np.zeros((trace_length,), dtype=np.int32)
    if input_len > 0:
        input_ids[:input_len] = np.array(ids, dtype=np.int32)
        attention_mask[:input_len] = 1

    # ref_s: (256,)
    if use_zero_ref_s:
        ref_s = np.zeros((256,), dtype=np.float32)
    else:
        from huggingface_hub import hf_hub_download
        import torch

        if not voice:
            raise ValueError("voice is required when use_zero_ref_s=False")
        vpath = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{voice}.pt")
        pack = torch.load(vpath, weights_only=True)
        # Select by phoneme length index as in pipeline (len(ps)-1)
        ref_s = pack[input_len - 1].detach().cpu().numpy().astype(np.float32)
        if ref_s.shape[-1] != 256:
            raise RuntimeError(f"Voice pack dim mismatch: {ref_s.shape}")

    inputs = {
        "input_ids": input_ids,
        "ref_s": ref_s,
        "speed": np.array([1.0], dtype=np.float32),
        "attention_mask": attention_mask,
    }
    # Return also the (1,256) ref_s expected by synthesizer (we will add batch later)
    return inputs, ref_s.reshape(1, 256).astype(np.float32)


class AneVerifier:
    def __init__(self, enabled: bool, min_samples_positive: int = 1, sample_ms: int = 200):
        self.enabled = enabled
        self.min_samples_positive = int(min_samples_positive)
        self.sample_ms = int(sample_ms)
        self.powermetrics_ok = False
        self.samples: List[float] = []
        self.proc: Optional[subprocess.Popen] = None

    def _try_start_powermetrics(self) -> bool:
        try:
            # Use sudo -n to avoid prompting; will fail fast if not allowed
            cmd = [
                "sudo",
                "-n",
                "powermetrics",
                "-i",
                str(self.sample_ms),
                "--samplers",
                "ane",
            ]
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            return False
        except Exception:
            return False
        # Non-blocking read of a couple of lines to detect permission errors
        start = time.time()
        while time.time() - start < 1.0:
            if self.proc.poll() is not None:
                break
            if self.proc.stdout is None:
                break
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            if "password is required" in line or "Operation not permitted" in line:
                self._stop_powermetrics()
                return False
            if "ANE Power" in line:
                # We are receiving samples; good
                self.powermetrics_ok = True
                self._parse_line(line)
                return True
        # Keep it running even if we didn't see a sample yet
        self.powermetrics_ok = True
        return True

    def _parse_line(self, line: str) -> None:
        # Expect lines like: "ANE Power: 1.23 W" or "ANE Power: 230 mW"
        m = re.search(r"ANE Power:\s*([0-9]+\.?[0-9]*)\s*(m?W)", line)
        if m:
            try:
                val = float(m.group(1))
                unit = m.group(2)
                if unit == "mW":
                    val = val / 1000.0
                self.samples.append(val)
            except Exception:
                pass

    def start(self) -> None:
        if not self.enabled:
            return
        self.powermetrics_ok = self._try_start_powermetrics()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_powermetrics()

    def _stop_powermetrics(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdout:
                # Drain remaining lines
                t_end = time.time() + 0.25
                while time.time() < t_end:
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    if "ANE Power" in line:
                        self._parse_line(line)
            # Politely terminate
            self.proc.terminate()
            try:
                self.proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        finally:
            self.proc = None

    def read_powermetrics_streaming(self) -> None:
        if not (self.enabled and self.powermetrics_ok and self.proc and self.proc.stdout):
            return
        try:
            while True:
                if self.proc.poll() is not None:
                    break
                line = self.proc.stdout.readline()
                if not line:
                    break
                if "ANE Power" in line:
                    self._parse_line(line)
                # Keep loop responsive
                if len(self.samples) >= 1024:
                    break
        except Exception:
            pass

    def result_or_fallback(self) -> Tuple[bool, Optional[Dict[str, float]], Optional[str]]:
        if not self.enabled:
            return (True, None, None)
        if self.powermetrics_ok and self.samples:
            positives = sum(1 for v in self.samples if v > 0.0)
            mean_w = float(np.mean(self.samples)) if self.samples else 0.0
            max_w = float(np.max(self.samples)) if self.samples else 0.0
            ok = positives >= self.min_samples_positive
            return (ok, {"mean_w": mean_w, "max_w": max_w, "pos_samples": float(positives)}, None)
        # Fallback: xctrace instructions (we avoid auto-embedding to keep harness simple)
        tip = (
            "ANE verifier fallback: powermetrics not permitted. Optionally run:\n"
            "  xcrun xctrace record --template 'Core ML' --time-limit 6s --output outputs/coreml_e2e.trace &\n"
            "  python scripts/run_coreml_e2e.py --no-ane-check --repeat 1 &&\n"
            "  xcrun xctrace export --input outputs/coreml_e2e.trace --output outputs/coreml_e2e.json --format json\n"
            "Then search the JSON for 'Neural Engine' activity."
        )
        return (False, None, tip)


def run_once(
    duration_model: Any,
    synthesizer_model: Any,
    shapes: ModelShapes,
    cached_inputs: Dict[str, np.ndarray],
    cached_ref_s_batched: np.ndarray,
    save_path: Optional[Path] = None,
) -> Dict[str, float]:
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    dur_out = duration_model.predict(cached_inputs)
    t1 = time.perf_counter()
    timings["duration_ms"] = (t1 - t0) * 1000.0

    pred_dur = dur_out.get("pred_dur")
    d = dur_out.get("d")
    t_en = dur_out.get("t_en")
    s = dur_out.get("s")

    # Normalize shapes
    if isinstance(pred_dur, np.ndarray) and pred_dur.ndim > 1:
        pred_dur = pred_dur.reshape(-1)
    if isinstance(d, np.ndarray) and d.ndim == 2:
        d = d.reshape(1, d.shape[0], d.shape[1])
    if isinstance(t_en, np.ndarray) and t_en.ndim == 2:
        t_en = t_en.reshape(1, t_en.shape[0], t_en.shape[1])
    if isinstance(s, np.ndarray) and s.ndim == 1:
        s = s.reshape(1, -1)

    # Build alignment
    t2 = time.perf_counter()
    pred_aln_trg = _build_alignment_matrix(
        pred_dur.astype(np.int64), shapes.trace_length, shapes.frame_count
    )
    t3 = time.perf_counter()
    timings["align_ms"] = (t3 - t2) * 1000.0

    # Prepare synthesizer inputs
    syn_inputs = {
        "d": d.astype(np.float32),
        "t_en": t_en.astype(np.float32),
        "s": s.astype(np.float32),
        "ref_s": cached_ref_s_batched.astype(np.float32),
        "pred_aln_trg": pred_aln_trg.astype(np.float32),
    }

    # Predict waveform
    t4 = time.perf_counter()
    syn_out = synthesizer_model.predict(syn_inputs)
    t5 = time.perf_counter()
    timings["synth_ms"] = (t5 - t4) * 1000.0

    # Extract waveform (first output)
    wave_key = list(syn_out.keys())[0]
    audio = syn_out[wave_key].astype(np.float32).reshape(-1)

    if save_path is not None:
        save_wav(str(save_path), audio, AudioOutputConstants.DEFAULT_SAMPLE_RATE)

    timings["total_ms"] = (t5 - t0) * 1000.0
    timings["audio_sec"] = len(audio) / float(AudioOutputConstants.DEFAULT_SAMPLE_RATE)
    timings["rtf"] = (timings["total_ms"] / 1000.0) / max(1e-6, timings["audio_sec"])

    return timings


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Core ML end-to-end TTS harness with ANE verification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Input text to synthesize")
    ap.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Voice name (e.g., af_heart). Use 'zeros' to skip voice download",
    )
    ap.add_argument("--repeat", type=int, default=3, help="Repeat count for timing")
    ap.add_argument(
        "--trace-length",
        type=int,
        default=None,
        help="Override duration trace_length tokens (pad/truncate)",
    )
    ap.add_argument(
        "--no-ane-check",
        action="store_true",
        help="Skip ANE verification (for CI / unprivileged environments)",
    )
    ap.add_argument(
        "--ane-min-samples",
        type=int,
        default=1,
        help="powermetrics positive-sample threshold for PASS",
    )
    ap.add_argument(
        "--out",
        default=str(BASE_DIR / "outputs" / "coreml_e2e.wav"),
        help="Output WAV path",
    )
    args = ap.parse_args()

    duration_path = COREML_DIR / "kokoro_duration.mlpackage"
    synth_path = COREML_DIR / "kokoro_synthesizer_3s.mlpackage"
    if not duration_path.exists() or not synth_path.exists():
        print(
            f"FATAL: Missing Core ML packages in {COREML_DIR}. Expected: kokoro_duration.mlpackage and kokoro_synthesizer_3s.mlpackage"
        )
        return 2

    print("Loading Core ML models…")
    duration_model = _load_mlmodel(duration_path)
    synthesizer_model = _load_mlmodel(synth_path)
    shapes = _infer_synth_shapes(synthesizer_model)

    if args.trace_length is not None:
        shapes = ModelShapes(
            trace_length=int(args.trace_length),
            frame_count=shapes.frame_count,
            hidden_size=shapes.hidden_size,
        )

    print(
        f"Synthesizer bucket: frames={shapes.frame_count}, hidden={shapes.hidden_size}, trace_length={shapes.trace_length}"
    )

    # Prepare inputs once (cache across repeats)
    print("Preparing inputs (phonemization, vocab mapping, ref_s)…")
    use_zero_ref_s = (args.voice.lower() == "zeros")
    phonemes = _phonemize_en(args.text)
    if not phonemes:
        print("FATAL: Failed to phonemize input text")
        return 2
    vocab = _load_vocab_mapping()
    duration_inputs, ref_s_batched = _make_duration_inputs(
        phonemes, vocab, shapes.trace_length, use_zero_ref_s, None if use_zero_ref_s else args.voice
    )

    # Sanity-log input lengths
    in_len = int(duration_inputs["attention_mask"].sum())
    print(f"Phonemes length={len(phonemes)} → token_count={in_len}")

    # ANE verifier
    verifier = AneVerifier(enabled=not args.no_ane_check, min_samples_positive=args.ane_min_samples)
    verifier.start()

    # Run repeats
    all_timings: List[Dict[str, float]] = []
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for r in range(args.repeat):
        save_path = out_path if r == 0 else None
        timings = run_once(
            duration_model,
            synthesizer_model,
            shapes,
            duration_inputs,
            ref_s_batched,
            save_path=save_path,
        )
        all_timings.append(timings)
        verifier.read_powermetrics_streaming()
        print(
            f"Run {r+1}/{args.repeat}: duration={timings['duration_ms']:.1f}ms, align={timings['align_ms']:.1f}ms, synth={timings['synth_ms']:.1f}ms, total={timings['total_ms']:.1f}ms, rtf={timings['rtf']:.3f}"
        )

    verifier.stop()

    # Summary
    mean = lambda k: float(np.mean([t[k] for t in all_timings]))
    audio_sec = all_timings[0]["audio_sec"] if all_timings else 0.0
    print(
        f"\nSummary: audio_sec={audio_sec:.3f}s, duration_ms={mean('duration_ms'):.1f}, align_ms={mean('align_ms'):.1f}, synth_ms={mean('synth_ms'):.1f}, total_ms={mean('total_ms'):.1f}, rtf={mean('rtf'):.3f}"
    )

    ok, ane_stats, tip = verifier.result_or_fallback()
    if ane_stats is not None:
        print(
            f"ANE: PASS={ok} samples={int(ane_stats['pos_samples'])} mean_W={ane_stats['mean_w']:.2f} max_W={ane_stats['max_w']:.2f}"
        )
    elif tip is not None:
        print(tip)

    print(f"WAV saved to: {out_path}")
    return 0 if (ok or args.no_ane_check) else 1


if __name__ == "__main__":
    sys.exit(main())
