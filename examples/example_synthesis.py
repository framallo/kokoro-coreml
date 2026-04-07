#!/usr/bin/env python3
"""CLI example: synthesize one line with ``HybridTTSPipeline`` (CoreML or PyTorch).

Run from repo root::

    python examples/example_synthesis.py --text "Hello world" --voice af_heart
    python examples/example_synthesis.py --engine pytorch --text "Debug" --voice af_heart

See ``--help`` for ``--out`` and ``--speed``. For automated checks, use ``pytest`` under
``tests/``; this script is for manual demos and local integration smoke tests.
"""

import argparse
import time
import numpy as np
import wave
from pathlib import Path
from kokoro.coreml_pipeline import HybridTTSPipeline


class AudioOutputConstants:
    """16-bit mono WAV at Kokoro sample rate."""

    DEFAULT_SAMPLE_RATE = 24000
    PCM_BIT_DEPTH = 16
    CHANNELS = 1
    PEAK_SAFETY_MARGIN = 1e-7
    NORMALIZATION_SCALE = 32767.0
    AUDIO_CLIP_MIN = -1.0
    AUDIO_CLIP_MAX = 1.0


def save_wav(path: str, audio: np.ndarray, sample_rate: int = AudioOutputConstants.DEFAULT_SAMPLE_RATE):
    """Write float32 mono audio to 16-bit PCM WAV; creates parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Normalize to int16 safely using defined constants
    if audio.size == 0:
        data = np.zeros((0,), dtype=np.int16)
    else:
        peak = max(AudioOutputConstants.PEAK_SAFETY_MARGIN, float(np.max(np.abs(audio))))
        scaled = np.clip(audio / peak, AudioOutputConstants.AUDIO_CLIP_MIN, AudioOutputConstants.AUDIO_CLIP_MAX)
        data = (scaled * AudioOutputConstants.NORMALIZATION_SCALE).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(AudioOutputConstants.CHANNELS)
        wf.setsampwidth(AudioOutputConstants.PCM_BIT_DEPTH // 8)  # Convert bits to bytes
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def main():
    """Parse argv, run ``HybridTTSPipeline``, write WAV, print timing line. Exit 0/1."""
    ap = argparse.ArgumentParser(
        description='Kokoro TTS Command-Line Interface with Hybrid Engine Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --text "Hello world" --voice af_heart
  %(prog)s --engine pytorch --text "Debug test" --voice af_bella
  %(prog)s --text "Fast speech" --speed 1.5 --out fast.wav

Engines:
  coreml   - Apple Neural Engine optimized (default, recommended)
  pytorch  - Native PyTorch implementation (debugging, development)

Common Voices:
  af_heart, af_bella, am_adam, am_david, bf_emma, bm_george
  (Voice availability depends on selected engine)
        """
    )
    
    ap.add_argument('--engine', choices=['coreml', 'pytorch'], default='coreml',
                   help='TTS synthesis engine (default: coreml)')
    ap.add_argument('--text', required=True,
                   help='Text to synthesize (required)')
    ap.add_argument('--voice', default='af_heart',
                   help='Voice model name (default: af_heart)')
    ap.add_argument('--speed', type=float, default=1.0,
                   help='Speech speed multiplier (default: 1.0)')
    ap.add_argument('--out', default='outputs/out.wav',
                   help='Output WAV file path (default: outputs/out.wav)')
    
    args = ap.parse_args()

    # Initialize synthesis pipeline with engine selection
    # HybridTTSPipeline handles engine availability and fallback logic
    try:
        p = HybridTTSPipeline(force_engine=args.engine)
    except Exception as e:
        print(f"FAIL engine initialization: {e}")
        return 1

    # Execute synthesis with precision timing for performance analysis
    print(f"Synthesizing with {args.engine} engine: '{args.text[:50]}{'...' if len(args.text) > 50 else ''}'")
    t0 = time.time()
    
    try:
        audio, sr = p.synthesize(args.text, voice=args.voice, speed=args.speed)
    except Exception as e:
        print(f"FAIL synthesis error: {e}")
        return 1
    
    t1 = time.time()

    # Validate synthesis output before proceeding to file output
    if audio is None or len(audio) == 0:
        print('FAIL synthesis returned no audio')
        return 1

    # Save audio with professional formatting and error handling
    try:
        save_wav(args.out, audio, sr)
    except Exception as e:
        print(f"FAIL audio output error: {e}")
        return 1

    # Calculate and report comprehensive performance metrics
    audio_len = len(audio) / sr  # Duration in seconds
    synth_time = t1 - t0  # Synthesis time in seconds
    rtf = synth_time / audio_len if audio_len > 0 else float('inf')
    
    # Performance report in parseable format for automation and monitoring
    print(f"engine={args.engine} time_sec={synth_time:.3f} audio_sec={audio_len:.3f} rtf={rtf:.3f} out={args.out}")
    
    return 0


if __name__ == '__main__':
    main()
