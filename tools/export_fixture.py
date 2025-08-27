#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import argparse

# Reuse existing pipeline code
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from test_ane_pipeline import HybridTTSPipeline  # type: ignore


def flatten(arr: np.ndarray):
    return arr.reshape(-1).astype(np.float32).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--text', default='Hello Matt, this is Kokoro running on Apple Neural Engine.')
    p.add_argument('--voice', default='af_heart')
    p.add_argument('--speed', type=float, default=1.0)
    p.add_argument('--out', default=str(BASE / 'Swift/KokoroPhase2/Fixtures/fixture_decoder_only_5s.json'))
    args = p.parse_args()

    pipeline = HybridTTSPipeline(force_engine='coreml')
    vi = pipeline.extract_vocoder_inputs(args.text, voice=args.voice, speed=args.speed)

    # Prepare fixed 5s bucket shapes
    shapes = {
        'asr': [1, 512, 1, 200],
        'f0_curve': [1, 1, 1, 400],
        'n': [1, 1, 1, 400],
        's': [1, 128],
    }

    # Pad/truncate
    asr = vi['asr'].astype(np.float32)
    asr_pad = np.zeros((1, 512, 200), dtype=np.float32)
    t_asr = min(200, asr.shape[-1])
    if t_asr > 0:
        asr_pad[:, :, :t_asr] = asr[:, :, :t_asr]

    f0 = vi['f0_curve'].astype(np.float32)  # (1, T_f0)
    f0_pad = np.zeros((1, 400), dtype=np.float32)
    t_f0 = min(400, f0.shape[-1])
    if t_f0 > 0:
        f0_pad[:, :t_f0] = f0[:, :t_f0]

    n = vi['n'].astype(np.float32)  # (1, T_f0)
    n_pad = np.zeros((1, 400), dtype=np.float32)
    t_n = min(400, n.shape[-1])
    if t_n > 0:
        n_pad[:, :t_n] = n[:, :t_n]

    s = vi['s'].astype(np.float32)

    out = {
        'text': args.text,
        'voice': args.voice,
        'shapes': shapes,
        'asr': flatten(asr_pad.reshape(1, 512, 1, 200)),
        'f0_curve': flatten(f0_pad.reshape(1, 1, 1, 400)),
        'n': flatten(n_pad.reshape(1, 1, 1, 400)),
        's': flatten(s.reshape(1, 128)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f"wrote fixture: {out_path}")


if __name__ == '__main__':
    main()
