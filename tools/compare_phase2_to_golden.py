#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


def dbfs(x: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(x) + 1e-12))
    return 20.0 * np.log10(max(rms, 1e-12))


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size:
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt(np.sum(vx*vx) * np.sum(vy*vy)) + 1e-12
    return float(np.sum(vx*vy) / denom)


def load_latest_golden(golden_root: Path) -> Path:
    if not golden_root.exists():
        raise SystemExit(f"No golden folder at {golden_root}")
    cands = sorted([p for p in golden_root.iterdir() if p.is_dir() and p.name.startswith('golden_')])
    if not cands:
        raise SystemExit("No golden_ directories found")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase2_dir', required=True)
    ap.add_argument('--golden_root', default='outputs/golden')
    args = ap.parse_args()

    phase2_dir = Path(args.phase2_dir)
    cand_wav = phase2_dir / 'output.wav'
    if not cand_wav.exists():
        raise SystemExit(f"missing output.wav in {phase2_dir}")

    golden_dir = load_latest_golden(Path(args.golden_root))
    golden_wav = golden_dir / 'output.wav'
    if not golden_wav.exists():
        raise SystemExit(f"missing golden output.wav in {golden_dir}")

    cand, sr1 = sf.read(cand_wav)
    gold, sr2 = sf.read(golden_wav)
    if sr1 != sr2:
        raise SystemExit(f"sample rate mismatch: {sr1} vs {sr2}")
    if cand.ndim > 1:
        cand = cand[:,0]
    if gold.ndim > 1:
        gold = gold[:,0]

    n = min(cand.size, gold.size)
    cand = cand[:n]
    gold = gold[:n]

    metrics = {
        'waveform': {
            'mse': float(np.mean((cand - gold)**2)),
            'mae': float(np.mean(np.abs(cand - gold))),
            'corr': corr(cand, gold),
            'dbfs_candidate': dbfs(cand),
            'dbfs_golden': dbfs(gold),
        }
    }

    out = phase2_dir / 'comparison.json'
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote: {out}")


if __name__ == '__main__':
    main()
