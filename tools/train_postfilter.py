#!/usr/bin/env python3
"""
Train a tiny post-filter to map Swift HAR iSTFT audio to Golden waveform.

Data: Use existing outputs created by kokoro-phase2-cli and the latest golden.
- Input X: outputs/local/phase2_*/output.wav (Swift HAR with KOKORO_FORCE_HAR=1 KOKORO_PHASE_SCALE=0.3)
- Target Y: outputs/golden/latest/output.wav
Assumes 5s bucket (T=120000 at 24 kHz). We'll slice/center to min length.

Exports Core ML via coremltools if available; otherwise writes TorchScript for later.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from postfilter_model import TinyPostFilter, loss_fn

SR = 24000
T_FIXED = 120000  # 5s


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]]):
        self.data = []
        print(f'Loading {len(pairs)} pairs into memory...')
        for x_path, y_path in pairs:
            x, sr1 = sf.read(str(x_path))
            y, sr2 = sf.read(str(y_path))
            assert sr1 == sr2 == SR
            n = min(len(x), len(y), T_FIXED)
            x = x[:n].astype(np.float32)
            y = y[:n].astype(np.float32)
            if n < T_FIXED:
                x = np.pad(x, (0, T_FIXED - n))
                y = np.pad(y, (0, T_FIXED - n))
            x = torch.from_numpy(x).unsqueeze(0)  # (1,T)
            y = torch.from_numpy(y).unsqueeze(0)
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def find_pairs(local_root: Path, golden_root: Path) -> list[tuple[Path, Path]]:
    # Use latest golden dir
    gdirs = sorted([p for p in golden_root.iterdir() if p.is_dir() and p.name.startswith('golden_')])
    if not gdirs:
        raise SystemExit('No golden dirs found')
    g_latest = gdirs[-1]
    y_path = g_latest / 'output.wav'
    # Pair against all local phase2 runs with output.wav
    pairs = []
    for run in sorted([p for p in local_root.iterdir() if p.is_dir() and p.name.startswith('phase2_')]):
        x_path = run / 'output.wav'
        if x_path.exists() and y_path.exists():
            pairs.append((x_path, y_path))
    if not pairs:
        raise SystemExit('No training pairs found under outputs/local')
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=16)
    ap.add_argument('--local_root', default='outputs/local')
    ap.add_argument('--golden_root', default='outputs/golden')
    ap.add_argument('--outdir', default='coreml')
    args = ap.parse_args()

    pairs = find_pairs(Path(args.local_root), Path(args.golden_root))
    ds = PairDataset(pairs)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    model = TinyPostFilter(hidden_channels=args.hidden, num_blocks=args.blocks).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10, verbose=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        avg_loss = tot / len(dl)
        print(f'Epoch {epoch}: loss={avg_loss:.6f}')
        scheduler.step(avg_loss)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Save TorchScript for fallback and Core ML conversion
    example = torch.randn(1, 1, T_FIXED)
    traced = torch.jit.trace(model.eval(), example)
    ts_path = outdir / 'KokoroPostFilter.torchscript.pt'
    traced.save(str(ts_path))
    print('wrote', ts_path)

    try:
        import coremltools as ct
        import numpy as np
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name='audio_in', shape=(1, 1, T_FIXED), dtype=np.float32)],
            convert_to='mlprogram',
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
        )
        ml_path = outdir / 'KokoroPostFilter.mlpackage'
        mlmodel.save(str(ml_path))
        print('wrote', ml_path)
    except Exception as e:
        print('Core ML conversion skipped/failed:', e)

if __name__ == '__main__':
    main()
