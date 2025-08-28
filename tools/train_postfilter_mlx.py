#!/usr/bin/env python3
"""
Train a tiny post-filter (MLX version).
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import time

import mlx.core as mx
from mlx.core import value_and_grad
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import soundfile as sf

from postfilter_model_mlx import TinyPostFilter, loss_fn as mlx_loss_fn

SR = 24000
T_FIXED = 120000  # 5s

def find_pairs(local_root: Path, golden_root: Path) -> list[tuple[Path, Path]]:
    gdirs = sorted([p for p in golden_root.iterdir() if p.is_dir() and p.name.startswith('golden_')])
    if not gdirs:
        raise SystemExit('No golden dirs found')
    g_latest = gdirs[-1]
    y_path = g_latest / 'output.wav'
    pairs = []
    for run in sorted([p for p in local_root.iterdir() if p.is_dir() and p.name.startswith('phase2_')]):
        x_path = run / 'output.wav'
        if x_path.exists() and y_path.exists():
            pairs.append((x_path, y_path))
    if not pairs:
        raise SystemExit('No training pairs found under outputs/local')
    return pairs

def load_data(pairs: list[tuple[Path, Path]]) -> list[tuple[mx.array, mx.array]]:
    data = []
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
        # channels-last per sample: (T,1)
        x = mx.array(x).reshape(-1, 1)
        y = mx.array(y).reshape(-1, 1)
        data.append((x, y))
    return data

def batch_iterate(batch_size: int, data: list[tuple[mx.array, mx.array]]):
    while True:
        perm = np.random.permutation(len(data))
        for i in range(0, len(data) - batch_size + 1, batch_size):
            ids = perm[i:i+batch_size]
            batch_x = mx.stack([data[i][0] for i in ids])  # (B, T, 1)
            batch_y = mx.stack([data[i][1] for i in ids])  # (B, T, 1)
            yield batch_x, batch_y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=192)
    ap.add_argument('--blocks', type=int, default=14)
    ap.add_argument('--local_root', default='outputs/local')
    ap.add_argument('--golden_root', default='outputs/golden')
    ap.add_argument('--weights_dir', default='coreml')
    args = ap.parse_args()

    pairs = find_pairs(Path(args.local_root), Path(args.golden_root))
    data = load_data(pairs)
    
    model = TinyPostFilter(hidden_channels=args.hidden, num_blocks=args.blocks)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=args.lr)

    # Proper loss closure for value_and_grad
    def compute_loss(m: nn.Module, x: mx.array, y: mx.array) -> mx.array:
        y_pred = m(x)
        return mlx_loss_fn(y_pred, y, l1_weight=0.5, corr_weight=0.5, stft_weight=0.0)

    loss_and_grad_fn = value_and_grad(compute_loss)

    def step(x, y):
        loss, grads = loss_and_grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    batches = batch_iterate(args.batch, data)
    steps_per_epoch = len(data) // args.batch
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 10 # For LR scheduler

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        start_time = time.time()
        for i in range(steps_per_epoch):
            x, y = next(batches)
            loss = step(x, y)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / steps_per_epoch
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}: loss={avg_loss:.6f}, time={epoch_time:.2f}s")
        
        # LR Scheduler logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            patience_counter = 0
            optimizer.learning_rate *= 0.5
            print(f"Validation loss did not improve for {patience_limit} epochs. Reducing learning rate to {optimizer.learning_rate:.2e}")

    outdir = Path(args.weights_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    weights_path = outdir / 'kokoro_postfilter.safetensors'
    model.save_weights(str(weights_path))
    print(f"Saved weights to {weights_path}")


if __name__ == '__main__':
    main()
