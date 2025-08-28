#!/usr/bin/env python3
"""
Load weights from a trained MLX model into a PyTorch model and export to Core ML.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import coremltools as ct
import mlx.core as mx
import numpy as np
import torch

from postfilter_model import TinyPostFilter

T_FIXED = 120000  # 5s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=192)
    ap.add_argument('--blocks', type=int, default=14)
    ap.add_argument('--weights_path', default='coreml/kokoro_postfilter.safetensors')
    ap.add_argument('--outdir', default='coreml')
    args = ap.parse_args()

    # 1. Load MLX weights
    mlx_weights = mx.load(args.weights_path)
    print("Loaded MLX weights from:", args.weights_path)

    # 2. Instantiate PyTorch model
    pt_model = TinyPostFilter(hidden_channels=args.hidden, num_blocks=args.blocks)
    pt_model.eval()

    # 3. Transfer weights from MLX to PyTorch
    pt_state_dict = pt_model.state_dict()
    for name, mlx_w in mlx_weights.items():
        if name in pt_state_dict:
            # Convert MLX array -> NumPy array -> PyTorch tensor
            pt_tensor = torch.from_numpy(np.array(mlx_w))
            if pt_state_dict[name].shape != pt_tensor.shape:
                print(f"Shape mismatch for {name}: PT {pt_state_dict[name].shape}, MLX {pt_tensor.shape}")
                # MLX Conv1d weights are (out, in, ks), PyTorch are (out, in, ks)
                # No transpose needed for Conv1d, but good to be aware of for other layers
                continue
            pt_state_dict[name].copy_(pt_tensor)
        else:
            print(f"Warning: weight '{name}' from MLX not found in PyTorch model.")
    
    pt_model.load_state_dict(pt_state_dict)
    print("Successfully transferred weights from MLX to PyTorch model.")

    # 4. Export to Core ML
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    example = torch.randn(1, 1, T_FIXED)
    traced = torch.jit.trace(pt_model, example)

    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='audio_in', shape=(1, 1, T_FIXED), dtype=np.float32)],
        convert_to='mlprogram',
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    
    ml_path = outdir / 'KokoroPostFilter.mlpackage'
    mlmodel.save(str(ml_path))
    print('Wrote Core ML model to:', ml_path)

if __name__ == '__main__':
    main()
