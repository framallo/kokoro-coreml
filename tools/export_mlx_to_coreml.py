#!/usr/bin/env python3
"""
Load weights from a trained MLX model into a PyTorch model and export to Core ML.

Adds simple versioning and metadata recording alongside the exported .mlpackage.
Outputs:
- coreml/KokoroPostFilter_<YYYYmmdd_HHMMSS>_h{H}_b{B}.mlpackage
- coreml/KokoroPostFilter.latest.mlpackage (copy of latest for convenience)
- coreml/KokoroPostFilter_<...>.metadata.json (export metadata)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime

import coremltools as ct
import mlx.core as mx
import numpy as np
import torch

from postfilter_model import TinyPostFilter

# Audio processing constants
class AudioConstants:
    """Constants for audio processing and Core ML model configuration."""
    
    # Fixed-length audio processing for post-filter model
    # Represents exactly 5 seconds of audio at 24kHz sample rate
    # Used by post-filter models trained on fixed-duration segments
    # Calculation: 5 seconds × 24,000 samples/second = 120,000 samples
    T_FIXED_SAMPLES = 120000  # 5s at 24kHz
    
    # Standard audio sample rate for Kokoro TTS system
    # All models expect this sample rate for proper temporal alignment
    SAMPLE_RATE_HZ = 24000

T_FIXED = AudioConstants.T_FIXED_SAMPLES  # Backward compatibility

def main():
    """Convert MLX-trained post-filter model to Core ML with versioning and metadata.
    
    This script provides the complete pipeline for converting a post-filter model
    trained with MLX (Apple's machine learning framework) into a production-ready
    Core ML package suitable for deployment on Apple devices.
    
    Conversion Pipeline:
        1. Load MLX weights from safetensors checkpoint
        2. Create equivalent PyTorch model with matching architecture  
        3. Transfer weights with proper tensor layout conversion (channels-last → channels-first)
        4. Trace PyTorch model for Core ML compatibility
        5. Convert to Core ML with optimized precision settings
        6. Generate versioned output with metadata recording
    
    Post-Filter Model Architecture:
        - TinyPostFilter with configurable hidden channels and residual blocks
        - Processes fixed-length 5-second audio segments (120,000 samples)
        - Trained to reduce artifacts and improve naturalness of TTS output
    
    Output Files:
        - Versioned .mlpackage: KokoroPostFilter_YYYYmmdd_HHMMSS_hH_bB.mlpackage
        - Convenience symlink: KokoroPostFilter.latest.mlpackage  
        - Export metadata: KokoroPostFilter_YYYYmmdd_HHMMSS_hH_bB.metadata.json
    
    Called by:
        - Training scripts after MLX post-filter model completion
        - Manual export workflows for production deployment
        - CI/CD pipelines for automated model releases
    
    Cross-file Dependencies:
        - postfilter_model.py: TinyPostFilter PyTorch architecture definition
        - MLX framework: Loading and processing trained weights
        - Core ML tools: Model conversion and optimization
    """
    # Model architecture constants
    class ModelDefaults:
        """Default hyperparameters for post-filter model architecture."""
        
        # Hidden channel dimension for convolutional layers
        # Larger values improve model capacity but increase inference time
        DEFAULT_HIDDEN_CHANNELS = 192
        
        # Number of residual blocks in the post-filter architecture
        # More blocks provide better processing but slower inference
        DEFAULT_NUM_BLOCKS = 14
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=ModelDefaults.DEFAULT_HIDDEN_CHANNELS)
    ap.add_argument('--blocks', type=int, default=ModelDefaults.DEFAULT_NUM_BLOCKS)
    ap.add_argument('--weights_path', default='coreml/kokoro_postfilter.safetensors')
    ap.add_argument('--outdir', default='coreml')
    ap.add_argument('--float32', action='store_true', help='Export with FLOAT32 precision (debug/accuracy)')
    args = ap.parse_args()

    # 1. Load MLX weights
    mlx_weights = mx.load(args.weights_path)
    print("Loaded MLX weights from:", args.weights_path)

    # 2. Instantiate PyTorch model
    pt_model = TinyPostFilter(hidden_channels=args.hidden, num_blocks=args.blocks)
    pt_model.eval()

    # 3. Transfer weights from MLX to PyTorch
    pt_state_dict = pt_model.state_dict()
    def map_name(n: str) -> str:
        # Align MLX Sequential naming to PyTorch nn.Sequential
        if n.startswith('blocks.layers.'):
            n = n.replace('blocks.layers.', 'blocks.')
        return n

    for raw_name, mlx_w in mlx_weights.items():
        name = map_name(raw_name)
        if name in pt_state_dict:
            arr = np.array(mlx_w)
            # Conv1d weight layout: MLX uses (out, kernel, in) for channels-last; PyTorch expects (out, in, kernel)
            if name.endswith('.weight') and arr.ndim == 3:
                arr = np.transpose(arr, (0, 2, 1))
            pt_tensor = torch.from_numpy(arr)
            if pt_state_dict[name].shape != pt_tensor.shape:
                print(f"Shape mismatch for {name}: PT {pt_state_dict[name].shape}, MLX {pt_tensor.shape}")
                continue
            pt_state_dict[name].copy_(pt_tensor)
        else:
            print(f"Warning: weight '{raw_name}' from MLX not found in PyTorch model.")
    
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
        compute_precision=(ct.precision.FLOAT32 if args.float32 else ct.precision.FLOAT16),
    )
    # Versioned filename
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"KokoroPostFilter_{stamp}_h{args.hidden}_b{args.blocks}.mlpackage"
    ml_path = outdir / tag
    mlmodel.save(str(ml_path))
    print('Wrote Core ML model to:', ml_path)

    # Write/copy latest convenience link
    latest_path = outdir / 'KokoroPostFilter.latest.mlpackage'
    try:
        if latest_path.exists():
            if latest_path.is_dir():
                # Remove existing directory before copying new one
                import shutil
                shutil.rmtree(latest_path)
        # Core ML packages are directories; copytree
        import shutil
        shutil.copytree(ml_path, latest_path)
        print('Updated convenience copy:', latest_path)
    except Exception as e:
        print('Warning: failed to update latest copy:', e)

    # Metadata
    meta = {
        'timestamp': stamp,
        'weights_path': str(Path(args.weights_path).resolve()),
        'hidden_channels': args.hidden,
        'num_blocks': args.blocks,
        't_fixed': T_FIXED,
        'compute_precision': ('FLOAT32' if args.float32 else 'FLOAT16'),
        'out_mlpackage': str(ml_path.resolve()),
    }
    try:
        import json
        meta_path = outdir / (tag.replace('.mlpackage', '.metadata.json'))
        meta_path.write_text(json.dumps(meta, indent=2))
        print('Wrote metadata to:', meta_path)
    except Exception as e:
        print('Warning: failed to write metadata:', e)

if __name__ == '__main__':
    main()
