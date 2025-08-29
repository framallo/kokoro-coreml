#!/usr/bin/env python3
"""Convert Kokoro TTS model from safetensors to PyTorch checkpoint format.

This utility script converts pre-trained Kokoro TTS models from the safetensors format
(commonly used by MLX training frameworks) to PyTorch checkpoint format required by
the inference and CoreML export pipelines.

Conversion Process:
    1. Load safetensors weights from MLX training output
    2. Organize weights by neural module (BERT, predictor, decoder, etc.)
    3. Save in PyTorch checkpoint format expected by KModel loader
    4. Create directory structure for downstream processing

Module Organization:
    - bert: BERT-based phoneme encoder weights
    - bert_encoder: Linear projection from BERT to hidden dimensions
    - predictor: Prosody prediction network (duration, F0, noise)
    - text_encoder: Text feature extraction layers
    - decoder: Audio synthesis network (iSTFT-based)

Cross-file Dependencies:
    Input: Safetensors from MLX training workflow (kokoro-mlx-swift project)
    Output: PyTorch checkpoint consumed by:
        - kokoro/model.py: KModel weight loading in __init__()
        - examples/export_coreml.py: prepare_pytorch_models() function
        - Training scripts: Model initialization and fine-tuning workflows

Called by:
    - Manual conversion workflows when switching between MLX and PyTorch
    - CI/CD pipelines preparing models for deployment
    - examples/export_coreml.py: Automatic conversion when checkpoint missing

File Structure:
    Input:  {mlx_resources}/kokoro-v1_0.safetensors
    Output: checkpoints/kokoro-v1_0.pth
"""

import os
import torch
from safetensors.torch import load_file
from collections import OrderedDict

# File path constants for model conversion
class ConversionPaths:
    """File path configuration for safetensors to PyTorch conversion."""
    
    # MLX resources directory containing trained safetensors models
    # This path should match the MLX training output directory
    MLX_RESOURCES = "/Users/mattmireles/Documents/GitHub/kokoro-mlx-swift/kokoro-ios/mlxtest/mlxtest/Resources"
    
    # Source safetensors model filename
    SAFETENSORS_FILENAME = "kokoro-v1_0.safetensors"
    
    # Target PyTorch checkpoint path
    PYTORCH_CHECKPOINT_PATH = "checkpoints/kokoro-v1_0.pth"
    
    @classmethod
    def get_safetensors_path(cls) -> str:
        """Get full path to safetensors input file."""
        return os.path.join(cls.MLX_RESOURCES, cls.SAFETENSORS_FILENAME)

def convert_safetensors_to_checkpoint():
    """Convert safetensors model to organized PyTorch checkpoint format.
    
    This function handles the complete conversion pipeline from MLX safetensors
    format to the module-organized PyTorch checkpoint format expected by the
    Kokoro TTS inference system.
    
    Returns:
        str: Path to created PyTorch checkpoint file
        
    Raises:
        FileNotFoundError: If safetensors source file is not found
        OSError: If output directory cannot be created
    """
    safetensors_path = ConversionPaths.get_safetensors_path()
    checkpoint_path = ConversionPaths.PYTORCH_CHECKPOINT_PATH
    
    print(f"Loading safetensors from: {safetensors_path}")
    state_dict = load_file(safetensors_path)
    
    print(f"Found {len(state_dict)} parameters")
    
    # Organize weights by neural network module for KModel loading
    # This structure matches the expected format in kokoro/model.py
    organized_dict = OrderedDict((k, OrderedDict()) for k in ['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder'])
    for key, value in state_dict.items():
        module_name = key.split('.')[0]
        if module_name in organized_dict:
            organized_dict[module_name][key[len(module_name)+1:]] = value
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(organized_dict, checkpoint_path)
    print(f"✅ Saved PyTorch checkpoint to {checkpoint_path}")
    
    return checkpoint_path

if __name__ == "__main__":
    convert_safetensors_to_checkpoint()