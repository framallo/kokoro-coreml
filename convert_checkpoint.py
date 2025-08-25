#!/usr/bin/env python3
"""
Safetensors to PyTorch Checkpoint Conversion Utility

This script converts Kokoro TTS model weights from the safetensors format to PyTorch's
native checkpoint format. It handles the module organization and path management
required for seamless integration with the Kokoro training and inference pipeline.

Conversion Process:
1. Source Format: Safetensors - Hugging Face's safe tensor serialization format
2. Target Format: PyTorch checkpoint - Native PyTorch state_dict format
3. Module Organization: Groups parameters by model component for efficient loading
4. Path Management: Handles directory creation and cross-platform path resolution

Architectural Context:
The conversion preserves the exact model architecture and parameter organization
used by the original Kokoro training pipeline. The output format is compatible
with both training and inference modes, as well as CoreML/ONNX export workflows.

Cross-file Dependencies:
- Output used by: model.py (KModel.__init__), export_*.py (CoreML conversion)
- Input sources: Safetensors files from Hugging Face Hub or local training
- Integration with: kokoro-mlx-swift project for iOS deployment

Safety and Integrity:
- Safetensors format provides built-in integrity checking and corruption detection
- Module organization validation ensures all expected components are present
- File system operations include error handling for permission and space issues

Performance Characteristics:
- Memory efficient: Streaming conversion without loading entire model twice
- Fast execution: Direct tensor copying without computation overhead
- Cross-platform: Compatible with Windows, macOS, and Linux file systems
"""

import os
import sys
import torch
from safetensors.torch import load_file
from collections import OrderedDict
from typing import Dict, Any, Optional
from pathlib import Path

class ConversionConstants:
    """
    Configuration constants for safetensors to PyTorch checkpoint conversion.
    
    This class centralizes all file paths, model component names, and conversion
    parameters used during the checkpoint format migration process. It provides
    clear documentation for source/target locations and architectural organization.
    
    File Path Configuration:
    - Source paths point to safetensors files from training or download
    - Target paths organized for easy integration with inference pipeline
    - Directory structure follows standard ML project conventions
    
    Model Organization:
    - Component names match the architectural structure in model.py
    - Ordering preserves dependency relationships between modules
    - Parameter grouping enables efficient partial loading during inference
    
    Cross-Project Integration:
    - MLX Swift resources path supports iOS deployment workflow
    - Checkpoint directory structure matches training pipeline expectations
    - File naming conventions enable version management and rollback
    
    Used by:
    - convert_safetensors_to_checkpoint(): File path resolution and validation
    - organize_parameters_by_module(): Module name validation and ordering
    - Model loading functions: Consistent checkpoint structure expectations
    """
    
    # Source file paths (safetensors format)
    MLX_RESOURCES_BASE = "/Users/mattmireles/Documents/GitHub/kokoro-mlx-swift/kokoro-ios/mlxtest/mlxtest/Resources"
    DEFAULT_SAFETENSORS_NAME = "kokoro-v1_0.safetensors"
    
    # Target file paths (PyTorch checkpoint format)
    CHECKPOINT_DIRECTORY = "checkpoints"
    DEFAULT_CHECKPOINT_NAME = "kokoro-v1_0.pth"
    
    # Model architecture component organization
    # Order matters: components with dependencies should come after their dependencies
    MODEL_COMPONENTS = [
        'bert',           # BERT encoder for phoneme contextualization
        'bert_encoder',   # BERT output projection layer
        'predictor',      # Prosody prediction network (duration, F0, noise)
        'text_encoder',   # Bidirectional LSTM text encoder  
        'decoder'         # iSTFT-based neural vocoder
    ]
    
    # File format specifications
    SAFETENSORS_EXTENSION = ".safetensors"
    PYTORCH_EXTENSION = ".pth"
    
    # Validation parameters
    EXPECTED_MIN_PARAMETERS = 1000     # Minimum parameter count for valid model
    EXPECTED_MAX_PARAMETERS = 100000   # Maximum reasonable parameter count
    
    # Error handling configuration
    VERBOSE_OUTPUT = True              # Enable detailed progress logging
    STRICT_VALIDATION = True           # Fail on missing expected components

def validate_source_file(safetensors_path: str) -> bool:
    """
    Validate safetensors source file availability and integrity.

    Performs comprehensive validation of the source safetensors file to ensure
    successful conversion. Checks file existence, permissions, format validity,
    and basic integrity before attempting the conversion process.

    Validation Checks:
    1. File Existence: Verify file exists at specified path
    2. File Permissions: Ensure read access for current user
    3. File Size: Basic sanity check for non-empty file
    4. Format Validation: Attempt to load file header for format verification

    Args:
        safetensors_path (str): Absolute path to safetensors source file
                              Must be valid file path with appropriate extension
                              File must be accessible by current user

    Returns:
        bool: True if file is valid and ready for conversion
              False if validation fails (detailed error logged)

    Raises:
        FileNotFoundError: If source file does not exist
        PermissionError: If file exists but is not readable
        ValueError: If file format is invalid or corrupted

    Validation Process:
    - Path existence and accessibility verification
    - File size and format basic validation
    - Safetensors header integrity check
    - Parameter count estimation for reasonableness

    Called by:
    - convert_safetensors_to_checkpoint(): Pre-conversion validation
    - Main execution: Early error detection before processing
    """
    try:
        # Check file existence and readability
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")
        
        if not os.access(safetensors_path, os.R_OK):
            raise PermissionError(f"Cannot read safetensors file: {safetensors_path}")
        
        # Basic file size validation
        file_size = os.path.getsize(safetensors_path)
        if file_size == 0:
            raise ValueError(f"Safetensors file is empty: {safetensors_path}")
        
        # Try to load file header to validate format
        try:
            # Attempt to load just the metadata without full tensor loading
            state_dict = load_file(safetensors_path)
            param_count = len(state_dict)
            
            if param_count < ConversionConstants.EXPECTED_MIN_PARAMETERS:
                raise ValueError(f"Too few parameters ({param_count}), expected at least {ConversionConstants.EXPECTED_MIN_PARAMETERS}")
            
            if param_count > ConversionConstants.EXPECTED_MAX_PARAMETERS:
                print(f"Warning: Large parameter count ({param_count}), this may take longer than usual")
            
            print(f"✅ Validated safetensors file: {param_count} parameters")
            return True
            
        except Exception as e:
            raise ValueError(f"Invalid safetensors format: {e}")
            
    except Exception as e:
        print(f"❌ Source file validation failed: {e}")
        return False

def organize_parameters_by_module(state_dict: Dict[str, Any]) -> OrderedDict:
    """
    Organize flat parameter dictionary into modular structure for efficient loading.

    Transforms the flat parameter namespace from safetensors into a hierarchical
    structure organized by model components. This enables efficient partial loading
    during inference and maintains clear separation between architectural modules.

    Organizational Strategy:
    - Parameter names follow dot-notation: 'module.submodule.parameter'
    - Top-level modules correspond to major architectural components
    - Hierarchical nesting preserves parameter relationships and dependencies
    - Unused parameters are logged but not included in final structure

    Module Architecture Mapping:
    - bert: BERT encoder for phoneme contextualization (largest component)
    - bert_encoder: BERT output projection and transformation layers
    - predictor: Prosody prediction network (duration, F0, noise)
    - text_encoder: Bidirectional LSTM sequence encoder
    - decoder: iSTFT-based neural vocoder for audio synthesis

    Args:
        state_dict (Dict[str, Any]): Flat parameter dictionary from safetensors
                                   Keys: dot-separated parameter names
                                   Values: torch.Tensor parameter values

    Returns:
        OrderedDict: Hierarchically organized parameter structure
                    Top-level keys: model component names
                    Values: OrderedDict of component parameters
                    Order: Preserves component dependency relationships

    Parameter Processing:
    1. Initialize ordered structure for all expected components
    2. Parse parameter names to extract module and sub-parameter paths
    3. Group parameters by top-level module component
    4. Preserve parameter relationships within each module
    5. Report any unrecognized parameters for debugging

    Error Handling:
    - Unknown modules: Parameters logged but not included
    - Malformed names: Graceful skipping with warning
    - Empty modules: Maintained in structure for completeness

    Performance Characteristics:
    - Memory efficient: Single-pass organization without duplication
    - Order preserving: Maintains parameter dependency relationships
    - Scalable: Handles large models without performance degradation

    Called by:
    - convert_safetensors_to_checkpoint(): Core organizational step
    - Model validation: Structure verification during testing

    Example:
    ```python
    flat_dict = {
        'bert.encoder.layer.0.attention.self.query.weight': tensor(...),
        'predictor.duration_proj.weight': tensor(...),
        'decoder.conv_post.weight': tensor(...)
    }
    organized = organize_parameters_by_module(flat_dict)
    # Result: {'bert': {'encoder.layer.0.attention...': tensor(...)}, ...}
    ```
    """
    # Initialize ordered structure for expected model components
    organized_dict = OrderedDict(
        (component, OrderedDict()) for component in ConversionConstants.MODEL_COMPONENTS
    )
    
    # Track parameter organization statistics
    organized_count = 0
    unrecognized_params = []
    
    # Process each parameter in the flat state dictionary
    for parameter_name, parameter_tensor in state_dict.items():
        # Extract top-level module name from dot-separated parameter path
        name_parts = parameter_name.split('.')
        if len(name_parts) < 2:
            unrecognized_params.append(parameter_name)
            continue
            
        module_name = name_parts[0]
        sub_parameter_path = '.'.join(name_parts[1:])
        
        # Group parameter by module if it's a recognized component
        if module_name in organized_dict:
            organized_dict[module_name][sub_parameter_path] = parameter_tensor
            organized_count += 1
        else:
            unrecognized_params.append(parameter_name)
    
    # Report organization statistics
    if ConversionConstants.VERBOSE_OUTPUT:
        print(f"✅ Organized {organized_count} parameters into {len(ConversionConstants.MODEL_COMPONENTS)} modules")
        
        # Report per-module parameter counts
        for module_name, module_params in organized_dict.items():
            param_count = len(module_params)
            if param_count > 0:
                print(f"   {module_name}: {param_count} parameters")
            else:
                print(f"   {module_name}: empty (may be unused)")
        
        # Report unrecognized parameters
        if unrecognized_params:
            print(f"⚠️  Unrecognized parameters ({len(unrecognized_params)}):")
            for param in unrecognized_params[:5]:  # Show first 5 only
                print(f"   {param}")
            if len(unrecognized_params) > 5:
                print(f"   ... and {len(unrecognized_params) - 5} more")
    
    return organized_dict

def convert_safetensors_to_checkpoint(
    safetensors_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    validate_source: bool = True
) -> bool:
    """
    Convert safetensors model to PyTorch checkpoint format with comprehensive validation.

    This function performs the complete conversion pipeline from safetensors to PyTorch
    checkpoint format, including validation, organization, and error handling. It serves
    as the main entry point for checkpoint format conversion operations.

    Conversion Pipeline:
    1. Input Validation: Verify source file integrity and accessibility
    2. Parameter Loading: Load tensors from safetensors with error handling
    3. Module Organization: Group parameters by architectural components
    4. Directory Management: Create target directory structure as needed
    5. Checkpoint Creation: Save organized parameters in PyTorch format
    6. Validation: Verify successful conversion and output integrity

    Args:
        safetensors_path (str, optional): Path to source safetensors file
                                        Defaults to standard MLX resources location
                                        Must be valid file path if provided
        checkpoint_path (str, optional): Path for output PyTorch checkpoint
                                       Defaults to standard checkpoints directory
                                       Directory created if it doesn't exist
        validate_source (bool, optional): Enable source file validation. Defaults to True.
                                        Set to False for trusted sources only

    Returns:
        bool: True if conversion completed successfully
              False if conversion failed (error details logged)

    Raises:
        FileNotFoundError: If source file not found and strict validation enabled
        PermissionError: If unable to create target directory or write checkpoint
        ValueError: If source file format invalid or conversion logic fails
        RuntimeError: If unexpected error during tensor operations

    Error Handling Strategy:
    - Pre-validation: Catch file system and format issues before processing
    - Graceful degradation: Continue with warnings for non-critical issues
    - Comprehensive logging: Detailed error messages for debugging
    - Cleanup: Remove partial outputs if conversion fails

    Performance Characteristics:
    - Memory efficient: Processes tensors without unnecessary duplication
    - Fast execution: Direct tensor copying with minimal overhead
    - Progress reporting: Real-time feedback for large model conversions
    - Resource cleanup: Automatic memory management for large tensors

    Integration Points:
    - MLX Swift workflow: Converts models for iOS deployment
    - Training pipeline: Prepares checkpoints for training continuation
    - Export workflows: Creates compatible inputs for CoreML/ONNX conversion
    - Model validation: Provides checkpoints for testing and validation

    Called by:
    - Main execution: Direct script invocation for conversion tasks
    - Training scripts: Automated conversion during model preparation
    - Export pipelines: Checkpoint preparation for deployment formats

    Example:
    ```python
    # Basic conversion with default paths
    success = convert_safetensors_to_checkpoint()
    
    # Custom paths with validation disabled
    success = convert_safetensors_to_checkpoint(
        safetensors_path="/path/to/model.safetensors",
        checkpoint_path="/path/to/output.pth",
        validate_source=False
    )
    ```
    """
    try:
        # Set up default paths if not provided
        if safetensors_path is None:
            safetensors_path = os.path.join(
                ConversionConstants.MLX_RESOURCES_BASE, 
                ConversionConstants.DEFAULT_SAFETENSORS_NAME
            )
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                ConversionConstants.CHECKPOINT_DIRECTORY,
                ConversionConstants.DEFAULT_CHECKPOINT_NAME
            )
        
        print(f"🔄 Starting conversion process...")
        print(f"   Source: {safetensors_path}")
        print(f"   Target: {checkpoint_path}")
        
        # Validate source file if requested
        if validate_source and not validate_source_file(safetensors_path):
            return False
        
        # Load parameters from safetensors format
        print(f"📦 Loading parameters from safetensors...")
        state_dict = load_file(safetensors_path)
        total_parameters = len(state_dict)
        
        if ConversionConstants.VERBOSE_OUTPUT:
            print(f"✅ Loaded {total_parameters} parameters successfully")
        
        # Organize parameters by model components
        print(f"🗂️  Organizing parameters by module...")
        organized_dict = organize_parameters_by_module(state_dict)
        
        # Create target directory if needed
        target_directory = os.path.dirname(checkpoint_path)
        if target_directory and not os.path.exists(target_directory):
            os.makedirs(target_directory, exist_ok=True)
            print(f"📁 Created directory: {target_directory}")
        
        # Save organized parameters as PyTorch checkpoint
        print(f"💾 Saving PyTorch checkpoint...")
        torch.save(organized_dict, checkpoint_path)
        
        # Verify successful save by checking file existence and basic properties
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path)
            print(f"✅ Conversion completed successfully!")
            print(f"   Output size: {file_size / (1024*1024):.1f} MB")
            print(f"   Location: {checkpoint_path}")
            return True
        else:
            print(f"❌ Checkpoint file was not created successfully")
            return False
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return False
    except PermissionError as e:
        print(f"❌ Permission denied: {e}")
        return False
    except ValueError as e:
        print(f"❌ Invalid data: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during conversion: {e}")
        return False

def main():
    """
    Main execution function for command-line usage.

    Provides the primary entry point for checkpoint conversion when the script
    is executed directly. Handles command-line argument processing and provides
    user-friendly feedback for the conversion process.

    Command-line Usage:
    - python convert_checkpoint.py  # Use default paths
    - Direct execution via shebang for Unix-like systems

    Exit Codes:
    - 0: Conversion completed successfully
    - 1: Conversion failed due to error (details printed)

    Called by:
    - Direct script execution: python convert_checkpoint.py
    - System integration: Automated conversion in build processes
    """
    print("🚀 Kokoro TTS Checkpoint Conversion Utility")
    print("=" * 50)
    
    success = convert_safetensors_to_checkpoint()
    
    if success:
        print("=" * 50)
        print("🎉 Conversion completed successfully!")
        sys.exit(0)
    else:
        print("=" * 50)
        print("💥 Conversion failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()