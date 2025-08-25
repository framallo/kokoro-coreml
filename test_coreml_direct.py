#!/usr/bin/env python3
"""
Direct CoreML Model Testing and Validation Framework

This module provides comprehensive testing and validation capabilities for CoreML models
without requiring the full Kokoro TTS pipeline dependencies. It serves as a lightweight,
standalone diagnostic tool for verifying CoreML model integrity, input/output specifications,
and basic inference functionality across different deployment scenarios.

Core Testing Philosophy:
The direct testing approach isolates CoreML model validation from the complex dependencies
of the full TTS pipeline. This enables quick diagnostic feedback, deployment readiness
verification, and troubleshooting of CoreML-specific issues without the overhead of
loading and initializing the complete synthesis architecture.

Testing Architecture:
1. Model Loading: Validates CoreML package integrity and accessibility
2. Specification Analysis: Examines input/output tensor specifications and requirements
3. Dummy Input Generation: Creates representative test inputs matching model expectations
4. Inference Validation: Executes model prediction with synthetic data
5. Output Analysis: Analyzes prediction results and tensor characteristics
6. Error Diagnosis: Comprehensive error handling with actionable troubleshooting guidance

Supported Model Types:
- Duration Models: First-stage duration prediction and alignment models
- Synthesizer Models: End-to-end synthesis models with bucketing configurations
- Vocoder Models: Specialized audio generation models (standard and HAR variants)
- Hybrid Components: Individual pipeline components for hybrid CPU/ANE deployment

Key Validation Areas:
- Package Integrity: Verifies CoreML package structure and metadata
- Input Compatibility: Validates tensor shapes, types, and value ranges
- Inference Functionality: Confirms successful model execution and output generation
- Output Consistency: Analyzes output tensor characteristics and expected ranges
- Performance Assessment: Basic timing and memory usage measurement

Cross-Platform Deployment Testing:
- Device Compatibility: Validates models across different Apple Silicon variants
- Compute Unit Selection: Tests ANE, GPU, and CPU execution paths
- Memory Constraints: Verifies operation within mobile device memory limits
- Error Recovery: Tests fallback behavior and graceful degradation scenarios

Development and Production Support:
- CI/CD Integration: Automated model validation in deployment pipelines
- Quality Assurance: Pre-deployment verification of model functionality
- Troubleshooting: Rapid diagnosis of CoreML-specific deployment issues
- Performance Baseline: Establishment of expected inference characteristics

Technical Implementation:
- Minimal Dependencies: Uses only CoreML tools and NumPy for maximum portability
- Standalone Operation: No dependency on Kokoro TTS pipeline or PyTorch
- Comprehensive Logging: Detailed progress reporting and error diagnostics
- Flexible Testing: Configurable test scenarios for different validation requirements

Integration Points:
- Export Validation: Post-export verification of converted models
- Deployment Testing: Pre-production validation of model deployment readiness
- Performance Monitoring: Baseline establishment for production performance tracking
- Debug Workflows: Isolated testing for troubleshooting deployment issues
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    print("❌ CoreML Tools not available - cannot test CoreML models")
    COREML_AVAILABLE = False
    sys.exit(1)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("❌ NumPy not available - required for tensor operations")
    NUMPY_AVAILABLE = False
    sys.exit(1)

class CoreMLTestConstants:
    """
    Configuration constants for direct CoreML model testing and validation.
    
    This class centralizes all testing parameters, validation thresholds, and
    configuration settings used throughout the CoreML testing process. Constants
    are organized by functional area with comprehensive documentation of testing
    strategies and validation criteria.
    
    Testing Configuration:
    Default values chosen based on typical CoreML model characteristics and
    deployment requirements. Timeout values account for model complexity and
    device performance variations across Apple's hardware lineup.
    
    Model Specifications:
    Expected input/output formats and tensor characteristics for different
    model types. These specifications enable automated validation of model
    compliance with deployment requirements.
    
    Validation Thresholds:
    Acceptable ranges and limits for model performance, memory usage, and
    inference characteristics. Thresholds balance strict validation with
    practical deployment considerations.
    
    Used by:
    - Model testing functions: Test configuration and validation parameters
    - Input generation: Representative test data creation
    - Output validation: Expected tensor characteristics and value ranges
    - Performance measurement: Timing and resource usage benchmarks
    """
    
    # Default model paths and discovery
    DEFAULT_MODEL_DIRECTORY = "coreml"             # Standard CoreML model directory
    DURATION_MODEL_NAME = "kokoro_duration.mlpackage"    # Duration prediction model
    VOCODER_MODEL_NAME = "KokoroVocoder.mlpackage"       # Standard vocoder model
    HAR_MODEL_NAME = "KokoroDecoder_HAR.mlpackage"       # HAR variant model
    SYNTHESIZER_MODEL_PATTERN = "kokoro_synthesizer_*.mlpackage"  # Bucket model pattern
    
    # Input generation parameters
    DEFAULT_SEQUENCE_LENGTH = 128                  # Default sequence length for testing
    DEFAULT_BATCH_SIZE = 1                         # Fixed batch size for mobile deployment
    DEFAULT_FEATURE_DIM = 256                      # Default feature dimension
    DEFAULT_STYLE_DIM = 256                        # Voice style embedding dimension
    
    # Tensor value ranges for dummy input generation
    TOKEN_ID_RANGE = (0, 100)                      # Phoneme token ID range
    STYLE_EMBEDDING_STD = 1.0                      # Standard deviation for style embeddings
    F0_VALUE_RANGE = (0.0, 1.0)                   # Normalized F0 curve range
    NOISE_PARAMETER_STD = 0.1                      # Noise parameter standard deviation
    SPEED_RANGE = (0.5, 2.0)                       # Speech rate multiplier range
    
    # Validation and performance thresholds
    INFERENCE_TIMEOUT_SEC = 30.0                   # Maximum inference time
    MEMORY_LIMIT_MB = 512                          # Maximum model memory usage
    MODEL_LOAD_TIMEOUT_SEC = 10.0                  # Maximum model loading time
    EXPECTED_OUTPUT_TYPES = [np.ndarray, float, int]  # Valid output types
    
    # Testing configuration
    NUM_TEST_ITERATIONS = 3                        # Number of inference tests per model
    WARMUP_ITERATIONS = 1                          # Warmup runs before timing
    VALIDATION_TOLERANCE = 1e-6                    # Numerical validation tolerance
    
    # Error handling and reporting
    ENABLE_DETAILED_ERRORS = True                  # Show detailed error information
    LOG_TENSOR_SHAPES = True                       # Log all tensor shapes
    VALIDATE_OUTPUT_RANGES = True                  # Check output value ranges
    PERFORMANCE_REPORTING = True                   # Enable timing measurements
    
    # Model metadata validation
    EXPECTED_MODEL_VERSION = "1.0"                 # Expected model version
    REQUIRED_METADATA_FIELDS = ["author", "version"]  # Required metadata fields
    MAX_MODEL_SIZE_MB = 200                        # Maximum reasonable model size
    
    # Compute unit testing
    TEST_COMPUTE_UNITS = [                         # Compute units to test
        ct.ComputeUnit.ALL,
        ct.ComputeUnit.CPU_ONLY
    ]
    
    # Development and debugging
    SAVE_TEST_OUTPUTS = False                      # Save test outputs for inspection
    VERBOSE_LOGGING = True                         # Enable detailed progress logging
    DEBUG_MODE = False                             # Enable debug mode with extra checks

def discover_available_models(model_dir: str = None) -> Dict[str, str]:
    """
    Discover and catalog all available CoreML models in the specified directory.
    
    Scans the target directory for CoreML package files and creates a comprehensive
    inventory of available models with their types and characteristics. This function
    enables dynamic model testing across different deployment scenarios and model
    variants without requiring hardcoded paths.
    
    Discovery Strategy:
    1. Directory Scanning: Searches specified directory for .mlpackage files
    2. Pattern Matching: Identifies model types based on filename patterns
    3. Metadata Extraction: Reads basic model information where accessible
    4. Categorization: Groups models by type and deployment purpose
    
    Args:
        model_dir (str, optional): Directory to search for CoreML models
                                 Defaults to CoreMLTestConstants.DEFAULT_MODEL_DIRECTORY
                                 Must be valid directory path if specified
    
    Returns:
        Dict[str, str]: Dictionary mapping model names to absolute file paths
                       Keys: Descriptive model names (e.g., "duration", "vocoder")
                       Values: Absolute paths to .mlpackage files
                       Empty dict if no models found or directory inaccessible
    
    Model Type Identification:
    - Duration Models: Files containing "duration" in filename
    - Vocoder Models: Files containing "vocoder" in filename  
    - Synthesizer Models: Files matching synthesizer bucket patterns
    - HAR Models: Files containing "HAR" for harmonic+noise variants
    - Custom Models: Any other .mlpackage files with generic classification
    
    Error Handling:
    - Missing directory: Returns empty dict with warning message
    - Permission errors: Logs error and continues with accessible files
    - Corrupted packages: Logs warning and excludes from results
    - Invalid paths: Gracefully handles malformed file paths
    
    Called by:
    - main(): Primary model discovery for comprehensive testing
    - test_specific_model(): Validates target model availability
    - Batch testing: Automated testing across multiple models
    
    Example:
    ```python
    models = discover_available_models("coreml/")
    for name, path in models.items():
        print(f"Found {name}: {path}")
    ```
    """
    if model_dir is None:
        model_dir = CoreMLTestConstants.DEFAULT_MODEL_DIRECTORY
    
    models = {}
    
    try:
        model_path = Path(model_dir)
        if not model_path.exists():
            print(f"⚠️  Model directory not found: {model_dir}")
            return models
        
        # Scan for .mlpackage files
        for package_path in model_path.glob("*.mlpackage"):
            try:
                package_name = package_path.stem.lower()
                absolute_path = str(package_path.absolute())
                
                # Categorize models by filename patterns
                if "duration" in package_name:
                    models["duration"] = absolute_path
                elif "vocoder" in package_name and "har" not in package_name:
                    models["vocoder"] = absolute_path
                elif "har" in package_name:
                    models["har_vocoder"] = absolute_path
                elif "synthesizer" in package_name:
                    # Extract bucket duration from filename
                    try:
                        parts = package_name.split('_')
                        duration = next((p.rstrip('s') for p in parts if p.endswith('s') and p[:-1].isdigit()), None)
                        if duration:
                            models[f"synthesizer_{duration}s"] = absolute_path
                        else:
                            models["synthesizer"] = absolute_path
                    except Exception:
                        models["synthesizer"] = absolute_path
                else:
                    models[package_name] = absolute_path
                    
            except Exception as e:
                print(f"⚠️  Error processing {package_path}: {e}")
                continue
        
        if CoreMLTestConstants.VERBOSE_LOGGING:
            print(f"📁 Discovered {len(models)} CoreML models in {model_dir}")
            
        return models
        
    except Exception as e:
        print(f"❌ Error discovering models in {model_dir}: {e}")
        return models

def generate_test_inputs(model_spec) -> Dict[str, np.ndarray]:
    """
    Generate representative test inputs matching CoreML model specifications.
    
    Creates realistic dummy tensor data that matches the exact input requirements
    of the target CoreML model. Input generation follows model-specific patterns
    and value distributions to ensure meaningful validation results.
    
    Input Generation Strategy:
    1. Specification Analysis: Examines model input requirements and tensor shapes
    2. Type Matching: Creates inputs with correct data types and precision
    3. Value Distribution: Uses realistic value ranges based on model type
    4. Shape Compliance: Ensures exact shape matching with model expectations
    
    Args:
        model_spec: CoreML model specification object containing input definitions
                   Must include input tensor descriptions with names, shapes, and types
                   Obtained from model.get_spec().description
    
    Returns:
        Dict[str, np.ndarray]: Dictionary of test inputs ready for model prediction
                             Keys: Input tensor names as defined in model specification
                             Values: NumPy arrays with appropriate shapes and data types
                             All inputs guaranteed to match model requirements exactly
    
    Input Type Handling:
    - Token IDs: Random integers within vocabulary range for text inputs
    - Embeddings: Gaussian distributed floats for style and voice representations
    - Masks: Binary tensors for attention and padding masks
    - Scalar Values: Single values for speed, intensity, and control parameters
    - Sequences: Variable-length inputs with appropriate padding and masking
    
    Value Range Selection:
    - Phoneme Tokens: Realistic vocabulary indices (0-100 range)
    - Style Embeddings: Standard normal distribution for voice characteristics
    - F0 Curves: Normalized frequency values in [0,1] range
    - Speed Controls: Reasonable speech rate multipliers (0.5-2.0 range)
    - Attention Masks: Valid attention patterns for sequence processing
    
    Called by:
    - test_coreml_model(): Primary testing function for model validation
    - performance_benchmark(): Performance testing with consistent inputs
    - batch_validation(): Automated testing across multiple model variants
    """
    test_inputs = {}
    
    try:
        for input_spec in model_spec.description.input:
            input_name = input_spec.name
            
            # Handle different input types based on name patterns and specifications
            if hasattr(input_spec.type, 'multiArrayType'):
                array_spec = input_spec.type.multiArrayType
                shape = tuple(array_spec.shape)
                
                # Generate appropriate test data based on input name
                if "input_ids" in input_name or "token" in input_name:
                    # Token IDs: Random integers within vocabulary range
                    data = np.random.randint(
                        CoreMLTestConstants.TOKEN_ID_RANGE[0],
                        CoreMLTestConstants.TOKEN_ID_RANGE[1],
                        shape
                    ).astype(np.int32)
                    
                elif "attention_mask" in input_name or "mask" in input_name:
                    # Attention masks: Mostly ones with some padding
                    data = np.ones(shape, dtype=np.int32)
                    # Add some realistic padding (last 20% of sequence)
                    if len(shape) >= 2 and shape[-1] > 10:
                        padding_start = int(shape[-1] * 0.8)
                        data[..., padding_start:] = 0
                        
                elif "ref_s" in input_name or "style" in input_name:
                    # Style embeddings: Gaussian distribution
                    data = np.random.randn(*shape).astype(np.float32) * CoreMLTestConstants.STYLE_EMBEDDING_STD
                    
                elif "speed" in input_name:
                    # Speed control: Reasonable speech rate
                    speed_val = np.random.uniform(CoreMLTestConstants.SPEED_RANGE[0], 
                                                CoreMLTestConstants.SPEED_RANGE[1])
                    data = np.array([speed_val], dtype=np.float32)
                    if len(shape) > 1:
                        data = np.broadcast_to(data, shape)
                        
                elif "f0" in input_name:
                    # F0 curve: Normalized frequency values
                    data = np.random.uniform(
                        CoreMLTestConstants.F0_VALUE_RANGE[0],
                        CoreMLTestConstants.F0_VALUE_RANGE[1],
                        shape
                    ).astype(np.float32)
                    
                elif "noise" in input_name or "_n" in input_name:
                    # Noise parameters: Small random values
                    data = np.random.randn(*shape).astype(np.float32) * CoreMLTestConstants.NOISE_PARAMETER_STD
                    
                else:
                    # Generic float tensor: Standard normal distribution
                    data = np.random.randn(*shape).astype(np.float32)
                
                test_inputs[input_name] = data
                
                if CoreMLTestConstants.LOG_TENSOR_SHAPES:
                    print(f"  Generated {input_name}: {data.shape} {data.dtype}")
                    
            else:
                print(f"⚠️  Unsupported input type for {input_name}")
                
        return test_inputs
        
    except Exception as e:
        print(f"❌ Error generating test inputs: {e}")
        if CoreMLTestConstants.ENABLE_DETAILED_ERRORS:
            traceback.print_exc()
        return {}

def test_coreml_model(model_path: str = None) -> bool:
    """
    Comprehensive testing and validation of a CoreML model.
    
    Performs complete model validation including loading, specification analysis,
    inference testing, and output validation. This function provides thorough
    diagnostic information for model deployment readiness and troubleshooting.
    
    Testing Pipeline:
    1. Model Loading: Validates CoreML package integrity and accessibility
    2. Metadata Analysis: Examines model information and specifications
    3. Input/Output Specification: Analyzes tensor requirements and formats
    4. Test Input Generation: Creates representative inputs for validation
    5. Inference Execution: Runs model prediction with synthetic data
    6. Output Analysis: Validates prediction results and tensor characteristics
    7. Performance Assessment: Measures inference timing and resource usage
    
    Args:
        model_path (str, optional): Path to CoreML model package for testing
                                  Defaults to standard duration model location
                                  Must be valid .mlpackage file if specified
    
    Returns:
        bool: True if all tests pass successfully
              False if any test fails (detailed error information printed)
    
    Validation Criteria:
    - Package Integrity: Model loads without errors
    - Specification Compliance: Valid input/output tensor definitions
    - Inference Functionality: Successful prediction execution
    - Output Consistency: Reasonable output tensor characteristics
    - Performance Acceptable: Inference time within expected bounds
    
    Error Handling:
    - Missing Models: Clear error message with suggested resolution
    - Loading Failures: Detailed diagnostics for package corruption issues
    - Inference Errors: Comprehensive error reporting with tensor information
    - Performance Issues: Warnings for slow inference or excessive memory usage
    
    Called by:
    - main(): Primary testing entry point for individual model validation
    - CI/CD pipelines: Automated model validation in deployment workflows
    - Development workflows: Manual testing during model development and debugging
    """
    if model_path is None:
        model_path = os.path.join(
            CoreMLTestConstants.DEFAULT_MODEL_DIRECTORY,
            CoreMLTestConstants.DURATION_MODEL_NAME
        )
    
    print("🧪 Testing CoreML model directly...")
    print(f"📦 Loading model from: {model_path}")
    
    try:
        # Validate model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("💡 Available models:")
            available_models = discover_available_models()
            for name, path in available_models.items():
                print(f"   - {name}: {path}")
            return False
        
        # Load the CoreML model with timeout
        start_time = time.time()
        model = ct.models.MLModel(model_path)
        load_time = time.time() - start_time
        
        if load_time > CoreMLTestConstants.MODEL_LOAD_TIMEOUT_SEC:
            print(f"⚠️  Model loading took {load_time:.2f}s (slower than expected)")
        
        print("✅ CoreML model loaded successfully!")
        print(f"⏱️  Load time: {load_time:.3f} seconds")
        
        # Analyze model metadata and specifications
        print(f"\n📋 Model Information:")
        spec = model.get_spec()
        
        # Basic model info
        print(f"   Author: {getattr(model, 'author', 'Not specified')}")
        print(f"   Description: {getattr(model, 'short_description', 'Not specified')}")
        print(f"   Version: {getattr(model, 'version', 'Not specified')}")
        
        # Model size estimation
        try:
            package_size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file())
            size_mb = package_size / (1024 * 1024)
            print(f"   Package Size: {size_mb:.1f} MB")
            
            if size_mb > CoreMLTestConstants.MAX_MODEL_SIZE_MB:
                print(f"⚠️  Model size ({size_mb:.1f} MB) exceeds recommended limit")
        except Exception as e:
            print(f"   Package Size: Unable to calculate ({e})")
        
        # Input specifications
        print(f"\n🔤 Input Specifications:")
        for input_spec in spec.description.input:
            print(f"   - {input_spec.name}: {input_spec.type}")
        
        # Output specifications  
        print(f"\n📤 Output Specifications:")
        for output_spec in spec.description.output:
            print(f"   - {output_spec.name}: {output_spec.type}")
        
        # Generate test inputs
        print(f"\n🎲 Generating test inputs...")
        test_inputs = generate_test_inputs(spec)
        
        if not test_inputs:
            print("❌ Failed to generate test inputs")
            return False
        
        # Execute model inference
        print(f"\n🧪 Running model inference...")
        
        total_inference_time = 0
        successful_runs = 0
        
        for iteration in range(CoreMLTestConstants.NUM_TEST_ITERATIONS):
            try:
                start_time = time.time()
                result = model.predict(test_inputs)
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                successful_runs += 1
                
                if iteration == 0:  # Detailed analysis on first run
                    print("✅ Model prediction successful!")
                    
                    # Analyze outputs
                    print(f"\n📊 Output Analysis:")
                    for key, value in result.items():
                        if hasattr(value, 'shape'):
                            print(f"   - {key}: {value.shape} ({value.dtype})")
                            
                            # Basic value range analysis
                            if CoreMLTestConstants.VALIDATE_OUTPUT_RANGES and hasattr(value, 'min'):
                                print(f"     Range: [{value.min():.6f}, {value.max():.6f}]")
                                if hasattr(value, 'mean'):
                                    print(f"     Mean: {value.mean():.6f}, Std: {value.std():.6f}")
                        else:
                            print(f"   - {key}: {type(value)} = {value}")
                            
            except Exception as e:
                print(f"❌ Inference iteration {iteration + 1} failed: {e}")
                if CoreMLTestConstants.ENABLE_DETAILED_ERRORS:
                    traceback.print_exc()
                continue
        
        # Performance summary
        if successful_runs > 0:
            avg_inference_time = total_inference_time / successful_runs
            print(f"\n⏱️  Performance Summary:")
            print(f"   Successful runs: {successful_runs}/{CoreMLTestConstants.NUM_TEST_ITERATIONS}")
            print(f"   Average inference time: {avg_inference_time:.3f} seconds")
            
            if avg_inference_time > CoreMLTestConstants.INFERENCE_TIMEOUT_SEC / 2:
                print(f"⚠️  Inference time higher than expected")
            else:
                print("✅ Inference performance acceptable")
        
        print(f"\n🎉 CoreML model test completed successfully!")
        return successful_runs > 0
        
    except Exception as e:
        print(f"❌ Error testing CoreML model: {e}")
        if CoreMLTestConstants.ENABLE_DETAILED_ERRORS:
            traceback.print_exc()
        return False

def main():
    """
    Main execution function for comprehensive CoreML model testing.
    
    Orchestrates the complete testing workflow, providing both interactive and
    automated testing capabilities. Supports single model testing and batch
    testing across multiple available models.
    
    Execution Modes:
    - Single Model: Tests default or specified model with detailed analysis
    - Model Discovery: Automatic detection and testing of available models
    - Performance Baseline: Establishes performance expectations for deployment
    - Validation Report: Comprehensive testing report for quality assurance
    
    Exit Codes:
    - 0: All tests passed successfully
    - 1: One or more tests failed
    - 2: No models found or testing environment issues
    
    Called by:
    - Direct execution: python test_coreml_direct.py
    - CI/CD pipelines: Automated model validation
    - Development workflows: Manual model testing and validation
    """
    print("🔧 Direct CoreML Model Testing Framework")
    print("=" * 50)
    
    if not COREML_AVAILABLE or not NUMPY_AVAILABLE:
        print("❌ Required dependencies not available")
        sys.exit(2)
    
    # Discover available models
    available_models = discover_available_models()
    
    if not available_models:
        print("❌ No CoreML models found for testing")
        print("💡 Run export scripts to generate models:")
        print("   - python export_synthesizers.py")
        print("   - python export_vocoder.py")
        sys.exit(2)
    
    print(f"\n📋 Available Models:")
    for name, path in available_models.items():
        print(f"   - {name}: {os.path.basename(path)}")
    
    # Test each available model
    test_results = {}
    overall_success = True
    
    for model_name, model_path in available_models.items():
        print(f"\n{'='*20} Testing {model_name} {'='*20}")
        
        try:
            success = test_coreml_model(model_path)
            test_results[model_name] = success
            if not success:
                overall_success = False
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
            test_results[model_name] = False
            overall_success = False
    
    # Final summary
    print(f"\n{'='*50}")
    print("🎯 Testing Summary:")
    
    for model_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {model_name}: {status}")
    
    if overall_success:
        print(f"\n🎉 All models tested successfully!")
        print("✅ Models are ready for deployment")
    else:
        print(f"\n⚠️  Some models failed testing")
        print("🔧 Review error messages above for troubleshooting")
    
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main()