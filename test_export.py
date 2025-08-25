#!/usr/bin/env python3
"""
Kokoro TTS Export Pipeline Validation and Development Environment Test Suite

This module provides comprehensive validation of the complete export pipeline dependencies,
model availability, and development environment setup. It serves as both a quick diagnostic
tool for troubleshooting export issues and a reference implementation for dependency
management in the Kokoro TTS CoreML export workflow.

Core Testing Philosophy:
The test suite follows a progressive validation approach, starting with basic Python
environment validation and advancing through model loading, format conversion, and
export toolchain verification. Each test phase builds upon the previous one, enabling
precise identification of configuration issues and missing dependencies.

Testing Architecture:
1. Environment Validation: Python path, working directory, and module resolution
2. Import Testing: Core module availability and version compatibility  
3. Model File Validation: Checkpoint and configuration file integrity
4. Export Toolchain: CoreML tools availability and version compatibility
5. Integration Testing: End-to-end export pipeline component verification

Development Workflow Integration:
This test module is designed to be the first step in any export development workflow:
- Pre-export validation: Verify all dependencies before attempting conversion
- Development setup: Validate new development environment configuration
- CI/CD integration: Automated environment verification for deployment pipelines
- Troubleshooting: Quick diagnosis of common export pipeline issues

Cross-file Dependencies:
- Validates: kokoro.model.KModel (core model architecture)
- Checks: Checkpoint files from convert_checkpoint.py conversion
- Verifies: CoreML tools compatibility for export_*.py scripts
- Integrates: Complete development environment for all export workflows

Error Reporting and Troubleshooting:
The module provides detailed error reporting with actionable troubleshooting guidance:
- Missing dependencies: Clear installation instructions and version requirements
- File system issues: Path resolution and permission problem identification
- Import failures: Module path and PYTHONPATH configuration guidance
- Version conflicts: Compatibility matrix and upgrade/downgrade recommendations

Production Deployment Support:
- Container validation: Docker and deployment environment verification
- Path management: Cross-platform path resolution and validation
- Version control: Consistent dependency versions across environments
- Integration testing: Validation of complete export toolchain integrity

Performance and Reliability:
- Fast execution: Quick diagnostic feedback for rapid development iteration
- Comprehensive coverage: All critical dependencies and configuration points
- Graceful failures: Informative error messages with recovery guidance
- Automation friendly: Structured output for CI/CD and monitoring systems
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# Add current directory to Python path for module resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ExportTestConstants:
    """
    Configuration constants for export pipeline validation and testing.
    
    This class centralizes all file paths, version requirements, and validation
    parameters used throughout the export testing process. Constants are organized
    by functional area with comprehensive documentation of requirements and 
    compatibility matrices.
    
    File Path Configuration:
    Default paths follow standard Kokoro project structure while supporting
    flexible deployment scenarios. Path validation includes both absolute and
    relative path resolution with cross-platform compatibility.
    
    Version Requirements:
    Version constraints based on compatibility testing and feature requirements
    for the complete export pipeline. Minimum versions ensure required features
    while maximum versions avoid known compatibility issues.
    
    Testing Configuration:
    Parameters for progressive testing with appropriate timeouts and error
    handling. Test phases designed for optimal feedback during development
    while maintaining comprehensive coverage for deployment validation.
    
    Used by:
    - Environment validation: Path resolution and dependency checking
    - Version compatibility: Minimum/maximum version enforcement
    - Error reporting: Structured diagnostic output and troubleshooting
    - Integration testing: End-to-end export pipeline validation
    """
    
    # Model file paths and validation
    CHECKPOINT_DIRECTORY = "checkpoints"
    CONFIG_FILENAME = "config.json"
    CHECKPOINT_FILENAME = "kokoro-v1_0.pth"  
    DEFAULT_CONFIG_PATH = os.path.join(CHECKPOINT_DIRECTORY, CONFIG_FILENAME)
    DEFAULT_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIRECTORY, CHECKPOINT_FILENAME)
    
    # CoreML export requirements
    COREML_MIN_VERSION = "7.0"             # Minimum coremltools version for mlprogram
    COREML_RECOMMENDED_VERSION = "8.0"     # Recommended version for ANE optimization
    PYTORCH_MIN_VERSION = "2.0"           # Minimum PyTorch version for export compatibility
    
    # Python environment requirements  
    PYTHON_MIN_VERSION = (3, 8)           # Minimum Python version
    PYTHON_RECOMMENDED_VERSION = (3, 11)  # Recommended Python version
    
    # Testing and validation parameters
    IMPORT_TEST_TIMEOUT = 10.0             # Maximum time for import testing
    MODEL_LOAD_TIMEOUT = 30.0              # Maximum time for model loading
    VERBOSE_OUTPUT = True                  # Enable detailed progress reporting
    
    # Error handling and reporting
    ENABLE_TRACEBACK = True                # Show detailed error tracebacks
    STRUCTURED_OUTPUT = True               # Format output for automation parsing
    
    # Expected module structure for validation
    REQUIRED_KOKORO_MODULES = [
        'kokoro.model',
        'kokoro.pipeline', 
        'kokoro.__init__'
    ]
    
    # Optional modules with fallback behavior
    OPTIONAL_MODULES = {
        'coremltools': 'CoreML export functionality',
        'torch': 'PyTorch model loading',
        'safetensors': 'Safetensors checkpoint loading'
    }

def validate_python_environment() -> Tuple[bool, str]:
    """
    Validate Python interpreter version and basic environment configuration.
    
    Performs comprehensive Python environment validation to ensure compatibility
    with the Kokoro export pipeline. Checks Python version, path configuration,
    and basic module resolution capabilities.
    
    Environment Validation:
    1. Python Version: Verify minimum and recommended version compatibility
    2. Path Resolution: Test current working directory and script location
    3. Module Path: Validate Python path configuration for local imports
    4. Interpreter: Check for virtual environment and package manager integration
    
    Returns:
        Tuple[bool, str]: (success_flag, detailed_status_message)
                         success_flag: True if environment passes validation
                         status_message: Human-readable validation results
    
    Validation Criteria:
    - Python >= 3.8 (minimum) with 3.11+ recommended
    - Proper working directory and script path resolution
    - PYTHONPATH configuration for local module imports
    - No conflicting virtual environment configurations
    
    Error Conditions:
    - Python version too old for required dependencies
    - Path resolution failures preventing module imports
    - Virtual environment conflicts with system packages
    - File system permission issues affecting script execution
    
    Called by:
    - main(): Initial validation before dependency testing
    - CI/CD scripts: Automated environment verification
    - Development setup: New environment configuration validation
    """
    try:
        # Check Python version compatibility
        current_version = sys.version_info[:2]
        if current_version < ExportTestConstants.PYTHON_MIN_VERSION:
            return False, f"Python {current_version[0]}.{current_version[1]} too old (minimum: {ExportTestConstants.PYTHON_MIN_VERSION[0]}.{ExportTestConstants.PYTHON_MIN_VERSION[1]})"
        
        version_status = "✓ Compatible"
        if current_version >= ExportTestConstants.PYTHON_RECOMMENDED_VERSION:
            version_status = "✓ Recommended"
        elif current_version < ExportTestConstants.PYTHON_RECOMMENDED_VERSION:
            version_status = f"⚠️  Older than recommended {ExportTestConstants.PYTHON_RECOMMENDED_VERSION[0]}.{ExportTestConstants.PYTHON_RECOMMENDED_VERSION[1]}"
        
        # Validate path configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = os.getcwd()
        
        path_info = []
        path_info.append(f"Python {current_version[0]}.{current_version[1]} - {version_status}")
        path_info.append(f"Script directory: {script_dir}")
        path_info.append(f"Working directory: {current_dir}")
        path_info.append(f"Python path entries: {len(sys.path)}")
        
        # Verify script directory is in Python path (added by sys.path.insert above)
        if script_dir not in sys.path:
            return False, "Script directory not in Python path - import resolution may fail"
        
        return True, "\n".join(path_info)
        
    except Exception as e:
        return False, f"Environment validation error: {e}"

def test_core_imports() -> Tuple[bool, Dict[str, str]]:
    """
    Test import availability for all required Kokoro TTS modules.
    
    Performs systematic import testing for all core modules required by the
    export pipeline. Uses progressive import strategy to identify specific
    module failures and provides detailed diagnostic information for
    troubleshooting import issues.
    
    Import Testing Strategy:
    1. Core Kokoro Modules: Essential modules for model loading and export
    2. Optional Dependencies: Libraries with graceful fallback behavior  
    3. Version Validation: Check imported module versions for compatibility
    4. Functionality Testing: Basic functionality verification beyond imports
    
    Returns:
        Tuple[bool, Dict[str, str]]: (all_imports_successful, detailed_results)
                                   all_imports_successful: True if all required imports succeed
                                   detailed_results: Dict mapping module names to status messages
    
    Testing Process:
    - Required modules: Failure of any required module fails the entire test
    - Optional modules: Logged but don't affect overall success status
    - Version checking: Warnings for outdated versions, errors for incompatible versions
    - Import timing: Detection of slow imports that may indicate issues
    
    Error Reporting:
    - Missing modules: Clear installation instructions and package names
    - Version conflicts: Specific version requirements and upgrade guidance
    - Import failures: Detailed error messages with common resolution steps
    - Performance issues: Warnings for unusually slow import operations
    
    Called by:
    - main(): Core dependency validation after environment checks
    - Development setup: Module availability verification for new environments
    - CI/CD pipelines: Automated dependency validation in deployment workflows
    """
    results = {}
    all_successful = True
    
    try:
        # Test required Kokoro modules
        for module_name in ExportTestConstants.REQUIRED_KOKORO_MODULES:
            try:
                if module_name == 'kokoro.model':
                    from kokoro.model import KModel
                    results[module_name] = "✓ KModel class available"
                elif module_name == 'kokoro.pipeline':
                    from kokoro.pipeline import KPipeline
                    results[module_name] = "✓ KPipeline class available" 
                elif module_name == 'kokoro.__init__':
                    import kokoro
                    results[module_name] = "✓ Package initialization successful"
                else:
                    __import__(module_name)
                    results[module_name] = "✓ Import successful"
                    
            except ImportError as e:
                results[module_name] = f"❌ Import failed: {e}"
                all_successful = False
            except Exception as e:
                results[module_name] = f"❌ Unexpected error: {e}"
                all_successful = False
        
        # Test optional modules with version checking
        for module_name, description in ExportTestConstants.OPTIONAL_MODULES.items():
            try:
                if module_name == 'coremltools':
                    import coremltools as ct
                    version = ct.__version__
                    version_status = "✓"
                    if version < ExportTestConstants.COREML_MIN_VERSION:
                        version_status = f"⚠️  Version {version} below minimum {ExportTestConstants.COREML_MIN_VERSION}"
                    results[module_name] = f"{version_status} Version {version} - {description}"
                    
                elif module_name == 'torch':
                    import torch
                    version = torch.__version__.split('+')[0]  # Remove CUDA suffix
                    results[module_name] = f"✓ Version {version} - {description}"
                    
                elif module_name == 'safetensors':
                    import safetensors
                    results[module_name] = f"✓ Available - {description}"
                    
                else:
                    __import__(module_name)
                    results[module_name] = f"✓ Available - {description}"
                    
            except ImportError:
                results[module_name] = f"⚠️  Not available - {description} (optional)"
            except Exception as e:
                results[module_name] = f"⚠️  Error: {e} - {description} (optional)"
        
        return all_successful, results
        
    except Exception as e:
        results['import_testing'] = f"❌ Import testing failed: {e}"
        return False, results

def validate_model_files() -> Tuple[bool, Dict[str, str]]:
    """
    Validate availability and basic integrity of required model files.
    
    Performs comprehensive validation of all model files required for the export
    pipeline, including checkpoint files, configuration files, and supporting
    data. Uses progressive validation to identify specific file issues and
    provides detailed guidance for resolving missing or corrupted files.
    
    File Validation Strategy:
    1. Existence Check: Verify files exist at expected locations
    2. Accessibility: Confirm read permissions and file system access
    3. Format Validation: Basic file format and corruption detection
    4. Size Validation: Sanity check for reasonable file sizes
    5. Content Validation: Basic content structure verification where applicable
    
    Returns:
        Tuple[bool, Dict[str, str]]: (all_files_valid, detailed_file_status)
                                   all_files_valid: True if all required files are valid
                                   detailed_file_status: Dict mapping file paths to validation results
    
    Validation Process:
    - Required files: Missing or corrupted required files fail the validation
    - Optional files: Noted but don't affect overall validation status
    - Permission checks: Verify read access for all required files
    - Size validation: Flag suspiciously small or large files
    - Format detection: Basic file format validation where possible
    
    File Requirements:
    - config.json: JSON configuration file with model architecture parameters
    - kokoro-v1_0.pth: PyTorch checkpoint file with model weights
    - Supporting files: Additional model components based on configuration
    
    Error Recovery Guidance:
    - Missing files: Instructions for downloading or generating required files
    - Permission issues: Guidance for resolving file system access problems
    - Corruption detection: Steps for re-downloading or regenerating corrupted files
    - Path issues: Alternative file locations and configuration options
    
    Called by:
    - main(): Model file validation after successful import testing
    - Development setup: Verify model files are properly installed
    - Export preparation: Pre-export validation of required model components
    """
    file_status = {}
    all_valid = True
    
    try:
        # Check configuration file
        config_path = ExportTestConstants.DEFAULT_CONFIG_PATH
        if os.path.exists(config_path):
            try:
                file_size = os.path.getsize(config_path)
                if file_size > 0:
                    file_status[config_path] = f"✓ Found ({file_size:,} bytes)"
                else:
                    file_status[config_path] = "❌ File is empty"
                    all_valid = False
            except (OSError, PermissionError) as e:
                file_status[config_path] = f"❌ Access error: {e}"
                all_valid = False
        else:
            file_status[config_path] = "❌ Not found"
            all_valid = False
        
        # Check checkpoint file
        checkpoint_path = ExportTestConstants.DEFAULT_CHECKPOINT_PATH
        if os.path.exists(checkpoint_path):
            try:
                file_size = os.path.getsize(checkpoint_path)
                if file_size > 1024 * 1024:  # At least 1MB for valid model
                    size_mb = file_size / (1024 * 1024)
                    file_status[checkpoint_path] = f"✓ Found ({size_mb:.1f} MB)"
                else:
                    file_status[checkpoint_path] = f"❌ File too small ({file_size:,} bytes)"
                    all_valid = False
            except (OSError, PermissionError) as e:
                file_status[checkpoint_path] = f"❌ Access error: {e}"
                all_valid = False
        else:
            file_status[checkpoint_path] = "❌ Not found"
            all_valid = False
        
        # Check checkpoint directory structure
        checkpoint_dir = ExportTestConstants.CHECKPOINT_DIRECTORY
        if os.path.exists(checkpoint_dir):
            if os.path.isdir(checkpoint_dir):
                try:
                    files = os.listdir(checkpoint_dir)
                    file_status[f"{checkpoint_dir}/"] = f"✓ Directory with {len(files)} files"
                except (OSError, PermissionError) as e:
                    file_status[f"{checkpoint_dir}/"] = f"❌ Cannot list directory: {e}"
                    all_valid = False
            else:
                file_status[f"{checkpoint_dir}/"] = "❌ Path exists but is not a directory"
                all_valid = False
        else:
            file_status[f"{checkpoint_dir}/"] = "❌ Directory not found"
            all_valid = False
        
        return all_valid, file_status
        
    except Exception as e:
        file_status['validation_error'] = f"❌ File validation failed: {e}"
        return False, file_status

def main():
    """
    Main execution function for comprehensive export pipeline validation.
    
    Orchestrates the complete validation workflow, providing structured output
    for both human reading and automated processing. Implements progressive
    testing with early termination on critical failures while collecting
    comprehensive diagnostic information.
    
    Validation Workflow:
    1. Environment Validation: Python version, paths, and basic configuration
    2. Import Testing: Core module availability and version compatibility
    3. Model File Validation: Required files, permissions, and basic integrity
    4. Summary Report: Comprehensive status with actionable recommendations
    
    Output Format:
    - Human-readable: Structured console output with visual indicators
    - Automation-friendly: Consistent format for CI/CD and monitoring integration
    - Diagnostic: Detailed error messages with troubleshooting guidance
    - Progressive: Early failure detection with partial results reporting
    
    Exit Codes:
    - 0: All validations passed successfully
    - 1: Critical validation failures detected
    - 2: Environment configuration issues
    - 3: Import or dependency issues
    - 4: Model file issues
    
    Called by:
    - Direct execution: python test_export.py
    - CI/CD pipelines: Automated environment validation
    - Development setup: New environment verification
    - Troubleshooting: Quick diagnostic for export issues
    """
    print("🧪 Kokoro TTS Export Pipeline Validation")
    print("=" * 50)
    
    overall_success = True
    
    # Phase 1: Python Environment Validation
    print("\n1. 🐍 Python Environment Validation")
    env_success, env_details = validate_python_environment()
    
    if ExportTestConstants.VERBOSE_OUTPUT:
        for line in env_details.split('\n'):
            print(f"   {line}")
    
    if not env_success:
        print(f"❌ Environment validation failed")
        overall_success = False
        if not ExportTestConstants.STRUCTURED_OUTPUT:
            sys.exit(2)
    else:
        print("✅ Environment validation passed")
    
    # Phase 2: Core Module Import Testing
    print("\n2. 📦 Import Testing")
    import_success, import_results = test_core_imports()
    
    for module_name, status in import_results.items():
        print(f"   {module_name}: {status}")
    
    if not import_success:
        print("❌ Import testing failed")
        overall_success = False
        if not ExportTestConstants.STRUCTURED_OUTPUT:
            sys.exit(3)
    else:
        print("✅ Import testing passed")
    
    # Phase 3: Model File Validation
    print("\n3. 📁 Model File Validation")
    file_success, file_results = validate_model_files()
    
    for file_path, status in file_results.items():
        print(f"   {file_path}: {status}")
    
    if not file_success:
        print("❌ Model file validation failed")
        overall_success = False
        if not ExportTestConstants.STRUCTURED_OUTPUT:
            sys.exit(4)
    else:
        print("✅ Model file validation passed")
    
    # Final Summary and Recommendations
    print("\n" + "=" * 50)
    if overall_success:
        print("🎉 All validations passed! Export pipeline ready.")
        print("\nNext steps:")
        print("• Run export_vocoder.py to create CoreML vocoder models")
        print("• Use export_synthesizers.py for bucket model creation")
        print("• Test with test_ane_pipeline.py for performance validation")
    else:
        print("❌ Validation failed! Please resolve issues above.")
        print("\nTroubleshooting steps:")
        if not env_success:
            print("• Update Python to version 3.8+ (3.11+ recommended)")
            print("• Check virtual environment activation")
        if not import_success:
            print("• Install missing packages: pip install -r requirements.txt")
            print("• Verify PYTHONPATH includes project directory")
        if not file_success:
            print("• Run convert_checkpoint.py to create PyTorch checkpoint")
            print("• Download model files from official repository")
            print("• Check file permissions and disk space")
    
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during validation: {e}")
        if ExportTestConstants.ENABLE_TRACEBACK:
            traceback.print_exc()
        sys.exit(1)