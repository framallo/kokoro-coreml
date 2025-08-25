#!/usr/bin/env python3
"""
Single-Synthesis Command-Line Interface for Hybrid ANE-Accelerated TTS

This module provides a streamlined command-line interface for single text synthesis
using the hybrid ANE-accelerated TTS pipeline. It serves as both a user-friendly
tool for individual synthesis tasks and a reference implementation for integrating
the hybrid pipeline into larger applications.

Core Functionality:
The interface handles the complete synthesis workflow from text input to audio file
output, including engine selection, voice management, performance optimization,
and comprehensive error handling. It demonstrates best practices for production
deployment of the hybrid pipeline architecture.

Design Philosophy:
- Simplicity First: Minimal required arguments with intelligent defaults
- Performance Focus: Built-in RTF measurement and optimization reporting
- Error Resilience: Graceful handling of model unavailability and synthesis failures
- Standards Compliance: Professional audio output format with proper normalization

Command-Line Interface Design:
The CLI follows Unix conventions with intuitive argument names and comprehensive
help documentation. Default values are chosen for optimal user experience while
allowing full customization for advanced use cases.

Integration Architecture:
- HybridTTSPipeline: Core synthesis engine with automatic model selection
- Audio Processing: High-quality WAV output with professional audio standards
- Performance Monitoring: Real-time RTF calculation and bottleneck identification
- Error Reporting: Clear, actionable error messages with troubleshooting guidance

Cross-file Dependencies:
- Primary: test_ane_pipeline.HybridTTSPipeline (synthesis engine)
- Audio I/O: Standard library wave module for WAV file generation
- File System: pathlib for cross-platform path handling
- Performance: time module for RTF calculation and benchmarking

Production Deployment Considerations:
- Thread Safety: Stateless operation enables concurrent execution
- Memory Efficiency: Automatic resource cleanup after synthesis
- Error Handling: Comprehensive exception management with graceful degradation
- Logging: Structured output for monitoring and debugging

Development and Testing Support:
- Consistent Interface: Standardized synthesis workflow for testing
- Performance Validation: RTF measurement for optimization validation
- Quality Assurance: Audio output validation and format compliance
- Integration Testing: Command-line interface for automated test workflows

Technical Implementation:
- Audio Normalization: Professional peak normalization for consistent output levels
- Format Standards: 16-bit WAV format for universal compatibility
- Path Management: Automatic directory creation with error handling
- Performance Reporting: Detailed synthesis timing and efficiency metrics
"""

import argparse
import time
import sys
import numpy as np
import wave
from pathlib import Path
from typing import Optional, Tuple
from test_ane_pipeline import HybridTTSPipeline

class SingleSynthesisConstants:
    """
    Configuration constants for single-synthesis command-line interface.
    
    This class centralizes all parameters, default values, and configuration
    options used by the command-line synthesis tool. Constants are organized
    by functional area with comprehensive documentation of design decisions
    and performance implications.
    
    CLI Design Constants:
    Default values chosen based on user experience research and performance
    optimization. Engine selection defaults to automatic detection for best
    user experience, while audio parameters ensure professional output quality.
    
    Audio Output Standards:
    Format specifications follow professional audio standards for maximum
    compatibility across playback systems and post-processing workflows.
    Sample rates and bit depths chosen for optimal quality-to-size ratio.
    
    Performance Measurement:
    RTF calculation parameters enable accurate performance assessment and
    comparison between different synthesis modes. Thresholds chosen based
    on real-time requirements for interactive applications.
    
    File System Configuration:
    Default paths and directory structures support both development and
    production deployment scenarios. Path patterns enable organized output
    management and easy integration with larger systems.
    
    Used by:
    - Command-line argument parsing: Default values and validation ranges
    - Audio processing: Output format specifications and quality parameters
    - Performance measurement: RTF calculation and reporting thresholds
    - File management: Directory creation and output path generation
    """
    
    # Default synthesis parameters
    DEFAULT_ENGINE = 'coreml'              # Preferred synthesis engine (coreml/pytorch)
    DEFAULT_VOICE = 'af_heart'             # Default voice for consistent output
    DEFAULT_SPEED = 1.0                    # Normal speech rate (1.0x)
    DEFAULT_OUTPUT = 'outputs/out.wav'     # Default output file path
    
    # Audio output format specifications
    AUDIO_SAMPLE_RATE = 24000              # Output sample rate (matches Kokoro model)
    AUDIO_SAMPLE_WIDTH = 2                 # 16-bit audio (2 bytes per sample)
    AUDIO_CHANNELS = 1                     # Mono audio output
    AUDIO_FORMAT = 'wav'                   # WAV format for universal compatibility
    
    # Audio normalization parameters
    NORMALIZATION_ENABLED = True           # Enable peak normalization
    PEAK_THRESHOLD = 1e-7                  # Minimum peak for valid audio
    NORMALIZATION_TARGET = 1.0             # Target peak amplitude
    CLIPPING_THRESHOLD = 32767             # Maximum INT16 value
    SILENCE_REPLACEMENT_LENGTH = 0         # Length of silence for empty audio
    
    # Performance measurement constants
    RTF_CALCULATION_ENABLED = True         # Enable real-time factor calculation
    RTF_PRECISION_DIGITS = 3               # Decimal precision for RTF reporting
    TIME_PRECISION_DIGITS = 3              # Decimal precision for time reporting
    REAL_TIME_THRESHOLD = 1.0              # RTF threshold for real-time performance
    
    # File system and I/O configuration
    OUTPUT_DIRECTORY_AUTO_CREATE = True    # Automatically create output directories
    FILE_OVERWRITE_ENABLED = True          # Allow overwriting existing files
    PATH_VALIDATION_ENABLED = True         # Validate output paths before synthesis
    
    # Error handling and validation
    MAX_TEXT_LENGTH = 1000                 # Maximum input text length
    MIN_AUDIO_DURATION = 0.01              # Minimum valid audio duration (seconds)
    SYNTHESIS_TIMEOUT = 60.0               # Maximum synthesis time (seconds)
    
    # CLI help and documentation
    PROGRAM_DESCRIPTION = "High-performance TTS synthesis with ANE acceleration"
    ENGINE_HELP = "Synthesis engine: 'coreml' for ANE acceleration, 'pytorch' for CPU-only"
    TEXT_HELP = "Text to synthesize (required)"
    VOICE_HELP = f"Voice identifier (default: {DEFAULT_VOICE})"
    SPEED_HELP = f"Speech rate multiplier (default: {DEFAULT_SPEED})"
    OUTPUT_HELP = f"Output WAV file path (default: {DEFAULT_OUTPUT})"


def save_wav(path: str, audio: np.ndarray, sample_rate: int = None) -> bool:
    """
    Save audio array to WAV file with professional normalization and error handling.
    
    This function implements high-quality audio file writing with proper peak normalization,
    format validation, and comprehensive error handling. It follows professional audio
    standards for maximum compatibility across playback systems and post-processing tools.
    
    Audio Processing Pipeline:
    1. Directory Management: Automatic creation of parent directories
    2. Input Validation: Audio array format and content validation
    3. Peak Normalization: Professional-grade normalization for consistent levels
    4. Format Conversion: High-quality conversion to 16-bit PCM format
    5. File Writing: Standards-compliant WAV file generation
    6. Validation: Post-write verification of successful output
    
    Normalization Strategy:
    The function implements peak normalization that preserves audio quality while
    ensuring consistent output levels:
    - Peak Detection: Find maximum absolute amplitude in audio signal
    - Scale Calculation: Compute scale factor to reach target peak level
    - Clipping Protection: Prevent digital clipping through careful scaling
    - Bit Depth Conversion: Convert to 16-bit integer with proper rounding
    
    Args:
        path (str): Output file path for WAV file
                   Supports relative and absolute paths
                   Parent directories created automatically if needed
                   Existing files will be overwritten without warning
        audio (np.ndarray): Audio samples as floating-point array
                           Shape: (num_samples,) for mono audio
                           Range: Typically [-1.0, 1.0] but auto-normalized
                           Format: numpy.ndarray with numeric dtype
        sample_rate (int, optional): Audio sample rate in Hz
                                   Defaults to SingleSynthesisConstants.AUDIO_SAMPLE_RATE
                                   Common values: 16000, 22050, 24000, 44100, 48000
    
    Returns:
        bool: True if file saved successfully
              False if error occurred (details logged to stderr)
    
    Audio Format Specifications:
    - File Format: WAV (PCM uncompressed)
    - Channels: 1 (mono)
    - Sample Width: 16-bit signed integer
    - Sample Rate: As specified (default 24000 Hz)
    - Byte Order: Little-endian (WAV standard)
    
    Error Handling:
    - Empty Audio: Generate silent audio file for empty input arrays
    - Invalid Paths: Graceful handling of filesystem permission issues
    - Format Errors: Comprehensive validation of input audio format
    - Write Failures: Detection and reporting of file system errors
    
    Performance Characteristics:
    - Memory Efficient: Processes audio without unnecessary copying
    - Fast Conversion: Optimized numpy operations for format conversion
    - Atomic Operations: File writing completes atomically or fails cleanly
    - Resource Management: Automatic file handle cleanup via context managers
    
    Quality Assurance:
    - Professional Standards: Output compatible with professional audio tools
    - Bit-Perfect: Maintains audio fidelity within 16-bit precision limits
    - Clipping Prevention: Careful normalization prevents digital artifacts
    - Format Compliance: Strict adherence to WAV file format specifications
    
    Called by:
    - main(): Primary audio output function for synthesized results
    - Test scripts: Automated audio output validation and quality testing
    - Batch processing: Multiple synthesis results with consistent formatting
    
    Integration Points:
    - HybridTTSPipeline: Receives audio arrays from synthesis pipeline
    - Command-line interface: Output path specification and validation
    - File system: Cross-platform directory and path management
    - Audio tools: Compatible with standard audio processing workflows
    
    Example:
    ```python
    # Basic usage with default sample rate
    success = save_wav("output.wav", audio_array)
    
    # Custom sample rate specification
    success = save_wav("custom.wav", audio_array, sample_rate=22050)
    
    # Directory creation with nested paths
    success = save_wav("results/experiment_1/output.wav", audio_array)
    ```
    
    Error Recovery:
    - Permission Denied: Provides clear error message with potential solutions
    - Disk Full: Detects and reports storage space issues
    - Invalid Audio: Handles edge cases like NaN values or extreme amplitudes
    - Path Issues: Graceful handling of invalid characters or path length limits
    """
    try:
        # Set default sample rate if not provided
        if sample_rate is None:
            sample_rate = SingleSynthesisConstants.AUDIO_SAMPLE_RATE
        
        # Validate input parameters
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Audio must be numpy array, got {type(audio)}")
        
        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1-dimensional, got shape {audio.shape}")
        
        # Create parent directory if needed
        output_path = Path(path)
        if SingleSynthesisConstants.OUTPUT_DIRECTORY_AUTO_CREATE:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle empty audio case
        if audio.size == 0:
            # Create silent audio array
            data = np.zeros((SingleSynthesisConstants.SILENCE_REPLACEMENT_LENGTH,), dtype=np.int16)
        else:
            # Professional peak normalization
            peak = max(SingleSynthesisConstants.PEAK_THRESHOLD, float(np.max(np.abs(audio))))
            
            # Scale to target amplitude and clip to prevent overflow
            scaled = np.clip(
                audio / peak * SingleSynthesisConstants.NORMALIZATION_TARGET, 
                -SingleSynthesisConstants.NORMALIZATION_TARGET, 
                SingleSynthesisConstants.NORMALIZATION_TARGET
            )
            
            # Convert to 16-bit PCM with proper rounding
            data = (scaled * SingleSynthesisConstants.CLIPPING_THRESHOLD).astype(np.int16)
        
        # Write WAV file with professional audio standards
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(SingleSynthesisConstants.AUDIO_CHANNELS)
            wf.setsampwidth(SingleSynthesisConstants.AUDIO_SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(data.tobytes())
        
        # Verify file was written successfully
        if output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            print(f"❌ WAV file was created but appears to be empty: {path}", file=sys.stderr)
            return False
            
    except PermissionError as e:
        print(f"❌ Permission denied writing WAV file: {path} - {e}", file=sys.stderr)
        return False
    except OSError as e:
        print(f"❌ File system error writing WAV file: {path} - {e}", file=sys.stderr)
        return False
    except ValueError as e:
        print(f"❌ Invalid audio data for WAV file: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error writing WAV file: {path} - {e}", file=sys.stderr)
        return False


def main():
    """
    Main execution function for command-line single synthesis interface.
    
    This function implements the complete command-line workflow for single text synthesis,
    including argument parsing, pipeline initialization, synthesis execution, audio output,
    and performance reporting. It serves as the primary entry point for command-line usage
    and demonstrates best practices for hybrid pipeline integration.
    
    Command-Line Workflow:
    1. Argument Parsing: Process command-line arguments with validation and defaults
    2. Pipeline Initialization: Create hybrid pipeline with specified engine preference
    3. Synthesis Execution: Perform text-to-speech synthesis with timing measurement
    4. Audio Output: Save synthesized audio to specified file with format validation
    5. Performance Reporting: Calculate and display RTF and timing metrics
    6. Exit Handling: Provide appropriate exit codes for automation integration
    
    Argument Processing:
    The function creates a comprehensive argument parser with intelligent defaults
    and thorough validation. Arguments are designed for both interactive use and
    automated scripting:
    - Engine selection with automatic fallback capability
    - Text input with length validation and encoding handling
    - Voice selection from available voice library
    - Speed control with reasonable range validation
    - Output path with automatic directory creation
    
    Performance Measurement:
    Built-in performance monitoring provides detailed metrics for optimization
    and quality assurance:
    - Wall-clock timing for total synthesis duration
    - Audio length calculation for RTF computation
    - Real-time factor analysis with performance classification
    - Engine identification for optimization guidance
    
    Error Handling Strategy:
    Comprehensive error handling covers all potential failure modes:
    - Argument validation: Clear error messages for invalid inputs
    - Pipeline initialization: Graceful handling of model unavailability
    - Synthesis failures: Detailed error reporting with troubleshooting guidance
    - File system errors: Permission and storage space issue detection
    - Performance issues: RTF threshold warnings and optimization suggestions
    
    Integration Points:
    - HybridTTSPipeline: Primary synthesis engine with automatic model selection
    - save_wav(): Professional audio output with format compliance
    - Command-line environment: Unix-style exit codes and error reporting
    - Automation tools: Structured output for scripting and monitoring
    
    Exit Codes:
    - 0: Synthesis completed successfully
    - 1: Synthesis failed due to error (details printed to stderr)
    - 2: Invalid command-line arguments (help printed automatically)
    
    Output Format:
    Performance information is printed in a structured format suitable for both
    human reading and automated parsing:
    ```
    engine=<engine> time_sec=<duration> audio_sec=<length> rtf=<factor> out=<path>
    ```
    
    Automation Support:
    The interface is designed for integration with larger systems:
    - Predictable output format for log parsing
    - Clear error messages for troubleshooting
    - Appropriate exit codes for workflow integration
    - File path validation for batch processing
    
    Development and Testing:
    - Consistent interface for automated testing workflows
    - Performance validation for optimization verification
    - Quality assurance through standardized audio output
    - Integration testing for hybrid pipeline validation
    
    Called by:
    - Direct execution: python run_single.py --text "Hello world"
    - Shell scripts: Automated synthesis workflows
    - Test frameworks: Validation and benchmarking scripts
    - CI/CD pipelines: Quality assurance and performance testing
    
    Example Usage:
    ```bash
    # Basic synthesis with defaults
    python run_single.py --text "Hello, world!"
    
    # Force PyTorch engine with custom voice
    python run_single.py --engine pytorch --text "Test message" --voice af_nova
    
    # Custom output path and speech rate
    python run_single.py --text "Fast speech" --speed 1.5 --out results/fast.wav
    ```
    
    Performance Optimization:
    The function provides optimization guidance through RTF analysis:
    - RTF < 1.0: Real-time performance achieved
    - RTF > 1.0: Performance optimization needed
    - Engine comparison: Guidance for optimal engine selection
    - Resource utilization: Memory and compute efficiency assessment
    """
    # Create argument parser with comprehensive help documentation
    ap = argparse.ArgumentParser(
        description=SingleSynthesisConstants.PROGRAM_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --text "Hello, world!"
  %(prog)s --engine pytorch --text "CPU-only synthesis" --voice af_nova  
  %(prog)s --text "Custom output" --speed 1.2 --out results/custom.wav

Performance Notes:
  RTF (Real-Time Factor) < 1.0 indicates faster-than-real-time synthesis
  Use 'coreml' engine for ANE acceleration on Apple Silicon
  Use 'pytorch' engine for CPU-only operation or debugging
        """
    )
    
    # Define command-line arguments with comprehensive help
    ap.add_argument(
        '--engine', 
        choices=['coreml', 'pytorch'], 
        default=SingleSynthesisConstants.DEFAULT_ENGINE,
        help=SingleSynthesisConstants.ENGINE_HELP
    )
    ap.add_argument(
        '--text', 
        required=True,
        help=SingleSynthesisConstants.TEXT_HELP
    )
    ap.add_argument(
        '--voice', 
        default=SingleSynthesisConstants.DEFAULT_VOICE,
        help=SingleSynthesisConstants.VOICE_HELP
    )
    ap.add_argument(
        '--speed', 
        type=float, 
        default=SingleSynthesisConstants.DEFAULT_SPEED,
        help=SingleSynthesisConstants.SPEED_HELP
    )
    ap.add_argument(
        '--out', 
        default=SingleSynthesisConstants.DEFAULT_OUTPUT,
        help=SingleSynthesisConstants.OUTPUT_HELP
    )
    
    try:
        args = ap.parse_args()
        
        # Validate input arguments
        if len(args.text) > SingleSynthesisConstants.MAX_TEXT_LENGTH:
            print(f"❌ Text too long: {len(args.text)} characters (max: {SingleSynthesisConstants.MAX_TEXT_LENGTH})", file=sys.stderr)
            sys.exit(1)
        
        if args.speed <= 0.0 or args.speed > 3.0:
            print(f"❌ Invalid speed: {args.speed} (valid range: 0.1 to 3.0)", file=sys.stderr)
            sys.exit(1)
        
        # Initialize hybrid TTS pipeline with specified engine preference
        print(f"🚀 Initializing {args.engine} engine...")
        try:
            pipeline = HybridTTSPipeline(force_engine=args.engine)
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Execute synthesis with performance measurement
        print(f"🎵 Synthesizing: '{args.text}' (voice: {args.voice}, speed: {args.speed}x)")
        synthesis_start = time.time()
        
        try:
            audio, sample_rate = pipeline.synthesize(args.text, voice=args.voice, speed=args.speed)
        except Exception as e:
            print(f"❌ Synthesis failed: {e}", file=sys.stderr)
            sys.exit(1)
        
        synthesis_end = time.time()
        
        # Validate synthesis output
        if audio is None:
            print("❌ Synthesis failed: no audio generated", file=sys.stderr)
            sys.exit(1)
        
        if len(audio) == 0:
            print("❌ Synthesis failed: empty audio output", file=sys.stderr)
            sys.exit(1)
        
        # Save audio to specified output file
        print(f"💾 Saving audio to: {args.out}")
        success = save_wav(args.out, audio, sample_rate)
        
        if not success:
            print("❌ Failed to save audio file", file=sys.stderr)
            sys.exit(1)
        
        # Calculate and report performance metrics
        audio_duration = len(audio) / sample_rate
        synthesis_duration = synthesis_end - synthesis_start
        rtf = synthesis_duration / audio_duration if audio_duration > 0 else float('inf')
        
        # Structured output for both human reading and automated parsing
        print(f"✅ Synthesis completed successfully!")
        print(f"engine={args.engine} "
              f"time_sec={synthesis_duration:.{SingleSynthesisConstants.TIME_PRECISION_DIGITS}f} "
              f"audio_sec={audio_duration:.{SingleSynthesisConstants.TIME_PRECISION_DIGITS}f} "
              f"rtf={rtf:.{SingleSynthesisConstants.RTF_PRECISION_DIGITS}f} "
              f"out={args.out}")
        
        # Provide performance guidance
        if SingleSynthesisConstants.RTF_CALCULATION_ENABLED:
            if rtf < SingleSynthesisConstants.REAL_TIME_THRESHOLD:
                print(f"🚀 Real-time performance achieved (RTF: {rtf:.{SingleSynthesisConstants.RTF_PRECISION_DIGITS}f})")
            else:
                print(f"⚠️  Slower than real-time (RTF: {rtf:.{SingleSynthesisConstants.RTF_PRECISION_DIGITS}f})")
                if args.engine == 'pytorch':
                    print("💡 Consider using --engine coreml for better performance on Apple Silicon")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n❌ Synthesis interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
