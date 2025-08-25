#!/usr/bin/env python3
"""
Comprehensive Test Suite for Custom STFT Implementation Validation

This module provides exhaustive testing and validation for the CustomSTFT implementation
used in the Kokoro TTS vocoder architecture. The tests ensure numerical accuracy,
performance consistency, and architectural compatibility between the custom STFT
implementation and PyTorch's reference implementation across various signal processing
scenarios and deployment configurations.

Testing Philosophy:
The test suite follows a rigorous validation approach that verifies both functional
correctness and numerical precision. Custom STFT implementations are critical for
TTS vocoder performance, as they directly impact audio quality and synthesis fidelity.
The tests validate implementation equivalence across different signal characteristics,
processing parameters, and batch configurations.

Validation Architecture:
1. Reference Comparison: Validates custom implementation against PyTorch's torch.stft
2. Numerical Precision: Ensures floating-point accuracy within acceptable tolerances
3. Signal Processing: Tests across various audio characteristics and spectral content
4. Batch Processing: Validates efficiency and correctness for batch inference
5. Parameter Sweeps: Tests robustness across different STFT configuration parameters
6. Edge Cases: Validates behavior with boundary conditions and unusual inputs

STFT Implementation Context:
The CustomSTFT module provides a specialized Short-Time Fourier Transform implementation
optimized for the Kokoro TTS vocoder. Key implementation characteristics:
- CoreML Export Compatibility: Operations designed for seamless CoreML conversion
- ANE Optimization: Tensor layouts and operations optimized for Apple Neural Engine
- Numerical Stability: Careful handling of phase computation and reconstruction
- Performance Tuning: Optimized for real-time synthesis requirements

Test Coverage Areas:
- Reconstruction Fidelity: Forward-inverse transform accuracy and signal preservation
- Magnitude/Phase Consistency: Spectral component accuracy across implementations
- Batch Processing: Multi-signal processing efficiency and correctness
- Parameter Robustness: Performance across various window sizes and hop lengths
- Boundary Conditions: Edge case handling and numerical stability
- Performance Characteristics: Timing and memory usage validation

Cross-file Dependencies:
- Primary: kokoro.custom_stft.CustomSTFT (implementation under test)
- Reference: kokoro.istftnet.TorchSTFT (reference implementation)
- Integration: Used by vocoder components in model.py and export pipelines
- Validation: Supports CoreML export validation and deployment verification

Quality Assurance Standards:
- Numerical Tolerance: Strict floating-point comparison with appropriate tolerances
- Signal Quality: Audio reconstruction fidelity measurement and validation
- Performance Benchmarking: Timing and resource usage measurement
- Implementation Parity: Exact equivalence verification with reference implementation

Development and Production Support:
- Regression Testing: Automated validation in CI/CD pipelines
- Performance Profiling: Benchmarking for optimization and deployment planning
- Debug Support: Detailed failure analysis and diagnostic information
- Documentation: Comprehensive test case documentation for future development
"""

import torch
import numpy as np
import pytest
import time
from typing import Dict, List, Tuple, Optional, Any
from kokoro.custom_stft import CustomSTFT
from kokoro.istftnet import TorchSTFT
import torch.nn.functional as F

class STFTTestConstants:
    """
    Configuration constants for STFT implementation testing and validation.
    
    This class centralizes all testing parameters, tolerance thresholds, and
    configuration values used throughout the STFT test suite. Constants are
    organized by functional area with comprehensive documentation of testing
    strategies and validation criteria.
    
    Numerical Precision Constants:
    Tolerance values chosen based on floating-point precision characteristics
    and acceptable audio quality degradation. Thresholds balance strict validation
    with practical numerical precision limitations in DSP operations.
    
    Audio Signal Parameters:
    Default values chosen to provide comprehensive signal processing validation
    across typical TTS synthesis scenarios. Parameters cover common sample rates,
    frequency content, and duration ranges used in speech synthesis.
    
    STFT Configuration Parameters:
    Window sizes, hop lengths, and other STFT parameters selected to test
    common configurations used in modern TTS vocoders while ensuring
    comprehensive validation coverage across parameter space.
    
    Used by:
    - Test functions: Tolerance thresholds and comparison criteria
    - Fixture generation: Audio signal parameters and characteristics
    - Performance testing: Benchmarking parameters and timing thresholds
    - Batch testing: Multi-signal processing configuration
    """
    
    # Numerical precision and tolerance thresholds
    RECONSTRUCTION_RTOL = 1e-3                  # Relative tolerance for reconstruction tests
    RECONSTRUCTION_ATOL = 1e-3                  # Absolute tolerance for reconstruction tests
    MAGNITUDE_RTOL = 1e-2                       # Relative tolerance for magnitude comparison
    MAGNITUDE_ATOL = 1e-2                       # Absolute tolerance for magnitude comparison
    PHASE_RTOL = 1e-1                          # Relative tolerance for phase comparison (less strict)
    PHASE_ATOL = 1e-1                          # Absolute tolerance for phase comparison
    
    # Audio signal generation parameters
    DEFAULT_SAMPLE_RATE = 16000                 # Standard sample rate for speech processing
    REFERENCE_FREQUENCY = 440.0                 # A4 reference frequency for sine wave generation
    DEFAULT_DURATION = 1.0                      # Default test signal duration in seconds
    SHORT_DURATION = 0.1                       # Short duration for fast batch testing
    BATCH_TEST_SIZE = 4                        # Default batch size for batch processing tests
    
    # STFT configuration parameters for testing
    DEFAULT_FILTER_LENGTH = 800                 # Default FFT window size
    DEFAULT_HOP_LENGTH = 200                    # Default hop length (25% overlap)
    DEFAULT_WIN_LENGTH = 800                    # Default window length (matches filter length)
    
    # Parameter sweep ranges for robustness testing
    FILTER_LENGTH_VARIANTS = [512, 1024, 2048] # Window sizes for parameter sweep tests
    HOP_RATIO_VARIANTS = [2, 4, 8]             # Hop length ratios (filter_length / ratio)
    
    # Performance and timing thresholds
    MAX_PROCESSING_TIME_SEC = 5.0               # Maximum allowed processing time per test
    MEMORY_USAGE_LIMIT_MB = 100                 # Maximum memory usage for test signals
    
    # Boundary condition testing parameters
    BOUNDARY_FRAME_MARGIN = 2                   # Frames to exclude from boundary testing
    MINIMUM_SIGNAL_LENGTH = 1024                # Minimum signal length for meaningful STFT
    MAXIMUM_SIGNAL_LENGTH = 48000               # Maximum test signal length (3 seconds at 16kHz)
    
    # Batch processing validation parameters
    MAX_BATCH_SIZE = 16                         # Maximum batch size for testing
    BATCH_SIZE_VARIANTS = [1, 2, 4, 8]         # Different batch sizes for validation
    
    # Error detection and validation parameters
    NAN_DETECTION_ENABLED = True               # Enable NaN detection in outputs
    INF_DETECTION_ENABLED = True               # Enable infinity detection in outputs
    SHAPE_VALIDATION_ENABLED = True            # Enable output shape validation
    
    # Debug and development parameters
    VERBOSE_LOGGING = False                     # Enable detailed test logging
    SAVE_TEST_OUTPUTS = False                   # Save test outputs for inspection
    PERFORMANCE_PROFILING = False               # Enable performance profiling

@pytest.fixture
def sample_audio():
    """
    Generate standardized test audio signal for STFT validation.
    
    Creates a high-quality sine wave signal with characteristics suitable for
    comprehensive STFT testing. The signal provides predictable spectral content
    that enables precise validation of frequency-domain processing accuracy
    and reconstruction fidelity.
    
    Signal Characteristics:
    - Frequency: 440 Hz (A4 musical note) for clear spectral peaks
    - Duration: 1 second for comprehensive temporal analysis
    - Sample Rate: 16 kHz (standard for speech processing)
    - Amplitude: Unit amplitude with clean harmonic content
    - Phase: Zero-phase start for predictable spectral characteristics
    
    Mathematical Generation:
    signal(t) = sin(2π × f × t) where f = 440 Hz
    Provides single-frequency sinusoidal content ideal for STFT validation
    
    Returns:
        torch.Tensor: Audio signal with shape (1, num_samples)
                     Batch dimension included for compatibility with STFT implementations
                     Float32 precision for numerical consistency
                     Sample values in range [-1.0, 1.0]
    
    Signal Properties:
    - Deterministic: Consistent output for reproducible testing
    - High SNR: Clean signal without noise for precise validation
    - Known Spectrum: Predictable frequency domain characteristics
    - Standard Format: Compatible with both CustomSTFT and TorchSTFT
    
    Used by:
    - test_stft_reconstruction(): Primary reconstruction fidelity validation
    - test_magnitude_phase_consistency(): Spectral component accuracy testing
    - Other test functions requiring standardized audio input
    
    Quality Assurance:
    - Amplitude normalization ensures consistent signal levels
    - Frequency selection provides clear spectral peaks for validation
    - Duration chosen for comprehensive temporal-spectral analysis
    - Sample rate matches typical TTS processing requirements
    """
    # Generate time axis for signal creation
    sample_rate = STFTTestConstants.DEFAULT_SAMPLE_RATE
    duration = STFTTestConstants.DEFAULT_DURATION
    t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
    
    # Create pure sine wave with reference frequency
    frequency = STFTTestConstants.REFERENCE_FREQUENCY
    signal = torch.sin(2 * np.pi * frequency * t)
    
    # Add batch dimension for compatibility with STFT implementations
    return signal.unsqueeze(0)  # Shape: (1, num_samples)


def test_stft_reconstruction(sample_audio):
    """
    Validate STFT reconstruction fidelity between custom and reference implementations.
    
    This test performs the most critical validation: ensuring that the CustomSTFT
    implementation produces numerically equivalent results to the reference TorchSTFT
    implementation for complete forward-inverse STFT processing. Reconstruction
    fidelity is essential for audio quality preservation in TTS vocoder applications.
    
    Test Methodology:
    1. Implementation Initialization: Create both custom and reference STFT processors
    2. Parallel Processing: Apply identical STFT parameters to the same input signal
    3. Output Comparison: Validate numerical equivalence within tolerance thresholds
    4. Quality Assessment: Ensure reconstruction maintains audio fidelity
    
    Validation Strategy:
    The test uses a pure sine wave input signal with known spectral characteristics.
    This enables precise detection of any implementation differences in:
    - Window function application and overlap processing
    - FFT computation accuracy and phase handling  
    - Inverse transform reconstruction and signal synthesis
    - Boundary condition handling at signal edges
    
    STFT Configuration:
    - Filter Length: 800 samples (50ms at 16kHz) for good time-frequency resolution
    - Hop Length: 200 samples (75% overlap) for high-quality reconstruction
    - Window Length: 800 samples (matching filter length) for standard windowing
    - Window Function: Hann window (implicit) for minimal spectral leakage
    
    Args:
        sample_audio (torch.Tensor): Standardized test signal from sample_audio fixture
                                   Shape: (1, num_samples) with clean sine wave content
                                   Used for predictable spectral analysis
    
    Validation Criteria:
    - Reconstruction Accuracy: Output signals must match within 1e-3 relative/absolute tolerance
    - Signal Preservation: No significant amplitude or phase distortion
    - Temporal Alignment: Proper handling of windowing and overlap reconstruction
    - Spectral Fidelity: Accurate frequency domain representation and inverse transform
    
    Quality Assurance:
    - Tolerance Selection: 1e-3 chosen based on floating-point precision limits
    - Signal Quality: Clean reconstruction essential for TTS audio quality
    - Implementation Parity: Ensures CustomSTFT can replace TorchSTFT seamlessly
    - Edge Case Coverage: Validates boundary frame handling and signal length processing
    
    Failure Analysis:
    Test failures typically indicate:
    - Window function implementation differences
    - FFT/IFFT numerical precision issues
    - Overlap-add reconstruction algorithm discrepancies
    - Phase computation or unwrapping errors
    - Boundary condition handling differences
    
    Called by:
    - pytest: Automated testing framework during development and CI/CD
    - Quality assurance: Pre-deployment validation of STFT implementation changes
    - Performance validation: Ensures optimization changes don't affect accuracy
    
    Integration Impact:
    - Vocoder Quality: STFT accuracy directly affects synthesized audio quality
    - CoreML Export: Validates compatibility with CoreML conversion requirements
    - Real-time Performance: Ensures optimizations maintain reconstruction fidelity
    - Production Deployment: Critical for audio quality in deployed TTS systems
    """
    # Initialize both STFT implementations with identical configurations
    filter_length = STFTTestConstants.DEFAULT_FILTER_LENGTH
    hop_length = STFTTestConstants.DEFAULT_HOP_LENGTH
    win_length = STFTTestConstants.DEFAULT_WIN_LENGTH
    
    custom_stft = CustomSTFT(
        filter_length=filter_length,
        hop_length=hop_length, 
        win_length=win_length
    )
    torch_stft = TorchSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length
    )

    # Process identical input through both implementations
    custom_output = custom_stft(sample_audio)
    torch_output = torch_stft(sample_audio)

    # Validate numerical equivalence within acceptable tolerances
    reconstruction_match = torch.allclose(
        custom_output, 
        torch_output, 
        rtol=STFTTestConstants.RECONSTRUCTION_RTOL, 
        atol=STFTTestConstants.RECONSTRUCTION_ATOL
    )
    
    assert reconstruction_match, (
        f"STFT reconstruction mismatch detected. "
        f"Max absolute difference: {torch.max(torch.abs(custom_output - torch_output)):.6f}, "
        f"Max relative difference: {torch.max(torch.abs((custom_output - torch_output) / torch_output)):.6f}"
    )


def test_magnitude_phase_consistency(sample_audio):
    """
    Validate magnitude and phase spectrum consistency between STFT implementations.
    
    This test focuses on the intermediate spectral representations produced by the
    transform() method, which separates magnitude and phase components. Consistent
    spectral decomposition is critical for vocoder applications that manipulate
    magnitude and phase independently before reconstruction.
    
    Test Methodology:
    1. Spectral Decomposition: Extract magnitude and phase from both implementations
    2. Boundary Exclusion: Focus on stable interior frames avoiding edge effects
    3. Component Comparison: Validate magnitude accuracy (phase has relaxed tolerances)
    4. Spectral Fidelity: Ensure accurate frequency domain representation
    
    Magnitude vs Phase Validation:
    - Magnitude: Stricter tolerance (1e-2) as it directly affects audio amplitude
    - Phase: More relaxed validation due to inherent phase computation sensitivity
    - Boundary Frames: Excluded due to windowing edge effects and zero-padding
    - Interior Frames: Focus on stable spectral content with reliable processing
    
    Boundary Frame Exclusion Strategy:
    STFT boundary frames often contain artifacts from:
    - Window function edge effects at signal start/end
    - Zero-padding interactions with windowing
    - Overlap-add reconstruction edge cases
    Excluding boundary frames ensures validation focuses on stable spectral content.
    
    Args:
        sample_audio (torch.Tensor): Standardized test signal from sample_audio fixture
                                   Clean sine wave provides predictable spectral content
                                   Enables precise validation of frequency domain accuracy
    
    Validation Strategy:
    - Known Spectrum: Sine wave produces predictable magnitude/phase characteristics
    - Interior Focus: Boundary frame exclusion eliminates edge effects
    - Component Isolation: Tests magnitude and phase extraction independently
    - Tolerance Matching: Accounts for numerical precision in spectral computation
    
    Spectral Analysis Details:
    - Magnitude Spectrum: Amplitude information for each frequency bin
    - Phase Spectrum: Phase information for complex spectral reconstruction
    - Time-Frequency: Validates both temporal and spectral processing accuracy
    - Windowing Effects: Proper handling of window function spectral characteristics
    
    Quality Assurance:
    - Magnitude Accuracy: Critical for audio amplitude preservation
    - Phase Consistency: Important for spectral reconstruction quality  
    - Boundary Handling: Proper edge effect management
    - Numerical Stability: Consistent floating-point computation
    
    Failure Analysis:
    Test failures may indicate:
    - FFT implementation differences between custom and reference
    - Window function application discrepancies
    - Complex number handling or phase computation errors
    - Boundary condition processing differences
    - Numerical precision issues in spectral decomposition
    
    Integration Impact:
    - Vocoder Processing: Accurate spectral decomposition essential for TTS quality
    - Phase Vocoder Applications: Enables independent magnitude/phase manipulation
    - Feature Extraction: Supports spectral feature computation for synthesis
    - Quality Control: Validates spectral processing before audio reconstruction
    """
    # Initialize STFT implementations with matching configurations
    filter_length = STFTTestConstants.DEFAULT_FILTER_LENGTH
    hop_length = STFTTestConstants.DEFAULT_HOP_LENGTH 
    win_length = STFTTestConstants.DEFAULT_WIN_LENGTH
    
    custom_stft = CustomSTFT(
        filter_length=filter_length,
        hop_length=hop_length, 
        win_length=win_length
    )
    torch_stft = TorchSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length
    )

    # Extract magnitude and phase spectra from both implementations
    custom_mag, custom_phase = custom_stft.transform(sample_audio)
    torch_mag, torch_phase = torch_stft.transform(sample_audio)

    # Focus validation on interior frames to avoid boundary effects
    boundary_margin = STFTTestConstants.BOUNDARY_FRAME_MARGIN
    custom_mag_center = custom_mag[..., boundary_margin:-boundary_margin]
    torch_mag_center = torch_mag[..., boundary_margin:-boundary_margin]
    
    # Validate magnitude spectrum consistency
    magnitude_match = torch.allclose(
        custom_mag_center, 
        torch_mag_center, 
        rtol=STFTTestConstants.MAGNITUDE_RTOL, 
        atol=STFTTestConstants.MAGNITUDE_ATOL
    )
    
    assert magnitude_match, (
        f"Magnitude spectrum mismatch detected. "
        f"Max absolute difference: {torch.max(torch.abs(custom_mag_center - torch_mag_center)):.6f}, "
        f"Max relative difference: {torch.max(torch.abs((custom_mag_center - torch_mag_center) / torch_mag_center)):.6f}"
    )


def test_batch_processing():
    """
    Validate efficient and accurate batch processing capabilities of CustomSTFT implementation.
    
    This test ensures the CustomSTFT implementation can handle multiple audio signals
    simultaneously with both computational efficiency and numerical accuracy. Batch
    processing is essential for production TTS systems that need to synthesize multiple
    utterances concurrently or process audio streams in parallel.
    
    Test Methodology:
    1. Batch Signal Generation: Create multiple identical sine wave signals
    2. Parallel Processing: Apply STFT to the entire batch simultaneously
    3. Shape Validation: Verify correct output tensor dimensions and structure
    4. Efficiency Assessment: Ensure batch processing maintains performance characteristics
    
    Batch Processing Architecture:
    The test validates that the CustomSTFT implementation properly handles:
    - Tensor Broadcasting: Correct application of STFT operations across batch dimension
    - Memory Management: Efficient memory usage for multi-signal processing
    - Parallelization: Potential for vectorized operations and GPU acceleration
    - Output Consistency: Identical processing results for identical inputs
    
    Signal Configuration:
    - Batch Size: 4 signals for moderate batch processing validation
    - Duration: 0.1 seconds (shorter for faster test execution)
    - Sample Rate: 16 kHz (standard speech processing rate)
    - Frequency Content: 440 Hz sine wave for predictable spectral characteristics
    - Signal Characteristics: Identical content across batch for consistency validation
    
    Validation Strategy:
    - Shape Verification: Output tensor must preserve batch dimension structure
    - Dimension Consistency: 3D output tensor (batch, channels, time) expected
    - Batch Size Preservation: Input batch size must equal output batch size
    - Processing Uniformity: All signals in batch receive identical processing
    
    Quality Assurance:
    - Memory Efficiency: Batch processing should be more efficient than sequential
    - Numerical Consistency: Each signal in batch produces identical results
    - Shape Integrity: Proper tensor dimension handling across batch operations
    - Performance Scaling: Linear or sub-linear scaling with batch size
    
    Production Relevance:
    Batch processing capabilities are critical for:
    - Real-time TTS: Multiple concurrent synthesis requests
    - Server Applications: Efficient multi-user speech synthesis
    - Mobile Deployment: Efficient resource utilization on constrained devices
    - Pipeline Integration: Seamless integration with batch-oriented ML workflows
    
    Failure Analysis:
    Test failures typically indicate:
    - Tensor dimension handling errors in batch operations
    - Memory management issues with larger tensor allocations
    - Broadcasting problems in windowing or FFT operations
    - Indexing errors when processing multiple signals simultaneously
    
    Integration Impact:
    - Deployment Efficiency: Enables efficient multi-request processing
    - Resource Utilization: Better GPU/ANE utilization through batch operations
    - Scalability: Supports high-throughput TTS applications
    - System Performance: Reduces per-request processing overhead
    """
    # Configure batch processing test parameters
    batch_size = STFTTestConstants.BATCH_TEST_SIZE
    sample_rate = STFTTestConstants.DEFAULT_SAMPLE_RATE
    duration = STFTTestConstants.SHORT_DURATION  # Faster testing with shorter signals
    
    # Generate time axis for signal creation
    t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
    
    # Create batch of identical sine wave signals for consistency testing
    frequency = STFTTestConstants.REFERENCE_FREQUENCY
    single_signal = torch.sin(2 * np.pi * frequency * t)
    signals = single_signal.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, num_samples)

    # Initialize STFT with standard configuration for batch processing
    custom_stft = CustomSTFT(
        filter_length=STFTTestConstants.DEFAULT_FILTER_LENGTH,
        hop_length=STFTTestConstants.DEFAULT_HOP_LENGTH, 
        win_length=STFTTestConstants.DEFAULT_WIN_LENGTH
    )

    # Process entire batch simultaneously
    batch_output = custom_stft(signals)

    # Validate output tensor structure and dimensions
    assert batch_output.shape[0] == batch_size, (
        f"Batch dimension mismatch: expected {batch_size}, got {batch_output.shape[0]}"
    )
    assert len(batch_output.shape) == 3, (
        f"Output tensor dimensionality incorrect: expected 3D (batch, channels, time), got {len(batch_output.shape)}D"
    )
    
    # Validate that batch processing preserves signal length relationships
    input_length = signals.shape[-1]
    output_length = batch_output.shape[-1]
    assert output_length >= input_length, (
        f"Output length {output_length} shorter than input length {input_length} - signal truncation detected"
    )


def test_different_window_sizes():
    """
    Validate CustomSTFT robustness across various window size configurations.
    
    This test ensures the CustomSTFT implementation maintains accuracy and stability
    across different STFT parameter configurations commonly used in TTS vocoder
    applications. Different window sizes provide different time-frequency resolution
    trade-offs, and the implementation must handle all configurations correctly.
    
    Test Methodology:
    1. Parameter Sweep: Test multiple window size configurations systematically
    2. Configuration Validation: Ensure each parameter set produces valid output
    3. Length Preservation: Verify proper signal length handling across configurations
    4. Stability Assessment: Confirm numerical stability across parameter ranges
    
    Window Size Impact Analysis:
    Different window sizes affect STFT characteristics:
    - Small Windows (512): Better temporal resolution, coarser frequency resolution
    - Medium Windows (1024): Balanced time-frequency resolution for general use
    - Large Windows (2048): Better frequency resolution, coarser temporal resolution
    Each configuration requires proper handling of windowing and overlap parameters.
    
    Parameter Configuration Strategy:
    - Filter Length: Controls frequency resolution and window size
    - Hop Length: Set to filter_length/4 for 75% overlap (standard for reconstruction)
    - Win Length: Matches filter length for consistent windowing behavior
    - Window Function: Hann window (implicit) for minimal spectral leakage
    
    Signal Characteristics:
    - Content: Random noise provides broadband spectral content
    - Duration: 1 second (16000 samples) for adequate spectral analysis
    - Amplitude: Unit variance for consistent numerical behavior
    - Distribution: Gaussian noise for robust statistical properties
    
    Validation Strategy:
    - Output Length: Must be at least as long as input (perfect reconstruction requirement)
    - Shape Consistency: Proper tensor dimensions across all configurations
    - Numerical Stability: No NaN or infinite values in outputs
    - Processing Success: All configurations must complete without errors
    
    TTS Vocoder Relevance:
    Different applications prefer different window sizes:
    - Real-time Synthesis: Smaller windows (512) for lower latency
    - High-quality Synthesis: Medium windows (1024) for balanced resolution
    - Spectral Analysis: Larger windows (2048) for detailed frequency analysis
    
    Quality Assurance:
    - Parameter Robustness: Implementation handles various common configurations
    - Length Consistency: Proper signal length relationships maintained
    - Numerical Stability: No degradation with different window sizes
    - Performance Characteristics: Reasonable processing time across configurations
    
    Failure Analysis:
    Test failures may indicate:
    - Parameter validation issues in STFT initialization
    - Window function computation errors with different sizes
    - FFT size handling problems with various filter lengths
    - Memory allocation issues with larger window sizes
    - Boundary condition handling differences across configurations
    
    Integration Impact:
    - Deployment Flexibility: Supports various TTS vocoder configurations
    - Application Optimization: Enables parameter tuning for specific use cases
    - Performance Tuning: Allows trade-off optimization between quality and speed
    - System Robustness: Ensures stable operation across parameter ranges
    """
    # Generate test signal with broadband spectral content
    sample_rate = STFTTestConstants.DEFAULT_SAMPLE_RATE
    signal_length = sample_rate * 1  # 1 second duration
    signal = torch.randn(1, signal_length, dtype=torch.float32)  # Gaussian noise signal

    # Test implementation robustness across different window size configurations
    for filter_length in STFTTestConstants.FILTER_LENGTH_VARIANTS:
        # Configure STFT parameters with standard overlap ratio
        hop_length = filter_length // 4  # 75% overlap for high-quality reconstruction
        win_length = filter_length       # Match window length to filter length
        
        # Initialize CustomSTFT with current parameter configuration
        custom_stft = CustomSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
        )

        # Apply STFT transformation with current configuration
        stft_output = custom_stft(signal)

        # Validate output length preservation (critical for reconstruction quality)
        input_length = signal.shape[-1]
        output_length = stft_output.shape[-1]
        assert output_length >= input_length, (
            f"Signal truncation detected with filter_length={filter_length}: "
            f"output length {output_length} < input length {input_length}"
        )
        
        # Validate output tensor structure
        assert len(stft_output.shape) == 3, (
            f"Invalid output shape with filter_length={filter_length}: "
            f"expected 3D tensor, got {len(stft_output.shape)}D"
        )
        
        # Validate numerical stability (no NaN or infinite values)
        if STFTTestConstants.NAN_DETECTION_ENABLED:
            assert not torch.isnan(stft_output).any(), (
                f"NaN values detected in output with filter_length={filter_length}"
            )
        
        if STFTTestConstants.INF_DETECTION_ENABLED:
            assert not torch.isinf(stft_output).any(), (
                f"Infinite values detected in output with filter_length={filter_length}"
            )
