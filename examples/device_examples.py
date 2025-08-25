"""
Performance benchmarking and device selection validation for Kokoro TTS pipeline.

This script demonstrates device-specific performance characteristics and validates
that the Kokoro TTS pipeline can run on different compute devices (CPU, CUDA, auto).
It also tests model reuse patterns for memory efficiency.

Used by:
- Developers testing device compatibility before deployment
- Performance optimization workflows in CI/CD
- Debugging device-specific issues

Calls into:
- kokoro.KPipeline: Main TTS inference pipeline
- loguru.logger: Structured logging for performance metrics

This script is standalone and not called by other modules.
"""
import time
from kokoro import KPipeline
from loguru import logger


# --- Performance Testing Constants ---
# These constants define the behavior and expectations of the performance tests.

class TestConstants:
    """Named constants for performance testing to avoid magic numbers."""
    
    # Standard test phrase chosen for consistent phoneme distribution
    # Contains common English sounds and patterns for representative timing
    TEST_PHRASE = "The quick brown fox jumps over the lazy dog."
    
    # Voice selection for consistent testing
    # af_bella chosen for balanced quality/speed characteristics
    TEST_VOICE = "af_bella"
    
    # Language code for American English
    # 'a' maps to American English in Kokoro's language system
    LANGUAGE_CODE = "a"
    
    # Model reuse test phrase - shorter for faster execution
    REUSE_TEST_PHRASE = "Testing model reuse."
    
    # Display formatting constants
    DEVICE_COLUMN_WIDTH = 6  # Fixed width for device name alignment
    TIMING_COLUMN_WIDTH = 5  # Width for timing display (e.g., "123.4")
    SAMPLE_COLUMN_WIDTH = 6  # Width for sample count display
    SEPARATOR_LENGTH = 40    # Length of visual separators in output

def generate_audio(pipeline: KPipeline, text: str) -> int:
    """
    Generates audio from text using the provided pipeline and returns sample count.
    
    This function is called by:
    - time_synthesis() for device-specific performance testing
    - compare_shared_model() for model reuse validation
    
    The pipeline() call yields tuples of (phonemes, durations, audio).
    We only need the audio tensor to measure output size and validate generation.
    
    Args:
        pipeline: Initialized KPipeline instance configured for specific device
        text: Input text string to synthesize into audio
        
    Returns:
        int: Number of audio samples generated (validates non-zero output)
        
    Raises:
        AssertionError: If no audio samples are generated (pipeline failure)
    """
    # The pipeline yields (phonemes, durations, audio) tuples
    # We only need the audio component for validation
    for _, _, audio in pipeline(text, voice=TestConstants.TEST_VOICE):
        samples = audio.shape[0] if audio is not None else 0
        
        # Validate that audio generation succeeded
        # Zero samples indicates pipeline failure or silent output
        assert samples > 0, "No audio generated - pipeline may have failed"
        return samples

def time_synthesis(device: str = None) -> None:
    """
    Benchmarks TTS synthesis performance on a specific device.
    
    This function measures end-to-end latency from pipeline initialization
    through audio generation. It handles device-specific errors gracefully
    and logs structured performance metrics.
    
    Called by:
    - main() for each device type (auto, cuda, cpu)
    
    Calls into:
    - KPipeline() for pipeline initialization with device specification
    - generate_audio() for actual synthesis and validation
    - time.perf_counter() for high-precision timing measurements
    
    Args:
        device: Target compute device ('cpu', 'cuda', or None for auto-selection)
                None triggers Kokoro's automatic device detection
    
    Performance expectations:
    - CPU: 1000-5000ms for test phrase (varies by hardware)
    - CUDA: 100-500ms for test phrase (if GPU available)
    - Auto: Should select fastest available device
    """
    try:
        # Start high-precision timer before pipeline creation
        # Pipeline initialization includes model loading which affects timing
        start_time = time.perf_counter()
        
        # Initialize pipeline with specified device
        # device=None enables Kokoro's automatic device selection
        pipeline = KPipeline(lang_code=TestConstants.LANGUAGE_CODE, device=device)
        
        # Generate audio and validate output
        sample_count = generate_audio(pipeline, TestConstants.TEST_PHRASE)
        
        # Calculate total elapsed time in milliseconds
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Format device name for consistent logging alignment
        device_name = device or 'auto'
        
        # Log successful synthesis with structured metrics
        logger.info(
            f"✓ {device_name:<{TestConstants.DEVICE_COLUMN_WIDTH}} | "
            f"{elapsed_ms:>{TestConstants.TIMING_COLUMN_WIDTH}.1f}ms total | "
            f"{sample_count:>{TestConstants.SAMPLE_COLUMN_WIDTH},d} samples"
        )
        
    except RuntimeError as runtime_error:
        # Handle device-specific runtime errors (e.g., CUDA not available)
        error_message = str(runtime_error)
        
        # Detect CUDA-specific errors for better user messaging
        if 'CUDA' in error_message:
            device_display = 'cuda'
            status_message = 'not available'
        else:
            device_display = device or 'auto'
            status_message = error_message
        
        # Log error with consistent formatting
        logger.error(
            f"✗ {device_display:<{TestConstants.DEVICE_COLUMN_WIDTH}} | {status_message}"
        )

def compare_shared_model() -> None:
    """
    Tests memory efficiency of shared model instances across multiple pipelines.
    
    This function validates that the same underlying model can be reused
    across multiple pipeline instances without reloading weights from disk.
    This pattern is critical for memory-constrained deployments.
    
    Called by:
    - main() as part of the performance validation suite
    
    Calls into:
    - KPipeline() for creating primary and secondary pipeline instances
    - generate_audio() for validating synthesis on both pipelines
    
    Memory efficiency pattern:
    - First pipeline: Loads model from checkpoint (expensive)
    - Second pipeline: Reuses existing model instance (cheap)
    
    Expected behavior:
    - Both pipelines should produce valid audio output
    - Total time should be much less than 2x individual pipeline time
    - Memory usage should not double
    """
    try:
        # Start timing before any pipeline creation
        start_time = time.perf_counter()
        
        # Create primary pipeline - this loads the model from checkpoint
        # This is the expensive operation we want to avoid repeating
        primary_pipeline = KPipeline(lang_code=TestConstants.LANGUAGE_CODE)
        
        # Create secondary pipeline using the same model instance
        # model=en_us.model shares the loaded weights instead of reloading
        secondary_pipeline = KPipeline(
            lang_code=TestConstants.LANGUAGE_CODE, 
            model=primary_pipeline.model
        )
        
        # Test synthesis on both pipelines to validate functionality
        # Both should work identically since they share the same model
        pipeline_instances = [primary_pipeline, secondary_pipeline]
        for pipeline_instance in pipeline_instances:
            generate_audio(pipeline_instance, TestConstants.REUSE_TEST_PHRASE)
        
        # Calculate total elapsed time for both operations
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        # Log successful model reuse with timing metrics
        logger.info(
            f"✓ {'reuse':<{TestConstants.DEVICE_COLUMN_WIDTH}} | "
            f"{elapsed_ms:>{TestConstants.TIMING_COLUMN_WIDTH}.1f}ms for both models"
        )
        
    except Exception as general_error:
        # Handle any error in model reuse pattern
        # Could be memory issues, model incompatibility, or synthesis failures
        logger.error(f"✗ {'reuse':<{TestConstants.DEVICE_COLUMN_WIDTH}} | {str(general_error)}")

if __name__ == '__main__':
    """
    Main execution block for device performance validation.
    
    Execution flow:
    1. Test automatic device selection (lets Kokoro pick best available)
    2. Test explicit CUDA device (validates GPU acceleration if available)
    3. Test explicit CPU device (validates CPU-only fallback)
    4. Test model reuse pattern (validates memory efficiency)
    
    This script is intended to be run manually by developers or in CI/CD
    to validate device compatibility and measure performance characteristics.
    """
    # Print structured header for performance test results
    logger.info("Device Selection & Performance")
    logger.info("-" * TestConstants.SEPARATOR_LENGTH)
    
    # Test device-specific performance in order of preference
    # Auto-selection should pick the fastest available device
    time_synthesis()  # device=None enables auto-selection
    
    # Explicitly test CUDA if available (will gracefully fail if not)
    time_synthesis('cuda')
    
    # Test CPU fallback (should always work)
    time_synthesis('cpu')
    
    # Visual separator between device tests and model reuse test
    logger.info("-" * TestConstants.SEPARATOR_LENGTH)
    
    # Test memory-efficient model sharing pattern
    compare_shared_model()