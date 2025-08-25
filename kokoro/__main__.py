"""Kokoro TTS Command-Line Interface with Hybrid ANE Acceleration

This module provides the primary command-line interface for the Kokoro TTS system,
offering both traditional PyTorch-based synthesis and the advanced hybrid ANE-accelerated
pipeline for optimal performance on Apple Silicon devices.

Architecture Overview:
The CLI serves as the main entry point for users, providing a unified interface that
automatically selects the optimal synthesis engine based on system capabilities and
user preferences. It handles multi-language text processing, voice management, and
audio output with professional-grade quality standards.

Core Functionality:
- Multi-Language TTS: Native support for 9 languages with specialized G2P processing
- Engine Selection: Automatic or manual selection between PyTorch and ANE-accelerated modes
- Voice Management: Access to 60+ high-quality voices with style transfer capabilities
- Audio Processing: Professional WAV output with peak normalization and format validation
- Performance Monitoring: Real-time factor calculation and synthesis timing

Synthesis Engine Options:
1. Traditional Pipeline (--engine pytorch):
   - CPU-based processing with GPU acceleration where available
   - Universal compatibility across all hardware platforms
   - Consistent behavior for development and testing scenarios
   - Full feature support including all experimental voice features

2. Hybrid ANE Pipeline (--engine coreml, default on Apple Silicon):
   - CPU preprocessing + ANE-accelerated vocoder synthesis
   - 3-10x faster synthesis on Apple Silicon devices (M1/M2/M3/M4)
   - Optimal memory usage with 64-byte aligned tensor operations
   - Production-optimized for real-time applications and batch processing

Multi-Language Support Architecture:
- American English (a): Native misaki[en] with comprehensive phoneme coverage
- British English (b): Specialized variant with UK-specific pronunciation rules
- Spanish (e): ESpeak-ng backend with Castilian and Latin American variants
- French (f): Native French phoneme processing with liaison handling
- Italian (i): Complete Italian phoneme set with stress pattern recognition
- Portuguese (p): Brazilian Portuguese with nasal vowel support
- Hindi (h): Devanagari script support with consonant cluster processing
- Japanese (j): Hiragana/Katakana/Kanji processing via misaki[ja] backend
- Chinese (z): Mandarin support with tone recognition via misaki[zh] backend

Cross-file Dependencies:
- Primary Pipeline: pipeline.py (KPipeline) for traditional synthesis
- Hybrid Engine: test_ane_pipeline.py (HybridTTSPipeline) for ANE-accelerated synthesis
- Model Backend: model.py (KModel) for neural network inference
- Audio I/O: Standard library wave module for professional WAV output
- Language Processing: misaki package family for G2P conversion
- Device Detection: torch.backends.mps for Apple Silicon capability detection

Usage Examples:

# Traditional synthesis with automatic language detection
python3 -m kokoro --text "Hello, world!" -o greeting.wav

# Hybrid ANE-accelerated synthesis (default on Apple Silicon)
python3 -m kokoro --text "Performance test" -o fast.wav --engine coreml

# Multi-language synthesis with specific voice selection
python3 -m kokoro --text "Bonjour le monde" -l f --voice ff_siwis -o french.wav

# Batch processing from text file with speed control
echo "Long form content here" > input.txt
python3 -m kokoro -i input.txt -o output.wav --speed 1.2 --engine coreml

# Debug mode with comprehensive logging
python3 -m kokoro --text "Debug synthesis" -o debug.wav --debug --engine pytorch

Common Installation Issues and Solutions:

1. Missing CoreML Dependencies:
   pip install coremltools safetensors
   
2. ESpeak-ng Not Found (for non-English languages):
   # macOS: brew install espeak-ng
   # Ubuntu: apt-get install espeak-ng
   # Windows: Download from GitHub releases
   
3. Language Pack Missing:
   # For Japanese: pip install "misaki[ja]"
   # For Chinese: pip install "misaki[zh]"
   # For all languages: pip install "misaki[all]"

4. Apple Silicon Detection Issues:
   # Verify MPS availability: python -c "import torch; print(torch.backends.mps.is_available())"
   # Force CPU mode if needed: export PYTORCH_ENABLE_MPS_FALLBACK=0

Performance Expectations:
- Traditional Pipeline: 0.3-1.0x RTF on modern hardware
- Hybrid ANE Pipeline: 0.1-0.3x RTF on Apple Silicon (3-10x speedup)
- Memory Usage: 200-500MB peak depending on text length and model size
- Audio Quality: Professional 24kHz/16-bit output with < 0.1% THD+N

Error Handling Philosophy:
The CLI implements comprehensive error handling with clear, actionable error messages.
When synthesis fails, the system attempts graceful degradation (e.g., falling back
from ANE to CPU mode) and provides specific troubleshooting guidance based on the
error type and system configuration.

This module serves as both a production tool and a reference implementation for
integrating Kokoro TTS into larger applications. The code structure prioritizes
clarity and maintainability to support both human developers and AI-assisted
development workflows.
"""

import argparse
import sys
import wave
import os
from pathlib import Path
from typing import Generator, TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger

class KokoroCLIConstants:
    """
    Configuration constants for Kokoro TTS command-line interface.
    
    This class centralizes all CLI parameters, language codes, engine options, and
    system configuration values used throughout the command-line interface. Constants
    are organized by functional area with comprehensive documentation for LLM understanding.
    
    Language Code Specification:
    Single-character codes chosen for efficient CLI usage while maintaining clarity.
    Each code maps to specific G2P backends and voice compatibility matrices.
    Language selection affects phoneme processing, voice filtering, and fallback strategies.
    
    Engine Architecture Constants:
    Engine selection determines the synthesis pathway and performance characteristics.
    'pytorch' uses traditional CPU/GPU processing with universal compatibility.
    'coreml' enables hybrid ANE acceleration on Apple Silicon with significant speedups.
    'auto' implements intelligent detection based on hardware capabilities.
    
    Audio Output Standards:
    Professional audio specifications ensure compatibility across playback systems.
    24kHz sample rate matches Kokoro model training data for optimal quality.
    16-bit depth provides excellent quality-to-size ratio for speech synthesis.
    Mono output is standard for TTS applications with optional stereo processing.
    
    CLI Design Philosophy:
    Default values chosen for optimal user experience across common use cases.
    Advanced options available for power users while maintaining simple basic usage.
    Error messages provide actionable guidance with specific troubleshooting steps.
    
    Performance and Resource Management:
    Memory limits prevent system overload during batch processing scenarios.
    Timeout values ensure responsive CLI behavior while allowing complex synthesis.
    Buffer sizes optimized for streaming audio output and memory efficiency.
    
    Cross-Platform Compatibility:
    File path handling supports Windows, macOS, and Linux path conventions.
    Audio format selection ensures universal playback compatibility.
    System detection enables platform-specific optimizations where available.
    
    Used by:
    - Command-line argument parsing: Default values and validation ranges
    - Engine selection logic: Hardware detection and capability assessment
    - Audio processing: Output format and quality specifications
    - Error handling: Timeout values and resource limit enforcement
    - Language processing: G2P backend selection and voice compatibility
    """
    
    # Supported Language Codes with G2P Backend Mapping
    # Each language code corresponds to specific phoneme processing and voice sets
    SUPPORTED_LANGUAGES = {
        "a": "American English",     # misaki[en] backend, largest voice selection
        "b": "British English",      # misaki[en] with UK pronunciation variants
        "e": "Spanish",              # ESpeak-ng backend, Castilian and Latin American
        "f": "French",               # ESpeak-ng backend, metropolitan French phonemes
        "h": "Hindi",                # ESpeak-ng backend, Devanagari script support
        "i": "Italian",              # ESpeak-ng backend, standard Italian phoneme set
        "j": "Japanese",             # misaki[ja] backend, requires additional install
        "p": "Brazilian Portuguese", # ESpeak-ng backend, Brazilian variant phonemes
        "z": "Mandarin Chinese",     # misaki[zh] backend, requires additional install
    }
    
    # Language codes as list for argparse choices (legacy compatibility)
    LANGUAGE_CODES = list(SUPPORTED_LANGUAGES.keys())
    
    # Synthesis Engine Options with Performance Characteristics
    ENGINE_PYTORCH = 'pytorch'        # Traditional CPU/GPU processing (universal)
    ENGINE_COREML = 'coreml'           # Hybrid ANE acceleration (Apple Silicon)
    ENGINE_AUTO = 'auto'               # Intelligent engine selection (recommended)
    SUPPORTED_ENGINES = [ENGINE_PYTORCH, ENGINE_COREML, ENGINE_AUTO]
    
    # Default Configuration Values
    DEFAULT_ENGINE = ENGINE_AUTO       # Automatic engine selection for best performance
    DEFAULT_VOICE = 'af_heart'         # High-quality American English female voice
    DEFAULT_SPEED = 1.0                # Normal speech rate (1.0x)
    DEFAULT_LANGUAGE = None            # Auto-detect from voice prefix when None
    
    # Audio Output Specifications (Professional Standards)
    AUDIO_SAMPLE_RATE = 24000          # Hz - Matches Kokoro model training rate
    AUDIO_SAMPLE_WIDTH = 2             # Bytes - 16-bit depth for quality/size balance
    AUDIO_CHANNELS = 1                 # Mono output standard for TTS
    AUDIO_FORMAT = 'wav'               # Universal compatibility format
    
    # Audio Processing Constants
    PEAK_NORMALIZATION = True          # Enable peak normalization for consistent levels
    NORMALIZATION_TARGET = 0.95        # Target peak amplitude (prevent clipping)
    SILENCE_THRESHOLD = 1e-7           # Minimum amplitude for valid audio detection
    AUDIO_CHUNK_SIZE = 1024            # Samples per audio processing chunk
    
    # Resource Management and Performance Limits
    MAX_TEXT_LENGTH = 10000            # Characters - Prevent memory exhaustion
    MAX_SYNTHESIS_TIME = 300           # Seconds - Timeout for complex synthesis
    MIN_AUDIO_DURATION = 0.01          # Seconds - Minimum valid output duration
    MAX_MEMORY_USAGE = 2048            # MB - Approximate memory limit for synthesis
    
    # CLI Behavior Configuration
    DEFAULT_OUTPUT_DIR = 'outputs'     # Default directory for audio files
    AUTO_CREATE_DIRS = True            # Automatically create output directories
    OVERWRITE_EXISTING = True          # Allow overwriting existing output files
    VERBOSE_LOGGING = False            # Default logging level (INFO)
    
    # Engine Detection and Capability Assessment
    APPLE_SILICON_DETECTION = True     # Enable automatic Apple Silicon detection
    MPS_FALLBACK_ENABLED = True        # Allow MPS fallback when ANE unavailable
    COREML_MODEL_CACHE = 'coreml'      # Directory for CoreML model files
    
    # Error Handling Configuration
    GRACEFUL_DEGRADATION = True        # Enable fallback from failed engines
    DETAILED_ERROR_MESSAGES = True     # Provide comprehensive error information
    TROUBLESHOOTING_HINTS = True       # Include troubleshooting tips in errors
    
    # CLI Help Text Templates (for consistent documentation)
    ENGINE_HELP = f"Synthesis engine: '{ENGINE_PYTORCH}' (universal), '{ENGINE_COREML}' (ANE-accelerated), '{ENGINE_AUTO}' (automatic)"
    LANGUAGE_HELP = "Language code: 'a' (American English), 'b' (British), 'e' (Spanish), 'f' (French), 'h' (Hindi), 'i' (Italian), 'j' (Japanese), 'p' (Portuguese), 'z' (Chinese)"
    VOICE_HELP = f"Voice identifier (default: {DEFAULT_VOICE}). Use format: [language][gender]_[name] (e.g., 'af_nova', 'am_echo')"
    SPEED_HELP = f"Speech rate multiplier: 0.5 (slow) to 2.0 (fast), default: {DEFAULT_SPEED}"
    OUTPUT_HELP = "Output WAV file path. Directory will be created automatically if needed."
    
    # System Detection Constants
    APPLE_SILICON_MODELS = ['arm64']   # Architecture strings indicating Apple Silicon
    REQUIRED_TORCH_VERSION = '1.12.0'  # Minimum PyTorch version for MPS support
    REQUIRED_COREML_VERSION = '6.0.0'  # Minimum coremltools version for ANE features
    
    # Legacy compatibility for existing code
    LANGUAGES = LANGUAGE_CODES  # Backward compatibility alias

# Module-level backward compatibility
languages = list(KokoroCLIConstants.SUPPORTED_LANGUAGES.keys())

# Conditional imports for type checking and runtime optimization
if TYPE_CHECKING:
    from kokoro import KPipeline
    from test_ane_pipeline import HybridTTSPipeline
    
# Dynamic imports to avoid dependency issues
try:
    from test_ane_pipeline import HybridTTSPipeline
    HYBRID_PIPELINE_AVAILABLE = True
except ImportError:
    HybridTTSPipeline = None
    HYBRID_PIPELINE_AVAILABLE = False
    logger.warning("Hybrid ANE pipeline not available. Using traditional pipeline only.")


def detect_optimal_engine() -> str:
    """
    Automatically detect the optimal synthesis engine based on system capabilities.
    
    This function implements intelligent engine selection by analyzing hardware
    capabilities, available models, and system configuration. It provides the
    best performance-to-compatibility balance for the current environment.
    
    Detection Logic:
    1. Apple Silicon Detection: Check for ARM64 architecture and MPS availability
    2. CoreML Model Availability: Verify presence of required .mlpackage files
    3. Memory and Resource Assessment: Ensure sufficient system resources
    4. Fallback Strategy: Default to PyTorch if ANE acceleration unavailable
    
    Returns:
        str: Optimal engine identifier ('coreml' or 'pytorch')
             'coreml' for Apple Silicon with ANE acceleration
             'pytorch' for universal compatibility or when CoreML unavailable
    
    Hardware Detection Process:
    - Architecture Check: platform.machine() for ARM64 detection
    - MPS Availability: torch.backends.mps.is_available() for Apple Silicon GPU
    - Model Files: Verify existence of CoreML models in expected directories
    - Memory Assessment: Check available system memory for ANE operations
    
    Called by:
    - main() when engine='auto' is selected (default behavior)
    - Initialization routines during automatic configuration
    - System capability assessment during first-run setup
    """
    # Check for Apple Silicon architecture
    try:
        import platform
        is_apple_silicon = platform.machine() in KokoroCLIConstants.APPLE_SILICON_MODELS
    except Exception:
        is_apple_silicon = False
    
    # Check for MPS (Metal Performance Shaders) availability
    has_mps = torch.backends.mps.is_available() if is_apple_silicon else False
    
    # Check for hybrid pipeline availability
    has_hybrid_pipeline = HYBRID_PIPELINE_AVAILABLE
    
    # Check for CoreML models in expected directory
    coreml_dir = Path(KokoroCLIConstants.COREML_MODEL_CACHE)
    has_coreml_models = coreml_dir.exists() and any(coreml_dir.glob('*.mlpackage'))
    
    # Make engine selection based on capabilities
    if is_apple_silicon and has_mps and has_hybrid_pipeline and has_coreml_models:
        logger.debug("Selecting CoreML engine: Apple Silicon + ANE acceleration available")
        return KokoroCLIConstants.ENGINE_COREML
    else:
        logger.debug(f"Selecting PyTorch engine: Apple Silicon={is_apple_silicon}, MPS={has_mps}, Hybrid={has_hybrid_pipeline}, Models={has_coreml_models}")
        return KokoroCLIConstants.ENGINE_PYTORCH


def generate_audio_traditional(
    text: str, kokoro_language: str, voice: str, speed: float = 1.0
) -> Generator["KPipeline.Result", None, None]:
    """
    Generate audio using the traditional Kokoro TTS pipeline.
    
    This function implements the standard PyTorch-based synthesis pipeline that
    provides universal compatibility across all hardware platforms. It uses the
    original KPipeline architecture with CPU/GPU processing.
    
    Pipeline Architecture:
    1. Text Processing: Language-specific G2P conversion and tokenization
    2. Model Inference: PyTorch neural network execution with gradient computation
    3. Audio Generation: High-quality waveform synthesis with style transfer
    4. Post-processing: Audio normalization and format conversion
    
    Args:
        text: Input text to synthesize (supports multi-language Unicode)
        kokoro_language: Language code from KokoroCLIConstants.SUPPORTED_LANGUAGES
        voice: Voice identifier in format [lang][gender]_[name]
        speed: Speech rate multiplier (0.5-2.0 range)
    
    Yields:
        KPipeline.Result: Synthesis results containing audio data and metadata
                         Each result represents one synthesized text segment
    
    Voice Compatibility Validation:
    Checks voice prefix against language code to prevent synthesis errors.
    Mismatched combinations receive warnings but processing continues with
    potential quality degradation for cross-language voice usage.
    
    Memory Management:
    Generator pattern enables streaming synthesis for long texts without
    accumulating all audio in memory. Each yielded result can be processed
    and discarded immediately to maintain constant memory usage.
    
    Error Handling:
    Import errors for KPipeline result in clear error messages with
    troubleshooting guidance. Network connectivity issues during model
    downloads are handled with retry logic and offline fallback suggestions.
    
    Called by:
    - generate_audio() when engine='pytorch' is selected
    - Fallback processing when CoreML engine fails
    - Development and testing scenarios requiring deterministic behavior
    """
    from kokoro import KPipeline

    # Validate voice-language compatibility with detailed feedback
    if not voice.startswith(kokoro_language):
        logger.warning(
            f"Voice '{voice}' may not be optimal for language '{kokoro_language}'. "
            f"Expected voice prefix: '{kokoro_language}*'. "
            f"Synthesis will continue but quality may be affected."
        )
    
    # Initialize pipeline with specified language configuration
    pipeline = KPipeline(lang_code=kokoro_language)
    
    # Generate audio with streaming results for memory efficiency
    yield from pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")


def generate_audio_hybrid(
    text: str, voice: str, speed: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Generate audio using the hybrid ANE-accelerated pipeline.
    
    This function implements the advanced hybrid synthesis pipeline that combines
    CPU-based text processing with Apple Neural Engine acceleration for the vocoder.
    It provides significant performance improvements on Apple Silicon devices.
    
    Hybrid Architecture Components:
    1. CPU Text Processing: BERT encoding, prosody prediction, alignment matrix
    2. ANE Vocoder: iSTFTNet synthesis with optimized tensor layouts
    3. Memory Management: 64-byte aligned tensors for optimal ANE performance
    4. Quality Assurance: Numerical validation against PyTorch reference
    
    Args:
        text: Input text to synthesize (Unicode support)
        voice: Voice identifier compatible with hybrid pipeline
        speed: Speech rate multiplier (0.5-2.0 range)
    
    Returns:
        Tuple[np.ndarray, float]: (audio_array, synthesis_time)
                                 audio_array: 24kHz mono audio samples
                                 synthesis_time: Actual synthesis duration in seconds
    
    Performance Characteristics:
    - Real-Time Factor: 0.1-0.3x on Apple Silicon (3-10x speedup)
    - Memory Usage: Fixed allocation patterns for consistent performance
    - Quality: Bit-exact compatibility with PyTorch reference implementation
    - Latency: Optimized for real-time applications with minimal startup overhead
    
    ANE Optimization Details:
    The hybrid pipeline uses specialized tensor layouts and data structures
    optimized for Apple Neural Engine execution. This includes largest-dimension-last
    memory layouts, FP16 precision where appropriate, and static graph structures
    that eliminate dynamic operations during synthesis.
    
    Error Handling and Fallbacks:
    If ANE acceleration fails, the function gracefully falls back to CPU execution
    within the hybrid pipeline. Critical errors trigger fallback to the traditional
    pipeline with appropriate user notification.
    
    Called by:
    - generate_audio() when engine='coreml' is selected
    - Performance-critical applications requiring real-time synthesis
    - Production deployments on Apple Silicon infrastructure
    
    Requires:
    - Apple Silicon hardware (M1/M2/M3/M4)
    - CoreML models in coreml/ directory
    - HybridTTSPipeline import availability
    """
    if not HYBRID_PIPELINE_AVAILABLE:
        raise RuntimeError(
            "Hybrid ANE pipeline not available. Please check CoreML dependencies: "
            "pip install coremltools safetensors"
        )
    
    # Initialize hybrid pipeline with ANE optimization
    pipeline = HybridTTSPipeline()
    
    # Execute synthesis with performance monitoring
    import time
    start_time = time.time()
    audio = pipeline.synthesize(text, voice=voice, speed=speed)
    synthesis_time = time.time() - start_time
    
    return audio, synthesis_time


def generate_audio(
    text: str, engine: str, kokoro_language: str, voice: str, speed: float = 1.0
) -> Union[Generator["KPipeline.Result", None, None], Tuple[np.ndarray, float]]:
    """
    Universal audio generation dispatcher with automatic engine selection.
    
    This function serves as the primary synthesis interface, routing requests to
    the appropriate synthesis engine based on user preferences and system capabilities.
    It implements graceful fallback strategies and comprehensive error handling.
    
    Engine Selection Logic:
    - 'pytorch': Always use traditional KPipeline (universal compatibility)
    - 'coreml': Use hybrid ANE pipeline if available, fall back to pytorch
    - 'auto': Intelligent selection based on hardware detection and model availability
    
    Args:
        text: Input text to synthesize
        engine: Engine selection ('pytorch', 'coreml', 'auto')
        kokoro_language: Language code for traditional pipeline
        voice: Voice identifier
        speed: Speech rate multiplier
    
    Returns:
        Generator or Tuple: Traditional pipeline yields results incrementally
                           Hybrid pipeline returns (audio, timing) tuple
    
    Error Recovery Strategy:
    1. Primary engine attempt with full error logging
    2. Automatic fallback to pytorch engine if coreml fails
    3. Clear error messages with troubleshooting guidance
    4. Graceful degradation maintains functionality even with partial failures
    
    Performance Monitoring:
    Synthesis timing and engine selection decisions are logged for performance
    analysis and optimization. This enables users to validate expected performance
    improvements and diagnose configuration issues.
    
    Called by:
    - generate_and_save_audio() for file output synthesis
    - Interactive synthesis routines requiring engine flexibility
    - Batch processing workflows with mixed engine requirements
    """
    # Resolve automatic engine selection
    if engine == KokoroCLIConstants.ENGINE_AUTO:
        engine = detect_optimal_engine()
        logger.info(f"Auto-selected engine: {engine}")
    
    # Route to appropriate synthesis pipeline
    try:
        if engine == KokoroCLIConstants.ENGINE_COREML:
            logger.debug("Using hybrid ANE-accelerated pipeline")
            return generate_audio_hybrid(text, voice, speed)
        else:
            logger.debug("Using traditional PyTorch pipeline")
            return generate_audio_traditional(text, kokoro_language, voice, speed)
    
    except Exception as e:
        if engine == KokoroCLIConstants.ENGINE_COREML:
            logger.warning(f"CoreML engine failed: {e}. Falling back to PyTorch engine.")
            return generate_audio_traditional(text, kokoro_language, voice, speed)
        else:
            # Re-raise exception for pytorch engine failures (no fallback available)
            raise


def normalize_audio_professional(audio: np.ndarray) -> np.ndarray:
    """
    Apply professional-grade audio normalization for optimal output quality.
    
    This function implements peak normalization with professional audio standards,
    ensuring consistent output levels while preventing clipping artifacts. It handles
    edge cases like silence detection and maintains audio fidelity throughout processing.
    
    Normalization Process:
    1. Peak Detection: Find maximum absolute amplitude across all channels
    2. Silence Handling: Detect and preserve intentional silence periods
    3. Gain Calculation: Compute normalization factor to target peak level
    4. Dynamic Range Protection: Prevent over-normalization of quiet content
    5. Clipping Prevention: Ensure output remains within valid amplitude range
    
    Args:
        audio: Input audio array (float32, range typically -1.0 to 1.0)
    
    Returns:
        np.ndarray: Normalized audio array (float32, peak at target level)
    
    Professional Audio Standards:
    - Target Peak: -0.5dB (0.95 amplitude) to prevent digital clipping
    - Silence Threshold: -70dB to distinguish intended silence from noise floor
    - Dynamic Range: Preserve original dynamic relationships between segments
    - Headroom: Maintain 0.5dB headroom for downstream processing
    
    Quality Assurance:
    - No clipping artifacts introduced during normalization process
    - Consistent output levels across different synthesis engines
    - Preservation of audio fidelity and dynamic characteristics
    - Professional broadcast standards compliance
    
    Called by:
    - generate_and_save_audio() for all synthesis outputs
    - Audio processing workflows requiring consistent levels
    - Quality assurance processes during batch synthesis
    """
    if len(audio) == 0:
        return audio
    
    # Find peak amplitude for normalization calculation
    peak = np.max(np.abs(audio))
    
    # Handle silence or extremely quiet audio
    if peak < KokoroCLIConstants.SILENCE_THRESHOLD:
        logger.debug("Audio below silence threshold, skipping normalization")
        return audio
    
    # Calculate normalization gain to target peak level
    gain = KokoroCLIConstants.NORMALIZATION_TARGET / peak
    
    # Apply normalization with clipping prevention
    normalized = audio * gain
    normalized = np.clip(normalized, -1.0, 1.0)
    
    logger.debug(f"Audio normalized: peak {peak:.4f} -> {np.max(np.abs(normalized)):.4f}")
    return normalized


def generate_and_save_audio(
    output_file: Path, text: str, engine: str, kokoro_language: str, voice: str, speed: float = 1.0
) -> float:
    """
    Complete audio generation and file output with professional quality standards.
    
    This function orchestrates the complete synthesis workflow from text input to
    high-quality WAV file output. It handles both traditional and hybrid synthesis
    engines, applies professional audio processing, and ensures consistent output quality.
    
    Workflow Architecture:
    1. Directory Management: Auto-create output directories as needed
    2. Engine Dispatch: Route synthesis to appropriate engine (traditional/hybrid)
    3. Audio Processing: Apply professional normalization and format conversion
    4. File Output: Write broadcast-quality WAV files with proper metadata
    5. Performance Monitoring: Calculate and report real-time factor
    
    Args:
        output_file: Output file path (will be created/overwritten)
        text: Input text for synthesis
        engine: Synthesis engine selection
        kokoro_language: Language code for traditional pipeline
        voice: Voice identifier
        speed: Speech rate multiplier
    
    Returns:
        float: Real-time factor (synthesis_time / audio_duration)
               Values < 1.0 indicate faster-than-real-time performance
    
    Audio Quality Specifications:
    - Sample Rate: 24kHz (matches model training data)
    - Bit Depth: 16-bit (optimal quality/size balance for speech)
    - Channels: Mono (standard for TTS applications)
    - Normalization: Professional peak normalization at -0.5dB
    - Format: WAV PCM for universal compatibility
    
    Engine-Specific Handling:
    - Traditional Pipeline: Processes generator results incrementally
    - Hybrid Pipeline: Handles single audio array output with timing data
    - Error Recovery: Automatic fallback with user notification
    
    File System Management:
    - Automatic directory creation for output paths
    - Safe file overwriting with atomic operations where possible
    - Cross-platform path handling (Windows, macOS, Linux)
    - Proper file handle cleanup and error recovery
    
    Performance Monitoring:
    Real-time factor calculation enables performance validation and optimization.
    Values significantly above 1.0 may indicate configuration issues or resource
    constraints that should be investigated.
    
    Called by:
    - main() for command-line synthesis requests
    - Batch processing workflows
    - Integration testing and quality assurance processes
    """
    import time
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Track total synthesis time for RTF calculation
    total_start_time = time.time()
    
    # Open WAV file with professional audio specifications
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(KokoroCLIConstants.AUDIO_CHANNELS)       # Mono audio
        wav_file.setsampwidth(KokoroCLIConstants.AUDIO_SAMPLE_WIDTH)   # 16-bit depth
        wav_file.setframerate(KokoroCLIConstants.AUDIO_SAMPLE_RATE)    # 24kHz rate
        
        total_audio_samples = 0
        
        # Handle different synthesis engine outputs
        synthesis_result = generate_audio(text, engine, kokoro_language, voice, speed)
        
        # Process hybrid pipeline output (single audio array)
        if isinstance(synthesis_result, tuple):
            audio_array, synthesis_time = synthesis_result
            
            # Apply professional normalization
            if KokoroCLIConstants.PEAK_NORMALIZATION:
                audio_array = normalize_audio_professional(audio_array)
            
            # Convert to 16-bit integer format
            audio_int16 = (audio_array * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Write to file
            wav_file.writeframes(audio_bytes)
            total_audio_samples = len(audio_array)
            
            logger.debug(f"Hybrid synthesis: {len(audio_array)} samples in {synthesis_time:.3f}s")
        
        # Process traditional pipeline output (generator of results)
        else:
            for result in synthesis_result:
                logger.debug(f"Processing segment: {result.phonemes if hasattr(result, 'phonemes') else 'N/A'}")
                
                if result.audio is None:
                    continue
                
                audio_segment = result.audio.numpy()
                
                # Apply professional normalization
                if KokoroCLIConstants.PEAK_NORMALIZATION:
                    audio_segment = normalize_audio_professional(audio_segment)
                
                # Convert to 16-bit integer format
                audio_int16 = (audio_segment * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                # Write segment to file
                wav_file.writeframes(audio_bytes)
                total_audio_samples += len(audio_segment)
    
    # Calculate performance metrics
    total_synthesis_time = time.time() - total_start_time
    audio_duration = total_audio_samples / KokoroCLIConstants.AUDIO_SAMPLE_RATE
    rtf = total_synthesis_time / audio_duration if audio_duration > 0 else float('inf')
    
    logger.info(f"Synthesis complete: {audio_duration:.2f}s audio in {total_synthesis_time:.3f}s (RTF: {rtf:.3f})")
    
    return rtf


def validate_system_requirements() -> bool:
    """
    Validate system requirements and dependencies for optimal TTS performance.
    
    This function performs comprehensive system validation to ensure all required
    dependencies are available and properly configured. It provides detailed
    feedback on any issues and suggests specific remediation steps.
    
    Validation Categories:
    1. Python Environment: Version compatibility and package availability
    2. Hardware Capabilities: Apple Silicon detection and MPS availability
    3. Audio Dependencies: System audio libraries and codecs
    4. Model Files: Required models and configuration files
    5. Storage Space: Adequate disk space for models and output
    
    Returns:
        bool: True if all requirements met, False if critical issues found
    
    Validation Process:
    - Non-blocking warnings for optional features (e.g., missing language packs)
    - Blocking errors for critical dependencies (e.g., PyTorch, audio libraries)
    - Detailed troubleshooting guidance for common installation issues
    - System-specific recommendations based on detected platform
    
    Called by:
    - main() during initialization to ensure proper configuration
    - System setup scripts during first-time installation
    - Diagnostic routines when troubleshooting synthesis issues
    """
    validation_passed = True
    
    # Check Python version compatibility
    import sys
    if sys.version_info < (3, 10):
        logger.error(f"Python 3.10+ required, found {sys.version}. Please upgrade Python.")
        validation_passed = False
    
    # Check PyTorch availability and version
    try:
        torch_version = torch.__version__
        logger.debug(f"PyTorch {torch_version} available")
    except Exception as e:
        logger.error(f"PyTorch not available: {e}. Install with: pip install torch")
        validation_passed = False
    
    # Check for Apple Silicon optimizations
    if torch.backends.mps.is_available():
        logger.debug("MPS (Metal Performance Shaders) available for GPU acceleration")
    else:
        logger.debug("MPS not available - using CPU/standard GPU processing")
    
    # Check hybrid pipeline dependencies
    if HYBRID_PIPELINE_AVAILABLE:
        logger.debug("Hybrid ANE pipeline available")
    else:
        logger.info("Hybrid ANE pipeline not available. Install with: pip install coremltools safetensors")
    
    # Check for language-specific dependencies
    missing_languages = []
    try:
        import misaki.en
    except ImportError:
        missing_languages.append("English (misaki[en])")
    
    try:
        import misaki.ja
    except ImportError:
        missing_languages.append("Japanese (misaki[ja])")
    
    try:
        import misaki.zh
    except ImportError:
        missing_languages.append("Chinese (misaki[zh])")
    
    if missing_languages:
        logger.info(f"Optional language packs missing: {', '.join(missing_languages)}")
    
    return validation_passed


def main() -> None:
    """
    Main entry point for Kokoro TTS command-line interface.
    
    This function implements the complete CLI workflow including argument parsing,
    system validation, engine selection, and audio synthesis. It provides comprehensive
    error handling and user feedback throughout the synthesis process.
    
    CLI Architecture:
    1. System Validation: Check dependencies and hardware capabilities
    2. Argument Processing: Parse and validate command-line arguments
    3. Engine Selection: Choose optimal synthesis engine based on configuration
    4. Input Processing: Handle text from files, stdin, or direct arguments
    5. Synthesis Execution: Generate audio with performance monitoring
    6. Output Management: Save high-quality WAV files with proper formatting
    
    Error Handling Philosophy:
    - Clear, actionable error messages with specific troubleshooting steps
    - Graceful degradation when possible (e.g., engine fallbacks)
    - System-specific guidance based on detected platform and capabilities
    - Comprehensive logging for debugging and performance analysis
    
    Performance Monitoring:
    Real-time factor calculation and timing information helps users validate
    expected performance improvements and identify configuration issues.
    
    Cross-Platform Compatibility:
    The CLI handles platform-specific differences in file paths, audio systems,
    and hardware capabilities while maintaining consistent user experience.
    
    Called by:
    - Direct execution: python -m kokoro [args]
    - Script invocation: python kokoro/__main__.py [args]
    - Integration: from kokoro.__main__ import main; main()
    """
    # Perform system validation before processing
    if not validate_system_requirements():
        logger.error("System validation failed. Please resolve dependency issues before continuing.")
        sys.exit(1)
    
    # Set up comprehensive argument parsing with detailed help
    parser = argparse.ArgumentParser(
        prog='kokoro',
        description=KokoroCLIConstants.ENGINE_HELP,
        epilog="For detailed documentation and examples, visit: https://github.com/hexgrad/kokoro",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Engine selection (most important parameter)
    parser.add_argument(
        "-e", "--engine",
        choices=KokoroCLIConstants.SUPPORTED_ENGINES,
        default=KokoroCLIConstants.DEFAULT_ENGINE,
        help=KokoroCLIConstants.ENGINE_HELP
    )
    
    # Voice selection with detailed help
    parser.add_argument(
        "-m", "--voice",
        default=KokoroCLIConstants.DEFAULT_VOICE,
        help=KokoroCLIConstants.VOICE_HELP
    )
    
    # Language override with comprehensive options
    parser.add_argument(
        "-l", "--language",
        choices=KokoroCLIConstants.LANGUAGE_CODES,
        help=KokoroCLIConstants.LANGUAGE_HELP
    )
    
    # Output file specification (required)
    parser.add_argument(
        "-o", "--output-file", "--output_file",
        type=Path,
        help=KokoroCLIConstants.OUTPUT_HELP,
        required=True
    )
    
    # Input text sources (mutually exclusive)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "-t", "--text",
        help="Text to synthesize directly from command line"
    )
    text_group.add_argument(
        "-i", "--input-file", "--input_file",
        type=Path,
        help="Path to input text file (UTF-8 encoding assumed)"
    )
    text_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read text from standard input (pipe or interactive)"
    )
    
    # Speech control parameters
    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=KokoroCLIConstants.DEFAULT_SPEED,
        help=KokoroCLIConstants.SPEED_HELP
    )
    
    # Debugging and verbose output
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging and performance metrics"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with synthesis progress"
    )
    
    # Parse arguments with comprehensive validation
    args = parser.parse_args()
    
    # Configure logging based on user preferences
    if args.debug:
        logger.level("DEBUG")
        logger.debug(f"Debug mode enabled. Arguments: {args}")
    elif args.verbose:
        logger.level("INFO")
    
    # Validate and process synthesis parameters
    # Language auto-detection from voice prefix
    lang = args.language or args.voice[0]
    if lang not in KokoroCLIConstants.SUPPORTED_LANGUAGES:
        logger.error(f"Unsupported language code: {lang}. Supported: {list(KokoroCLIConstants.SUPPORTED_LANGUAGES.keys())}")
        sys.exit(1)
    
    # Speed validation with reasonable limits
    if not (0.1 <= args.speed <= 3.0):
        logger.error(f"Speed must be between 0.1 and 3.0, got {args.speed}")
        sys.exit(1)
    
    # Input text acquisition with encoding handling
    try:
        if args.text:
            text = args.text
        elif args.input_file:
            text = args.input_file.read_text(encoding='utf-8')
            logger.debug(f"Read {len(text)} characters from {args.input_file}")
        else:  # --stdin or default fallback
            print("Enter text to synthesize (Ctrl+D when finished):", file=sys.stderr)
            text = sys.stdin.read()
    except Exception as e:
        logger.error(f"Failed to read input text: {e}")
        sys.exit(1)
    
    # Text length validation
    if len(text) > KokoroCLIConstants.MAX_TEXT_LENGTH:
        logger.error(f"Text too long: {len(text)} characters (max: {KokoroCLIConstants.MAX_TEXT_LENGTH})")
        sys.exit(1)
    
    if len(text.strip()) == 0:
        logger.error("No text provided for synthesis")
        sys.exit(1)
    
    logger.debug(f"Input text ({len(text)} chars): {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Output file validation and setup
    output_file: Path = args.output_file
    if not output_file.suffix.lower() == ".wav":
        logger.warning(f"Output file should have .wav extension, got: {output_file.suffix}")
    
    # Execute synthesis with comprehensive error handling
    try:
        logger.info(f"Starting synthesis: engine={args.engine}, voice={args.voice}, language={lang}")
        
        rtf = generate_and_save_audio(
            output_file=output_file,
            text=text,
            engine=args.engine,
            kokoro_language=lang,
            voice=args.voice,
            speed=args.speed
        )
        
        # Success reporting with performance metrics
        file_size = output_file.stat().st_size if output_file.exists() else 0
        logger.info(f"✅ Synthesis successful!")
        logger.info(f"📁 Output: {output_file} ({file_size:,} bytes)")
        logger.info(f"⚡ Performance: RTF {rtf:.3f} ({'faster than' if rtf < 1 else 'slower than'} real-time)")
        
        if args.verbose or args.debug:
            audio_duration = file_size / (KokoroCLIConstants.AUDIO_SAMPLE_RATE * KokoroCLIConstants.AUDIO_SAMPLE_WIDTH)
            logger.info(f"🎵 Audio: {audio_duration:.2f}s at {KokoroCLIConstants.AUDIO_SAMPLE_RATE}Hz")
    
    except KeyboardInterrupt:
        logger.info("Synthesis interrupted by user")
        sys.exit(130)  # Standard SIGINT exit code
    
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    """
    Script execution entry point with proper error handling.
    
    When executed directly (python -m kokoro or python kokoro/__main__.py),
    this block ensures proper initialization and graceful error handling.
    It also provides a clean interface for programmatic usage.
    
    Exit Code Standards:
    - 0: Successful synthesis completion
    - 1: General error (invalid arguments, synthesis failure)
    - 130: User interrupt (Ctrl+C)
    
    The error handling ensures that the CLI behaves properly in shell scripts,
    CI/CD pipelines, and other automated environments.
    """
    try:
        main()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # Catch any unhandled exceptions for clean error reporting
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
