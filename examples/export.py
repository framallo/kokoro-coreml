"""
PyTorch to ONNX export pipeline for Kokoro TTS model deployment.

This module provides a complete conversion workflow from PyTorch to ONNX format,
enabling deployment on ONNX Runtime-compatible platforms. Unlike the Core ML
conversion pipeline, this approach creates a single monolithic model that
handles both text processing and audio synthesis.

Architecture Overview:
- Single-model approach: All operations in one ONNX graph
- Variable-length support: Dynamic axes for flexible input sizes
- Cross-platform compatibility: Runs on any ONNX Runtime backend

Used by:
- Cross-platform deployment workflows
- Non-Apple device deployments
- Development testing and validation
- Performance benchmarking against other backends

Calls into:
- kokoro.KModel: Base PyTorch model architecture
- kokoro.KPipeline: Text processing and voice loading utilities
- kokoro.model.KModelForONNX: ONNX-optimized model wrapper
- torch.onnx.export: PyTorch's ONNX conversion functionality

Output:
- kokoro.onnx: Complete TTS model ready for ONNX Runtime deployment

Trade-offs vs Core ML:
- Pros: Cross-platform, single model, simpler deployment
- Cons: Less optimized for Apple silicon, no ANE acceleration

This script provides three modes:
1. Export: Convert PyTorch model to ONNX
2. Check: Validate PyTorch model functionality
3. Inference: Test ONNX model with audio playback
"""

import argparse
import os
import torch
import onnx
import onnxruntime as ort
import sounddevice as sd

from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX


# --- ONNX Export Constants ---
# These constants define the conversion parameters and model behavior.
# All magic numbers are replaced with named constants for LLM understanding.

class ONNXConstants:
    """Named constants for ONNX export to avoid magic numbers."""
    
    # Model Architecture Dimensions
    # These match Kokoro's internal architecture and training data
    REFERENCE_STYLE_DIM = 256         # Full reference style vector size
    MAX_TEXT_LENGTH = 510             # Maximum supported text length (before truncation)
    MIN_TOKEN_COUNT = 48              # Minimum tokens for representative tracing
    MAX_TOKEN_COUNT = 100             # Maximum token ID for random generation
    SPEED_RANGE_MIN = 1               # Minimum speed multiplier
    SPEED_RANGE_MAX = 10              # Maximum speed multiplier for random testing
    
    # ONNX Export Parameters
    ONNX_OPSET_VERSION = 17           # ONNX operator set version (for compatibility)
    ONNX_FILENAME = "kokoro.onnx"     # Standard output filename
    
    # Audio Parameters
    SAMPLE_RATE = 24000               # Kokoro's native sample rate (24kHz)
    
    # Test Data Configuration
    # Sample texts for different languages and validation
    ENGLISH_TEST_TEXT = "In today's fast-paced tech world, building software applications has never been easier — thanks to AI-powered coding assistants."
    ENGLISH_ALT_TEXT = "The sky above the port was the color of television, tuned to a dead channel."
    CHINESE_TEST_TEXT = "2月15日晚，猫眼专业版数据显示，截至发稿，《哪吒之魔童闹海》（或称《哪吒2》）今日票房已达7.8亿元，累计票房（含预售）超过114亿元。"
    
    # Voice Configuration
    ENGLISH_VOICE_PATH = "checkpoints/voices/af_heart.pt"
    CHINESE_VOICE_PATH = "checkpoints/voices/zf_xiaoxiao.pt"
    
    # Language Codes
    # These map to Kokoro's internal language processing systems
    LANG_ENGLISH = 'a'  # American English
    LANG_CHINESE = 'z'  # Chinese (Simplified)
    
    # Model Paths
    DEFAULT_CONFIG_PATH = "checkpoints/config.json"
    DEFAULT_CHECKPOINT_PATH = "checkpoints/kokoro-v1_0.pth"
    DEFAULT_OUTPUT_DIR = "onnx"

def export_onnx(model: KModelForONNX, output_dir: str) -> None:
    """
    Exports a PyTorch model to ONNX format with dynamic input support.
    
    This function creates a complete ONNX representation of the Kokoro TTS model
    that supports variable-length text input and produces variable-length audio output.
    
    Called by:
    - main() when export mode is selected
    
    Calls into:
    - torch.onnx.export() for PyTorch to ONNX conversion
    - onnx.load() and onnx.checker.check_model() for validation
    
    Export Configuration:
    - Dynamic axes enable variable-length inputs and outputs
    - Constant folding optimizes static operations
    - ONNX opset 17 provides broad compatibility
    
    Args:
        model: KModelForONNX instance optimized for ONNX export
        output_dir: Directory path where ONNX file will be saved
        
    Outputs:
        - Creates kokoro.onnx in the specified directory
        - Validates the exported model for correctness
        
    Input Tensor Shapes:
        - input_ids: (1, variable_length) - Text tokens with BOS/EOS
        - style: (1, 256) - Reference voice embedding
        - speed: (1,) - Speech rate multiplier
        
    Output Tensor Shapes:
        - waveform: (variable_samples,) - Generated audio waveform
        - duration: (1, variable_length) - Duration per input token
    """
    # Construct output file path
    onnx_file = os.path.join(output_dir, ONNXConstants.ONNX_FILENAME)

    # Create representative input tensors for ONNX tracing
    # Generate random token sequence within vocabulary bounds
    random_tokens = torch.randint(1, ONNXConstants.MAX_TOKEN_COUNT, (ONNXConstants.MIN_TOKEN_COUNT,)).numpy()
    
    # Add BOS (0) and EOS (0) tokens for complete sequence
    input_ids = torch.LongTensor([[0, *random_tokens, 0]])
    
    # Random reference style vector (normally from voice embedding)
    style = torch.randn(1, ONNXConstants.REFERENCE_STYLE_DIM)
    
    # Random speed multiplier for synthesis rate control
    speed = torch.randint(
        ONNXConstants.SPEED_RANGE_MIN, 
        ONNXConstants.SPEED_RANGE_MAX, 
        (1,)
    ).int()

    # Export to ONNX with comprehensive configuration
    torch.onnx.export(
        model,  # The model to export
        args=(input_ids, style, speed),  # Representative input tuple
        f=onnx_file,  # Output file path
        export_params=True,  # Include trained parameters in export
        verbose=True,  # Print detailed export information
        input_names=['input_ids', 'style', 'speed'],  # Named inputs for clarity
        output_names=['waveform', 'duration'],  # Named outputs for clarity
        opset_version=ONNXConstants.ONNX_OPSET_VERSION,  # ONNX operator compatibility
        dynamic_axes={
            # Enable variable-length text input
            'input_ids': {1: 'input_ids_len'}, 
            # Enable variable-length audio output
            'waveform': {0: 'num_samples'}, 
        }, 
        do_constant_folding=True,  # Optimize static operations
    )

    print(f'✅ Export to {onnx_file} completed successfully!')

    # Validate the exported ONNX model
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print('✅ ONNX model validation passed!')

def load_input_ids(pipeline: KPipeline, text: str) -> tuple[list, torch.LongTensor]:
    """
    Converts text to model input tokens through phonemization and vocabulary lookup.
    
    This function handles the complete text preprocessing pipeline:
    1. Text → Phonemes (grapheme-to-phoneme conversion)
    2. Phonemes → Token IDs (vocabulary lookup)
    3. Add BOS/EOS tokens for model compatibility
    
    Called by:
    - load_sample() for creating test inputs
    
    Calls into:
    - pipeline.g2p() for phoneme conversion
    - pipeline.en_tokenize() for English tokenization (if applicable)
    - pipeline.model.vocab.get() for token ID lookup
    
    Language Support:
    - English ('a', 'b'): Uses en_tokenize for advanced processing
    - Other languages: Direct phoneme conversion
    
    Args:
        pipeline: Initialized KPipeline with language-specific settings
        text: Raw input text to convert
        
    Returns:
        tuple: (phoneme_list, token_tensor)
        - phoneme_list: List of phoneme strings for debugging
        - token_tensor: (1, seq_len) tensor of token IDs with BOS/EOS
        
    Processing Details:
    - Truncates to max 510 phonemes (leaves room for BOS/EOS tokens)
    - Filters out None values from vocabulary lookup
    - Adds BOS (0) and EOS (0) tokens for model compatibility
    - Places tensor on same device as model
    """
    # Handle language-specific phoneme processing
    if pipeline.lang_code in 'ab':  # English languages
        # English uses advanced tokenization with grapheme processing
        _, tokens = pipeline.g2p(text)
        for graphemes, phonemes, token_ids in pipeline.en_tokenize(tokens):
            if not phonemes:
                continue
            ps = phonemes  # Use processed phonemes
    else:
        # Other languages use direct phoneme conversion
        ps, _ = pipeline.g2p(text)

    # Truncate phoneme sequence to maximum supported length
    # Leave room for BOS (1) + EOS (1) = 2 tokens
    if len(ps) > ONNXConstants.MAX_TEXT_LENGTH:
        ps = ps[:ONNXConstants.MAX_TEXT_LENGTH]
        print(f"⚠️  Text truncated to {ONNXConstants.MAX_TEXT_LENGTH} phonemes")

    # Convert phonemes to token IDs through vocabulary lookup
    # Filter out None values (unknown phonemes)
    input_ids = list(filter(
        lambda token_id: token_id is not None, 
        map(lambda phoneme: pipeline.model.vocab.get(phoneme), ps)
    ))
    
    # Debug output showing conversion pipeline
    print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"Phonemes ({len(ps)}): {' '.join(ps[:10])}{'...' if len(ps) > 10 else ''}")
    print(f"Token IDs ({len(input_ids)}): {input_ids[:10]}{'...' if len(input_ids) > 10 else ''}")
    
    # Add BOS (Beginning Of Sequence) and EOS (End Of Sequence) tokens
    # Format: [BOS, token1, token2, ..., tokenN, EOS]
    input_ids_with_special = [0, *input_ids, 0]
    
    # Convert to tensor and place on model device
    input_ids_tensor = torch.LongTensor([input_ids_with_special]).to(pipeline.model.device)
    
    return ps, input_ids_tensor

def load_voice(pipeline: KPipeline, voice_path: str, phonemes: list) -> torch.Tensor:
    """
    Loads voice embedding and selects appropriate conditioning vector.
    
    Voice embeddings in Kokoro are learned representations that capture
    speaker characteristics, speaking style, and prosodic patterns.
    Different phoneme counts may use different embedding indices.
    
    Called by:
    - load_sample() for voice conditioning setup
    
    Calls into:
    - pipeline.load_voice() for voice embedding loading
    
    Args:
        pipeline: KPipeline instance for voice loading
        voice_path: Path to voice embedding file (.pt format)
        phonemes: List of phonemes (length used for embedding selection)
        
    Returns:
        torch.Tensor: Voice conditioning vector (256 dimensions)
        
    Voice Embedding Details:
    - Each voice file contains multiple embeddings for different contexts
    - Embedding selection based on phoneme count (len(phonemes) - 1)
    - Moved to CPU for consistent device handling
    - Used for style conditioning throughout synthesis pipeline
    """
    # Load voice embedding pack from checkpoint file
    # Voice packs contain multiple embeddings for different contexts
    voice_pack = pipeline.load_voice(voice_path).to('cpu')
    
    # Select embedding based on phoneme count
    # -1 adjustment accounts for model-specific indexing
    embedding_index = len(phonemes) - 1
    
    # Extract and return the appropriate voice embedding
    return voice_pack[embedding_index]

def load_sample(model: KModelForONNX) -> tuple[torch.LongTensor, torch.Tensor, torch.IntTensor]:
    """
    Creates sample input tensors for model testing and validation.
    
    This function sets up a complete test configuration with representative
    text, voice, and synthesis parameters. It demonstrates the full
    preprocessing pipeline from raw text to model inputs.
    
    Called by:
    - check_model() for PyTorch model validation
    - inference_onnx() for ONNX model testing
    
    Calls into:
    - KPipeline() for text and voice processing
    - load_input_ids() for text tokenization
    - load_voice() for voice embedding loading
    
    Sample Configuration:
    This function includes multiple language examples for comprehensive testing.
    Currently configured for Chinese text with appropriate voice pairing.
    
    Language Examples:
    - English: Technical and literary text samples
    - Chinese: News text with complex characters and numbers
    
    Args:
        model: KModelForONNX instance (used to access base kmodel)
        
    Returns:
        tuple: (input_ids, style, speed)
        - input_ids: Tokenized text with BOS/EOS (1, seq_len)
        - style: Voice embedding vector (1, 256)
        - speed: Synthesis speed multiplier (1,) - 1 = normal speed
        
    Device Configuration:
    All processing is done on CPU for consistency with ONNX export expectations.
    """
    # Configuration 1: English language testing
    # Create pipeline for American English processing
    english_pipeline = KPipeline(lang_code=ONNXConstants.LANG_ENGLISH, model=model.kmodel, device='cpu')
    
    # English test texts - variety of complexity and content
    english_text_technical = ONNXConstants.ENGLISH_TEST_TEXT
    english_text_literary = ONNXConstants.ENGLISH_ALT_TEXT
    english_voice = ONNXConstants.ENGLISH_VOICE_PATH

    # Configuration 2: Chinese language testing (currently active)
    # Create pipeline for Chinese processing
    chinese_pipeline = KPipeline(lang_code=ONNXConstants.LANG_CHINESE, model=model.kmodel, device='cpu')
    
    # Chinese test text - news content with numbers, dates, and complex characters
    chinese_text = ONNXConstants.CHINESE_TEST_TEXT
    chinese_voice = ONNXConstants.CHINESE_VOICE_PATH

    # Select active configuration for testing
    # Currently using Chinese for comprehensive Unicode testing
    active_pipeline = chinese_pipeline
    active_text = chinese_text
    active_voice = chinese_voice

    # Process text through complete preprocessing pipeline
    phonemes, input_ids = load_input_ids(active_pipeline, active_text)
    
    # Load voice conditioning vector
    style = load_voice(active_pipeline, active_voice, phonemes)
    
    # Set normal speech speed (1.0 = baseline rate)
    speed = torch.IntTensor([1])

    print(f"\n🎤 Sample loaded:")
    print(f"Language: {active_pipeline.lang_code}")
    print(f"Voice: {os.path.basename(active_voice)}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Style shape: {style.shape}")
    print(f"Speed: {speed.item()}x")

    return input_ids, style, speed

def inference_onnx(model: KModelForONNX, output_dir: str) -> None:
    """
    Tests ONNX model inference with audio playback validation.
    
    This function performs end-to-end validation of the exported ONNX model
    by running inference and playing back the generated audio. It serves as
    both a correctness test and a demonstration of ONNX Runtime usage.
    
    Called by:
    - main() when inference mode is selected
    
    Calls into:
    - ort.InferenceSession() for ONNX Runtime execution
    - load_sample() for test input generation
    - sounddevice.play() for audio playback
    
    Validation Process:
    1. Load ONNX model into runtime session
    2. Generate test inputs using same pipeline as PyTorch
    3. Run inference through ONNX Runtime
    4. Convert output to audio and validate shape
    5. Play audio for subjective quality assessment
    
    Args:
        model: Original PyTorch model (used for input generation)
        output_dir: Directory containing the exported ONNX file
        
    Expected Behavior:
    - Should generate audio with similar quality to PyTorch model
    - Audio length should match text complexity
    - No runtime errors or shape mismatches
    
    Performance Notes:
    - ONNX Runtime performance varies by backend (CPU/CUDA/etc.)
    - First inference may be slower due to optimization
    - Subsequent calls should be faster due to caching
    """
    # Construct path to exported ONNX model
    onnx_file = os.path.join(output_dir, ONNXConstants.ONNX_FILENAME)
    
    print(f"\n📋 Loading ONNX model: {onnx_file}")
    
    # Initialize ONNX Runtime inference session
    # This loads and optimizes the model for the target backend
    session = ort.InferenceSession(onnx_file)
    
    print(f"📋 Available providers: {session.get_providers()}")
    print(f"📋 Active provider: {session.get_providers()[0]}")

    # Generate test inputs using the same pipeline as PyTorch
    print(f"\n🚀 Generating test inputs...")
    input_ids, style, speed = load_sample(model)

    # Convert PyTorch tensors to NumPy arrays for ONNX Runtime
    onnx_inputs = {
        'input_ids': input_ids.numpy(), 
        'style': style.numpy(), 
        'speed': speed.numpy(), 
    }
    
    print(f"\n⏱️ Running ONNX inference...")
    
    # Run inference through ONNX Runtime
    # None as first argument means "return all outputs"
    outputs = session.run(None, onnx_inputs)
    
    # Extract waveform output (first output tensor)
    waveform_output = torch.from_numpy(outputs[0])
    
    # Duration output (second output tensor) - optional validation
    duration_output = torch.from_numpy(outputs[1]) if len(outputs) > 1 else None
    
    print(f"\n🎵 Inference complete!")
    print(f"Waveform shape: {waveform_output.shape}")
    print(f"Sample count: {waveform_output.shape[0]:,}")
    print(f"Duration: {waveform_output.shape[0] / ONNXConstants.SAMPLE_RATE:.2f} seconds")
    
    if duration_output is not None:
        print(f"Duration output shape: {duration_output.shape}")
        print(f"Average duration per token: {duration_output.mean():.2f}")

    # Convert to NumPy for audio playback
    audio_array = waveform_output.numpy()
    
    print(f"\n🔊 Playing generated audio...")
    print(f"Sample rate: {ONNXConstants.SAMPLE_RATE} Hz")
    print(f"Press Ctrl+C to stop playback")
    
    # Play audio through default audio device
    sd.play(audio_array, ONNXConstants.SAMPLE_RATE)
    sd.wait()  # Block until playback completes
    
    print(f"✅ ONNX inference validation complete!")

def check_model(model: KModelForONNX) -> None:
    """
    Validates PyTorch model functionality before ONNX conversion.
    
    This function performs a complete forward pass through the PyTorch model
    to ensure it's working correctly before attempting ONNX export. It also
    provides audio playback for subjective quality assessment.
    
    Called by:
    - main() when check mode is selected
    
    Calls into:
    - load_sample() for test input generation
    - model() for forward pass inference
    - sounddevice.play() for audio validation
    
    Validation Process:
    1. Generate representative test inputs
    2. Run forward pass through PyTorch model
    3. Validate output shapes and ranges
    4. Play generated audio for quality assessment
    
    Args:
        model: KModelForONNX instance to validate
        
    Expected Outputs:
    - Waveform: Variable-length audio samples
    - Duration: Per-token timing information
    
    This validation is crucial because:
    - Ensures model loads correctly with expected architecture
    - Confirms text processing pipeline works end-to-end
    - Validates voice conditioning is properly applied
    - Provides baseline for comparing ONNX conversion quality
    """
    print(f"\n🚀 Validating PyTorch model...")
    
    # Generate test inputs using standard pipeline
    input_ids, style, speed = load_sample(model)
    
    print(f"\n⏱️ Running PyTorch inference...")
    
    # Run forward pass through PyTorch model
    # Model returns (waveform, duration) tuple
    waveform_output, duration_output = model(input_ids, style, speed)

    print(f"\n🎵 PyTorch inference complete!")
    print(f"Waveform shape: {waveform_output.shape}")
    print(f"Duration shape: {duration_output.shape}")
    print(f"Sample count: {waveform_output.shape[0]:,}")
    print(f"Audio duration: {waveform_output.shape[0] / ONNXConstants.SAMPLE_RATE:.2f} seconds")
    print(f"Average duration per token: {duration_output.mean():.2f}")
    
    # Show sample of waveform values for debugging
    print(f"\nWaveform sample (first 10 values):")
    print(waveform_output[:10])
    print(f"Waveform range: [{waveform_output.min():.3f}, {waveform_output.max():.3f}]")

    # Convert to NumPy for audio playback
    audio_array = waveform_output.numpy()
    
    print(f"\n🔊 Playing generated audio...")
    print(f"Sample rate: {ONNXConstants.SAMPLE_RATE} Hz")
    print(f"Press Ctrl+C to stop playback")
    
    # Play audio through default audio device
    sd.play(audio_array, ONNXConstants.SAMPLE_RATE)
    sd.wait()  # Block until playback completes
    
    print(f"✅ PyTorch model validation complete!")

if __name__ == "__main__":
    """
    Main execution block for Kokoro → ONNX export pipeline.
    
    This script provides three operational modes:
    1. Export (default): Convert PyTorch model to ONNX format
    2. Check: Validate PyTorch model functionality
    3. Inference: Test exported ONNX model with audio playback
    
    Usage Examples:
        python export.py                           # Export to ONNX
        python export.py --check                   # Validate PyTorch model
        python export.py --inference               # Test ONNX model
        python export.py -o ./models -c ./config.json  # Custom paths
        
    The script handles complete model loading, validation, and conversion
    with comprehensive error handling and progress reporting.
    
    Output:
    - Default mode: Creates kokoro.onnx in specified output directory
    - Check mode: Validates model and plays test audio
    - Inference mode: Tests ONNX model and plays generated audio
    """
    # Set up command line argument parsing with detailed help
    parser = argparse.ArgumentParser(
        "Export Kokoro Model to ONNX", 
        add_help=True,
        description="Convert Kokoro TTS model from PyTorch to ONNX format with validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export.py                              # Export to ONNX (default)
  python export.py --check                      # Validate PyTorch model
  python export.py --inference                  # Test ONNX inference
  python export.py -o ./models -c ./my_config   # Custom paths
        """
    )
    
    # Operational mode selection
    parser.add_argument(
        "--inference", "-t", 
        help="Test exported ONNX model with audio playback", 
        action="store_true"
    )
    parser.add_argument(
        "--check", "-m", 
        help="Validate PyTorch model functionality before export", 
        action="store_true"
    )
    
    # Model configuration paths
    parser.add_argument(
        "--config_file", "-c", 
        type=str, 
        default=ONNXConstants.DEFAULT_CONFIG_PATH, 
        help="Path to model configuration JSON file"
    )
    parser.add_argument(
        "--checkpoint_path", "-p", 
        type=str, 
        default=ONNXConstants.DEFAULT_CHECKPOINT_PATH, 
        help="Path to PyTorch model checkpoint file"
    )
    parser.add_argument(
        "--output_dir", "-o", 
        type=str, 
        default=ONNXConstants.DEFAULT_OUTPUT_DIR, 
        help="Output directory for ONNX files"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Extract configuration with clear variable names
    config_file = args.config_file
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    
    print(f"🚀 Kokoro → ONNX Export Pipeline")
    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory ready: {output_dir}")

    # Load and initialize PyTorch model
    print(f"\n📦 Loading PyTorch model...")
    try:
        kmodel = KModel(config=config_file, model=checkpoint_path, disable_complex=True)
        model = KModelForONNX(kmodel).eval()
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"Please check that config and checkpoint files exist and are valid")
        exit(1)

    # Execute selected operation mode
    try:
        if args.inference:
            print(f"\n📋 Mode: ONNX Inference Testing")
            inference_onnx(model, output_dir)
        elif args.check:
            print(f"\n🔍 Mode: PyTorch Model Validation")
            check_model(model)
        else:
            print(f"\n🔄 Mode: PyTorch → ONNX Export")
            export_onnx(model, output_dir)
            
        print(f"\n✅ Operation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        print(f"Please check the logs above for detailed error information")
        exit(1)
