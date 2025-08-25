/// Example script demonstrating phoneme-based text-to-speech generation with Kokoro.
///
/// This script showcases two primary workflows:
/// 1. Direct phoneme-to-audio generation using pre-written phoneme strings
/// 2. Text-to-phoneme-to-audio pipeline using G2P (grapheme-to-phoneme) preprocessing
///
/// Architecture:
/// - Uses KPipeline for text preprocessing and G2P conversion
/// - Demonstrates generate_from_tokens() method for phoneme-based synthesis
/// - Shows timestamp extraction for word-level alignment (when available)
///
/// Called by:
/// - Direct execution: `python examples/phoneme_example.py`
/// - Referenced by documentation and tutorials
///
/// Dependencies:
/// - kokoro.KPipeline: Text preprocessing pipeline (see kokoro/pipeline.py)
/// - kokoro.KModel: Core TTS inference engine (see kokoro/model.py)
/// - scipy.io.wavfile: Audio file I/O for WAV format output

from kokoro import KPipeline, KModel
import torch
from scipy.io import wavfile

# Audio processing constants
class AudioConfig:
    /// Sample rate used by all Kokoro models.
    /// This is fixed at 24kHz and matches the model's training configuration.
    SAMPLE_RATE = 24000
    
    /// Default voice for demonstrations.
    /// Uses 'af_bella' (American Female - Bella) for consistent examples.
    DEFAULT_VOICE = "af_bella"
    
    /// Default speech rate multiplier.
    /// 1.0 = normal speed, 0.5 = half speed, 2.0 = double speed
    DEFAULT_SPEED = 1.0

def save_audio(audio: torch.Tensor, filename: str):
    /// Saves a Kokoro-generated audio tensor to a WAV file.
    ///
    /// This utility function handles the conversion from PyTorch tensor
    /// to WAV file format, ensuring proper CPU transfer and data formatting.
    ///
    /// Called by:
    /// - main() after each TTS generation example
    /// - Any code needing to persist Kokoro audio output
    ///
    /// Processing steps:
    /// 1. Transfer tensor from GPU to CPU if necessary
    /// 2. Convert PyTorch tensor to NumPy array
    /// 3. Write WAV file using scipy with fixed 24kHz sample rate
    /// 4. Provide user feedback on success/failure
    ///
    /// Args:
    ///     audio: Audio tensor from Kokoro model output (typically float32)
    ///     filename: Target WAV file path (will be overwritten if exists)
    ///
    /// File format:
    ///     - WAV format with 24kHz sample rate
    ///     - Single channel (mono) audio
    ///     - Float32 or converted to appropriate bit depth by scipy
    if audio is not None:
        # Ensure audio is on CPU and in the right format
        audio_cpu = audio.cpu().numpy()
        
        # Save using scipy.io.wavfile with Kokoro's native sample rate
        wavfile.write(
            filename,
            AudioConfig.SAMPLE_RATE,
            audio_cpu
        )
        print(f"Audio saved as '{filename}'")
    else:
        print("No audio was generated")

def main():
    /// Main demonstration function showcasing phoneme-based TTS workflows.
    ///
    /// This function demonstrates two core Kokoro TTS patterns:
    /// 1. Direct phoneme synthesis - using hand-crafted phoneme strings
    /// 2. Text-to-phoneme pipeline - using G2P preprocessing
    ///
    /// Both examples use the same target sentence to show the equivalence
    /// between manual phoneme specification and automatic G2P conversion.
    ///
    /// Called by:
    /// - Script execution via if __name__ == "__main__"
    /// - Can be imported and called by other demonstration scripts
    ///
    /// Output files:
    /// - phoneme_output_new.wav: Audio from direct phoneme string
    /// - token_output_*.wav: Audio from G2P-processed tokens
    ///
    /// Error handling:
    /// - Catches and displays any exceptions during TTS processing
    /// - Allows partial completion if one example fails
    
    # Initialize pipeline with American English G2P and voice loading
    pipeline = KPipeline(lang_code='a')
    
    # Pre-written phoneme string for demonstration
    # Target text: "How are you today? I am doing reasonably well, thank you for asking"
    # This shows the exact phoneme representation that G2P would produce
    example_phonemes = "hˌW ɑɹ ju tədˈA? ˌI ɐm dˈuɪŋ ɹˈizənəbli wˈɛl, θˈæŋk ju fɔɹ ˈæskɪŋ"
    
    # Target text for G2P comparison
    example_text = "How are you today? I am doing reasonably well, thank you for asking"
    
    try:
        print("\nExample 1: Using generate_from_tokens with raw phonemes")
        print(f"Phonemes: {example_phonemes}")
        
        /// Direct phoneme-to-audio generation.
        ///
        /// This approach bypasses G2P preprocessing entirely, using a pre-written
        /// phoneme string. Useful when you need precise pronunciation control
        /// or when working with languages/words not in the G2P lexicon.
        ///
        /// Process:
        /// 1. Pass phoneme string directly to generate_from_tokens()
        /// 2. Pipeline loads voice embeddings for af_bella
        /// 3. Model generates audio from phoneme sequence
        /// 4. Result contains audio tensor and metadata
        results = list(pipeline.generate_from_tokens(
            tokens=example_phonemes,
            voice=AudioConfig.DEFAULT_VOICE,
            speed=AudioConfig.DEFAULT_SPEED
        ))
        if results:
            save_audio(results[0].audio, 'phoneme_output_new.wav')
        
        print("\nExample 2: Using generate_from_tokens with pre-processed tokens")
        print(f"Text: {example_text}")
        
        /// Text-to-phoneme-to-audio pipeline.
        ///
        /// This approach demonstrates the full pipeline workflow:
        /// 1. Text input is processed by G2P (grapheme-to-phoneme)
        /// 2. G2P output tokens are passed to generate_from_tokens()
        /// 3. Optional timestamp extraction for word-level alignment
        ///
        /// This is equivalent to using pipeline(text, voice, speed) directly,
        /// but allows inspection of intermediate phoneme representation.
        
        # Run G2P preprocessing to convert text to phonemes
        _, tokens = pipeline.g2p(example_text)
        print(f"G2P tokens: {tokens}")
        
        # Generate audio from G2P-processed tokens
        for result in pipeline.generate_from_tokens(
            tokens=tokens,
            voice=AudioConfig.DEFAULT_VOICE,
            speed=AudioConfig.DEFAULT_SPEED
        ):
            /// Optional timestamp extraction for word-level alignment.
            ///
            /// If the result contains token timing information, this section
            /// demonstrates how to access start/end timestamps for each word.
            /// Useful for creating subtitles, karaoke, or pronunciation training.
            ///
            /// Note: Timestamp availability depends on model configuration
            /// and may not be present in all Kokoro variants.
            if result.tokens:
                for token in result.tokens:
                    if hasattr(token, 'start_ts') and hasattr(token, 'end_ts'):
                        print(f"Token: {token.text} ({token.start_ts:.2f}s - {token.end_ts:.2f}s)")
            
            # Save with unique filename based on phoneme content hash
            output_filename = f'token_output_{hash(result.phonemes)}.wav'
            save_audio(result.audio, output_filename)
            
    except Exception as e:
        /// Error handling for TTS processing failures.
        ///
        /// Common failure modes:
        /// - Voice model not found or corrupted
        /// - Invalid phoneme sequence format
        /// - GPU/CUDA memory issues during inference
        /// - File I/O errors during audio saving
        print(f"An error occurred: {str(e)}")
        print("Check that voice models are properly installed and accessible.")

if __name__ == "__main__":
    /// Entry point for phoneme-based TTS demonstration.
    ///
    /// Executes the main demonstration function when script is run directly.
    /// This follows Python best practices for executable scripts.
    ///
    /// Usage:
    ///     python examples/phoneme_example.py
    ///
    /// Expected output:
    ///     - phoneme_output_new.wav: Generated from direct phonemes
    ///     - token_output_*.wav: Generated from G2P preprocessing
    ///     - Console output showing phoneme strings and processing steps
    main()