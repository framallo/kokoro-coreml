"""Kokoro TTS: Neural Text-to-Speech with CoreML Optimization

This package provides a complete text-to-speech solution optimized for Apple devices,
featuring neural voice synthesis with advanced phoneme processing and CoreML export
capabilities. Kokoro TTS combines transformer-based text encoding with high-quality
audio synthesis for natural-sounding speech generation.

Package Architecture:
    - model.py: Core neural TTS model (KModel) with BERT-based phoneme encoding
    - pipeline.py: Language processing pipeline (KPipeline) with G2P and voice management
    - modules.py: Neural network components (encoders, decoders, attention layers)
    - istftnet.py: Audio synthesis decoder with inverse STFT operations
    - custom_stft.py: CoreML-compatible STFT implementations

Key Entry Points:
    KModel: Neural TTS model for direct phoneme-to-audio conversion
    KPipeline: Complete text processing pipeline with language support

Cross-Package Dependencies:
    - examples/: Model export scripts for CoreML conversion
    - tools/: Training and post-processing utilities
    - Swift/: iOS/macOS applications and testing frameworks
    - demo/: Interactive web interface and example usage

External Dependencies:
    - transformers: BERT tokenization and model architecture
    - torch: Neural network inference and training
    - misaki: Grapheme-to-phoneme conversion for multiple languages
    - coremltools: Apple Neural Engine optimization and deployment

Usage Patterns:
    Simple synthesis: KPipeline(lang_code='a')("Hello world", voice="af_bella")
    Direct model: KModel().forward(phonemes, voice_embedding)
    CoreML export: See examples/export_coreml.py for deployment pipelines
"""

__version__ = '0.9.4'

from loguru import logger
import sys

# Centralized logging configuration for AI-first debugging
class LoggingConfig:
    """Logging configuration constants for development and debugging."""
    
    # Log format optimized for AI development workflows
    # Includes module:line for precise error location and time for performance analysis
    LOG_FORMAT = "<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>"
    
    # Default log level balances information with performance
    # Change to "DEBUG" for detailed inference tracing
    # Change to "ERROR" for production deployment
    DEFAULT_LEVEL = "INFO"

# Remove default handler
logger.remove()

# Add custom handler with clean format including module and line number
logger.add(
    sys.stderr,
    format=LoggingConfig.LOG_FORMAT,
    colorize=True,
    level=LoggingConfig.DEFAULT_LEVEL
)

# Disable before release or as needed
logger.disable("kokoro")

# Main package exports
from .model import KModel
from .pipeline import KPipeline
