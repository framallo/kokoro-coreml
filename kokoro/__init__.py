"""
Kokoro TTS Package Initialization and Logging Configuration

This module serves as the main entry point for the Kokoro text-to-speech package,
providing centralized logging configuration and exposing the core API components.
It establishes a consistent logging infrastructure used throughout the codebase
and imports the primary user-facing classes.

Core Components Exported:
- KModel: Neural network model for text-to-speech synthesis
- KPipeline: Language-aware pipeline for complete TTS functionality

Logging Architecture:
- Structured logging with module/line information for debugging
- Configurable log levels for development vs production use
- Centralized disable capability for release builds
- Color-coded output for improved developer experience

Cross-file dependencies:
- Imported by: All user-facing scripts, demo applications, test suites
- Configures logging for: All modules in the kokoro package
- Exposes APIs from: model.py (KModel), pipeline.py (KPipeline)

Design Philosophy:
- Single import point for external users: `from kokoro import KModel, KPipeline`
- Centralized configuration reduces boilerplate across modules
- Explicit version management for compatibility tracking
- Logging setup happens at import time for consistent behavior
"""

__version__ = '0.9.4'

from loguru import logger
import sys

class LoggingConstants:
    """Logging configuration constants for consistent behavior across environments."""
    
    # Log level hierarchy for different deployment scenarios
    DEVELOPMENT_LEVEL = "DEBUG"    # Verbose logging for development and debugging
    PRODUCTION_LEVEL = "INFO"      # Standard operational logging for production
    ERROR_ONLY_LEVEL = "ERROR"     # Minimal logging for high-performance scenarios
    
    # Default configuration optimized for development
    DEFAULT_LEVEL = PRODUCTION_LEVEL
    
    # Format components for structured logging output
    TIME_FORMAT = "<green>{time:HH:mm:ss}</green>"           # Green timestamp for visibility
    MODULE_FORMAT = "<cyan>{module:>16}:{line}</cyan>"       # Cyan module:line for source tracking
    LEVEL_FORMAT = "<level>{level: >8}</level>"              # Color-coded log level
    MESSAGE_FORMAT = "<level>{message}</level>"              # Color-coded message content
    
    # Complete format string combining all components
    LOG_FORMAT = f"{TIME_FORMAT} | {MODULE_FORMAT} | {LEVEL_FORMAT} | {MESSAGE_FORMAT}"
    
    # Output configuration
    STDERR_OUTPUT = sys.stderr      # Standard error stream for log output
    COLOR_ENABLED = True            # Enable ANSI color codes for terminal output
    
    # Package-specific settings
    PACKAGE_NAME = "kokoro"         # Package identifier for disable functionality

def configure_logging():
    """
    Configure the loguru logger with Kokoro-specific settings.
    
    This function sets up structured logging that provides clear debugging information
    while remaining lightweight enough for production use. The configuration includes
    color-coded output, module/line tracking, and centralized control.
    
    Logging Configuration:
    1. Remove default loguru handler to avoid duplicate output
    2. Add custom handler with structured format for debugging
    3. Enable colorization for improved developer experience
    4. Set appropriate log level for current deployment context
    5. Provide package-wide disable capability for release builds
    
    Log Format Example:
    12:34:56 | model:142 |     INFO | Successfully loaded model weights
    12:34:57 | pipeline:89 |    DEBUG | Processing phonemes: 'hɛloʊ wɜrld'
    
    Called by:
    - Package import: Automatic configuration when `import kokoro` is executed
    - Used throughout: All modules inherit this logging configuration
    - Test suites: Consistent logging behavior across test environments
    
    Performance Notes:
    - Minimal overhead when logging is disabled via logger.disable()
    - Structured format aids in log parsing and monitoring systems
    - Color codes are automatically stripped when output is redirected
    """
    # Remove default handler to prevent duplicate log entries
    # This ensures clean output when package is imported in different contexts
    logger.remove()

    # Configure structured logging with comprehensive debugging information
    logger.add(
        LoggingConstants.STDERR_OUTPUT,
        format=LoggingConstants.LOG_FORMAT,
        colorize=LoggingConstants.COLOR_ENABLED,
        level=LoggingConstants.DEFAULT_LEVEL
    )

    # Disable package logging by default to prevent noise in production
    # This can be enabled during development by calling: logger.enable("kokoro")
    logger.disable(LoggingConstants.PACKAGE_NAME)

# Configure logging immediately upon import for consistent behavior
configure_logging()

# Import and expose core API components
# These are the primary classes that external users interact with
from .model import KModel      # Neural TTS model for direct inference
from .pipeline import KPipeline  # Complete TTS pipeline with language support
