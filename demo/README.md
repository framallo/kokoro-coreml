# Kokoro TTS Demo

Interactive Gradio-based demonstration of the Kokoro Text-to-Speech system with multi-voice synthesis capabilities.

## Overview

This demo provides a comprehensive web interface for testing Kokoro TTS functionality, featuring:

- **28 high-quality voices** across American and British English variants
- **Real-time audio generation** with CPU/GPU acceleration options
- **Advanced pronunciation control** via markdown syntax and phonetic notation
- **Streaming synthesis** for long-form content
- **Interactive tokenization** to inspect model processing

## Features

### Voice Selection
The demo includes diverse voice options:

**American English (🇺🇸)**:
- **Female voices (11)**: Heart ❤️, Bella 🔥, Nicole 🎧, Aoede, Kore, Sarah, Nova, Sky, Alloy, Jessica, River
- **Male voices (9)**: Michael, Fenrir, Puck, Echo, Eric, Liam, Onyx, Santa, Adam

**British English (🇬🇧)**:
- **Female voices (4)**: Emma, Isabella, Alice, Lily  
- **Male voices (4)**: George, Fable, Lewis, Daniel

### Advanced Pronunciation Control

#### Phonetic Override Syntax
```
[Kokoro](/kˈOkəɹO/)  - Custom IPA pronunciation
```

#### Stress Adjustment
```
[or](+2)         - Raise stress 1 level or 2 levels (works on less stressed words)
[1 level](-1)    - Lower stress 1 level
[2 levels](-2)   - Lower stress 2 levels
```

#### Intonation Control
Use punctuation for natural speech patterns:
- `;:,.!?—…` - Standard punctuation effects
- `""()` - Quote and parenthetical intonation
- `ˈ` - Primary stress marker
- `ˌ` - Secondary stress marker

### Generation Modes

#### Standard Generation
- Single audio output with complete processing
- Token inspection for debugging
- Automatic quality optimization

#### Streaming Mode  
- Real-time audio generation for long text
- Progressive output as synthesis proceeds
- Stop/start controls for interactive use

## File Structure

```
demo/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies (espeak-ng)
├── en.txt             # Random quote collection (2,122 inspirational quotes)
├── gatsby5k.md        # The Great Gatsby excerpt (~5k characters)
├── frankenstein5k.md  # Frankenstein excerpt (~5k characters)
└── README.md          # This documentation
```

## Requirements

### Python Dependencies
- `kokoro>=0.7.13` - Core TTS engine
- `gradio` - Web interface framework  
- `pip` - Package management (listed in requirements.txt)
- Additional dependencies (torch, etc.) resolved automatically by kokoro package

### System Dependencies
- `espeak-ng` - Phoneme processing backend

## Usage

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python app.py
```

The application will start on `http://localhost:40001` with full API access enabled.

### Hugging Face Spaces Deployment
The demo is configured for Hugging Face Spaces deployment with the following settings:

- **Runtime**: Gradio 5.12.0
- **Hardware**: ZeroGPU support with CPU fallback
- **API**: Public API endpoints available
- **Caching**: Voice model preloading for performance

## API Integration

### Prediction Endpoint
```python
def predict(text, voice='af_heart', speed=1):
    """
    Generate speech from text using specified voice.
    Returns audio tuple (sample_rate, audio_array) - 24kHz audio data.
    """
```

### Tokenization Endpoint  
```python
def tokenize_first(text, voice='af_heart'):
    """
    Convert text to phonetic tokens without synthesis.
    Returns phonetic token string for inspection.
    """
```

## Performance Characteristics

### Hardware Acceleration
- **ZeroGPU**: ~2-3x faster synthesis with usage quotas
- **CPU**: Consistent performance, no usage limits
- **Automatic fallback**: GPU errors automatically retry on CPU

### Memory Usage
- **Model loading**: ~500MB per voice (cached after first use)
- **Synthesis**: ~50-100MB working memory per request
- **Concurrent users**: Gradio queue manages resource contention

### Quality Settings
- **Sample rate**: 24kHz for all outputs
- **Bit depth**: 32-bit float internally, 16-bit output
- **Latency**: ~500ms-2s depending on text length and hardware

## Content Examples

### Random Quotes (`en.txt`)
2,122 inspirational and philosophical quotes for quick testing:
- Buddha and mindfulness teachings
- Business and entrepreneurship wisdom  
- Personal development insights
- Scientific and philosophical observations

### Literary Excerpts
- **Gatsby excerpt**: Classic American literature sample (~5k chars)
- **Frankenstein excerpt**: Gothic literature sample (~5k chars)

Perfect for testing longer synthesis and streaming capabilities.

## Development Notes

### Voice Loading Strategy
All voices are preloaded on startup to minimize first-synthesis latency:
```python
for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)
```

### Error Handling
- GPU failures automatically fallback to CPU
- User-friendly warnings for quota limitations
- Graceful degradation maintains functionality

### Extensibility
- Easy voice addition via `CHOICES` dictionary
- Pluggable content via text file loading
- Configurable hardware preferences

## Troubleshooting

### Common Issues
1. **No audio on first stream**: Known Gradio streaming bug, click again
2. **GPU quota exceeded**: Switch to CPU mode in hardware dropdown  
3. **Slow synthesis**: Check hardware selection and system resources
4. **Config.MAX_TOKEN_LENGTH error**: The demo references an undefined Config variable (line 137) - this may cause display issues in the tokenization interface

### Performance Optimization
- Use GPU mode when available for faster synthesis
- Shorter text generates faster than long passages
- Preloaded voices avoid cold-start delays

---

**Note**: This demo showcases Kokoro TTS v1.0 capabilities. For production integration, consider using the direct Python API rather than the Gradio interface for optimal performance.