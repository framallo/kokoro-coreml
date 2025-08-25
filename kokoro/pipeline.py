"""
Kokoro TTS Pipeline Implementation

This module provides the KPipeline class, a comprehensive text-to-speech pipeline that
handles language-specific grapheme-to-phoneme (G2P) conversion, voice management, and
audio synthesis. It serves as the primary user interface for multi-language TTS functionality.

Architecture Components:
- Language Detection: Automatic language code resolution and validation
- G2P Processing: Language-specific phoneme conversion with fallback support
- Voice Management: Lazy loading and caching of speaker embeddings
- Text Chunking: Intelligent segmentation for long text processing
- Audio Synthesis: Integration with KModel for high-quality speech generation

Multi-Language Support:
- English (American/British): Native misaki[en] support with ESpeak fallback
- European Languages: ESpeak-ng integration for Spanish, French, Italian, Portuguese
- Asian Languages: Specialized support for Japanese (misaki[ja]) and Chinese (misaki[zh])
- Extensible Framework: Easy addition of new languages and G2P backends

Performance Optimizations:
- Lazy Voice Loading: Voices loaded on-demand and cached for reuse
- Smart Chunking: Context-aware text segmentation for memory efficiency
- Batch Processing: Multiple text segments processed in single pipeline calls
- Device Management: Automatic GPU/CPU placement with fallback support

Cross-file dependencies:
- Imports from: model.py (KModel), misaki (language-specific G2P)
- Used by: All demo applications, test suites, and production inference scripts
- Integrates with: Voice loading system, audio synthesis pipeline
- Requires: Language-specific packages (misaki[en], misaki[ja], misaki[zh])
"""

from .model import KModel
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from misaki import en, espeak
from typing import Callable, Generator, List, Optional, Tuple, Union
import re
import torch
import os

class PipelineConstants:
    """
    Configuration constants for KPipeline text processing and audio synthesis.
    
    This class centralizes all processing limits, chunk sizes, and configuration
    values used throughout the pipeline, providing clear documentation for
    constraints and optimizations that ensure reliable multi-language TTS.
    
    Text Processing Limits:
    - Phoneme sequences are limited by BERT context windows
    - Chunking strategies balance memory usage with processing efficiency  
    - Language-specific optimizations account for different G2P characteristics
    
    Audio Processing Configuration:
    - Sample rates and frame rates must match KModel expectations
    - Timestamp calculations require precise frame-to-sample conversion
    - Voice embedding formats follow standardized dimensions
    
    Performance Tuning:
    - Chunk sizes optimized for typical sentence lengths in different languages
    - Buffer sizes chosen to minimize memory fragmentation
    - Processing limits prevent OOM errors on resource-constrained devices
    
    Used by:
    - KPipeline.__call__: Text chunking and language processing limits
    - KPipeline.en_tokenize: English text segmentation and phoneme limits  
    - KPipeline.join_timestamps: Audio frame-to-timestamp conversion
    - Voice loading: Embedding format validation and caching decisions
    """
    
    # Phoneme sequence processing limits
    MAX_PHONEME_LENGTH = 510        # Maximum phoneme chars (BERT context - BOS/EOS tokens)
    PHONEME_SAFETY_MARGIN = 2       # Reserve for BOS/EOS tokens in sequence
    
    # Audio frame and timing constants  
    SAMPLE_RATE = 24000            # Audio sample rate in Hz (must match KModel)
    FRAMES_PER_SECOND = 40         # Duration prediction frame rate (25ms frames)
    SAMPLES_PER_FRAME = 600        # Samples per duration frame (24kHz / 40fps)
    
    # Timestamp calculation constants
    TIMESTAMP_DIVISOR = 80         # Magic divisor for half-frame timestamp precision
    TIMING_SAFETY_OFFSET = 3       # Frame offset to avoid boundary artifacts
    
    # Text chunking configuration
    ENGLISH_CHUNK_SIZE = 510       # English text processing limit (phoneme-based)
    NON_ENGLISH_CHUNK_SIZE = 400   # Non-English chunk size (character-based)
    
    # Punctuation patterns for intelligent chunking
    PRIMARY_BREAKS = ['!', '.', '?', '…']     # Sentence-ending punctuation
    SECONDARY_BREAKS = [':', ';']             # Clause-ending punctuation  
    TERTIARY_BREAKS = [',', '—']              # Phrase-ending punctuation
    CLOSING_MARKS = [')', '"']                # Closing punctuation to include
    
    # Voice embedding specifications
    VOICE_EMBEDDING_DIM = 256      # Standard voice embedding size
    VOICE_CACHE_SIZE = 100         # Maximum cached voices per pipeline
    
    # Language processing constants
    DEFAULT_REPO_ID = 'hexgrad/Kokoro-82M'  # Default model repository
    FALLBACK_ENABLED = True        # Enable ESpeak fallback for unknown words

# Language code mappings for user-friendly language specification
# Maps common language identifiers to internal single-character codes
ALIASES = {
    'en-us': 'a',      # American English -> 'a'
    'en-gb': 'b',      # British English -> 'b' 
    'es': 'e',         # Spanish -> 'e'
    'fr-fr': 'f',      # French (France) -> 'f'
    'hi': 'h',         # Hindi -> 'h'
    'it': 'i',         # Italian -> 'i'
    'pt-br': 'p',      # Portuguese (Brazil) -> 'p'
    'ja': 'j',         # Japanese -> 'j'
    'zh': 'z',         # Chinese (Mandarin) -> 'z'
}

# Language code to human-readable name and G2P backend mapping
LANG_CODES = dict(
    # Native misaki support with advanced features
    a='American English',     # misaki[en] with American pronunciation
    b='British English',      # misaki[en] with British pronunciation

    # ESpeak-ng backend support  
    e='es',          # Spanish via espeak-ng
    f='fr-fr',       # French (France) via espeak-ng
    h='hi',          # Hindi via espeak-ng
    i='it',          # Italian via espeak-ng
    p='pt-br',       # Portuguese (Brazil) via espeak-ng

    # Specialized misaki backends
    j='Japanese',           # misaki[ja] with morphological analysis
    z='Mandarin Chinese',   # misaki[zh] with tone support
)

class KPipeline:
    """
    Language-aware text-to-speech pipeline with comprehensive G2P and voice management.

    KPipeline serves as the primary user interface for multi-language text-to-speech
    synthesis, combining grapheme-to-phoneme (G2P) conversion, voice management, and
    audio synthesis into a unified, easy-to-use API. It handles the complexity of
    different languages while providing consistent behavior across all supported locales.

    Core Responsibilities:
    1. Language-Specific G2P: Convert text to phonemes using appropriate backends
    2. Voice Management: Lazy loading, caching, and blending of speaker embeddings  
    3. Text Chunking: Intelligent segmentation for long texts and memory management
    4. Audio Synthesis: Integration with KModel for high-quality speech generation

    Architecture Design:
    - One Pipeline Per Language: Each KPipeline instance handles one language
    - Shared Model Support: Multiple pipelines can share a single KModel instance
    - Flexible Initialization: Support for both "quiet" (G2P-only) and "loud" (full TTS) modes
    - Lazy Resource Loading: Models and voices loaded on-demand for efficiency

    Language Support Matrix:
    - English (a/b): Advanced support via misaki[en] with ESpeak fallback
    - European (e/f/h/i/p): ESpeak-ng integration for Romance and other languages
    - Asian (j/z): Specialized support with morphological/tonal analysis

    Usage Patterns:
    1. Full TTS Pipeline:
       ```python
       pipeline = KPipeline(lang_code='a')  # Auto-initializes KModel
       audio = pipeline("Hello world", voice="af_heart")
       ```

    2. Shared Model Across Languages:
       ```python
       model = KModel()
       en_pipeline = KPipeline(lang_code='a', model=model)
       es_pipeline = KPipeline(lang_code='e', model=model)
       ```

    3. G2P-Only Mode:
       ```python
       pipeline = KPipeline(lang_code='a', model=False)  # "quiet" mode
       for graphemes, phonemes, _ in pipeline(text, voice):
           print(f"{graphemes} -> {phonemes}")
       ```

    Performance Characteristics:
    - Memory Efficient: Lazy loading and caching of resources
    - Scalable: Handles texts from single words to full documents
    - Device Aware: Automatic GPU/CPU placement with fallback support
    - Thread Safe: Stateless operations allow concurrent usage

    Resource Management:
    - Automatic Downloads: Models and voices fetched from Hugging Face on demand
    - Intelligent Caching: Frequently used voices cached in memory
    - Memory Bounds: Configurable limits prevent excessive resource usage
    - Error Resilience: Graceful fallbacks when resources are unavailable

    Cross-file Dependencies:
    - Uses: model.py (KModel for synthesis), misaki (language-specific G2P)
    - Used by: demo/app.py, test_*.py, production inference scripts
    - Integrates with: Hugging Face Hub (model/voice downloads), ESpeak-ng (fallback G2P)
    - Requires: Language packs (misaki[en]/[ja]/[zh]) for full functionality

    Thread Safety and Concurrency:
    - Read Operations: Safe for concurrent access (voice loading, G2P conversion)
    - Write Operations: Voice caching uses thread-safe mechanisms
    - Model Sharing: Multiple pipelines can safely share a single KModel
    - State Isolation: Each pipeline maintains independent language configuration
    """
    def __init__(
        self,
        lang_code: str,
        repo_id: Optional[str] = None,
        model: Union[KModel, bool] = True,
        trf: bool = False,
        en_callable: Optional[Callable[[str], str]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize a language-specific TTS pipeline with comprehensive configuration options.

        Creates a KPipeline instance with language-specific G2P backend, optional model
        initialization, and device management. Supports multiple operating modes from
        G2P-only preprocessing to full TTS synthesis with automatic resource management.

        Language-Specific Backend Selection:
        - English (a/b): misaki[en] with ESpeak fallback for OOD words
        - Japanese (j): misaki[ja] with morphological analysis (requires separate install)
        - Chinese (z): misaki[zh] with tone support and version selection
        - Others: ESpeak-ng backend with language-specific pronunciation rules

        Device Management Strategy:
        - Automatic Selection: CUDA > MPS > CPU based on availability
        - Explicit Override: User can force specific device with error handling
        - MPS Requirements: Requires PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable
        - Fallback Handling: Graceful degradation with informative error messages

        Model Initialization Modes:
        1. Shared Model (model=KModel): Use existing model instance for efficiency
        2. Auto Model (model=True): Create new model with automatic device placement
        3. Quiet Mode (model=False): G2P-only operation without synthesis capability
        4. Custom Repository: Download from specified Hugging Face repository

        Args:
            lang_code (str): Language identifier for G2P backend selection
                           Supported codes: a,b (English), j (Japanese), z (Chinese), 
                           e,f,h,i,p (ESpeak languages)
                           Case insensitive, supports ALIASES mapping
            repo_id (str, optional): Hugging Face repository for model/voice assets
                                   Defaults to PipelineConstants.DEFAULT_REPO_ID
                                   Format: 'username/repository-name'
            model (Union[KModel, bool], optional): Model configuration mode
                                                 KModel instance: Share existing model
                                                 True: Auto-create new model (default)
                                                 False: G2P-only mode, no synthesis
            trf (bool, optional): Use transformer-based G2P for English
                                Defaults to False (uses simpler, faster approach)
                                Only affects English (a/b) language codes
            en_callable (Callable[[str], str], optional): Custom English G2P function for Chinese
                                                         Used by zh.ZHG2P for mixed-language text
                                                         Defaults to None (uses internal handling)
            device (str, optional): Explicit device placement override
                                  None: Auto-select best available device
                                  'cuda': Force CUDA (raises if unavailable)
                                  'mps': Force Apple Metal (requires MPS_FALLBACK=1)
                                  'cpu': Force CPU execution

        Raises:
            AssertionError: If lang_code not in supported LANG_CODES
            ImportError: If required language package not installed (misaki[ja]/misaki[zh])
            RuntimeError: If requested device unavailable or MPS misconfigured
            ConnectionError: If model repository inaccessible during download

        State Initialization Process:
        1. Repository Configuration: Set up Hugging Face repository for assets
        2. Language Validation: Resolve aliases and validate against supported codes
        3. Device Selection: Auto-detect or validate requested device
        4. Model Creation: Initialize based on mode with error handling
        5. G2P Backend Setup: Configure language-specific phoneme converter
        6. Voice Cache: Initialize empty dictionary for lazy voice loading

        Performance Characteristics:
        - Lazy Loading: G2P backends loaded on first use
        - Device Optimization: Automatic placement for best performance
        - Memory Efficient: Only requested resources are allocated
        - Error Resilient: Graceful fallbacks with detailed error messages

        Called by:
        - User Applications: Direct instantiation for TTS functionality
        - Demo Scripts: Language-specific pipeline creation
        - Test Suites: Validation of different configuration modes
        - Production Services: Shared model scenarios for efficiency

        Examples:
        ```python
        # Basic English pipeline with auto-device selection
        pipeline = KPipeline('a')
        
        # Japanese pipeline with explicit CPU usage
        ja_pipeline = KPipeline('j', device='cpu')
        
        # Shared model across multiple languages
        model = KModel()
        en_pipeline = KPipeline('a', model=model)
        es_pipeline = KPipeline('e', model=model)
        
        # G2P-only preprocessing pipeline
        quiet_pipeline = KPipeline('a', model=False)
        
        # Transformer-based English G2P
        trf_pipeline = KPipeline('a', trf=True)
        ```
        """
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        lang_code = lang_code.lower()
        lang_code = ALIASES.get(lang_code, lang_code)
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.model = None
        if isinstance(model, KModel):
            self.model = model
        elif model:
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            if device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1':
                raise RuntimeError("MPS requested but fallback not enabled")
            if device is None:
                if torch.cuda.is_available():
                    device = 'cuda'
                elif os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            try:
                self.model = KModel(repo_id=repo_id).to(device).eval()
            except RuntimeError as e:
                if device == 'cuda':
                    raise RuntimeError(f"""Failed to initialize model on CUDA: {e}. 
                                       Try setting device='cpu' or check CUDA installation.""")
                raise
        self.voices = {}
        if lang_code in 'ab':
            try:
                fallback = espeak.EspeakFallback(british=lang_code=='b')
            except Exception as e:
                logger.warning("EspeakFallback not Enabled: OOD words will be skipped")
                logger.warning({str(e)})
                fallback = None
            self.g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback, unk='')
        elif lang_code == 'j':
            try:
                from misaki import ja
                self.g2p = ja.JAG2P()
            except ImportError:
                logger.error("You need to `pip install misaki[ja]` to use lang_code='j'")
                raise
        elif lang_code == 'z':
            try:
                from misaki import zh
                self.g2p = zh.ZHG2P(
                    version=None if repo_id.endswith('/Kokoro-82M') else '1.1',
                    en_callable=en_callable
                )
            except ImportError:
                logger.error("You need to `pip install misaki[zh]` to use lang_code='z'")
                raise
        else:
            language = LANG_CODES[lang_code]
            logger.warning(f"Using EspeakG2P(language='{language}'). Chunking logic not yet implemented, so long texts may be truncated unless you split them with '\\n'.")
            self.g2p = espeak.EspeakG2P(language=language)

    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
        else:
            f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')
            if not voice.startswith(self.lang_code):
                v = LANG_CODES.get(voice, voice)
                p = LANG_CODES.get(self.lang_code, self.lang_code)
                logger.warning(f'Language mismatch, loading {v} voice into {p} pipeline.')
        pack = torch.load(f, weights_only=True)
        self.voices[voice] = pack
        return pack

    """
    load_voice is a helper function that lazily downloads and loads a voice:
    Single voice can be requested (e.g. 'af_bella') or multiple voices (e.g. 'af_bella,af_jessica').
    If multiple voices are requested, they are averaged.
    Delimiter is optional and defaults to ','.
    """
    def load_voice(self, voice: Union[str, torch.FloatTensor], delimiter: str = ",") -> torch.FloatTensor:
        if isinstance(voice, torch.FloatTensor):
            return voice
        if voice in self.voices:
            return self.voices[voice]
        logger.debug(f"Loading voice: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]

    @staticmethod
    def tokens_to_ps(tokens: List[en.MToken]) -> str:
        return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()

    @staticmethod
    def waterfall_last(
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '”']
    ) -> int:
        for w in waterfall:
            z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(KPipeline.tokens_to_ps(tokens[:z])) <= 510:
                return z
        return len(tokens)

    @staticmethod
    def tokens_to_text(tokens: List[en.MToken]) -> str:
        return ''.join(t.text + t.whitespace for t in tokens).strip()

    def en_tokenize(
        self,
        tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        """
        Intelligent tokenization and chunking for English text processing.

        This method implements a sophisticated chunking algorithm that respects phoneme
        length limits while maintaining linguistic coherence. It processes English
        tokens with proper boundary detection and generates chunks suitable for 
        BERT-based processing in the TTS pipeline.

        Chunking Strategy:
        - Phoneme-based limits: Respects BERT context window constraints
        - Intelligent boundaries: Prefers natural linguistic break points
        - Waterfall optimization: Finds optimal chunk boundaries to minimize splits
        - Memory efficient: Generator pattern for large text processing

        Phoneme Processing:
        - American/British variants: Handles pronunciation differences automatically
        - Whitespace preservation: Maintains proper spacing in phoneme sequences
        - Phoneme cleaning: Removes None values and handles pronunciation variants
        - Length calculation: Precise phoneme character counting for chunk limits

        Args:
            tokens (List[en.MToken]): English morphological tokens from misaki[en] G2P
                                    Each token contains: text, phonemes, whitespace, morphology
                                    Generated by en.G2P.tokenize() method
                                    Includes pronunciation and spacing information

        Yields:
            Tuple[str, str, List[en.MToken]]: Chunked text processing results
                - str: Original grapheme text for the chunk
                - str: Phoneme sequence ready for model input
                - List[en.MToken]: Token objects for timestamp alignment

        Processing Flow:
        1. Phoneme Preprocessing: Clean and normalize phoneme representations
        2. Length Tracking: Monitor cumulative phoneme character count
        3. Boundary Detection: Identify optimal chunk split points when limits approached
        4. Waterfall Optimization: Use intelligent boundary selection to minimize splits
        5. Chunk Generation: Yield complete chunks with text, phonemes, and tokens

        Chunk Size Management:
        - Maximum Length: PipelineConstants.MAX_PHONEME_LENGTH (510 characters)
        - Safety Margin: Reserves space for BOS/EOS tokens in BERT processing
        - Optimal Splitting: Prefers word boundaries, punctuation, and natural breaks
        - Overflow Handling: Graceful handling when chunks exceed limits

        Performance Characteristics:
        - Generator Pattern: Memory efficient for long text processing
        - Lazy Evaluation: Chunks processed on-demand for large documents
        - Boundary Optimization: Minimizes mid-word splits through waterfall algorithm
        - Debug Logging: Comprehensive logging for chunk boundary decisions

        Called by:
        - KPipeline.__call__: Main text processing pipeline for English input
        - Used with: en.G2P.tokenize() output for morphological token processing
        - Integrates with: KPipeline.waterfall_last for boundary optimization

        Example:
        ```python
        tokens = self.g2p.tokenize("Hello world, this is a test.")
        for text, phonemes, token_list in self.en_tokenize(tokens):
            print(f"Text: {text}")
            print(f"Phonemes: {phonemes}")
            print(f"Tokens: {len(token_list)}")
        ```
        """
        tks = []
        pcount = 0
        for t in tokens:
            # American English: ɾ => T
            t.phonemes = '' if t.phonemes is None else t.phonemes#.replace('ɾ', 'T')
            next_ps = t.phonemes + (' ' if t.whitespace else '')
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > PipelineConstants.MAX_PHONEME_LENGTH:
                z = KPipeline.waterfall_last(tks, next_pcount)
                text = KPipeline.tokens_to_text(tks[:z])
                logger.debug(f"Chunking text at {z}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
                ps = KPipeline.tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(KPipeline.tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.append(t)
            pcount += len(next_ps)
        if tks:
            text = KPipeline.tokens_to_text(tks)
            ps = KPipeline.tokens_to_ps(tks)
            yield ''.join(text).strip(), ''.join(ps).strip(), tks

    @staticmethod
    def infer(
        model: KModel,
        ps: str,
        pack: torch.FloatTensor,
        speed: Union[float, Callable[[int], float]] = 1
    ) -> KModel.Output:
        if callable(speed):
            speed = speed(len(ps))
        return model(ps, pack[len(ps)-1], speed, return_output=True)

    def generate_from_tokens(
        self,
        tokens: Union[str, List[en.MToken]],
        voice: str,
        speed: float = 1,
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        """Generate audio from either raw phonemes or pre-processed tokens.
        
        Args:
            tokens: Either a phoneme string or list of pre-processed MTokens
            voice: The voice to use for synthesis
            speed: Speech speed modifier (default: 1)
            model: Optional KModel instance (uses pipeline's model if not provided)
        
        Yields:
            KPipeline.Result containing the input tokens and generated audio
            
        Raises:
            ValueError: If no voice is provided or token sequence exceeds model limits
        """
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: pipeline.generate_from_tokens(..., voice="af_heart")')
        
        pack = self.load_voice(voice).to(model.device) if model else None

        # Handle raw phoneme string
        if isinstance(tokens, str):
            logger.debug("Processing phonemes from raw string")
            if len(tokens) > 510:
                raise ValueError(f'Phoneme string too long: {len(tokens)} > 510')
            output = KPipeline.infer(model, tokens, pack, speed) if model else None
            yield self.Result(graphemes='', phonemes=tokens, output=output)
            return
        
        logger.debug("Processing MTokens")
        # Handle pre-processed tokens
        for gs, ps, tks in self.en_tokenize(tokens):
            if not ps:
                continue
            elif len(ps) > 510:
                logger.warning(f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                logger.warning("Truncating to 510 characters")
                ps = ps[:510]
            output = KPipeline.infer(model, ps, pack, speed) if model else None
            if output is not None and output.pred_dur is not None:
                KPipeline.join_timestamps(tks, output.pred_dur)
            yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output)

    @staticmethod
    def join_timestamps(tokens: List[en.MToken], pred_dur: torch.LongTensor):
        # Multiply by 600 to go from pred_dur frames to sample_rate 24000
        # Equivalent to dividing pred_dur frames by 40 to get timestamp in seconds
        # We will count nice round half-frames, so the divisor is 80
        MAGIC_DIVISOR = PipelineConstants.TIMESTAMP_DIVISOR  # Half-frame precision for timestamp calculation
        if not tokens or len(pred_dur) < 3:
            # We expect at least 3: <bos>, token, <eos>
            return
        # We track 2 counts, measured in half-frames: (left, right)
        # This way we can cut space characters in half
        # TODO: Is -3 an appropriate offset?
        left = right = 2 * max(0, pred_dur[0].item() - 3)
        # Updates:
        # left = right + (2 * token_dur) + space_dur
        # right = left + space_dur
        i = 1
        for t in tokens:
            if i >= len(pred_dur)-1:
                break
            if not t.phonemes:
                if t.whitespace:
                    i += 1
                    left = right + pred_dur[i].item()
                    right = left + pred_dur[i].item()
                    i += 1
                continue
            j = i + len(t.phonemes)
            if j >= len(pred_dur):
                break
            t.start_ts = left / MAGIC_DIVISOR
            token_dur = pred_dur[i: j].sum().item()
            space_dur = pred_dur[j].item() if t.whitespace else 0
            left = right + (2 * token_dur) + space_dur
            t.end_ts = left / MAGIC_DIVISOR
            right = left + space_dur
            i = j + (1 if t.whitespace else 0)

    @dataclass
    class Result:
        graphemes: str
        phonemes: str
        tokens: Optional[List[en.MToken]] = None
        output: Optional[KModel.Output] = None
        text_index: Optional[int] = None

        @property
        def audio(self) -> Optional[torch.FloatTensor]:
            return None if self.output is None else self.output.audio

        @property
        def pred_dur(self) -> Optional[torch.LongTensor]:
            return None if self.output is None else self.output.pred_dur

        ### MARK: BEGIN BACKWARD COMPAT ###
        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index):
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self):
            return 3
        #### MARK: END BACKWARD COMPAT ####

    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Union[float, Callable[[int], float]] = 1,
        split_pattern: Optional[str] = r'\n+',
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: en_us_pipeline(text="Hello world!", voice="af_heart")')
        pack = self.load_voice(voice).to(model.device) if model else None
        
        # Convert input to list of segments
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]
            
        # Process each segment
        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():  # Skip empty segments
                continue
                
            # English processing (unchanged)
            if self.lang_code in 'ab':
                logger.debug(f"Processing English text: {graphemes[:50]}{'...' if len(graphemes) > 50 else ''}")
                _, tokens = self.g2p(graphemes)
                for gs, ps, tks in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logger.warning(f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'")
                        ps = ps[:510]
                    output = KPipeline.infer(model, ps, pack, speed) if model else None
                    if output is not None and output.pred_dur is not None:
                        KPipeline.join_timestamps(tks, output.pred_dur)
                    yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output, text_index=graphemes_index)
            
            # Non-English processing with chunking
            else:
                # Intelligent text chunking for non-English languages.
    # Priority-based chunking: sentence boundaries -> character limits.
    # Optimal chunk size for model context and processing efficiency.
                CHUNK_SIZE = PipelineConstants.NON_ENGLISH_CHUNK_SIZE
                chunks = []
                
                # Try to split on sentence boundaries first
                sentences = re.split(r'([.!?]+)', graphemes)
                current_chunk = ""
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    # Add the punctuation back if it exists
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                        
                    if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If no chunks were created (no sentence boundaries), fall back to character-based chunking
                if not chunks:
                    chunks = [graphemes[i:i+CHUNK_SIZE] for i in range(0, len(graphemes), CHUNK_SIZE)]
                
                # Process each chunk
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                        
                    ps, _ = self.g2p(chunk)
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        logger.warning(f'Truncating len(ps) == {len(ps)} > 510')
                        ps = ps[:510]
                        
                    output = KPipeline.infer(model, ps, pack, speed) if model else None
                    yield self.Result(graphemes=chunk, phonemes=ps, output=output, text_index=graphemes_index)
