from .model import KModel
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from misaki import en, espeak
from typing import Callable, Generator, List, Optional, Tuple, Union
import re
import torch
import os

ALIASES = {
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j',
    'zh': 'z',
}

LANG_CODES = dict(
    # pip install misaki[en]
    a='American English',
    b='British English',

    # espeak-ng
    e='es',
    f='fr-fr',
    h='hi',
    i='it',
    p='pt-br',

    # pip install misaki[ja]
    j='Japanese',

    # pip install misaki[zh]
    z='Mandarin Chinese',
)

class KPipeline:
    '''
    KPipeline is a language-aware support class with 2 main responsibilities:
    1. Perform language-specific G2P, mapping (and chunking) text -> phonemes
    2. Manage and store voices, lazily downloaded from HF if needed

    You are expected to have one KPipeline per language. If you have multiple
    KPipelines, you should reuse one KModel instance across all of them.

    KPipeline is designed to work with a KModel, but this is not required.
    There are 2 ways to pass an existing model into a pipeline:
    1. On init: us_pipeline = KPipeline(lang_code='a', model=model)
    2. On call: us_pipeline(text, voice, model=model)

    By default, KPipeline will automatically initialize its own KModel. To
    suppress this, construct a "quiet" KPipeline with model=False.

    A "quiet" KPipeline yields (graphemes, phonemes, None) without generating
    any audio. You can use this to phonemize and chunk your text in advance.

    A "loud" KPipeline _with_ a model yields (graphemes, phonemes, audio).
    '''
    def __init__(
        self,
        lang_code: str,
        repo_id: Optional[str] = None,
        model: Union[KModel, bool] = True,
        trf: bool = False,
        en_callable: Optional[Callable[[str], str]] = None,
        device: Optional[str] = None
    ):
        """Initialize a KPipeline.
        
        Args:
            lang_code: Language code for G2P processing
            model: KModel instance, True to create new model, False for no model
            trf: Whether to use transformer-based G2P
            device: Override default device selection ('cuda' or 'cpu', or None for auto)
                   If None, will auto-select cuda if available
                   If 'cuda' and not available, will explicitly raise an error
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

    def load_single_voice(self, voice: str) -> torch.FloatTensor:
        """Load a single voice embedding from Hugging Face Hub or local path.
        
        This method handles the complete voice loading pipeline:
        1. Checks local cache for previously loaded voices
        2. Downloads from HF Hub if not cached and voice doesn't end with .pt
        3. Validates language compatibility and warns on mismatches
        4. Loads PyTorch tensor with weights_only=True for security
        
        Voice Naming Convention:
            Format: {lang_code}{gender}_{name}
            Examples: af_bella, am_adam, bf_alice
            Where: a=American English, f=female, m=male, b=British English
        
        Called by:
            - load_voice(): For single voice requests or voice averaging
            - Internal voice management during pipeline operations
        
        Cross-file Dependencies:
            - Relies on HF Hub download from huggingface_hub library
            - Uses LANG_CODES mapping defined at module level (lines 23-40)
            - Voice files stored at {repo_id}/voices/{voice}.pt on HF Hub
        
        Args:
            voice: Voice identifier (e.g., 'af_bella') or local .pt file path
        
        Returns:
            Voice embedding tensor with shape (256,) containing baseline and style components
            
        Raises:
            FileNotFoundError: If local .pt file path doesn't exist
            HFValidationError: If voice file not found on Hugging Face Hub
        """
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

    def load_voice(self, voice: Union[str, torch.FloatTensor], delimiter: str = ",") -> torch.FloatTensor:
        """Load voice embedding(s) with support for voice blending via averaging.
        
        This method provides flexible voice loading capabilities:
        1. Pass-through for pre-loaded voice tensors (no processing needed)
        2. Single voice loading from identifier or file path
        3. Multi-voice blending by averaging multiple voice embeddings
        
        Multi-Voice Blending:
            - Voices separated by delimiter (default comma: 'af_bella,af_jessica')
            - All voices loaded individually via load_single_voice()
            - Final embedding computed as arithmetic mean of all voices
            - Useful for creating custom voice styles or gender blending
        
        Called by:
            - generate_from_tokens(): Voice preparation before inference
            - __call__(): Main pipeline voice loading
            - External code needing voice embedding preparation
        
        Cross-file Dependencies:
            - Delegates to load_single_voice() for individual voice downloads
            - Uses torch.mean() and torch.stack() for voice averaging
            - Results cached in self.voices dictionary for performance
        
        Args:
            voice: Single voice ID, comma-separated voice IDs, file path, or pre-loaded tensor
            delimiter: Character used to separate multiple voice identifiers (default: ",")
            
        Returns:
            Voice embedding tensor with shape (256,) ready for model inference
            
        Examples:
            Single voice: pipeline.load_voice('af_bella')
            Voice blending: pipeline.load_voice('af_bella,af_jessica')
            Custom delimiter: pipeline.load_voice('af_bella;af_jessica', delimiter=';')
        """
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
        """Convert tokenized text to phoneme string suitable for TTS model input.
        
        This method transforms a list of misaki MToken objects into a continuous
        phoneme string that can be fed directly to the Kokoro TTS model. It handles
        phoneme concatenation and whitespace preservation to maintain natural speech
        rhythm and timing.
        
        Processing Logic:
            1. Extract phonemes field from each MToken
            2. Add space after phonemes if original token had trailing whitespace
            3. Join all phonemes into continuous string
            4. Strip leading/trailing spaces from final result
        
        Called by:
            - en_tokenize(): During English text chunking and processing
            - generate_from_tokens(): When processing pre-tokenized input
            - Internal text processing workflows
        
        Cross-file Dependencies:
            - Operates on en.MToken objects from misaki.en module
            - Output consumed by KModel.forward() in model.py
            - String length validated against model context_length (512 tokens)
        
        Args:
            tokens: List of MToken objects from misaki English G2P processing
            
        Returns:
            Phoneme string ready for TTS model input (e.g., "hˈɛloʊ wˈɜːld")
            
        Performance Notes:
            - Uses generator expression with str.join() for memory efficiency
            - Output length should not exceed 510 characters for model compatibility
        """
        return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()

    @staticmethod
    def waterfall_last(
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '"']
    ) -> int:
        """Find optimal text chunking boundary using hierarchical punctuation priority.
        
        This method implements intelligent text chunking by finding the best place to split
        text based on punctuation hierarchy. It ensures chunks stay within model limits
        while maintaining natural linguistic boundaries for better TTS quality.
        
        Chunking Strategy:
            1. Search for sentence endings (!.?…) - highest priority
            2. Fall back to clause boundaries (:;) - medium priority  
            3. Use comma/dash breaks (,—) - lowest priority
            4. Handle closing punctuation bumps ()", ')
            5. Ensure result fits within 510-character phoneme limit
        
        Algorithm Flow:
            - Iterate through waterfall priorities (sentence → clause → comma)
            - For each priority level, search backward from end of token list
            - Find last occurrence of punctuation from current priority group
            - Check if resulting chunk would fit within phoneme budget
            - Apply "bumps" adjustment for closing punctuation
            - Return first valid boundary found, or full length if none work
        
        Called by:
            - en_tokenize(): When phoneme count would exceed model limit (510 chars)
            - Text chunking logic during long text processing
        
        Cross-file Dependencies:
            - Operates on en.MToken objects from misaki.en module
            - Phoneme length calculated via tokens_to_ps() method
            - 510-character limit corresponds to model.context_length in model.py
        
        Args:
            tokens: List of MToken objects to find chunking boundary within
            next_count: Total phoneme count including next token to be processed
            waterfall: Punctuation priority groups, highest to lowest importance
            bumps: Closing punctuation that should be included after main boundary
            
        Returns:
            Index of optimal chunk boundary (0 to len(tokens))
            
        Constants Explained:
            - 510: Maximum phoneme string length for model (512 - 2 for BOS/EOS tokens)
            - Waterfall priorities: sentence > clause > phrase boundaries
            - Bumps handle cases like: "Hello." → "Hello.)" (include closing quote)
        """
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
        """Reconstruct original text from tokenized MToken objects.
        
        This method reverses the tokenization process by concatenating the original
        text and whitespace from each MToken to recreate the source text exactly
        as it appeared before G2P processing.
        
        Reconstruction Process:
            1. Extract original text from each MToken.text field
            2. Append whitespace from MToken.whitespace field  
            3. Join all components to form continuous string
            4. Strip leading/trailing whitespace from final result
        
        Called by:
            - en_tokenize(): To generate grapheme output for each text chunk
            - Text processing workflows that need original text reconstruction
            - Debugging and validation code that compares input/output text
        
        Cross-file Dependencies:
            - Operates on en.MToken objects from misaki.en module
            - Used alongside tokens_to_ps() for parallel text/phoneme generation
            - Output used in Result.graphemes field for client code
        
        Args:
            tokens: List of MToken objects containing original text and spacing
            
        Returns:
            Reconstructed original text string with proper spacing preserved
            
        Invariant:
            For any text T processed through misaki G2P → tokens_to_text(tokens) ≈ T
        """
        return ''.join(t.text + t.whitespace for t in tokens).strip()

    def en_tokenize(
        self,
        tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        """Intelligently chunk English text tokens to fit within model context limits.
        
        This method processes pre-tokenized English text and splits it into chunks that
        respect both linguistic boundaries and the TTS model's 510-character phoneme limit.
        It implements sophisticated chunking logic to maintain natural speech patterns.
        
        Processing Pipeline:
            1. Iterate through tokens, building phoneme count incrementally
            2. When approaching 510-character limit, find optimal split using waterfall_last()
            3. Yield text chunk as (graphemes, phonemes, tokens) tuple
            4. Continue processing remaining tokens until all are consumed
        
        Phoneme Processing:
            - Filters None phonemes from G2P processing  
            - Preserves whitespace for natural speech timing
            - Applies optional phoneme transformations (e.g., ɾ → T for American English)
        
        Chunking Intelligence:
            - Uses waterfall_last() to find sentence/clause/phrase boundaries
            - Avoids splitting mid-word or mid-phrase when possible
            - Handles edge case where single token exceeds limit (strips leading spaces)
        
        Called by:
            - generate_from_tokens(): When processing pre-tokenized MToken lists
            - __call__(): During main pipeline English text processing
        
        Cross-file Dependencies:
            - Uses waterfall_last() for intelligent boundary detection
            - Uses tokens_to_ps() and tokens_to_text() for conversion
            - Operates on en.MToken objects from misaki.en module
            - Phoneme output fed to KModel.forward() in model.py
        
        Args:
            tokens: Pre-tokenized English text as list of MToken objects
            
        Yields:
            Tuple of (original_text, phonemes, token_chunk) for each chunk
            
        Constants:
            - 510: Maximum phoneme characters (model context_length 512 - 2 for special tokens)
        """
        tks = []
        pcount = 0
        for t in tokens:
            # American English: ɾ => T
            t.phonemes = '' if t.phonemes is None else t.phonemes#.replace('ɾ', 'T')
            next_ps = t.phonemes + (' ' if t.whitespace else '')
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > 510:
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
        """Execute TTS inference on phoneme string using loaded voice embedding.
        
        This method provides a streamlined interface for running TTS inference by
        wrapping the KModel.forward() call with flexible speed control and proper
        output handling. It's the core inference primitive used throughout the pipeline.
        
        Speed Control Modes:
            1. Fixed speed: Pass float value (e.g., 1.0 for normal, 0.8 for slower)
            2. Dynamic speed: Pass callable that takes phoneme length and returns speed
               Useful for adaptive speed based on text complexity or length
        
        Voice Selection Logic:
            - Uses phoneme length to index into voice embedding pack
            - pack[len(ps)-1] selects appropriate voice variant based on text length
            - This allows voice packs to contain length-dependent style variations
        
        Called by:
            - generate_from_tokens(): For each text chunk during pipeline processing
            - __call__(): During main pipeline inference execution
            - External code needing direct phoneme-to-audio conversion
        
        Cross-file Dependencies:
            - Delegates to KModel.forward() in model.py for actual neural inference
            - Requires voice pack from load_voice() or load_single_voice()
            - Output contains audio tensor and duration predictions for downstream use
        
        Args:
            model: Loaded KModel instance ready for inference
            ps: Phoneme string for TTS synthesis (e.g., "hˈɛloʊ wˈɜːld")
            pack: Voice embedding tensor from voice loading functions
            speed: Fixed speed multiplier or length-dependent speed function
            
        Returns:
            KModel.Output containing synthesized audio and predicted phoneme durations
            
        Performance Notes:
            - Inference time scales with phoneme length and model size
            - Voice pack indexing is O(1) but pack loading may involve disk I/O
        """
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
    def join_timestamps(tokens: List[en.MToken], pred_dur: torch.LongTensor) -> None:
        """Attach precise timing information to MToken objects using model duration predictions.
        
        This method calculates word-level timestamps by correlating the model's predicted
        phoneme durations with the original token boundaries. It enables precise timing
        for applications like subtitles, lip-syncing, and audio visualization.
        
        Duration Conversion Logic:
            - Model outputs durations in 40-fps frames (25ms per frame at 24kHz audio)
            - Each frame represents 600 samples at 24kHz sample rate
            - Half-frame precision used for space character timing (12.5ms resolution)
            - MAGIC_DIVISOR = 80 converts from half-frames to seconds: (frames / 40) / 2
        
        Timestamp Calculation Algorithm:
            1. Initialize left/right boundaries with BOS token duration offset
            2. For each token with phonemes, map to corresponding duration predictions
            3. Calculate token start_ts from current left boundary
            4. Advance boundaries by token duration + space duration
            5. Set token end_ts and update boundaries for next iteration
            6. Handle space characters by splitting duration between adjacent tokens
        
        Timing Precision:
            - start_ts/end_ts stored in seconds with ~12.5ms precision
            - Accounts for model's BOS/EOS tokens in duration sequence alignment
            - Handles variable-length phoneme sequences per token gracefully
        
        Called by:
            - generate_from_tokens(): After successful TTS inference with duration output
            - __call__(): During main pipeline processing when pred_dur is available
        
        Cross-file Dependencies:
            - Modifies en.MToken objects from misaki.en module in-place
            - Uses pred_dur output from KModel.forward_with_tokens() in model.py
            - Timestamp precision tied to model's 40-fps duration prediction rate
        
        Args:
            tokens: List of MToken objects to annotate with timing information
            pred_dur: Duration predictions from model, shape (sequence_length,)
            
        Side Effects:
            Modifies tokens in-place, setting start_ts and end_ts attributes
            
        Constants:
            - MAGIC_DIVISOR = 80: Converts half-frames to seconds (frames/40/2)
            - 40-fps: Model's internal frame rate for duration predictions
            - 24kHz: Audio sample rate, 600 samples per 40-fps frame
        """
        # Duration conversion: 40-fps frames → half-frames → seconds
        # Model predicts in 40-fps frames (600 samples at 24kHz = 25ms)
        # Half-frame precision allows fine-grained space timing
        MAGIC_DIVISOR = 80  # (frames / 40) / 2 = seconds
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
                
                # Text processing constants for non-English chunking
                class ChunkingConstants:
                    """Constants for intelligent text chunking in non-English languages."""
                    
                    # Maximum character count per chunk before forced splitting
                    # Chosen to balance processing efficiency with natural language boundaries
                    # Value accounts for average phoneme expansion ratio across languages
                    MAX_CHUNK_SIZE = 400
                
                CHUNK_SIZE = ChunkingConstants.MAX_CHUNK_SIZE
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
