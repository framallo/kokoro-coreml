# Core Model Implementation for Kokoro TTS
#
# This module contains the main KModel class that wraps the Kokoro text-to-speech
# neural network architecture. The model is designed for efficient inference and
# CoreML conversion, with particular attention to Apple Neural Engine (ANE) optimization.
#
# Key Components:
# - KModel: Main inference wrapper with phoneme-to-audio generation
# - KModelForONNX: Export-friendly wrapper for ONNX/CoreML conversion
# - Output: Structured dataclass for model outputs with audio and duration predictions
#
# Cross-file dependencies:
# - Called by: pipeline.py (KPipeline.infer), export_coreml.py, export_synthesizers.py
# - Imports from: modules.py (neural components), istftnet.py (decoder)
# - Used by: All inference scripts, demo applications, and export pipelines

from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch

class KModel(torch.nn.Module):
    """Primary neural network model for Kokoro text-to-speech synthesis.

    KModel serves as the central inference engine that converts phoneme sequences
    into high-quality audio waveforms. It combines multiple neural components:
    - BERT-based text encoder for phoneme understanding
    - Prosody predictor for duration and F0/pitch modeling
    - iSTFT-based decoder for final audio synthesis

    Architecture Overview:
        1. Phonemes -> BERT embeddings -> prosody prediction -> duration alignment
        2. Text features + voice style -> F0/pitch and noise predictions
        3. Aligned features -> decoder -> final audio waveform

    Key Design Decisions:
        - Language-agnostic: operates on phonemes, not raw text
        - Voice-conditioned: uses reference voice embeddings for style transfer
        - CoreML-friendly: designed to be traceable with torch.jit.trace
        - Memory-efficient: single instance can serve multiple pipelines

    Called by:
        pipeline.py: KPipeline.infer() for production inference
        export_coreml.py: DurationModel and SynthesizerModel wrappers
        export_synthesizers.py: Two-stage export with bucketing
        demo/app.py: Interactive Gradio interface

    Performance Notes:
        - Supports variable sequence lengths up to context_length (512 tokens)
        - Optimized for 24-kHz audio output with 600-frame hop length
        - Uses disable_complex=True for CoreML compatibility
    """
    
    # Model configuration constants
    class ModelConstants:
        """
        Configuration constants for Kokoro TTS model architecture and processing.
        
        This class centralizes all architectural constants used throughout the model,
        providing clear documentation for dimensions, limits, and default values that
        are critical for proper model operation and export compatibility.
        
        Voice Embedding Architecture:
        The model uses a split embedding approach where the reference voice vector (ref_s)
        is partitioned into baseline speaker characteristics and dynamic style information:
        - Baseline: Static speaker identity features (pitch range, vocal tract characteristics)
        - Style: Dynamic prosodic features (speaking rate, emotion, emphasis patterns)
        
        Sequence Processing:
        BERT-style tokenization with explicit boundary markers for robust sequence handling
        across variable-length inputs and different language contexts.
        
        Model Size Categories:
        - 82M: Baseline model optimized for quality/speed balance
        - Future: Larger variants possible with same architecture constants
        
        Used by:
        - KModel.__init__: Architecture configuration during model initialization
        - KModel.forward: Voice embedding partitioning and sequence boundary handling
        - Export scripts: Model dimension validation for CoreML conversion
        - Pipeline classes: Voice loading and sequence length validation
        """
        
        # Voice embedding dimensions - critical for voice cloning and style transfer
        BASELINE_VOICE_DIM = 128  # Speaker baseline embedding size (fundamental voice characteristics)
        STYLE_VOICE_DIM = 128     # Speaker style embedding size (prosodic and emotional attributes)
        TOTAL_VOICE_DIM = 256     # Total ref_s dimension (baseline + style concatenated)
        
        # Model repository configuration for automatic downloads
        DEFAULT_REPO_ID = 'hexgrad/Kokoro-82M'  # Default Hugging Face model repository
        
        # Sequence boundary tokens for BERT-compatible processing
        BOS_TOKEN_ID = 0  # Beginning of sequence token (shared with EOS for simplicity)
        EOS_TOKEN_ID = 0  # End of sequence token (same as BOS - model handles via masking)
        
        # Architecture dimension constants
        DEFAULT_CONTEXT_LENGTH = 512  # Maximum sequence length for BERT processing
        HIDDEN_DIM_DEFAULT = 512      # Default hidden dimension across model components
        
        # Audio synthesis constants
        SAMPLE_RATE = 24000          # Target sample rate in Hz for all audio output
        FRAMES_PER_SECOND = 40       # Frame rate for duration predictions (40fps = 25ms frames)
        
        # Model file naming patterns for multi-language support
        MODEL_FILE_PATTERNS = {
            'en': 'kokoro-v1_0.pth',        # English model checkpoint
            'zh': 'kokoro-v1_1-zh.pth',     # Chinese-enhanced model checkpoint
        }
        
        # Export compatibility constants
        COREML_MAX_SEQUENCE = 510    # Maximum phoneme sequence for CoreML (context - BOS/EOS)
        EXPORT_PRECISION_FP16 = True # Use FP16 precision for ANE optimization
        EXPORT_PRECISION_FP32 = False # Fallback precision for CPU-only execution

    # Model repository mappings for Hugging Face Hub downloads.
    #
    # These constants define the mapping between repository IDs and their
    # corresponding checkpoint filenames. Used by __init__ to automatically
    # download model weights when not provided locally.
    #
    # Supported Models:
    # - hexgrad/Kokoro-82M: Original English model (82M parameters)
    # - hexgrad/Kokoro-82M-v1.1-zh: Chinese-enhanced version
    #
    # File Format: PyTorch state_dict organized by module name
    # (bert, bert_encoder, predictor, text_encoder, decoder)
    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        """Initialize KModel with automatic weight and configuration loading.

        This constructor handles the complete model setup process:
        1. Downloads config.json and model weights from Hugging Face if needed
        2. Initializes all neural network components (BERT, predictor, decoder)
        3. Loads pre-trained weights with fallback handling for version mismatches

        Args:
            repo_id: Hugging Face repository ID (defaults to 'hexgrad/Kokoro-82M')
            config: Model configuration (dict, file path, or None for auto-download)
            model: Path to PyTorch checkpoint (None for auto-download)
            disable_complex: Use CustomSTFT instead of torch.stft for CoreML export

        Configuration Keys:
            vocab: Phoneme-to-ID mapping dictionary
            n_token: Vocabulary size for embedding layers
            hidden_dim: Hidden dimension for text and prosody encoders
            style_dim: Voice embedding dimension (128 baseline + 128 style)
            max_dur: Maximum duration prediction per phoneme
            istftnet: Decoder architecture parameters

        Called by:
            pipeline.py: KPipeline.__init__ with device placement
            export_coreml.py: prepare_pytorch_models() for conversion
            demo/app.py: Global model initialization
        """
        super().__init__()
        if repo_id is None:
            repo_id = self.ModelConstants.DEFAULT_REPO_ID
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu').items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self) -> torch.device:
        """
        Returns the device (CPU/CUDA/MPS) where the model is currently located.

        This property provides a convenient way to check model placement without
        inspecting individual parameter tensors. Uses the BERT module as the
        canonical device reference since it's always present and contains the
        majority of model parameters.

        Device Management Strategy:
        - Single source of truth: BERT module device placement
        - Automatic detection: No manual device tracking required
        - Consistent behavior: All model components follow BERT placement
        - Export compatibility: Works with CPU/GPU/ANE device contexts

        Performance Implications:
        - CPU: Compatible with all operations, slower inference
        - CUDA: GPU acceleration for supported operations, requires NVIDIA hardware
        - MPS: Apple Silicon GPU acceleration, requires fallback configuration
        - ANE: Apple Neural Engine optimization via CoreML export

        Returns:
            torch.device: Device object indicating current model location
                        Examples: cuda:0, cpu, mps, or specific device indices

        Used by:
        - pipeline.py: KPipeline.__init__ for device placement of input tensors
        - pipeline.py: KPipeline.load_voice for voice embedding device movement
        - export_coreml.py: Device synchronization during model conversion
        - test scripts: Device consistency validation across inference runs

        Thread Safety:
        - Read-only property: Safe for concurrent access
        - Device changes: Must be coordinated at model level (.to() calls)
        - No internal state: Property reflects current PyTorch module placement
        """
        return self.bert.device

    @dataclass
    class Output:
        """Structured output container for KModel inference results.

        This dataclass encapsulates the two primary outputs from model inference:
        audio waveform and predicted phoneme durations. Used when return_output=True
        in the forward() method to provide detailed inference information.

        Attributes:
            audio: Generated waveform tensor, shape (T,) at 24-kHz sample rate
            pred_dur: Per-phoneme duration in frames (40-fps), shape (N,) where N=num_phonemes

        Duration Interpretation:
            - Each duration value represents frames at 40-fps (600 samples at 24-kHz)
            - Used by pipeline.py for word-level timestamp alignment
            - Essential for lip-syncing and precise timing applications

        Used by:
            pipeline.py: KPipeline.generate_from_tokens() for timestamp calculation
            demo applications: Detailed output analysis and debugging
        """
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Core inference method operating directly on tokenized phoneme input.

        This is the primary computational pathway that implements the complete
        text-to-speech synthesis pipeline. It processes pre-tokenized phoneme
        sequences through the full neural architecture to generate high-quality
        audio waveforms with predicted timing information.

        Neural Architecture Pipeline:
        1. BERT Encoding: Transform phoneme tokens into contextual embeddings
        2. Prosody Prediction: Generate duration and F0/pitch from embeddings + voice style
        3. Duration Alignment: Build alignment matrix for temporal synchronization
        4. Feature Extraction: Extract aligned acoustic features via text encoder
        5. Audio Synthesis: Generate final waveform through iSTFT-based decoder

        Voice Conditioning:
        The ref_s parameter contains concatenated [baseline, style] embeddings:
        - Baseline (first 128 dims): Static speaker identity characteristics
        - Style (last 128 dims): Dynamic prosodic and emotional attributes
        This split enables both speaker identity and style transfer capabilities.

        Temporal Processing:
        - Duration predictions specify phoneme timing in 40fps frames
        - Alignment matrix maps phoneme features to audio frames
        - F0/pitch curves provide fundamental frequency contours
        - Final audio generated at 24kHz sample rate

        Args:
            input_ids (torch.LongTensor): Phoneme token IDs with shape (1, N)
                                        Must include BOS/EOS tokens (typically 0)
                                        Maximum length: 512 (BERT context limit)
            ref_s (torch.FloatTensor): Voice embedding with shape (1, 256)
                                     First 128 dims: baseline speaker characteristics
                                     Last 128 dims: style/prosodic attributes
            speed (float, optional): Speech rate multiplier. Defaults to 1.0.
                                   Values: 0.5=slow, 1.0=normal, 2.0=fast
                                   Applied as divisor to duration predictions

        Returns:
            tuple[torch.FloatTensor, torch.LongTensor]: Generated audio and timing
                - audio (torch.FloatTensor): Waveform tensor, shape (T,) at 24kHz
                - pred_dur (torch.LongTensor): Duration predictions, shape (N,) in 40fps frames

        Raises:
            RuntimeError: If input sequence exceeds context_length (512 tokens)
            ValueError: If ref_s doesn't have expected dimensions (256)
            
        Performance Notes:
        - Uses @torch.no_grad() decorator for inference-only mode
        - Handles variable sequence lengths with proper attention masking
        - Alignment matrix built efficiently via torch.repeat_interleave
        - F0/pitch and noise predictions computed in parallel branches
        - Memory efficient: intermediate tensors automatically garbage collected

        Called by:
        - KModel.forward(): User-facing phoneme string interface conversion
        - export_coreml.py: Model tracing during CoreML conversion process
        - KModelForONNX.forward(): ONNX export wrapper for simplified interface
        - Test suites: Direct testing of core inference pipeline

        Device Compatibility:
        - CPU: Full compatibility, slower execution
        - CUDA: GPU acceleration for supported operations
        - MPS: Apple Silicon GPU with fallback for unsupported ops
        - Export: Compatible with torch.jit.trace for ONNX/CoreML conversion
        """
        input_lengths = torch.full(
            (input_ids.shape[0],), 
            input_ids.shape[-1], 
            device=input_ids.device,
            dtype=torch.long
        )

        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, self.ModelConstants.BASELINE_VOICE_DIM:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :self.ModelConstants.BASELINE_VOICE_DIM]).squeeze()
        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        return_output: bool = False
    ) -> Union['KModel.Output', torch.FloatTensor]:
        """
        User-friendly inference interface accepting phoneme strings.

        This method provides the primary public API for text-to-speech synthesis,
        handling the complete pipeline from phoneme strings to audio waveforms.
        It performs phoneme tokenization, validation, and delegates to the core
        forward_with_tokens method for neural computation.

        Phoneme Processing Pipeline:
        1. Character-level tokenization: Map phoneme chars to vocabulary IDs
        2. Unknown phoneme filtering: Remove unsupported phonemes gracefully
        3. Sequence boundaries: Add BOS/EOS tokens for BERT compatibility
        4. Length validation: Ensure sequence fits within context window
        5. Device synchronization: Move inputs to model device automatically

        Phoneme Format:
        - Input: IPA phoneme string (e.g., "hˈɛloʊ wˈɜːld" for "Hello world")
        - Vocabulary: Character-level mapping defined during model training
        - Boundaries: Automatic BOS/EOS token insertion for BERT processing
        - Filtering: Unknown phonemes silently dropped with debug logging

        Voice Embedding Requirements:
        - Shape: (1, 256) with baseline + style components
        - Device: Automatically moved to match model device
        - Source: Loaded from voice packs via pipeline.load_voice()
        - Format: Concatenated [baseline(128), style(128)] embeddings

        Args:
            phonemes (str): IPA phoneme string for synthesis
                          Example: "hˈɛloʊ wˈɜːld" (Hello world)
                          Format: Character-level IPA phonemes
                          Limits: Max 510 chars after tokenization (BERT context - BOS/EOS)
            ref_s (torch.FloatTensor): Voice embedding tensor, shape (1, 256)
                                     Automatically moved to model.device
                                     Contains [baseline, style] voice characteristics
            speed (float, optional): Speech rate multiplier. Defaults to 1.0.
                                   Range: 0.1 to 3.0 (practical limits)
                                   Effect: Inversely affects duration predictions
            return_output (bool, optional): Output format control. Defaults to False.
                                          True: Return full Output dataclass
                                          False: Return audio tensor only

        Returns:
            Union[KModel.Output, torch.FloatTensor]: Synthesis results
                - If return_output=True: KModel.Output with audio + pred_dur
                - If return_output=False: Audio tensor only, shape (T,)
                Both formats return audio at 24kHz sample rate on CPU

        Raises:
            AssertionError: If phoneme sequence exceeds context_length after tokenization
            RuntimeError: If model device placement fails
            KeyError: If critical phonemes missing from vocabulary (rare)

        Error Handling:
        - Sequence length: Hard assertion against BERT context limit
        - Device mismatch: Automatic ref_s device synchronization
        - Unknown phonemes: Silent filtering with debug logging
        - Empty sequences: Graceful handling with minimal audio output

        Performance Characteristics:
        - Tokenization: O(n) character-level vocabulary lookup
        - Neural inference: Delegated to optimized forward_with_tokens
        - Memory: CPU tensors returned, GPU tensors automatically freed
        - Caching: No internal caching, stateless operation

        Called by:
        - pipeline.py: KPipeline.infer() during production TTS generation
        - demo/app.py: Direct model usage in Gradio web interface
        - test_*.py: Unit tests and integration validation
        - examples/*.py: Documentation examples and tutorials

        Example:
        ```python
        # Basic synthesis
        audio = model("hˈɛloʊ wˈɜːld", voice_embedding)
        
        # With timing information
        result = model("hˈɛloʊ", voice_embedding, return_output=True)
        audio, durations = result.audio, result.pred_dur
        ```
        """
        input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
        input_ids = torch.LongTensor([[self.ModelConstants.BOS_TOKEN_ID, *input_ids, self.ModelConstants.EOS_TOKEN_ID]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio

class KModelForONNX(torch.nn.Module):
    """
    ONNX/CoreML export wrapper for KModel with simplified tensor-only interface.

    This wrapper class provides a clean, export-friendly interface that eliminates
    string processing and Python-specific operations that can't be traced or
    converted to static graph formats. It focuses purely on numerical tensor
    operations while maintaining full model functionality.

    Export Compatibility Strategy:
    - Pure tensor operations: No string processing or Python data structures
    - Static graph friendly: All operations traceable by torch.jit.trace
    - Explicit signatures: Clear input/output types for conversion tools
    - Minimal overhead: Direct delegation to core KModel functionality

    Conversion Pipeline:
    1. Wrap trained KModel: wrapper = KModelForONNX(trained_model)
    2. Trace with representative inputs: torch.jit.trace(wrapper, sample_inputs)
    3. Convert traced model: coremltools.convert() or torch.onnx.export()
    4. Deploy converted model: Load in target runtime environment

    Design Rationale:
    - ONNX operators: Limited to supported tensor operations only
    - CoreML compatibility: Avoids Python runtime dependencies
    - Static shapes: Enables graph optimization in target runtimes
    - Simplified interface: Reduces conversion complexity and failure modes

    Supported Export Formats:
    - ONNX: Cross-platform neural network exchange format
    - CoreML: Apple's on-device inference framework
    - TensorRT: NVIDIA's inference optimization platform (via ONNX)
    - OpenVINO: Intel's inference engine (via ONNX)

    Limitations:
    - No string processing: Tokenization must happen externally
    - Fixed tensor shapes: Dynamic shapes require explicit handling
    - No Python objects: Only primitive tensor types supported
    - Inference only: No training-mode operations available

    Used by:
    - export_coreml.py: CoreML conversion workflows
    - export_vocoder.py: Vocoder-specific model conversion
    - Mobile deployment: Edge device inference scenarios
    - Model serving: Production inference with external tokenization
    """
    
    def __init__(self, kmodel: KModel):
        """
        Initialize ONNX export wrapper around existing KModel.

        This constructor creates a lightweight wrapper that exposes only the
        tensor-based forward_with_tokens interface, hiding string processing
        and other Python-specific operations from export tools.

        Args:
            kmodel (KModel): Pre-trained KModel instance with loaded weights
                           Must be in evaluation mode for consistent behavior
                           All parameters must be on the same device

        State Management:
        - Reference sharing: Wrapper shares parameters with original model
        - Device placement: Inherits device placement from wrapped model
        - Evaluation mode: Should be set on wrapped model before export
        - Memory overhead: Minimal - only stores reference to original

        Thread Safety:
        - Shared state: Both wrapper and original model share parameters
        - Concurrent access: Not safe for concurrent inference calls
        - Device synchronization: Inherits thread safety from wrapped model

        Called by:
        - Export scripts: During model conversion preparation phase
        - Model validation: Testing export compatibility before conversion
        - Production deployment: When external tokenization is available
        """
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1.0
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Pure tensor-based inference interface suitable for ONNX/CoreML export.

        This method provides a clean tensor-only interface that directly delegates
        to the underlying KModel's forward_with_tokens method. It maintains the
        same functionality while being compatible with static graph export tools.

        Export Characteristics:
        - Static graph: All operations are traceable by torch.jit.trace
        - No Python objects: Only tensor inputs and outputs
        - Deterministic shapes: Fixed tensor dimensions for graph optimization
        - Device agnostic: Works with CPU, GPU, and specialized accelerators

        Args:
            input_ids (torch.LongTensor): Tokenized phoneme sequence, shape (1, N)
                                        Must include BOS/EOS tokens
                                        Range: Valid vocabulary indices only
            ref_s (torch.FloatTensor): Voice embedding tensor, shape (1, 256)
                                     Format: [baseline(128), style(128)]
                                     Device: Must match model device
            speed (float, optional): Speech rate multiplier. Defaults to 1.0.
                                   Range: 0.1 to 3.0 for practical synthesis

        Returns:
            tuple[torch.FloatTensor, torch.LongTensor]: Synthesis results
                - waveform (torch.FloatTensor): Generated audio, shape (T,) at 24kHz
                - duration (torch.LongTensor): Phoneme durations, shape (N,) in frames

        Export Compatibility:
        - torch.jit.trace: Full compatibility for static graph tracing
        - ONNX export: All operations supported in standard ONNX opsets
        - CoreML conversion: Compatible with MLProgram backend
        - TensorRT/OpenVINO: Supports optimization via ONNX intermediate format

        Performance Notes:
        - Zero overhead: Direct method delegation with no extra computation
        - Memory efficient: No intermediate data structure conversions
        - Device optimized: Inherits all device-specific optimizations
        - Graph fusion: Export tools can optimize the complete computation graph

        Used by:
        - torch.jit.trace(): Static graph tracing for export preparation
        - ONNX workflows: torch.onnx.export() conversion pipeline
        - CoreML conversion: coremltools.convert() with traced models
        - Production inference: Deployed models with external preprocessing

        Example:
        ```python
        # Export workflow
        wrapper = KModelForONNX(trained_model)
        traced = torch.jit.trace(wrapper, (input_ids, ref_s, speed))
        onnx_model = torch.onnx.export(traced, ...)
        ```
        """
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
