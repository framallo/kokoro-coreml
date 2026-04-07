#!/usr/bin/env python3
"""Kokoro hybrid PyTorch + CoreML TTS pipeline (production orchestration).

Loads models and shared CPU helpers (``extract_vocoder_inputs``, bucket pick, alignment).
Synthesis backends and fallback order live in ``kokoro.synthesis_backends``; this class
delegates ``synthesize()`` to ``run_synthesis_chain``.

Consumed by ``examples/example_synthesis.py`` (and shim ``run_single.py``) and
``demo_ane_pipeline.py`` (CLI demo).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from kokoro import KModel, KPipeline
from kokoro.synthesis_backends import (
    VOCODER_CHUNK_SAMPLES,
    TextBackend,
    ViBackend,
    decoder_har_bucket_impl,
    decoder_har_grouped_impl,
    decoder_har_sliding_impl,
    pytorch_fallback_impl,
    run_synthesis_chain,
    synth_bucket_impl,
    vocoder_windows_impl,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
COREML_MODEL_PATH = str(_REPO_ROOT / "coreml" / "KokoroVocoder.mlpackage")
COREML_DECODER_HAR_PATH = str(_REPO_ROOT / "coreml" / "KokoroDecoder_HAR.mlpackage")
COREML_AVAILABLE = os.path.exists(COREML_MODEL_PATH) or os.path.exists(COREML_DECODER_HAR_PATH)

ct = None  # coremltools, when available
if COREML_AVAILABLE:
    try:
        import coremltools as ct
    except ImportError:
        COREML_AVAILABLE = False
        ct = None

__all__ = [
    "HybridTTSPipeline",
    "COREML_MODEL_PATH",
    "COREML_DECODER_HAR_PATH",
    "COREML_AVAILABLE",
    "VOCODER_CHUNK_SAMPLES",
]

class HybridTTSPipeline:
    """
    Hybrid TTS pipeline that uses PyTorch for text processing and CoreML for vocoding.
    
    This class demonstrates the optimal architecture split:
    - PyTorch: Handles BERT, LSTM, and text encoding (CPU-optimized)
    - CoreML: Handles iSTFTNet vocoder (ANE-optimized)
    
    The pipeline maintains compatibility with the original Kokoro interface
    while providing significant performance improvements for the vocoder stage.
    """
    
    def __init__(
        self,
        force_engine: str | None = None,
        *,
        text_backends: Sequence[TextBackend] | None = None,
        vi_backends: Sequence[ViBackend] | None = None,
    ):
        """Initialize the hybrid pipeline with both PyTorch and CoreML components.

        ``text_backends`` / ``vi_backends`` override the default synthesis order (see
        ``kokoro.synthesis_backends``). Pass a new list to insert a custom backend without
        editing this class.
        """
        print("🚀 Initializing Hybrid ANE-Accelerated TTS Pipeline...")
        self._text_backends = text_backends
        self._vi_backends = vi_backends
        
        # Initialize PyTorch components for text processing
        print("📦 Loading PyTorch text processing components...")
        self.pytorch_model = KModel().to('cpu').eval()
        self.pipeline = KPipeline(lang_code='a', model=False)  # English pipeline
        print("✅ PyTorch components loaded")
        
        self.coreml_vocoder = None
        self.coreml_decoder_har = None
        self.coreml_synth_buckets = {}
        self.coreml_decoder_har_buckets = {}
        self.use_coreml = False

        if force_engine == "coreml" and not COREML_AVAILABLE:
            raise RuntimeError(
                "force_engine='coreml' but CoreML is unavailable "
                "(install coremltools and/or add .mlpackage files under coreml/)."
            )

        # Initialize CoreML models if available
        if COREML_AVAILABLE and (force_engine is None or force_engine == "coreml"):
            print("🍎 Loading CoreML models...")
            if os.path.exists(COREML_MODEL_PATH):
                self.coreml_vocoder = ct.models.MLModel(COREML_MODEL_PATH)
                print("✅ CoreML vocoder loaded successfully")
            if os.path.exists(COREML_DECODER_HAR_PATH):
                self.coreml_decoder_har = ct.models.MLModel(COREML_DECODER_HAR_PATH)
                print("✅ CoreML Decoder_HAR loaded successfully (exact hn-nsf parity)")

            # Bucket models: any path returned by glob must load or __init__ fails (no silent skip).
            synth_globs = [
                str(_REPO_ROOT / "coreml" / "kokoro_synthesizer_*s.mlpackage"),
                str((_REPO_ROOT.parent / "coreml" / "kokoro_synthesizer_*s.mlpackage")),
            ]
            for g in synth_globs:
                for path in glob.glob(g):
                    model = ct.models.MLModel(path)
                    base = os.path.basename(path)
                    sec_str = base.split("_")[-1].replace("s.mlpackage", "").replace(".mlpackage", "")
                    sec = int(sec_str.replace("s", "")) if sec_str.endswith("s") else int(sec_str)
                    self.coreml_synth_buckets[sec] = model
                    print(f"✅ Loaded Synthesizer bucket: {sec}s → {path}")

            har_globs = [
                str(_REPO_ROOT / "coreml" / "KokoroDecoder_HAR_*s.mlpackage"),
                str((_REPO_ROOT.parent / "coreml" / "KokoroDecoder_HAR_*s.mlpackage")),
            ]
            for g in har_globs:
                for path in glob.glob(g):
                    model = ct.models.MLModel(path)
                    base = os.path.basename(path)
                    sec = int(base.split("_")[-1].replace("s.mlpackage", ""))
                    self.coreml_decoder_har_buckets[sec] = model
                    print(f"✅ Loaded Decoder_HAR bucket: {sec}s → {path}")

            synth_n = len(self.coreml_synth_buckets)
            har_n = len(self.coreml_decoder_har_buckets)
            self.use_coreml = (
                self.coreml_vocoder is not None
                or self.coreml_decoder_har is not None
                or synth_n > 0
                or har_n > 0
            )
            if synth_n or har_n:
                print(f"✅ Buckets → synth: {synth_n}, decoder_har: {har_n}")

            if self.coreml_vocoder is not None:
                print("\n📋 CoreML Vocoder Info:")
                for input_spec in self.coreml_vocoder.get_spec().description.input:
                    print(f"  Input - {input_spec.name}: {input_spec.type}")
                for output_spec in self.coreml_vocoder.get_spec().description.output:
                    print(f"  Output - {output_spec.name}: {output_spec.type}")

            if force_engine == "coreml" and not self.use_coreml:
                raise RuntimeError(
                    "force_engine='coreml' but no CoreML models loaded. "
                    f"Expected at least one of: {COREML_MODEL_PATH!r}, {COREML_DECODER_HAR_PATH!r}, "
                    "or bucket packages matching kokoro_synthesizer_*s / KokoroDecoder_HAR_*s under coreml/."
                )
        else:
            print("⚠️ CoreML not used (unavailable or engine=pytorch); PyTorch-only pipeline")

        print(f"\n🎯 Pipeline Mode: {'Hybrid (PyTorch + CoreML)' if self.use_coreml else 'PyTorch Only'}")
    
    def extract_vocoder_inputs(self, text, voice='af_heart', speed=1.0):
        """
        Extract vocoder inputs using PyTorch text processing pipeline.
        
        This method runs the first part of the TTS pipeline (text → spectrogram)
        using the original PyTorch implementation, then extracts the inputs
        needed for the CoreML vocoder.
        
        Args:
            text: Input text to synthesize
            voice: Voice ID to use
            speed: Speech rate multiplier
            
        Returns:
            dict: Vocoder inputs (asr, f0_curve, n, s) or None if extraction fails
        """
        print(f"\n🔤 Processing text with PyTorch: '{text}'")
        
        try:
            # Load voice pack
            voice_pack = self.pipeline.load_voice(voice)
            
            # Process text through the pipeline to get phonemes
            phonemes = None
            for _, ps, _ in self.pipeline(text, voice, speed):
                phonemes = ps
                break
                
            if not phonemes:
                print("❌ Failed to extract phonemes")
                return None
                
            print(f"🔊 Phonemes: {phonemes}")
            
            # Get voice reference style
            ref_s = voice_pack[len(phonemes)-1]  # Reference style for this length
            
            # Run through the PyTorch model up to the vocoder stage
            # We need to extract the inputs that would normally go to the decoder
            input_ids = list(filter(lambda i: i is not None, 
                                  map(lambda p: self.pytorch_model.vocab.get(p), phonemes)))
            input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.pytorch_model.device)
            ref_s = ref_s.to(self.pytorch_model.device)
            
            # Run forward pass up to decoder inputs
            with torch.no_grad():
                input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], 
                                         device=input_ids.device, dtype=torch.long)
                text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
                    input_lengths.shape[0], -1).type_as(input_lengths)
                text_mask = torch.gt(text_mask+1, input_lengths.unsqueeze(1)).to(self.pytorch_model.device)
                
                # BERT encoding
                bert_dur = self.pytorch_model.bert(input_ids, attention_mask=(~text_mask).int())
                d_en = self.pytorch_model.bert_encoder(bert_dur).transpose(-1, -2)
                s = ref_s[:, 128:]  # Style embedding
                
                # Prosody prediction
                d = self.pytorch_model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
                x, _ = self.pytorch_model.predictor.lstm(d)
                duration = self.pytorch_model.predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1) / speed
                pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
                
                # Duration alignment
                indices = torch.repeat_interleave(
                    torch.arange(input_ids.shape[1], device=self.pytorch_model.device), pred_dur)
                pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), 
                                         device=self.pytorch_model.device)
                pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
                pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.pytorch_model.device)
                
                # Generate F0 and noise predictions
                en = d.transpose(-1, -2) @ pred_aln_trg
                F0_pred, N_pred = self.pytorch_model.predictor.F0Ntrain(en, s)
                
                # Text encoder features
                t_en = self.pytorch_model.text_encoder(input_ids, input_lengths, text_mask)
                asr = t_en @ pred_aln_trg
                
                # Extract vocoder inputs
                vocoder_inputs = {
                    'asr': asr.cpu().numpy().astype(np.float32),
                    'f0_curve': F0_pred.cpu().numpy().astype(np.float32), 
                    'n': N_pred.cpu().numpy().astype(np.float32),
                    's': ref_s[:, :128].cpu().numpy().astype(np.float32),  # Style embedding
                    # Additional intermediates for synthesizer bucket path
                    'd': d.cpu().numpy().astype(np.float32),
                    't_en': t_en.cpu().numpy().astype(np.float32),
                    'pred_dur': pred_dur.cpu().numpy().astype(np.int64),
                    'ref_s': ref_s.cpu().numpy().astype(np.float32),
                }
                
                print("✅ Successfully extracted vocoder inputs")
                print(f"  - ASR features: {vocoder_inputs['asr'].shape}")
                print(f"  - F0 curve: {vocoder_inputs['f0_curve'].shape}")
                print(f"  - Noise: {vocoder_inputs['n'].shape}")
                print(f"  - Style: {vocoder_inputs['s'].shape}")
                
                return vocoder_inputs
                
        except Exception as e:
            print(f"❌ Error extracting vocoder inputs: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_bucket_seconds(self, total_seconds: float) -> int | None:
        """Pick the smallest available bucket >= total_seconds from any loaded bucket set."""
        candidates = []
        if getattr(self, 'coreml_synth_buckets', None):
            candidates.extend(self.coreml_synth_buckets.keys())
        if getattr(self, 'coreml_decoder_har_buckets', None):
            candidates.extend(self.coreml_decoder_har_buckets.keys())
        candidates = sorted(set(candidates))
        if not candidates:
            return None
        for sec in candidates:
            if sec >= int(np.ceil(total_seconds)):
                return sec
        return candidates[-1]

    def _build_alignment_matrix(self, pred_dur_tokens: np.ndarray, trace_length: int, frame_count: int) -> np.ndarray:
        """Construct pred_aln_trg of shape (trace_length, frame_count) with one-hot repeats."""
        assert trace_length >= 1 and frame_count >= 1, (trace_length, frame_count)
        # Pad or truncate token durations to trace_length
        pred_dur = np.zeros((trace_length,), dtype=np.int64)
        L = min(trace_length, pred_dur_tokens.shape[-1])
        pred_dur[:L] = pred_dur_tokens[:L]
        # Total frames limited by frame_count
        repeat_idx = np.repeat(np.arange(trace_length), pred_dur)
        if repeat_idx.size > frame_count:
            repeat_idx = repeat_idx[:frame_count]
        else:
            # pad with last valid token index
            pad = frame_count - repeat_idx.size
            last_idx = int(repeat_idx[-1]) if repeat_idx.size > 0 else 0
            last_idx = int(np.clip(last_idx, 0, trace_length - 1))
            repeat_idx = np.concatenate([repeat_idx, np.full((pad,), last_idx, dtype=repeat_idx.dtype)])
        repeat_idx = np.clip(repeat_idx, 0, trace_length - 1)
        assert repeat_idx.shape == (frame_count,)
        assert bool(np.all((repeat_idx >= 0) & (repeat_idx < trace_length))), (
            repeat_idx.min(),
            repeat_idx.max(),
            trace_length,
        )
        mat = np.zeros((trace_length, frame_count), dtype=np.float32)
        mat[repeat_idx, np.arange(frame_count)] = 1.0
        return mat

    def run_coreml_synth_bucket(self, text, voice='af_heart', speed=1.0):
        """Single-shot bucketed synthesis using CoreML synthesizer model."""
        return synth_bucket_impl(self, text, voice, speed)

    def run_coreml_decoder_har(self, vocoder_inputs):
        """Run CoreML Decoder_HAR (exact hn-nsf parity). PyTorch computes har_spec/har_phase."""
        return decoder_har_sliding_impl(self, vocoder_inputs)

    def run_coreml_decoder_har_bucket(self, text, voice='af_heart', speed=1.0):
        """Single-shot Decoder_HAR bucket: compute har once, call CoreML once, inverse STFT once."""
        return decoder_har_bucket_impl(self, text, voice, speed)

    def run_coreml_decoder_har_grouped(self, vocoder_inputs):
        """Greedy large-bucket segmentation with minimal calls and seam crossfades."""
        return decoder_har_grouped_impl(self, vocoder_inputs)

    def run_coreml_vocoder(self, vocoder_inputs):
        """Run the windowed CoreML vocoder (KokoroVocoder) on extracted inputs."""
        return vocoder_windows_impl(self, vocoder_inputs)

    def run_pytorch_fallback(self, text, voice='af_heart', speed=1.0):
        """Run the complete pipeline using PyTorch only as a fallback."""
        return pytorch_fallback_impl(self, text, voice, speed)

    def synthesize(self, text, voice='af_heart', speed=1.0):
        """
        Main synthesis method that orchestrates the hybrid pipeline.

        Order is defined by ``kokoro.synthesis_backends.DEFAULT_*_BACKENDS`` unless
        overridden via constructor ``text_backends=`` / ``vi_backends=``.

        Returns:
            tuple: (audio_array, sample_rate) or (None, None) if failed
        """
        return run_synthesis_chain(
            self,
            text,
            voice,
            speed,
            text_backends=self._text_backends,
            vi_backends=self._vi_backends,
        )
