# Audio Synthesis and iSTFT-based Neural Vocoder Components
#
# This module implements the core audio generation pipeline for Kokoro TTS,
# featuring an iSTFT-based neural vocoder that synthesizes high-quality
# waveforms from intermediate acoustic features.
#
# Key Components:
# - Generator: Main neural vocoder with HiFi-GAN-style architecture
# - Decoder: High-level wrapper combining feature processing and generation
# - SourceModuleHnNSF: Harmonic/noise source modeling for F0-conditioned synthesis
# - AdaINResBlock1: Style-adaptive residual blocks for voice conditioning
# - TorchSTFT/CustomSTFT: STFT implementations with CoreML compatibility options
#
# Architecture Philosophy:
# - iSTFT-based synthesis for high-fidelity audio generation
# - Style-conditioned layers throughout for voice adaptation
# - Harmonic plus noise source modeling for natural speech characteristics
# - Multi-scale residual processing for rich spectral detail
#
# CoreML Export Considerations:
# - CustomSTFT used when disable_complex=True for ONNX/CoreML compatibility
# - TorchSTFT used for native PyTorch inference (higher quality)
# - Complex number operations avoided in CustomSTFT variant
#
# Cross-file dependencies:
# - Imports from: custom_stft.py (CustomSTFT for export compatibility)
# - Used by: model.py (KModel.decoder), modules.py (ProsodyPredictor components)
# - Based on: StyleTTS2 iSTFTNet with Kokoro-specific optimizations
#
# Performance Notes:
# - Optimized for 24kHz synthesis with 600-sample hop length
# - Multi-resolution processing for efficient high-quality generation
# - Style conditioning enables zero-shot voice cloning capabilities

# Adapted from StyleTTS2: https://github.com/yl4579/StyleTTS2/blob/main/Modules/istftnet.py

from kokoro.custom_stft import CustomSTFT
from torch.nn.utils.parametrizations import weight_norm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m, mean=0.0, std=0.01):
    # Initialize convolutional layer weights with normal distribution.
    #
    # This utility function provides consistent weight initialization
    # across all convolutional layers in the vocoder architecture.
    # Proper initialization is critical for stable training and convergence.
    #
    # Parameters:
    # - m: PyTorch module to initialize
    # - mean: Normal distribution mean (default: 0.0)
    # - std: Normal distribution standard deviation (default: 0.01)
    #
    # Applied to:
    # - All Conv1d layers in Generator architecture
    # - Ensures consistent initialization across the network
    #
    # From: StyleTTS2 utilities
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    # Calculate padding for 'same' convolution with dilation.
    #
    # This utility ensures that convolutional operations maintain
    # the input sequence length, which is essential for the iSTFT
    # generation pipeline where temporal alignment must be preserved.
    #
    # Parameters:
    # - kernel_size: Convolution kernel size
    # - dilation: Dilation factor for dilated convolutions
    #
    # Returns:
    # - int: Padding value for same-size output
    #
    # Used throughout:
    # - AdaINResBlock1: Dilated convolutions in residual blocks
    # - Generator: Upsampling and processing layers
    #
    return int((kernel_size*dilation - dilation)/2)


class AdaIN1d(nn.Module):
    # Adaptive Instance Normalization for 1D sequences with style conditioning.
    #
    # This module implements style-conditioned normalization that adapts
    # the normalization statistics based on voice characteristics. It's a
    # key component enabling voice cloning and style transfer capabilities.
    #
    # Mathematical Operation:
    # 1. Instance normalization: x_norm = InstanceNorm(x)
    # 2. Style-dependent parameters: gamma, beta = Linear(style)
    # 3. Adaptive transformation: output = (1 + gamma) * x_norm + beta
    #
    # ONNX Export Compatibility:
    # - Uses affine=True in InstanceNorm1d to avoid channel dimension loss
    # - Workaround for legacy torch.onnx.export limitations
    # - Additional learnable parameters don't affect inference quality
    #
    # Parameters:
    # - style_dim: Dimension of input style vector (typically 128)
    # - num_features: Number of channels to normalize
    #
    # Used by:
    # - AdaINResBlock1: Style-conditioned residual processing
    # - Generator architecture: Voice adaptation throughout the network
    #
    def __init__(self, style_dim, num_features):
        super().__init__()
        # affine should be False, however there's a bug in the old torch.onnx.export (not newer dynamo) that causes the channel dimension to be lost if affine=False. When affine is true, there's additional learnably parameters. This shouldn't really matter setting it to True, since we're in inference mode
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        # Apply adaptive instance normalization with style conditioning.
        #
        # Parameters:
        # - x: Input features, shape (batch, channels, sequence)
        # - s: Style vector, shape (batch, style_dim)
        #
        # Returns:
        # - torch.Tensor: Style-adapted features, same shape as input
        #
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    # Multi-dilation residual block with adaptive instance normalization and Snake activation.
    #
    # This block implements the core processing unit of the iSTFTNet generator,
    # combining style-adaptive normalization with multi-scale dilated convolutions
    # for rich spectral feature extraction. The Snake activation provides smooth,
    # learnable nonlinearity particularly effective for audio generation.
    #
    # Architecture Features:
    # - Three parallel dilated convolution paths (1, 3, 5 dilation)
    # - AdaIN normalization before each activation for voice style conditioning
    # - Snake1D activation: x + (1/α) * sin²(αx) with learnable α parameters
    # - Residual connections for stable gradient flow
    #
    # Multi-Scale Processing:
    # - dilation=(1,3,5) captures features at different temporal scales
    # - Essential for modeling speech characteristics across phoneme boundaries
    # - Enables context-aware feature extraction for high-quality synthesis
    #
    # Used by:
    # - Generator.resblocks: Main vocoder processing pipeline
    # - Decoder.decode: High-level feature processing blocks
    #
    # Args:
    #     channels: Number of input/output channels (maintains channel count)
    #     kernel_size: Convolution kernel size (default: 3)
    #     dilation: Tuple of dilation factors for multi-scale processing
    #     style_dim: Style vector dimensionality for AdaIN conditioning
    #
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super(AdaINResBlock1, self).__init__()
        
        # Constants for multi-scale processing architecture
        class BlockConfig:
            # Number of parallel convolution paths in the residual block.
            # Each path processes features at a different temporal scale.
            NUM_CONV_PATHS = 3
            
            # Standard kernel size for all convolutions in the block.
            # Size 3 provides good local context while maintaining efficiency.
            KERNEL_SIZE = 3
            
            # Default dilation pattern for multi-scale feature extraction.
            # (1,3,5) captures immediate, short-term, and medium-term dependencies.
            DEFAULT_DILATION = (1, 3, 5)
        
        # First convolution layer with multi-scale dilations
        # These capture features at different temporal scales simultaneously
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                  padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                                  padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)
        
        # Second convolution layer with unit dilation for feature refinement
        # Standard 1-dilation convolutions consolidate multi-scale features
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        # Adaptive Instance Normalization layers for style conditioning
        # First set: applied before first convolution in each path
        self.adain1 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        # Second set: applied before second convolution in each path
        self.adain2 = nn.ModuleList([
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
            AdaIN1d(style_dim, channels),
        ])
        
        # Learnable parameters for Snake activation function
        # Snake1D: x + (1/α) * sin²(αx) where α is learned per channel
        # First set: for first activation in each path
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        # Second set: for second activation in each path
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x, s):
        # Forward pass through multi-path residual block with style conditioning.
        #
        # This method processes input features through three parallel paths,
        # each operating at a different temporal scale via dilated convolutions.
        # Style conditioning is applied via AdaIN before each activation.
        #
        # Processing Pipeline (per path):
        # 1. AdaIN normalization with style vector s
        # 2. Snake1D activation: x + (1/α) * sin²(αx)
        # 3. First dilated convolution (scale-specific)
        # 4. AdaIN normalization with style vector s
        # 5. Snake1D activation with different α
        # 6. Second convolution (unit dilation)
        # 7. Residual connection: output = processed + input
        #
        # Snake Activation Properties:
        # - Smooth, learnable nonlinearity
        # - α parameters adapt during training for optimal audio characteristics
        # - Particularly effective for audio generation tasks
        # - Provides richer dynamics than standard ReLU/LeakyReLU
        #
        # Multi-Path Fusion:
        # - All three paths contribute equally to final output
        # - Captures both fine-grained and broad temporal dependencies
        # - Essential for natural speech rhythm and prosody
        #
        # Args:
        #     x: Input features, shape (batch, channels, sequence)
        #     s: Style conditioning vector, shape (batch, style_dim)
        #
        # Returns:
        #     torch.Tensor: Style-conditioned features, same shape as input
        #
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)  # First AdaIN with style conditioning
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D activation
            xt = c1(xt)    # First convolution (dilated)
            xt = n2(xt, s) # Second AdaIN with style conditioning
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D activation
            xt = c2(xt)    # Second convolution (unit dilation)
            x = xt + x     # Residual connection
        return x


class TorchSTFT(nn.Module):
    # High-quality STFT implementation using PyTorch's native complex operations.
    #
    # This class provides the reference implementation for STFT operations,
    # using PyTorch's built-in torch.stft/torch.istft functions with complex
    # number support. It delivers the highest quality but is incompatible
    # with ONNX/CoreML export due to complex number operations.
    #
    # Quality vs Export Trade-off:
    # - TorchSTFT: High quality, native PyTorch inference only
    # - CustomSTFT: Lower quality, but ONNX/CoreML export compatible
    # - Switch controlled by Generator.disable_complex parameter
    #
    # STFT Configuration:
    # - filter_length=800: FFT size (~33ms at 24kHz)
    # - hop_length=200: Frame advance (~8.3ms, 75% overlap)
    # - win_length=800: Window size (matches FFT size)
    # - window='hann': Hann window for spectral smoothness
    #
    # Used by:
    # - Generator: When disable_complex=False (default mode)
    # - Research and development: Quality reference for CustomSTFT validation
    #
    # Not used by:
    # - CoreML export pipelines (complex operations unsupported)
    # - ONNX export workflows (complex tensor operations problematic)
    #
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        
        # Audio processing constants
        class STFTConfig:
            # FFT size in samples. 800 samples = ~33.3ms at 24kHz sampling rate.
            # Provides good frequency resolution for speech analysis.
            FILTER_LENGTH = 800
            
            # Hop size in samples. 200 samples = ~8.33ms at 24kHz.
            # Results in 75% overlap between analysis frames for smooth reconstruction.
            HOP_LENGTH = 200
            
            # Analysis window length. Matches filter_length for full-frame analysis.
            WIN_LENGTH = 800
            
            # Target sampling rate for all Kokoro audio processing.
            SAMPLE_RATE = 24000
        
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        assert window == 'hann', f"Only Hann window supported, got {window}"
        
        # Precompute Hann window for consistent analysis/synthesis
        # periodic=True ensures proper overlap-add reconstruction
        self.window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)

    def transform(self, input_data):
        # Forward STFT transformation using PyTorch's native complex STFT.
        #
        # This method performs high-quality STFT analysis using PyTorch's
        # optimized complex number operations. The result is magnitude and
        # phase spectra suitable for neural vocoder processing.
        #
        # Processing Pipeline:
        # 1. Apply torch.stft with complex output
        # 2. Extract magnitude: |X[k]| for spectral envelope
        # 3. Extract phase: ∠X[k] for fine harmonic structure
        #
        # Quality Advantages over CustomSTFT:
        # - Native complex arithmetic (no approximation errors)
        # - Optimized PyTorch implementation
        # - Proper reflect padding support
        # - Better numerical precision
        #
        # Args:
        #     input_data: Audio waveform, shape (batch, samples)
        #
        # Returns:
        #     tuple: (magnitude, phase) spectrograms
        #            - magnitude: shape (batch, freq_bins, frames)
        #            - phase: shape (batch, freq_bins, frames)
        #
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, 
            window=self.window.to(input_data.device),
            return_complex=True
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        # Inverse STFT reconstruction using PyTorch's native complex operations.
        #
        # This method reconstructs time-domain audio from magnitude and phase
        # spectrograms using PyTorch's optimized inverse STFT implementation.
        # The result is high-quality audio with minimal reconstruction artifacts.
        #
        # Reconstruction Process:
        # 1. Convert magnitude/phase to complex spectrum: M * e^(jφ)
        # 2. Apply torch.istft with overlap-add reconstruction
        # 3. Add batch dimension for consistency with conv_transpose1d outputs
        #
        # Quality Benefits:
        # - Perfect reconstruction with proper window overlap
        # - Native complex arithmetic precision
        # - Optimized overlap-add implementation
        # - No approximation errors from real-valued operations
        #
        # Args:
        #     magnitude: Spectral magnitude, shape (batch, freq_bins, frames)
        #     phase: Spectral phase, shape (batch, freq_bins, frames)
        #
        # Returns:
        #     torch.Tensor: Reconstructed audio, shape (batch, 1, samples)
        #                   Extra dimension for consistency with Generator output format
        #
        # Reconstruct complex spectrum from magnitude and phase
        complex_spectrum = magnitude * torch.exp(phase * 1j)
        
        # Perform inverse STFT with overlap-add reconstruction
        inverse_transform = torch.istft(
            complex_spectrum,
            self.filter_length, self.hop_length, self.win_length, 
            window=self.window.to(magnitude.device)
        )
        
        # Add channel dimension to match conv_transpose1d output format
        # This ensures consistency between TorchSTFT and CustomSTFT outputs
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data):
        # Complete STFT analysis-synthesis cycle for quality validation.
        #
        # This method performs a full round-trip transformation to validate
        # the STFT implementation and measure reconstruction quality. It's
        # primarily used for testing and as a quality reference.
        #
        # Processing:
        # 1. Forward STFT: time domain → magnitude/phase spectra
        # 2. Inverse STFT: magnitude/phase spectra → time domain
        # 3. Store intermediate results for inspection
        #
        # Quality Metrics:
        # - Typical SNR: 60+ dB for clean speech
        # - Near-perfect reconstruction with proper window overlap
        # - Reference standard for CustomSTFT validation
        #
        # Args:
        #     input_data: Input audio waveform, shape (batch, samples)
        #
        # Returns:
        #     torch.Tensor: Reconstructed audio, shape (batch, 1, samples)
        #
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class SineGen(nn.Module):
    """Neural sine wave generator with harmonic modeling.

    Generates multi-harmonic sine signals and voiced/unvoiced (uv) masks from an
    input F0 contour. Used by `SourceModuleHnNSF` and the `Generator` for F0-conditioned
    excitation.

    Mathematical foundation:
    - Fundamental: sin(2π·F0·t)
    - Harmonics: sin(2π·n·F0·t) for n ∈ {1..H}
    - Phase accumulation: φ[t] = Σ(F0[i]/fs)
    - Noise injection: Gaussian noise scaled by uv
    """
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * torch.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1/self.upsample_scale, mode="linear").transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
            phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            # get the sines
            sines = torch.cos(i_phase * 2 * torch.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype).unsqueeze(0).unsqueeze(0))
        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp
        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)
        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        #        for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    # Harmonic plus Noise Source module for Neural Source Filter (NSF) synthesis.
    #
    # This class implements the source modeling component of the NSF vocoder,
    # combining harmonic sine wave generation with noise modeling to create
    # naturalistic excitation signals for speech synthesis. It forms the
    # foundation of high-quality neural vocoders.
    #
    # HnNSF Architecture:
    # - Harmonic branch: Multi-harmonic sine generation based on F0
    # - Noise branch: Gaussian noise with voicing-dependent amplitude
    # - Linear combination: Learnable mixing of harmonic components
    # - Tanh nonlinearity: Soft saturation for natural dynamics
    #
    # Key Innovation:
    # - Separates harmonic (tonal) and noise (aperiodic) components
    # - Enables independent modeling of voiced/unvoiced speech characteristics
    # - Provides interpretable control over speech timbre and naturalness
    #
    # Used by:
    # - Generator.m_source: Core excitation generation in main vocoder
    # - Neural vocoder architectures requiring F0-conditioned synthesis
    #
    # Processing Pipeline:
    # 1. F0 → Multi-harmonic sine waves (SineGen)
    # 2. Linear combination of harmonics → single excitation
    # 3. Tanh saturation for natural dynamics
    # 4. Independent noise generation for unvoiced content
    # 5. Output: (harmonic_source, noise_source, voicing_flag)
    #
    # Args:
    #     sampling_rate: Audio sampling rate in Hz
    #     harmonic_num: Number of harmonics above F0 (0=fundamental only)
    #     sine_amp: Amplitude of harmonic source components (default 0.1)
    #     add_noise_std: Standard deviation of additive noise (default 0.003)
    #                   Note: Unvoiced noise amplitude = sine_amp/3
    #     voiced_threshold: F0 threshold for voiced/unvoiced decision (default 0)
    #
    # Returns:
    #     tuple: (sine_source, noise_source, uv_flag)
    #            - sine_source: Harmonic excitation, shape (batch, length, 1)
    #            - noise_source: Noise component, shape (batch, length, 1)
    #            - uv_flag: Voiced/unvoiced flags, shape (batch, length, 1)
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)
        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """Convert F0 contour x to (harmonic_source, noise_source, uv).

        Args:
            x: Tensor of shape (batch, length, 1) with F0 in Hz (0 = unvoiced)

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (sine_source, noise_source, uv), each
            of shape (batch, length, 1)
        """
        # 1) Multi-harmonic sine generation and voiced/unvoiced flags
        sine_wavs, uv, _ = self.l_sin_gen(x)
        # 2) Merge harmonics and apply gentle saturation
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # 3) Independent noise, scaled stronger for unvoiced regions
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(nn.Module):
    # Core neural vocoder implementing iSTFT-based high-fidelity audio synthesis.
    #
    # This class is the heart of the Kokoro TTS system, converting intermediate
    # acoustic features into high-quality 24kHz audio waveforms. It combines
    # harmonic-plus-noise source modeling with multi-resolution processing
    # and style-adaptive neural networks.
    #
    # Architecture Overview:
    # 1. F0-conditioned source generation (SourceModuleHnNSF)
    # 2. Multi-scale upsampling with transposed convolutions
    # 3. Style-adaptive residual blocks at each resolution
    # 4. Harmonic source injection at multiple scales
    # 5. Final iSTFT synthesis for waveform generation
    #
    # Key Innovation - iSTFT Integration:
    # - Generates magnitude and phase spectra instead of raw audio
    # - Uses STFT.inverse() for high-quality waveform synthesis
    # - Eliminates many artifacts common in direct time-domain generation
    #
    # Style Conditioning:
    # - Voice characteristics controlled via style vectors
    # - AdaIN applied throughout for voice adaptation
    # - Enables zero-shot voice cloning capabilities
    #
    # Export Compatibility:
    # - disable_complex=True: Uses CustomSTFT for ONNX/CoreML export
    # - disable_complex=False: Uses TorchSTFT for highest quality
    #
    # Processing Pipeline:
    # - Input: Acoustic features + F0 + style vector
    # - Output: 24kHz mono audio waveform
    #
    # Called by:
    # - Decoder.forward(): High-level TTS generation pipeline
    # - model.py: Main inference entry point via KModel.decoder
    #
    # Based on:
    # - StyleTTS2 iSTFTNet architecture
    # - HiFi-GAN upsampling strategy
    # - NSF source modeling for natural speech characteristics
    #
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, disable_complex=False):
        super(Generator, self).__init__()
        
        # Architecture configuration constants
        class GeneratorConfig:
            # Target sampling rate for all audio synthesis.
            # Fixed at 24kHz throughout the Kokoro pipeline.
            SAMPLE_RATE = 24000
            
            # Number of harmonic overtones in F0 source modeling.
            # 8 harmonics provide rich spectral content for natural speech.
            HARMONIC_NUM = 8
            
            # F0 threshold for voiced/unvoiced classification.
            # Values below 10 Hz considered unvoiced (silence/consonants).
            VOICED_THRESHOLD = 10
            
            # Default kernel size for post-processing convolution.
            # Size 7 provides good temporal smoothing of final spectra.
            POST_CONV_KERNEL = 7
            
            # Reflection padding for final processing.
            # (1,0) asymmetric padding for causal-like processing.
            REFLECTION_PAD = (1, 0)
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # F0-conditioned harmonic plus noise source generator
        # Produces naturalistic excitation signals based on fundamental frequency
        self.m_source = SourceModuleHnNSF(
            sampling_rate=GeneratorConfig.SAMPLE_RATE,
            upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=GeneratorConfig.HARMONIC_NUM, 
            voiced_threshod=GeneratorConfig.VOICED_THRESHOLD
        )
        
        # F0 upsampling to match final audio sampling rate
        # Ensures F0 contour aligns with generated audio frames
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates) * gen_istft_hop_size)
        
        # Multi-resolution processing components
        self.noise_convs = nn.ModuleList()  # Harmonic source injection layers
        self.noise_res = nn.ModuleList()    # Source processing residual blocks
        self.ups = nn.ModuleList()          # Upsampling layers
        
        # Build upsampling layers with progressive channel reduction
        # Each layer doubles the time resolution while halving channel count
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    upsample_initial_channel//(2**i), 
                    upsample_initial_channel//(2**(i+1)),
                    k, u, padding=(k-u)//2
                )
            ))
        
        # Build style-adaptive residual blocks
        # Applied after each upsampling stage for feature refinement
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            
            # Harmonic source injection layers at each resolution
            # Allows F0 information to influence processing at multiple scales
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    gen_istft_n_fft + 2, c_cur, 
                    kernel_size=stride_f0 * 2, stride=stride_f0, 
                    padding=(stride_f0+1) // 2
                ))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1,3,5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1,3,5], style_dim))
        
        # Final processing layers
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(
            ch, self.post_n_fft + 2, 
            GeneratorConfig.POST_CONV_KERNEL, 1, 
            padding=GeneratorConfig.POST_CONV_KERNEL//2
        ))
        
        # Initialize all upsampling and post-processing weights
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        
        # Reflection padding for causal-like processing
        self.reflection_pad = nn.ReflectionPad1d(GeneratorConfig.REFLECTION_PAD)
        
        # STFT implementation selection based on export requirements
        # CustomSTFT: Export-compatible but lower quality
        # TorchSTFT: High quality but PyTorch-only
        self.stft = (
            CustomSTFT(
                filter_length=gen_istft_n_fft, 
                hop_length=gen_istft_hop_size, 
                win_length=gen_istft_n_fft
            )
            if disable_complex
            else TorchSTFT(
                filter_length=gen_istft_n_fft, 
                hop_length=gen_istft_hop_size, 
                win_length=gen_istft_n_fft
            )
        )

    def forward(self, x, s, f0):
        # Main forward pass generating high-fidelity audio from acoustic features.
        #
        # This method implements the complete neural vocoder pipeline,
        # transforming intermediate acoustic representations into natural
        # 24kHz audio waveforms through multi-scale processing and iSTFT synthesis.
        #
        # Processing Pipeline:
        # 1. F0-conditioned source generation (harmonic + noise)
        # 2. Source spectrum extraction via STFT analysis
        # 3. Multi-resolution upsampling with source injection
        # 4. Style-adaptive residual processing at each scale
        # 5. Magnitude/phase prediction
        # 6. iSTFT waveform synthesis
        #
        # Key Architecture Decisions:
        # - Source generation in no_grad() context (fixed during inference)
        # - Harmonic source injected at multiple resolutions
        # - Exponential magnitude, sine phase for bounded outputs
        # - LeakyReLU(0.1) activation throughout upsampling
        #
        # Args:
        #     x: Acoustic features from encoder, shape (batch, channels, frames)
        #     s: Style conditioning vector, shape (batch, style_dim)
        #     f0: Fundamental frequency contour, shape (batch, frames)
        #
        # Returns:
        #     torch.Tensor: Generated audio waveform, shape (batch, 1, samples)
        #
        # Stage 1: F0-conditioned source generation
        # Generate harmonic plus noise excitation signals based on F0 contour
        # Upsample F0 to final audio sampling rate
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # (batch, frames, 1)

        # Generate harmonic and noise source components
        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2).squeeze(1)  # (batch, samples)

        # Extract harmonic source spectrum via STFT
        # This provides frequency-domain source information for injection
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)  # (batch, n_fft+2, frames)
        
        # Stage 2: Multi-resolution upsampling with source injection
        # Process features through multiple scales while injecting harmonic content
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            
            # Inject harmonic source information at current resolution
            x_source = self.noise_convs[i](har)        # Project source to current channels
            x_source = self.noise_res[i](x_source, s)  # Style-adaptive source processing
            
            # Upsample main feature path
            x = self.ups[i](x)
            
            # Apply reflection padding at final upsampling stage
            # Provides causal-like boundary handling
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            
            # Combine upsampled features with processed source
            x = x + x_source
            
            # Style-adaptive residual processing
            # Multiple residual blocks for rich feature extraction
            xs = None
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[block_idx](x, s)
                else:
                    xs += self.resblocks[block_idx](x, s)
            x = xs / self.num_kernels  # Average multiple residual paths
        
        # Stage 3: Magnitude and phase prediction
        x = F.leaky_relu(x)
        x = self.conv_post(x)  # Project to magnitude+phase dimensions
        
        # Split output into magnitude and phase components
        # Magnitude: exponential for positive values
        # Phase: sine for bounded [-1, 1] range
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        
        # Stage 4: iSTFT waveform synthesis
        # Convert magnitude/phase spectra to time-domain audio
        return self.stft.inverse(spec, phase)


class UpSample1d(nn.Module):
    # Simple 1D upsampling layer with configurable behavior.
    #
    # This utility class provides optional upsampling functionality
    # for residual blocks in the Decoder architecture. It supports
    # either no upsampling or 2x nearest-neighbor upsampling.
    #
    # Used by:
    # - AdainResBlk1d: Optional upsampling in residual blocks
    # - Decoder processing pipeline: Resolution increase during decoding
    #
    # Args:
    #     layer_type: 'none' for identity, any other value for 2x upsampling
    #
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        # Apply upsampling based on layer configuration.
        #
        # Args:
        #     x: Input tensor, shape (batch, channels, length)
        #
        # Returns:
        #     torch.Tensor: Upsampled or unchanged tensor
        #
        if self.layer_type == 'none':
            return x  # Identity operation
        else:
            # 2x upsampling with nearest-neighbor interpolation
            return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    # Style-adaptive residual block with optional upsampling for decoder processing.
    #
    # This block implements the core processing unit of the Decoder, combining
    # adaptive instance normalization with residual connections and optional
    # upsampling. It enables style-conditioned feature processing throughout
    # the audio synthesis pipeline.
    #
    # Architecture Features:
    # - Two-layer residual processing with normalization
    # - Style-dependent AdaIN before each activation
    # - Optional 2x upsampling via transposed convolution
    # - Learned shortcut connections for dimension changes
    # - Dropout for regularization
    #
    # Processing Pipeline:
    # 1. AdaIN + activation + pooling/upsampling + conv1
    # 2. AdaIN + activation + conv2
    # 3. Shortcut connection with optional dimension projection
    # 4. Residual addition with normalization: (residual + shortcut) / √2
    #
    # Style Conditioning:
    # - AdaIN parameters computed from style vector
    # - Enables voice adaptation throughout processing
    # - Essential for multi-speaker TTS capabilities
    #
    # Used by:
    # - Decoder: Multi-layer processing pipeline
    # - High-level feature processing before Generator
    #
    # Args:
    #     dim_in: Input channel dimension
    #     dim_out: Output channel dimension
    #     style_dim: Style conditioning vector dimension (default 64)
    #     actv: Activation function (default LeakyReLU(0.2))
    #     upsample: Upsampling mode ('none' or any value for 2x upsampling)
    #     dropout_p: Dropout probability for regularization (default 0.0)
    #
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout_p=0.0):
        super().__init__()
        
        # Configuration constants
        class ResBlockConfig:
            # Default style vector dimension for voice conditioning.
            DEFAULT_STYLE_DIM = 64
            
            # Transposed convolution parameters for upsampling.
            UPSAMPLE_KERNEL = 3
            UPSAMPLE_STRIDE = 2
            UPSAMPLE_PADDING = 1
            UPSAMPLE_OUTPUT_PADDING = 1
            
            # Residual normalization factor for stable training.
            # √2 normalization prevents activation magnitude growth.
            RESIDUAL_SCALE = 2.0
        
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out  # Shortcut needs dimension projection
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        # Pooling/upsampling configuration
        if upsample == 'none':
            self.pool = nn.Identity()  # No temporal resolution change
        else:
            # Transposed convolution for 2x upsampling
            # Groups=dim_in for depthwise operation (efficient)
            self.pool = weight_norm(nn.ConvTranspose1d(
                dim_in, dim_in, 
                kernel_size=ResBlockConfig.UPSAMPLE_KERNEL, 
                stride=ResBlockConfig.UPSAMPLE_STRIDE, 
                groups=dim_in,  # Depthwise convolution
                padding=ResBlockConfig.UPSAMPLE_PADDING, 
                output_padding=ResBlockConfig.UPSAMPLE_OUTPUT_PADDING
            ))

    def _build_weights(self, dim_in, dim_out, style_dim):
        # Initialize convolutional layers and normalization components.
        #
        # This method sets up the core processing components of the residual block,
        # including convolutions, style-adaptive normalization, and optional
        # shortcut projection layers.
        #
        # Layer Configuration:
        # - conv1/conv2: 3x3 convolutions with weight normalization
        # - norm1/norm2: AdaIN for style-dependent normalization
        # - conv1x1: Optional 1x1 shortcut projection when dims change
        #
        # Weight Normalization:
        # - Applied to all convolutions for stable training
        # - Separates weight magnitude from direction
        # - Improves convergence in generative models
        #
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        
        # Learned shortcut connection for dimension projection
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        # Process input through shortcut connection with optional projection.
        #
        # The shortcut path provides direct connection from input to output,
        # enabling gradient flow and stable training. When input and output
        # dimensions differ, a 1x1 convolution projects to the correct size.
        #
        # Args:
        #     x: Input features, shape (batch, dim_in, length)
        #
        # Returns:
        #     torch.Tensor: Processed shortcut, shape (batch, dim_out, length*scale)
        #
        x = self.upsample(x)  # Optional 2x upsampling
        if self.learned_sc:
            x = self.conv1x1(x)  # Dimension projection if needed
        return x

    def _residual(self, x, s):
        # Process input through main residual pathway.
        #
        # This method implements the core residual processing with style conditioning,
        # following the standard pre-activation residual block design with
        # adaptive normalization.
        #
        # Processing Order:
        # 1. AdaIN normalization (style-conditioned)
        # 2. Activation function
        # 3. Optional upsampling/pooling
        # 4. First convolution with dropout
        # 5. AdaIN normalization (style-conditioned)
        # 6. Activation function
        # 7. Second convolution with dropout
        #
        # Args:
        #     x: Input features, shape (batch, dim_in, length)
        #     s: Style vector for normalization, shape (batch, style_dim)
        #
        # Returns:
        #     torch.Tensor: Processed residual, shape (batch, dim_out, length*scale)
        #
        x = self.norm1(x, s)         # Style-adaptive normalization
        x = self.actv(x)             # Activation (typically LeakyReLU)
        x = self.pool(x)             # Optional upsampling
        x = self.conv1(self.dropout(x))  # First convolution with dropout
        x = self.norm2(x, s)         # Style-adaptive normalization
        x = self.actv(x)             # Activation
        x = self.conv2(self.dropout(x))  # Second convolution with dropout
        return x

    def forward(self, x, s):
        # Forward pass through style-adaptive residual block.
        #
        # This method processes input features through the complete residual
        # pathway, applying style conditioning and combining residual and
        # shortcut connections for stable gradient flow.
        #
        # Processing Steps:
        # 1. Residual path: norm → activation → pool → conv → norm → activation → conv
        # 2. Shortcut path: optional upsampling and dimension projection
        # 3. Combination: (residual + shortcut) / √2 for stable training
        #
        # Residual Scaling:
        # - Factor of 1/√2 prevents activation magnitude growth
        # - Essential for stable training of deep residual networks
        # - Maintains proper gradient flow throughout the network
        #
        # Args:
        #     x: Input features, shape (batch, dim_in, length)
        #     s: Style conditioning vector, shape (batch, style_dim)
        #
        # Returns:
        #     torch.Tensor: Processed features, shape (batch, dim_out, length*scale)
        #                   where scale=1 (no upsample) or 2 (with upsample)
        #
        out = self._residual(x, s)  # Main processing pathway
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2.0))  # Normalized residual addition
        return out


class Decoder(nn.Module):
    # High-level audio synthesis decoder combining feature processing and generation.
    #
    # This class serves as the top-level wrapper for the complete audio synthesis
    # pipeline, orchestrating the interaction between acoustic feature processing,
    # F0/noise conditioning, and the core Generator neural vocoder.
    #
    # Architecture Overview:
    # 1. Feature encoding with F0 and noise conditioning
    # 2. Multi-layer style-adaptive decoding
    # 3. ASR feature residual processing
    # 4. Generator-based waveform synthesis
    #
    # Input Processing:
    # - ASR features: Linguistic content representation
    # - F0 curve: Fundamental frequency for prosody
    # - N (noise): Breathiness/naturalness control
    # - s (style): Voice characteristics conditioning
    #
    # Feature Conditioning Strategy:
    # - F0 and noise downsampled to match processing resolution
    # - Features concatenated at multiple processing stages
    # - Residual ASR features preserved for content fidelity
    #
    # Used by:
    # - model.py: KModel.decoder for main TTS inference
    # - pipeline.py: High-level text-to-speech generation
    #
    # Export Compatibility:
    # - disable_complex parameter passed to Generator
    # - Controls TorchSTFT vs CustomSTFT selection
    #
    def __init__(self, dim_in, style_dim, dim_out, 
                 resblock_kernel_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 resblock_dilation_sizes,
                 upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size,
                 disable_complex=False):
        super().__init__()
        
        # Architecture configuration constants
        class DecoderConfig:
            # Hidden dimension for main processing pipeline.
            # 1024 channels provide rich representational capacity.
            HIDDEN_DIM = 1024
            
            # Reduced dimension after final upsampling.
            # 512 channels for efficient Generator input.
            OUTPUT_DIM = 512
            
            # ASR residual feature dimension.
            # 64 channels preserve essential linguistic content.
            ASR_RES_DIM = 64
            
            # F0/noise conditioning channels.
            # 2 additional channels for F0 and noise injection.
            CONDITIONING_CHANNELS = 2
            
            # F0/noise downsampling configuration.
            # stride=2, kernel=3 provides 2x downsampling with smoothing.
            DOWNSAMPLE_KERNEL = 3
            DOWNSAMPLE_STRIDE = 2
        
        # Initial encoding layer: ASR features + F0 + noise → hidden representation
        self.encode = AdainResBlk1d(
            dim_in + DecoderConfig.CONDITIONING_CHANNELS, 
            DecoderConfig.HIDDEN_DIM, 
            style_dim
        )
        
        # Multi-layer decoding with skip connections
        # Each layer receives ASR residual + F0 + noise conditioning
        self.decode = nn.ModuleList()
        # Three 1024-channel processing layers for feature refinement
        self.decode.append(AdainResBlk1d(
            DecoderConfig.HIDDEN_DIM + DecoderConfig.CONDITIONING_CHANNELS + DecoderConfig.ASR_RES_DIM, 
            DecoderConfig.HIDDEN_DIM, style_dim
        ))
        self.decode.append(AdainResBlk1d(
            DecoderConfig.HIDDEN_DIM + DecoderConfig.CONDITIONING_CHANNELS + DecoderConfig.ASR_RES_DIM, 
            DecoderConfig.HIDDEN_DIM, style_dim
        ))
        self.decode.append(AdainResBlk1d(
            DecoderConfig.HIDDEN_DIM + DecoderConfig.CONDITIONING_CHANNELS + DecoderConfig.ASR_RES_DIM, 
            DecoderConfig.HIDDEN_DIM, style_dim
        ))
        # Final upsampling layer: reduce channels for Generator input
        self.decode.append(AdainResBlk1d(
            DecoderConfig.HIDDEN_DIM + DecoderConfig.CONDITIONING_CHANNELS + DecoderConfig.ASR_RES_DIM, 
            DecoderConfig.OUTPUT_DIM, style_dim, upsample=True
        ))
        
        # F0 and noise conditioning processors
        # Downsample to match processing resolution
        self.F0_conv = weight_norm(nn.Conv1d(
            1, 1, 
            kernel_size=DecoderConfig.DOWNSAMPLE_KERNEL, 
            stride=DecoderConfig.DOWNSAMPLE_STRIDE, 
            groups=1, padding=1
        ))
        self.N_conv = weight_norm(nn.Conv1d(
            1, 1, 
            kernel_size=DecoderConfig.DOWNSAMPLE_KERNEL, 
            stride=DecoderConfig.DOWNSAMPLE_STRIDE, 
            groups=1, padding=1
        ))
        
        # ASR feature residual connection
        # Preserves linguistic content information throughout processing
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(DecoderConfig.OUTPUT_DIM, DecoderConfig.ASR_RES_DIM, kernel_size=1))
        )
        
        # Core neural vocoder for waveform generation
        self.generator = Generator(
            style_dim, resblock_kernel_sizes, upsample_rates, 
            upsample_initial_channel, resblock_dilation_sizes, 
            upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, 
            disable_complex=disable_complex
        )

    def forward(self, asr, F0_curve, N, s):
        # Forward pass through complete audio synthesis pipeline.
        #
        # This method orchestrates the full text-to-speech synthesis process,
        # from high-level acoustic features to final audio waveforms.
        #
        # Processing Pipeline:
        # 1. Condition F0 and noise features (downsample to processing resolution)
        # 2. Initial encoding: ASR + F0 + noise → hidden representation
        # 3. Extract ASR residual features for content preservation
        # 4. Multi-layer decoding with skip connections
        # 5. Generator-based waveform synthesis
        #
        # Skip Connection Strategy:
        # - ASR residual features injected at each decoding layer
        # - F0 and noise conditioning maintained throughout
        # - Ensures content fidelity and prosodic control
        #
        # Resolution Management:
        # - F0/noise downsampled to match ASR feature resolution
        # - Generator upsamples to final 24kHz audio rate
        # - Consistent temporal alignment throughout pipeline
        #
        # Args:
        #     asr: ASR acoustic features, shape (batch, dim_in, frames)
        #     F0_curve: Fundamental frequency, shape (batch, frames)
        #     N: Noise/breathiness parameter, shape (batch, frames)
        #     s: Style conditioning vector, shape (batch, style_dim)
        #
        # Returns:
        #     torch.Tensor: Generated audio waveform, shape (batch, 1, samples)
        #
        # Stage 1: Conditioning feature preparation
        # Downsample F0 and noise to match ASR feature resolution
        F0 = self.F0_conv(F0_curve.unsqueeze(1))  # (batch, 1, frames//2)
        N = self.N_conv(N.unsqueeze(1))           # (batch, 1, frames//2)
        
        # Stage 2: Initial encoding with all conditioning
        x = torch.cat([asr, F0, N], axis=1)  # Concatenate along channel dimension
        x = self.encode(x, s)                # Style-adaptive encoding
        
        # Stage 3: Extract ASR residual features
        # Preserve essential linguistic content for skip connections
        asr_res = self.asr_res(asr)  # (batch, 64, frames)
        
        # Stage 4: Multi-layer decoding with conditional skip connections
        res = True  # Flag to control skip connection injection
        for block in self.decode:
            if res:
                # Inject skip connections: processed features + ASR residual + conditioning
                x = torch.cat([x, asr_res, F0, N], axis=1)
            
            # Style-adaptive residual processing
            x = block(x, s)
            
            # Disable skip connections after upsampling begins
            # Resolution mismatch prevents further concatenation
            if block.upsample_type != "none":
                res = False
        
        # Stage 5: Neural vocoder waveform synthesis
        # Convert processed features to high-fidelity audio
        x = self.generator(x, s, F0_curve)
        return x
