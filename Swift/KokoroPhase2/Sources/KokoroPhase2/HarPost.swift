//
// HarPost.swift
// KokoroPhase2
//
// HAR (Harmonic-plus-Additive Representation) post-processor for Kokoro TTS models.
// Converts HAR network output (magnitude/phase spectrograms) back to time-domain audio.
//
// System Role:
// - Reconstructs waveforms from HAR-based Core ML model outputs
// - Implements inverse STFT with proper windowing and overlap-add (COLA) reconstruction
// - Handles complex spectrum reconstruction from magnitude/phase components
// - Provides environment-controlled parameter tuning for research and debugging
//
// Called by:
// - main.swift (kokoro-phase2-cli) when using HAR models or KOKORO_FORCE_HAR=1
// - Any Swift code needing to reconstruct audio from HAR spectrograms
//
// Technical Background:
// HAR models output log-magnitude and phase information for each frequency bin.
// This processor:
// 1. Converts log-magnitude back to linear magnitude via exp()
// 2. Reconstructs complex spectrum using magnitude * exp(i * phase)
// 3. Applies Hermitian symmetry for real-valued IFFT
// 4. Performs inverse FFT to get time-domain frames
// 5. Applies windowing and overlap-add to reconstruct continuous waveform
//
// Environment Variables:
// - KOKORO_USE_RAW_PHASE=1: Treat phase as raw angles instead of sin() values
// - KOKORO_PHASE_SCALE=N: Scale factor for phase values (default 0.3)
// - KOKORO_PACKING=interleaved: Channel packing format for input tensor
// - KOKORO_DISABLE_HALF_SCALE=1: Skip 1/2 scaling for interior frequency bins
//

import Foundation
import Accelerate
import CoreML

/// HAR (Harmonic-plus-Additive Representation) post-processor for audio reconstruction.
///
/// This processor converts HAR model output tensors (log-magnitude and phase spectrograms)
/// back to time-domain audio waveforms using inverse STFT with proper windowing.
///
/// **Architecture:**
/// - Pre-computes STFT window (periodic Hann) and IFFT twiddle factors for efficiency
/// - Supports various input tensor packing formats via environment variables
/// - Implements full complex spectrum reconstruction with Hermitian symmetry
/// - Uses overlap-add (COLA) reconstruction for seamless audio output
///
/// **Performance Notes:**
/// - Twiddle factors are pre-computed at initialization for fast IFFT
/// - Window function uses periodic Hann to match PyTorch torch.hann_window(periodic=True)
/// - Direct complex arithmetic implementation (no vDSP) for research flexibility
public struct HarPostProcessor {
    /// STFT frame size (number of frequency bins for two-sided spectrum).
    /// Determines frequency resolution: freq_res = sample_rate / nFFT
    public let nFFT: Int
    
    /// STFT hop length in samples (frame advance per time step).
    /// Determines time resolution and overlap: overlap = (nFFT - hop) / nFFT
    public let hop: Int
    
    /// Window length for STFT analysis (typically equals nFFT).
    /// Used for generating the analysis window function.
    public let winLength: Int
    
    /// Pre-computed Hann window for STFT reconstruction.
    /// Uses periodic=True to match PyTorch: w[n] = 0.5 * (1 - cos(2πn/N))
    private let window: [Float]
    
    /// Pre-computed cosine twiddle factors for IFFT: cos(2πkn/N)
    /// Indexed as cosTable[k * nFFT + n] for frequency k, time n
    private let cosTable: [Float]
    
    /// Pre-computed sine twiddle factors for IFFT: sin(2πkn/N)
    /// Indexed as sinTable[k * nFFT + n] for frequency k, time n
    private let sinTable: [Float]

    /// Initializes HAR post-processor with specified STFT parameters.
    ///
    /// **Parameter Selection Guidelines:**
    /// - nFFT: Should be power of 2 for efficiency, determines frequency resolution
    /// - hop: Typically nFFT/4 for 75% overlap (good reconstruction quality)
    /// - winLength: Usually equals nFFT; can be smaller for zero-padding
    ///
    /// **Computational Complexity:**
    /// - Pre-computes O(nFFT²) twiddle factors for direct IFFT implementation
    /// - Uses periodic Hann window matching PyTorch's torch.hann_window(periodic=True)
    ///
    /// - Parameter nFFT: STFT frame size (default: 800 for ~27ms frames at 24kHz)
    /// - Parameter hop: STFT hop length (default: 200 for 75% overlap)
    /// - Parameter winLength: Window function length (default: 800, matching nFFT)
    public init(nFFT: Int = Constants.STFT.defaultNFFT, hop: Int = Constants.STFT.defaultHop, winLength: Int = Constants.STFT.defaultWinLength) {
        self.nFFT = nFFT
        self.hop = hop
        self.winLength = winLength
        
        // Generate periodic Hann window matching torch.hann_window(periodic=True)
        // Formula: w[n] = 0.5 * (1 - cos(2πn/N)), n ∈ [0, N-1]
        // This differs from symmetric Hann which uses N+1 in denominator
        var w = [Float](repeating: 0, count: winLength)
        let N = Float(winLength)
        let twoPiOverN = Constants.Math.twoPi / N
        for n in 0..<winLength {
            w[n] = Constants.Window.hannAmplitude * (1.0 - cosf(twoPiOverN * Float(n)))
        }
        self.window = w
        // Pre-compute IFFT twiddle factors for all frequency-time combinations
        // Direct IFFT implementation: X[n] = Σ(k=0 to N-1) x[k] * e^(i*2πkn/N)
        // For inverse transform, we use negative exponent: e^(-i*2πkn/N)
        var c = [Float](repeating: 0, count: nFFT * nFFT)
        var s = [Float](repeating: 0, count: nFFT * nFFT)
        let twoPiOverN_ifft = Constants.Math.twoPi / Float(nFFT)
        
        for k in 0..<nFFT {
            for n in 0..<nFFT {
                // Negative sign for inverse transform: e^(-i*2πkn/N) = cos(-θ) + i*sin(-θ)
                let angle = -twoPiOverN_ifft * Float(k * n)
                c[k * nFFT + n] = cosf(angle)
                s[k * nFFT + n] = sinf(angle)
            }
        }
        self.cosTable = c
        self.sinTable = s
    }

    /// Reconstructs time-domain waveform from HAR Core ML model output.
    ///
    /// This method implements the complete HAR-to-audio pipeline:
    /// 1. Parses input tensor containing log-magnitude and phase information
    /// 2. Reconstructs complex spectrum for each time frame
    /// 3. Applies Hermitian symmetry for real-valued IFFT output
    /// 4. Performs inverse FFT to generate time-domain frames
    /// 5. Applies windowing and overlap-add (COLA) for smooth reconstruction
    ///
    /// **Input Tensor Format:**
    /// - Shape: (1, channels, frames) where channels = 2 * frequency_bins
    /// - Content: Interleaved or blocked magnitude/phase data (see environment variables)
    /// - Magnitude: Log-scale values that get converted via exp() to linear scale
    /// - Phase: Either raw angles or values to be processed via sin() function
    ///
    /// **Environment Controls for Research/Debugging:**
    /// - `KOKORO_USE_RAW_PHASE=1`: Treat phase channel as raw radians (skip sin() processing)
    /// - `KOKORO_PACKING=interleaved`: Input format [mag0, phase0, mag1, phase1, ...]
    ///   Default: blocked format [mag0, mag1, ..., phase0, phase1, ...]
    /// - `KOKORO_PHASE_SCALE=N`: Scaling factor for phase values (default 0.3, tuned empirically)
    /// - `KOKORO_DISABLE_HALF_SCALE=1`: Skip 1/2 scaling of interior bins for two-sided spectrum
    ///
    /// **Called by:**
    /// - `main.swift` CLI when KOKORO_FORCE_HAR=1 or using HAR models
    /// - HAR model inference pipelines
    ///
    /// - Parameter output: Core ML tensor with shape (1, channels, frames)
    /// - Parameter channels: Number of channels in input (should equal 2 * frequency_bins)
    /// - Parameter frames: Number of time frames in spectrogram
    /// - Returns: Reconstructed audio samples as Float array
    /// - Throws: Errors if tensor processing fails
    public func inverseFromNetworkOutput(_ output: MLMultiArray, channels: Int, frames: Int) throws -> [Float] {
        // Extract tensor data and validate dimensions
        let data = try DecoderOnly5sRunner.flattenFloatArrayStatic(output)
        let strideT = frames
        let halfBins = nFFT / 2 + 1  // One-sided spectrum bins including DC and Nyquist
        
        // HAR models output log-magnitude and phase for each frequency bin
        // Channel count should be 2 * frequency_bins (magnitude + phase)
        let bins = channels / 2
        precondition(bins == halfBins - 1 || bins == halfBins, 
                    "Channel count mismatch: expected \(2 * halfBins) or \(2 * (halfBins - 1)), got \(channels)")

        // Allocate output buffers for overlap-add reconstruction
        // Total length accounts for hop size and final frame: (frames-1) * hop + nFFT
        let totalLen = (frames - 1) * hop + nFFT
        var out = [Float](repeating: 0, count: totalLen)   // Accumulated audio samples
        var acc = [Float](repeating: 0, count: totalLen)   // Accumulated window weights (for COLA normalization)

        // Working buffers for complex spectrum reconstruction
        var real = [Float](repeating: 0, count: nFFT)  // Real part of complex spectrum
        var imag = [Float](repeating: 0, count: nFFT)  // Imaginary part of complex spectrum

        // Parse environment variables for processing configuration
        let env = ProcessInfo.processInfo.environment
        let useRawPhase = (env[Constants.Environment.useRawPhase] == "1")
        
        // Phase scaling factor - empirically tuned for best correlation with ground truth
        let phaseScale: Float = {
            if let s = env[Constants.Environment.phaseScale], let v = Float(s) { return v }
            return Constants.Processing.defaultPhaseScale  // ~0.3 maximizes correlation on 5s fixtures
        }()
        
        let packingInterleaved = (env[Constants.Environment.packing]?.lowercased() == Constants.Packing.interleaved)
        let halfScale = (env[Constants.Environment.disableHalfScale] == "1") ? false : true
        // Process each time frame independently
        for t in 0..<frames {
            // Reconstruct complex spectrum from log-magnitude and phase components
            for k in 0..<bins {
                let (logMag, phaseRaw): (Float, Float)
                
                if packingInterleaved {
                    // Interleaved packing: [mag0, phase0, mag1, phase1, ...]
                    logMag = data[(2*k)*strideT + t]
                    phaseRaw = data[(2*k + 1)*strideT + t]
                } else {
                    // Blocked packing: [mag0, mag1, ..., phase0, phase1, ...]
                    logMag = data[k*strideT + t]
                    phaseRaw = data[(bins + k)*strideT + t]
                }
                
                // Convert log-magnitude back to linear magnitude
                let mag = expf(logMag)
                // Handle special frequency bins and general complex reconstruction
                if k == 0 || k == bins - 1 {
                    // DC (k=0) and Nyquist (k=nFFT/2) components are purely real
                    // This ensures real-valued output from IFFT
                    real[k] = mag
                    imag[k] = 0
                } else {
                    // General frequency bins: reconstruct complex values
                    // Phase processing: either raw angles or sin() of scaled values
                    let angle = useRawPhase ? (phaseScale * phaseRaw) : sinf(phaseScale * phaseRaw)
                    real[k] = mag * cosf(angle)
                    imag[k] = mag * sinf(angle)
                }
            }
            
            // Optional scaling for two-sided spectrum construction
            // Interior bins get scaled by 1/2 because they appear twice in full spectrum
            if halfScale && bins > 2 {
                for k in 1..<(bins - 1) {
                    real[k] *= Constants.Processing.halfScaleFactor
                    imag[k] *= Constants.Processing.halfScaleFactor
                }
            }
            // Apply Hermitian symmetry to create full two-sided spectrum
            // For real-valued IFFT output: X[N-k] = conj(X[k]) for k = 1, 2, ..., N/2-1
            // This means real[N-k] = real[k] and imag[N-k] = -imag[k]
            let mirrorUpper = (bins == halfBins) ? (bins - 1) : bins
            if mirrorUpper > 1 {
                for k in 1..<mirrorUpper {
                    let rk = real[k]
                    let ik = imag[k]
                    real[nFFT - k] = rk     // Mirror real part
                    imag[nFFT - k] = -ik    // Conjugate: negate imaginary part
                }
            }
            // DC (k=0) and Nyquist (k=N/2) bins don't need mirroring

            // Compute inverse FFT using pre-computed twiddle factors
            // IFFT formula: x[n] = (1/N) * Σ(k=0 to N-1) X[k] * e^(i*2πkn/N)
            // For complex multiplication: (a+ib) * (c+is) = (ac-bs) + i(ad+bc)
            // We only need the real part since we enforced Hermitian symmetry
            var sigReal = [Float](repeating: 0, count: nFFT)
            let invN: Float = 1.0 / Float(nFFT)
            
            for n in 0..<nFFT {
                var sum: Float = 0
                for k in 0..<nFFT {
                    let c = cosTable[k * nFFT + n]  // cos(2πkn/N)
                    let s = sinTable[k * nFFT + n]  // sin(2πkn/N)
                    // Real part of X[k] * e^(i*2πkn/N) = real[k]*cos - imag[k]*sin
                    sum += real[k] * c - imag[k] * s
                }
                sigReal[n] = sum * invN
            }

            // Apply windowing and overlap-add (COLA) reconstruction
            let start = t * hop
            for i in 0..<nFFT {
                let w = window[i]
                let sample = sigReal[i] * w
                out[start + i] += sample      // Accumulate windowed signal
                acc[start + i] += w * w       // Accumulate squared window weights
            }
        }
        
        // Normalize by accumulated window weights for proper COLA reconstruction
        for i in 0..<totalLen {
            let a = max(acc[i], Constants.Processing.colaEpsilon)  // Avoid division by zero
            out[i] /= a
        }
        
        // Apply centered STFT alignment and trim to expected output length
        // Remove padding of nFFT/2 at both ends, trim to (frames-1)*hop samples
        let pad = nFFT / 2
        let desired = max(0, (frames - 1) * hop)
        let start = min(pad, out.count)
        let end = min(out.count, start + desired)
        
        if start < end {
            return Array(out[start..<end])
        }
        return out
    }
}

// MARK: - Constants

/// Constants for HAR post-processing operations.
/// Centralizes magic numbers and configuration values used throughout the processor.
private enum Constants {
    
    /// STFT analysis parameters with sensible defaults for 24kHz audio.
    enum STFT {
        /// Default FFT size: 800 samples ≈ 33ms frames at 24kHz.
        /// Provides good frequency resolution for speech signals.
        static let defaultNFFT = 800
        
        /// Default hop length: 200 samples ≈ 8.3ms hop at 24kHz.
        /// Creates 75% overlap (800-200)/800 for smooth reconstruction.
        static let defaultHop = 200
        
        /// Default window length: matches nFFT for standard STFT.
        /// Can be set smaller than nFFT for zero-padded analysis.
        static let defaultWinLength = 800
    }
    
    /// Mathematical constants used in calculations.
    enum Math {
        /// 2π constant for trigonometric calculations.
        static let twoPi: Float = 2.0 * Float.pi
    }
    
    /// Window function parameters.
    enum Window {
        /// Amplitude factor for Hann window: 0.5 in w[n] = 0.5 * (1 - cos(2πn/N)).
        static let hannAmplitude: Float = 0.5
    }
    
    /// Signal processing parameters and tuning constants.
    enum Processing {
        /// Default phase scaling factor, empirically tuned for best correlation.
        /// Value ~0.3 maximizes correlation with ground truth on 5s test fixtures.
        static let defaultPhaseScale: Float = 0.3
        
        /// Scaling factor for interior frequency bins in two-sided spectrum.
        /// Interior bins are scaled by 1/2 because they appear twice in full spectrum.
        static let halfScaleFactor: Float = 0.5
        
        /// Small epsilon to prevent division by zero in COLA normalization.
        /// Ensures numerical stability when window weights are very small.
        static let colaEpsilon: Float = 1e-6
    }
    
    /// Environment variable names for runtime configuration.
    enum Environment {
        /// Use raw phase values instead of sin(phase_scale * raw_phase).
        static let useRawPhase = "KOKORO_USE_RAW_PHASE"
        
        /// Scaling factor for phase values (default 0.3).
        static let phaseScale = "KOKORO_PHASE_SCALE"
        
        /// Tensor packing format: "interleaved" or blocked (default).
        static let packing = "KOKORO_PACKING"
        
        /// Disable 1/2 scaling of interior frequency bins.
        static let disableHalfScale = "KOKORO_DISABLE_HALF_SCALE"
    }
    
    /// Tensor packing format identifiers.
    enum Packing {
        /// Interleaved format: [mag0, phase0, mag1, phase1, ...].
        static let interleaved = "interleaved"
    }
}
