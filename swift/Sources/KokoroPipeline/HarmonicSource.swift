/// Native Swift implementation of the hn-nsf harmonic source generation path.
///
/// Replaces the PyTorch ``SourceModuleHnNSF`` / ``SineGen`` + STFT pipeline
/// that runs on CPU in ``build_decoder_har_post_inputs_np()``.
///
/// This path has correlation ~0.00 in CoreML (see ``README/Notes/debug-notes.md``)
/// and MUST stay on CPU. The computation is pure DSP: sine wave generation at
/// harmonic frequencies + FFT.
///
/// ## Architecture (matching PyTorch)
///
/// 1. F0 upsample: nearest-neighbor interpolation, scale_factor=300
/// 2. SineGen: 9 harmonics (fundamental + 8 overtones)
///    - Downsample phase increments → cumsum → upsample (with scale correction)
///    - ``sin(2π * accumulated_phase)``
/// 3. SourceModuleHnNSF: merge 9 harmonics via learned Linear(9→1) + Tanh
/// 4. STFT: n_fft=20, hop=5, Hann window, center padding (replicate)
/// 5. Output: ``[magnitude, phase]`` concatenated → shape ``(1, 22, n_frames)``
///
/// ## Critical: Double-precision phase accumulation
///
/// The phase integrator in SineGen MUST use Float64 (Double). This is load-bearing.
/// At 24 kHz over 8.35 s (~200,000 samples of phase integration), Float32 drift
/// compounds and corrupts the harmonic spectrum. Float32 is used only at the final
/// output stage.
///
/// Called by:
/// - ``KokoroPipeline.synthesize()`` to generate the ``har`` tensor for
///   ``GeneratorFromHar`` CoreML model input.
///
/// Reference implementation:
/// - ``kokoro/istftnet.py:227-389`` (SineGen, SourceModuleHnNSF)
/// - ``kokoro/istftnet.py:199-226`` (TorchSTFT)
/// - ``kokoro/custom_stft.py:43-382`` (CustomSTFT, DFT math)
/// - ``kokoro/synthesis_backends.py:95-99`` (usage in build_decoder_har_post_inputs_np)

import Foundation
import Accelerate

// MARK: - Constants

/// Parameters matching the Kokoro Generator configuration.
///
/// Source: ``kokoro/istftnet.py`` Generator.__init__
public enum HarmonicConstants {
    /// Audio sample rate in Hz.
    public static let sampleRate: Double = 24000.0
    /// F0 upsample factor = prod(upsample_rates) * gen_istft_hop_size = 10*6*5 = 300.
    public static let upsampleScale: Int = 300
    /// Number of harmonic overtones above fundamental. Total harmonics = harmonicNum + 1 = 9.
    public static let harmonicNum: Int = 8
    /// Total number of harmonic components (fundamental + overtones).
    public static let harmonicDim: Int = 9
    /// Amplitude of sine source signal.
    public static let sineAmp: Float = 0.1
    /// Std of additive Gaussian noise for voiced regions.
    public static let noiseStd: Float = 0.003
    /// F0 threshold (Hz) for voiced/unvoiced classification.
    public static let voicedThreshold: Float = 10.0
    /// STFT window size (n_fft).
    public static let stftNfft: Int = 20
    /// STFT hop length.
    public static let stftHop: Int = 5
    /// STFT frequency bins = n_fft / 2 + 1.
    public static let stftFreqBins: Int = 11
    /// Output har channels = 2 * freq_bins (magnitude + phase).
    public static let harChannels: Int = 22
}

// MARK: - F0 Upsample

/// Nearest-neighbor upsample of F0 curve by ``HarmonicConstants.upsampleScale``.
///
/// Matches ``nn.Upsample(scale_factor=300)`` with default mode ``'nearest'``.
///
/// - Parameter f0: F0 curve in Hz, length T. Unvoiced frames should be 0.
/// - Returns: Upsampled F0, length T * 300.
public func f0Upsample(_ f0: [Float]) -> [Float] {
    let scale = HarmonicConstants.upsampleScale
    var result = [Float](repeating: 0, count: f0.count * scale)
    for (i, val) in f0.enumerated() {
        let start = i * scale
        for j in 0..<scale {
            result[start + j] = val
        }
    }
    return result
}

// MARK: - SineGen

/// Generate harmonic sine waves from an upsampled F0 curve.
///
/// Matches ``SineGen._f02sine`` + ``SineGen.forward`` from ``kokoro/istftnet.py:261-343``.
///
/// ## Phase accumulation (load-bearing Double precision)
///
/// The `long` bakeoff input at 8.35 s means ~200,000 samples of cumulative phase.
/// Float32 has ~7 decimal digits. Phase wraps at 2pi ~= 6.28, so after ~1M increments
/// the accumulated error is significant. We use Float64 for the phase integrator and
/// only downcast to Float32 at the final sin() output.
///
/// - Parameters:
///   - f0Upsampled: F0 in Hz at 24 kHz sample rate, length L. Unvoiced = 0.
///   - seed: Random seed for reproducible initial phase noise on overtones.
///           Pass nil for random (non-reproducible) behavior matching training.
/// - Returns: Merged sine waveform, length L (after SourceModuleHnNSF linear merge).
///            Shape semantics: (L,) matching the squeezed PyTorch output.
public func sineGen(
    f0Upsampled: [Float],
    linearWeights: [Float],
    linearBias: Float,
    seed: UInt64? = nil
) -> [Float] {
    let L = f0Upsampled.count
    let dim = HarmonicConstants.harmonicDim // 9
    let sr = HarmonicConstants.sampleRate   // 24000.0
    let scale = HarmonicConstants.upsampleScale // 300
    let sineAmp = HarmonicConstants.sineAmp
    let noiseStd = HarmonicConstants.noiseStd
    let threshold = HarmonicConstants.voicedThreshold

    // --- Step 1: Build harmonic frequencies (L, 9) ---
    // fn[t, h] = f0[t] * (h + 1) for h in 0..<9
    // Normalize to phase increment: rad[t, h] = (fn[t, h] / sr) % 1
    // We work in Double for the phase accumulation path.

    // --- Step 2: Downsample phase increments ---
    let downLen = max(1, (L + scale - 1) / scale)
    let upLen = downLen * scale

    // Pre-allocate reusable buffers (avoids 9 harmonics × 6 allocations)
    var radValues = [Double](repeating: 0, count: L)
    var radDS = [Double](repeating: 0, count: downLen)
    var cumPhase = [Double](repeating: 0, count: downLen)
    var phaseScaled = [Double](repeating: 0, count: downLen)
    var phaseUp = [Double](repeating: 0, count: max(L, upLen))
    var sinResult = [Double](repeating: 0, count: L)
    var floatSines = [Float](repeating: 0, count: L)

    // Flat buffer for all harmonics: sineWaves[h * L ..< (h+1) * L]
    var sineWaves = [Float](repeating: 0, count: dim * L)

    // RNG for initial phase noise
    var rng: RandomNumberGenerator = seed.map { SeededRNG(seed: $0) as RandomNumberGenerator } ?? SystemRandomNumberGenerator()

    let twoPiTimesScale = 2.0 * Double.pi * Double(scale)

    for h in 0..<dim {
        let invSr = Double(h + 1) / sr

        // Compute phase increments in Double
        for t in 0..<L {
            let r = (Double(f0Upsampled[t]) * invSr).truncatingRemainder(dividingBy: 1.0)
            radValues[t] = r < 0 ? r + 1.0 : r
        }

        // Add initial phase noise for overtones (h > 0), not fundamental
        if h > 0 {
            radValues[0] += Double.random(in: 0..<1, using: &rng)
        }

        // Downsample via linear interpolation (in-place into radDS)
        linearInterpolateInto(from: radValues, count: L, into: &radDS, targetLen: downLen)

        // Cumulative sum in Double precision (THE critical accumulator)
        if downLen > 0 {
            cumPhase[0] = radDS[0]
            for t in 1..<downLen {
                cumPhase[t] = cumPhase[t - 1] + radDS[t]
            }
        }

        // Multiply by 2*pi*scale and upsample
        vDSP_vsmulD(cumPhase, 1, [twoPiTimesScale], &phaseScaled, 1, vDSP_Length(downLen))

        // Upsample phase to upLen (in-place into phaseUp)
        linearInterpolateInto(from: phaseScaled, count: downLen, into: &phaseUp, targetLen: upLen)

        // Use first L elements (upLen >= L since upLen = downLen * scale and L <= downLen * scale)
        // Vectorized sin using vForce
        var n = Int32(L)
        vvsin(&sinResult, phaseUp, &n)

        // Convert Double -> Float, scale by sineAmp, write into flat sineWaves buffer
        vDSP_vdpsp(sinResult, 1, &floatSines, 1, vDSP_Length(L))
        var ampScalar = sineAmp
        vDSP_vsmul(floatSines, 1, &ampScalar, &sineWaves[h * L], 1, vDSP_Length(L))
    }

    // --- Step 3: Apply voiced/unvoiced mask + noise ---
    // uv[t] = f0[t] > threshold ? 1 : 0
    // For voiced: sine_waves * uv
    // For unvoiced: noise with amplitude sine_amp / 3
    // noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3

    // Pre-compute voiced/unvoiced mask
    var uvMask = [Float](repeating: 0, count: L)
    for t in 0..<L {
        uvMask[t] = f0Upsampled[t] > threshold ? 1.0 : 0.0
    }

    // Pre-generate all Gaussian noise at once (fast, vectorized approach)
    // Total noise needed: dim * L values
    let totalNoise = dim * L
    var gaussianNoise = [Float](repeating: 0, count: totalNoise)
    generateGaussianNoise(into: &gaussianNoise, count: totalNoise, seed: seed)

    // Apply voiced/unvoiced mask + noise to each harmonic
    let unvoicedNoiseAmp = sineAmp / 3.0
    for h in 0..<dim {
        let sineOffset = h * L
        let noiseOffset = h * L
        for t in 0..<L {
            let uv = uvMask[t]
            let noiseAmp = uv * noiseStd + (1.0 - uv) * unvoicedNoiseAmp
            sineWaves[sineOffset + t] = sineWaves[sineOffset + t] * uv + noiseAmp * gaussianNoise[noiseOffset + t]
        }
    }

    // --- Step 4: Linear merge (9 → 1) + Tanh ---
    // l_linear: Linear(9, 1) with learned weights and bias
    // Vectorized: for each time step, dot product of 9 harmonic values with weights
    assert(linearWeights.count == dim, "Linear weights must have \(dim) elements")

    var merged = [Float](repeating: 0, count: L)

    // Compute weighted sum across harmonics
    // merged[t] = bias + sum_h(sineWaves[h*L+t] * weights[h])
    for t in 0..<L {
        var sum: Float = linearBias
        for h in 0..<dim {
            sum += sineWaves[h * L + t] * linearWeights[h]
        }
        merged[t] = tanh(sum)
    }

    return merged
}

// MARK: - STFT

/// Forward STFT matching ``TorchSTFT.transform`` / ``CustomSTFT.transform``.
///
/// Parameters: n_fft=20, hop=5, Hann window, center=True (replicate padding).
///
/// - Parameter signal: Real-valued signal, length S.
/// - Returns: Tuple of (magnitude, phase), each with shape (freqBins, nFrames)
///            where freqBins = 11, nFrames depends on signal length.
///            Stored as flat arrays in frequency-major order (freq outer, time inner).
public func stftTransform(_ signal: [Float]) -> (magnitude: [Float], phase: [Float]) {
    let nfft = HarmonicConstants.stftNfft   // 20
    let hop = HarmonicConstants.stftHop     // 5
    let freqBins = HarmonicConstants.stftFreqBins // 11
    let padLen = nfft / 2                   // 10

    // Center padding (replicate mode, matching PyTorch)
    var padded = [Float](repeating: 0, count: signal.count + 2 * padLen)
    // Left pad: replicate first sample
    for i in 0..<padLen {
        padded[i] = signal[0]
    }
    // Copy signal
    for i in 0..<signal.count {
        padded[padLen + i] = signal[i]
    }
    // Right pad: replicate last sample
    let lastSample = signal.last ?? 0
    for i in 0..<padLen {
        padded[padLen + signal.count + i] = lastSample
    }

    // Number of frames
    let paddedLen = padded.count
    let nFrames = (paddedLen - nfft) / hop + 1

    // Hann window (periodic, matching torch.hann_window(20, periodic=True))
    var window = [Float](repeating: 0, count: nfft)
    for n in 0..<nfft {
        window[n] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(n) / Float(nfft)))
    }

    // DFT basis (matching custom_stft.py DFT math)
    // For k = 0..<freqBins, n = 0..<nfft:
    //   real_basis[k][n] = window[n] * cos(2*pi*k*n / nfft)
    //   imag_basis[k][n] = window[n] * (-sin(2*pi*k*n / nfft))
    let twoPiOverN = 2.0 * Float.pi / Float(nfft)
    var realBasis = [[Float]](repeating: [Float](repeating: 0, count: nfft), count: freqBins)
    var imagBasis = [[Float]](repeating: [Float](repeating: 0, count: nfft), count: freqBins)
    for k in 0..<freqBins {
        for n in 0..<nfft {
            let angle = twoPiOverN * Float(k) * Float(n)
            realBasis[k][n] = window[n] * cos(angle)
            imagBasis[k][n] = window[n] * (-sin(angle))
        }
    }

    // Compute STFT frame by frame
    var magnitude = [Float](repeating: 0, count: freqBins * nFrames)
    var phase = [Float](repeating: 0, count: freqBins * nFrames)

    for frame in 0..<nFrames {
        let offset = frame * hop
        for k in 0..<freqBins {
            var realSum: Float = 0
            var imagSum: Float = 0
            for n in 0..<nfft {
                let sample = padded[offset + n]
                realSum += sample * realBasis[k][n]
                imagSum += sample * imagBasis[k][n]
            }
            let mag = sqrt(realSum * realSum + imagSum * imagSum + 1e-14)
            let ph = atan2(imagSum, realSum)
            magnitude[k * nFrames + frame] = mag
            phase[k * nFrames + frame] = ph
        }
    }

    return (magnitude: magnitude, phase: phase)
}

/// Build the full ``har`` tensor from an F0 curve.
///
/// This is the top-level function that replaces the PyTorch hn-nsf path in
/// ``build_decoder_har_post_inputs_np()``.
///
/// - Parameters:
///   - f0Padded: F0 curve in Hz, padded to bucket geometry (length = full_f0_len).
///   - linearWeights: Learned weights from ``SourceModuleHnNSF.l_linear``, shape (9,).
///   - linearBias: Learned bias from ``SourceModuleHnNSF.l_linear``, scalar.
///   - seed: Optional random seed for reproducibility.
/// - Returns: Flat array of shape (22, nFrames) in channel-major order,
///            matching ``torch.cat([har_spec, har_phase], dim=1)`` squeezed to 2D.
public func buildHar(
    f0Padded: [Float],
    linearWeights: [Float],
    linearBias: Float,
    seed: UInt64? = nil
) -> (har: [Float], nFrames: Int) {
    // 1. F0 upsample (nearest, x300)
    let f0Up = f0Upsample(f0Padded)

    // 2. SineGen + SourceModuleHnNSF merge
    let harSource = sineGen(
        f0Upsampled: f0Up,
        linearWeights: linearWeights,
        linearBias: linearBias,
        seed: seed
    )

    // 3. STFT transform
    let (mag, ph) = stftTransform(harSource)
    let freqBins = HarmonicConstants.stftFreqBins // 11
    let nFrames = mag.count / freqBins

    // 4. Concatenate [magnitude, phase] along channel dim
    // Output layout: (22, nFrames) in channel-major order
    // First 11 channels = magnitude, next 11 = phase
    var har = [Float](repeating: 0, count: HarmonicConstants.harChannels * nFrames)
    // magnitude channels (0..<11)
    for k in 0..<freqBins {
        for t in 0..<nFrames {
            har[k * nFrames + t] = mag[k * nFrames + t]
        }
    }
    // phase channels (11..<22)
    for k in 0..<freqBins {
        for t in 0..<nFrames {
            har[(freqBins + k) * nFrames + t] = ph[k * nFrames + t]
        }
    }

    return (har: har, nFrames: nFrames)
}

// MARK: - Interpolation Helpers

/// Linear interpolation (downsample) matching ``F.interpolate(mode='linear')``.
///
/// PyTorch's align_corners=False (default for 1D) interpolation.
func linearInterpolateDown(_ input: [Double], targetLen: Int) -> [Double] {
    if input.isEmpty || targetLen <= 0 { return [] }
    if targetLen == 1 {
        // PyTorch: single output = mean of all inputs for align_corners=False
        let sum = input.reduce(0, +)
        return [sum / Double(input.count)]
    }
    if input.count == targetLen { return input }

    var result = [Double](repeating: 0, count: targetLen)
    let srcLen = Double(input.count)
    let dstLen = Double(targetLen)

    // PyTorch F.interpolate align_corners=False:
    // src_idx = (dst_idx + 0.5) * src_len / dst_len - 0.5
    for i in 0..<targetLen {
        let srcIdx = (Double(i) + 0.5) * srcLen / dstLen - 0.5
        let srcIdxClamped = max(0, min(srcIdx, srcLen - 1))
        let lo = Int(srcIdxClamped)
        let hi = min(lo + 1, input.count - 1)
        let frac = srcIdxClamped - Double(lo)
        result[i] = input[lo] * (1.0 - frac) + input[hi] * frac
    }
    return result
}

/// Linear interpolation (upsample) matching ``F.interpolate(mode='linear')``.
func linearInterpolateUp(_ input: [Double], targetLen: Int) -> [Double] {
    // Same logic as down — F.interpolate is symmetric
    return linearInterpolateDown(input, targetLen: targetLen)
}

/// In-place linear interpolation into a pre-allocated buffer.
///
/// Avoids allocation per call — critical for the 9-harmonic inner loop.
func linearInterpolateInto(from input: [Double], count srcCount: Int, into output: inout [Double], targetLen: Int) {
    if srcCount == 0 || targetLen <= 0 { return }
    if targetLen == 1 {
        var sum = 0.0
        for i in 0..<srcCount { sum += input[i] }
        output[0] = sum / Double(srcCount)
        return
    }
    if srcCount == targetLen {
        for i in 0..<srcCount { output[i] = input[i] }
        return
    }

    let srcLen = Double(srcCount)
    let dstLen = Double(targetLen)
    let ratio = srcLen / dstLen

    for i in 0..<targetLen {
        let srcIdx = (Double(i) + 0.5) * ratio - 0.5
        let srcIdxClamped = max(0, min(srcIdx, srcLen - 1))
        let lo = Int(srcIdxClamped)
        let hi = min(lo + 1, srcCount - 1)
        let frac = srcIdxClamped - Double(lo)
        output[i] = input[lo] * (1.0 - frac) + input[hi] * frac
    }
}

// MARK: - Random Number Generation

/// Seeded RNG for reproducible benchmarks.
struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

/// Fast bulk Gaussian noise generation using Box-Muller transform.
///
/// Generates `count` Gaussian random samples into a pre-allocated buffer.
/// Much faster than per-sample generation because it avoids protocol dispatch
/// overhead on `RandomNumberGenerator` and can vectorize the math.
func generateGaussianNoise(into buffer: inout [Float], count: Int, seed: UInt64? = nil) {
    // Generate pairs of uniform random numbers, then Box-Muller transform
    var rng = SeededRNG(seed: seed ?? UInt64.random(in: 0..<UInt64.max))

    // Generate uniform random pairs and apply Box-Muller
    var i = 0
    while i < count - 1 {
        let u1 = max(Float.ulpOfOne, Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF))
        let u2 = Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF)
        let r = sqrt(-2.0 * log(u1))
        let theta = 2.0 * Float.pi * u2
        buffer[i] = r * cos(theta)
        buffer[i + 1] = r * sin(theta)
        i += 2
    }
    // Handle odd count
    if i < count {
        let u1 = max(Float.ulpOfOne, Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF))
        let u2 = Float(rng.next() & 0xFFFFFF) / Float(0xFFFFFF)
        buffer[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
    }
}
