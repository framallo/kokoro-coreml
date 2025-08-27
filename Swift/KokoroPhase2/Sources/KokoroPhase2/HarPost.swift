import Foundation
import Accelerate
import CoreML

public struct HarPostProcessor {
    public let nFFT: Int
    public let hop: Int
    public let winLength: Int
    private let window: [Float]
    private let cosTable: [Float]
    private let sinTable: [Float]

    public init(nFFT: Int = 800, hop: Int = 200, winLength: Int = 800) {
        self.nFFT = nFFT
        self.hop = hop
        self.winLength = winLength
        // Periodic Hann window matching torch.hann_window(periodic=True):
        // w[n] = 0.5 * (1 - cos(2*pi*n/N)), n in [0, N-1]
        var w = [Float](repeating: 0, count: winLength)
        let N = Float(winLength)
        let twoPiOverN = 2.0 * Float.pi / N
        for n in 0..<winLength {
            w[n] = 0.5 * (1.0 - cosf(twoPiOverN * Float(n)))
        }
        self.window = w
        // Precompute IFFT twiddle factors for N and all n
        var c = [Float](repeating: 0, count: nFFT * nFFT)
        var s = [Float](repeating: 0, count: nFFT * nFFT)
        let twoPiOverN_ifft = 2.0 * Float.pi / Float(nFFT)
        for k in 0..<nFFT {
            for n in 0..<nFFT {
                // Use negative sign in exponent for inverse transform: e^{-i 2πkn/N}
                let angle = -twoPiOverN_ifft * Float(k * n)
                c[k * nFFT + n] = cosf(angle)
                s[k * nFFT + n] = sinf(angle)
            }
        }
        self.cosTable = c
        self.sinTable = s
    }

    /// Reconstructs waveform from HAR CoreML network output.
    ///
    /// Environment controls:
    /// - `KOKORO_USE_RAW_PHASE=1`     → treat phase channel as raw angle (no sin)
    /// - `KOKORO_PACKING=interleaved` → channels ordered [mag0, phase0, mag1, phase1, ...]
    ///                                 default is blocked packing [mag bins..., phase bins...]
    /// - `KOKORO_DISABLE_HALF_SCALE=1`→ do not half interior bins before mirroring
    public func inverseFromNetworkOutput(_ output: MLMultiArray, channels: Int, frames: Int) throws -> [Float] {
        let data = try DecoderOnly5sRunner.flattenFloatArrayStatic(output)
        let strideT = frames
        let halfBins = nFFT/2 + 1 // spec bins (onesided)
        // HAR CoreML output packs [log|phase] pairs per bin in channel dimension: C_out = nFFT + 2
        // Use exact spec bin count
        let bins = halfBins
        precondition(bins == halfBins - 1 || bins == halfBins, "Unexpected HAR channels: \(channels)")

        // Output buffer with COLA normalization
        let totalLen = (frames - 1) * hop + nFFT
        var out = [Float](repeating: 0, count: totalLen)
        var acc = [Float](repeating: 0, count: totalLen)

        // Working buffers
        var real = [Float](repeating: 0, count: nFFT)
        var imag = [Float](repeating: 0, count: nFFT)

        let env = ProcessInfo.processInfo.environment
        let useRawPhase = (env["KOKORO_USE_RAW_PHASE"] == "1")
        let phaseScale: Float = {
            if let s = env["KOKORO_PHASE_SCALE"], let v = Float(s) { return v }
            // Empirically, ~0.3 maximizes corr on 5s fixture; keep tunable.
            return 0.3
        }()
        let packingInterleaved = (env["KOKORO_PACKING"]?.lowercased() == "interleaved")
        let halfScale = (env["KOKORO_DISABLE_HALF_SCALE"] == "1") ? false : true
        for t in 0..<frames {
            // Build complex spectrum from magnitude/phase
            for k in 0..<bins {
                let (logMag, phaseRaw): (Float, Float)
                if packingInterleaved {
                    // [mag0, phase0, mag1, phase1, ...]
                    logMag = data[(2*k)*strideT + t]
                    phaseRaw = data[(2*k + 1)*strideT + t]
                } else {
                    // [mag all bins..., phase all bins...]
                    logMag = data[k*strideT + t]
                    phaseRaw = data[(bins + k)*strideT + t]
                }
                let mag = expf(logMag)
                if k == 0 || k == bins - 1 {
                    // DC and Nyquist are purely real in onesided representation
                    real[k] = mag
                    imag[k] = 0
                } else {
                    let angle = useRawPhase ? (phaseScale * phaseRaw) : sinf(phaseScale * phaseRaw)
                    real[k] = mag * cosf(angle)
                    imag[k] = mag * sinf(angle)
                }
            }
            // Scale interior bins by 1/2 when constructing two-sided spectrum
            if halfScale && bins > 2 {
                for k in 1..<(bins - 1) {
                    real[k] *= 0.5
                    imag[k] *= 0.5
                }
            }
            // Hermitian symmetry for full spectrum synthesis
            // Mirror interior bins 1..bins-2 when Nyquist present; 1..bins-1 otherwise
            let mirrorUpper = (bins == halfBins) ? (bins - 1) : bins
            if mirrorUpper > 1 {
                for k in 1..<(mirrorUpper) {
                    let rk = real[k]
                    let ik = imag[k]
                    real[nFFT - k] = rk
                    imag[nFFT - k] = -ik
                }
            }
            // k=0 and k=Nyquist already set, mirror not needed

            // IFFT to time domain using precomputed twiddles (N is small)
            // Using definition with 1/N scaling on the inverse transform
            var sigReal = [Float](repeating: 0, count: nFFT)
            let invN: Float = 1.0 / Float(nFFT)
            for n in 0..<nFFT {
                var sum: Float = 0
                for k in 0..<nFFT {
                    let c = cosTable[k * nFFT + n]
                    let s = sinTable[k * nFFT + n]
                    // Re{ X[k] * e^{i 2πkn/N} } = Re{ (a+ib)(c+is) } = a c - b s
                    sum += real[k] * c - imag[k] * s
                }
                sigReal[n] = sum * invN
            }

            // Overlap-add with window
            let start = t * hop
            for i in 0..<nFFT {
                let w = window[i]
                let sample = sigReal[i] * w
                out[start + i] += sample
                acc[start + i] += w * w
            }
        }
        // Normalize by accumulated window
        for i in 0..<totalLen {
            let a = max(acc[i], 1e-6)
            out[i] /= a
        }
        // Centered STFT alignment: remove pad of nFFT/2 at both ends and trim to (frames-1)*hop samples
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
