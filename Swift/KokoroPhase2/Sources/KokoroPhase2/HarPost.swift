import Foundation
import Accelerate
import CoreML

public struct HarPostProcessor {
    public let nFFT: Int
    public let hop: Int
    public let winLength: Int
    private let dft: vDSP.DiscreteFourierTransform<Float>?
    private let window: [Float]

    public init(nFFT: Int = 800, hop: Int = 200, winLength: Int = 800) {
        self.nFFT = nFFT
        self.hop = hop
        self.winLength = winLength
        self.dft = try? vDSP.DiscreteFourierTransform<Float>(count: nFFT,
                                                             direction: .inverse,
                                                             transformType: .complexComplex,
                                                             ofType: Float.self)
        // Periodic Hann window
        var w = [Float](repeating: 0, count: winLength)
        vDSP_hann_window(&w, vDSP_Length(winLength), Int32(vDSP_HANN_NORM))
        self.window = w
    }

    public func inverseFromNetworkOutput(_ output: MLMultiArray, channels: Int, frames: Int) throws -> [Float] {
        let data = try DecoderOnly5sRunner.flattenFloatArrayStatic(output)
        let strideT = frames
        let halfBins = nFFT/2 + 1 // 401 when nFFT=800
        // HAR CoreML output packs [log|phase] pairs per bin in channel dimension: C = 2*halfBins-? ; decode from provided channels
        let bins = channels / 2
        precondition(bins == halfBins - 1 || bins == halfBins, "Unexpected HAR channels: \(channels)")

        // Output buffer with COLA normalization
        let totalLen = (frames - 1) * hop + nFFT
        var out = [Float](repeating: 0, count: totalLen)
        var acc = [Float](repeating: 0, count: totalLen)

        // Working buffers
        var real = [Float](repeating: 0, count: nFFT)
        var imag = [Float](repeating: 0, count: nFFT)
        var sigReal = [Float](repeating: 0, count: nFFT)
        var sigImag = [Float](repeating: 0, count: nFFT)

        for t in 0..<frames {
            // Build complex spectrum from magnitude/phase
            for k in 0..<bins {
                let logMag = data[k*strideT + t]
                let phaseRaw = data[(bins + k)*strideT + t]
                let mag = expf(logMag)
                let angle = sinf(phaseRaw) // mimic torch.sin before inverse
                real[k] = mag * cosf(angle)
                imag[k] = mag * sinf(angle)
            }
            // Hermitian symmetry for k=1..bins-1
            if bins >= 2 {
                for k in 1..<(bins) {
                    let rk = real[k]
                    let ik = imag[k]
                    real[nFFT - k] = rk
                    imag[nFFT - k] = -ik
                }
            }
            // k=0 and k=Nyquist already set, mirror not needed

            // IFFT to time domain
            real.withUnsafeBufferPointer { rPtr in
                imag.withUnsafeBufferPointer { iPtr in
                    sigReal.withUnsafeMutableBufferPointer { srPtr in
                        sigImag.withUnsafeMutableBufferPointer { siPtr in
                            dft?.transform(inputReal: rPtr, inputImaginary: iPtr, outputReal: &srPtr, outputImaginary: &siPtr)
                        }
                    }
                }
            }
            // Scale (inverse DFT is unnormalized)
            let scale: Float = 1.0 / Float(nFFT)
            for i in 0..<nFFT { sigReal[i] *= scale }

            // Overlap-add with window
            let start = t * hop
            for i in 0..<nFFT {
                let sample = sigReal[i] * window[i]
                out[start + i] += sample
                acc[start + i] += window[i]
            }
        }
        // Normalize by accumulated window
        for i in 0..<totalLen {
            let a = max(acc[i], 1e-6)
            out[i] /= a
        }
        return out
    }
}
