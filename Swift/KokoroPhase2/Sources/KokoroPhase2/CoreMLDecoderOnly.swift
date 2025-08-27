import Foundation
import CoreML
import AVFoundation

public enum KokoroPhase2Error: Error {
    case modelNotFound(String)
    case shapeMismatch(String)
    case predictionFailed(String)
}

/// Loads and runs the decoder-only 5s Core ML model.
///
/// Expected inputs:
/// - asr: (1,512,1,200) float32
/// - f0_curve: (1,1,1,400) float32
/// - n: (1,1,1,400) float32
/// - s: (1,128) float32
/// Output: audio samples as MLMultiArray, commonly (1,1,120000) float16/float32
public final class DecoderOnly5sRunner {
    private let model: MLModel
    public var rawModel: MLModel { model }

    public struct InputShapes {
        public let asr: [Int]
        public let f0: [Int]
        public let n: [Int]
        public let s: [Int]
        public init(asr: [Int] = [1,512,1,200], f0: [Int] = [1,1,1,400], n: [Int] = [1,1,1,400], s: [Int] = [1,128]) {
            self.asr = asr; self.f0 = f0; self.n = n; self.s = s
        }
    }

    /// Initializes the runner with a compiled Core ML model.
    ///
    /// Environment overrides:
    /// - `KOKORO_COMPUTE_UNITS` ∈ {"all","cpuAndGPU","cpuOnly"}
    /// - `KOKORO_CPU_ONLY` ∈ {"1","true"} (legacy shortcut; forces CPU only)
    public init(mlpackageURL: URL) throws {
        let config = MLModelConfiguration()
        // Default to ALL (ANE+GPU+CPU), allow override via env for diagnostics
        config.computeUnits = .all
        let env = ProcessInfo.processInfo.environment
        if let cpuOnly = env["KOKORO_CPU_ONLY"], ["1","true","yes"].contains(cpuOnly.lowercased()) {
            config.computeUnits = .cpuOnly
        } else if let units = env["KOKORO_COMPUTE_UNITS"]?.lowercased() {
            switch units {
            case "all": config.computeUnits = .all
            case "cpuandgpu", "cpu_gpu", "cpu+gpu": config.computeUnits = .cpuAndGPU
            case "cpuonly", "cpu": config.computeUnits = .cpuOnly
            default: break
            }
        }
        // Ensure model is compiled (supports .mlmodel and .mlpackage)
        let compiled = try MLModel.compileModel(at: mlpackageURL)
        self.model = try MLModel(contentsOf: compiled, configuration: config)
    }

    public func predict(asr: MLMultiArray, f0: MLMultiArray, n: MLMultiArray, s: MLMultiArray) throws -> (audio: [Float], sampleRate: Int) {
        let provider = try self.makeInputProvider(asr: asr, f0: f0, n: n, s: s)
        let out = try self.model.prediction(from: provider)
        guard let firstOut = out.featureNames.first, let arr = out.featureValue(for: firstOut)?.multiArrayValue else {
            throw KokoroPhase2Error.predictionFailed("No output multiArray")
        }
        // Flatten to Float
        let floats = try Self.flattenFloatArrayStatic(arr)
        return (floats, 24000)
    }

    private func makeInputProvider(asr: MLMultiArray, f0: MLMultiArray, n: MLMultiArray, s: MLMultiArray) throws -> MLDictionaryFeatureProvider {
        return try MLDictionaryFeatureProvider(dictionary: [
            "asr": MLFeatureValue(multiArray: asr),
            "f0_curve": MLFeatureValue(multiArray: f0),
            "n": MLFeatureValue(multiArray: n),
            "s": MLFeatureValue(multiArray: s),
        ])
    }

    public static func flattenFloatArrayStatic(_ arr: MLMultiArray) throws -> [Float] {
        switch arr.dataType {
        case .float32:
            return Array(UnsafeBufferPointer(start: arr.dataPointer.assumingMemoryBound(to: Float.self), count: arr.count))
        case .float16:
            // Convert Float16 to Float
            var out: [Float] = []
            out.reserveCapacity(arr.count)
            let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
            for i in 0..<arr.count {
                out.append(Float(Float16(bitPattern: ptr[i])))
            }
            return out
        default:
            // Fallback: cast via NSNumber
            return (0..<arr.count).map { i in arr[i].floatValue }
        }
    }

    private func flattenFloatArray(_ arr: MLMultiArray) throws -> [Float] {
        return try Self.flattenFloatArrayStatic(arr)
    }
}

public enum WAV {
    public static func writePCM16(fileURL: URL, samples: [Float], sampleRate: Int = 24000) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { throw KokoroPhase2Error.predictionFailed("buffer alloc") }
        buffer.frameLength = frameCount
        let dst = buffer.floatChannelData![0]
        for i in 0..<samples.count { dst[i] = samples[i] }
        let file = try AVAudioFile(forWriting: fileURL, settings: format.settings)
        try file.write(from: buffer)
    }
}
