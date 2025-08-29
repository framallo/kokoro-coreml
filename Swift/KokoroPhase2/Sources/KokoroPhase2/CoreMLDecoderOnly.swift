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
    private var postFilter: MLModel?
    public var hasPostFilter: Bool { postFilter != nil }
    public let usedComputeUnits: MLComputeUnits

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
    public convenience init(mlpackageURL: URL) throws {
        try self.init(mlpackageURL: mlpackageURL, postFilterURL: nil)
    }

    /// Designated initializer enabling an explicit post-filter path.
    /// If `postFilterURL` is nil, will look for env overrides or default PF in ./coreml.
    public init(mlpackageURL: URL, postFilterURL: URL?) throws {
        let config = MLModelConfiguration()
        // Default to ALL (ANE+GPU+CPU), allow override via env for diagnostics
        var chosenUnits: MLComputeUnits = .all
        let env = ProcessInfo.processInfo.environment
        if let cpuOnly = env["KOKORO_CPU_ONLY"], ["1","true","yes"].contains(cpuOnly.lowercased()) {
            chosenUnits = .cpuOnly
        } else if let units = env["KOKORO_COMPUTE_UNITS"]?.lowercased() {
            switch units {
            case "all": chosenUnits = .all
            case "cpuandgpu", "cpu_gpu", "cpu+gpu": chosenUnits = .cpuAndGPU
            case "cpuonly", "cpu": chosenUnits = .cpuOnly
            default: break
            }
        }
        config.computeUnits = chosenUnits
        // Accept either compiled (.mlmodelc) or source (.mlmodel/.mlpackage)
        let compiledURL: URL
        switch mlpackageURL.pathExtension.lowercased() {
        case "mlmodelc": compiledURL = mlpackageURL
        default: compiledURL = try MLModel.compileModel(at: mlpackageURL)
        }
        do {
            self.model = try MLModel(contentsOf: compiledURL, configuration: config)
            self.usedComputeUnits = chosenUnits
        } catch {
            // Retry with CPU+GPU to avoid ANE constraints
            let fallback = MLModelConfiguration()
            fallback.computeUnits = .cpuAndGPU
            self.model = try MLModel(contentsOf: compiledURL, configuration: fallback)
            self.usedComputeUnits = .cpuAndGPU
        }
        // Post-filter loading: explicit URL takes precedence
        let disablePF = (env["KOKORO_DISABLE_POSTFILTER"] == "1")
        if !disablePF {
            if let pfURL = postFilterURL {
                let pfCompiled = try MLModel.compileModel(at: pfURL)
                self.postFilter = try? MLModel(contentsOf: pfCompiled, configuration: config)
            } else if let pfPath = env["KOKORO_POSTFILTER_PATH"], FileManager.default.fileExists(atPath: pfPath) {
                let pfCompiled = try MLModel.compileModel(at: URL(fileURLWithPath: pfPath))
                self.postFilter = try? MLModel(contentsOf: pfCompiled, configuration: config)
            } else {
                let defaultPF = URL(fileURLWithPath: "coreml/KokoroPostFilter.mlpackage")
                if FileManager.default.fileExists(atPath: defaultPF.path) {
                    let pfCompiled = try MLModel.compileModel(at: defaultPF)
                    self.postFilter = try? MLModel(contentsOf: pfCompiled, configuration: config)
                }
            }
        }
    }

    public func predict(asr: MLMultiArray, f0: MLMultiArray, n: MLMultiArray, s: MLMultiArray) throws -> (audio: [Float], sampleRate: Int) {
        let provider = try self.makeInputProvider(asr: asr, f0: f0, n: n, s: s)
        let out = try self.model.prediction(from: provider)
        guard let firstOut = out.featureNames.first, let arr = out.featureValue(for: firstOut)?.multiArrayValue else {
            throw KokoroPhase2Error.predictionFailed("No output multiArray")
        }
        // Flatten to Float
        var floats = try Self.flattenFloatArrayStatic(arr)
        floats = applyPostFilterIfAvailable(samples: floats)
        return (floats, 24000)
    }
    public func applyPostFilterIfAvailable(samples: [Float]) -> [Float] {
        guard let pf = postFilter else { return samples }
        // The post-filter expects a fixed length (5s=120000). If input differs, pad/crop.
        let expected: Int = 120000
        var work = samples
        if work.count < expected {
            work.append(contentsOf: repeatElement(0.0, count: expected - work.count))
        } else if work.count > expected {
            work = Array(work.prefix(expected))
        }
        let T = work.count
        guard let inArr = try? MLMultiArray(shape: [1,1,NSNumber(value:T)], dataType: .float32) else { return samples }
        let ptr = UnsafeMutableBufferPointer(start: inArr.dataPointer.assumingMemoryBound(to: Float.self), count: inArr.count)
        for i in 0..<T { ptr[i] = work[i] }
        guard let out = try? pf.prediction(from: MLDictionaryFeatureProvider(dictionary:["audio_in": MLFeatureValue(multiArray: inArr)])),
              let outArr = out.featureValue(for: out.featureNames.first!)?.multiArrayValue,
              outArr.count == T,
              let outFloats = try? Self.flattenFloatArrayStatic(outArr) else {
            return samples
        }
        // Trim back to original length if padded
        if samples.count < T {
            return Array(outFloats.prefix(samples.count))
        }
        return outFloats
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
    public static func writePCM16(fileURL: URL, samples: [Float], sampleRate: Int = Constants.Audio.sampleRate) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { 
            throw KokoroPhase2Error.predictionFailed("Failed to allocate audio buffer") 
        }
        buffer.frameLength = frameCount
        let dst = buffer.floatChannelData![0]
        for i in 0..<samples.count { dst[i] = samples[i] }
        // Write buffer to file
        let file = try AVAudioFile(forWriting: fileURL, settings: format.settings)
        try file.write(from: buffer)
    }
}

// MARK: - Constants

/// Centralized constants for Kokoro Phase 2 operations.
/// All magic numbers and string literals are defined here with clear documentation.
private enum Constants {
    
    /// Audio-related constants used throughout the system.
    enum Audio {
        /// Standard sample rate for Kokoro TTS output (24 kHz).
        /// Used by all models and audio processing components.
        static let sampleRate: Int = 24000
        
        /// Number of samples in a 5-second audio clip at 24kHz.
        /// Used by post-filter and fixed-length processing components.
        static let fiveSecondSamples: Int = 120000
    }
    
    /// Environment variable names for runtime configuration.
    enum Environment {
        /// Override compute unit selection: "all", "cpuAndGPU", "cpuOnly".
        static let computeUnits = "KOKORO_COMPUTE_UNITS"
        
        /// Legacy flag to force CPU-only execution.
        static let cpuOnly = "KOKORO_CPU_ONLY"
        
        /// Disable post-filter loading when set to "1".
        static let disablePostFilter = "KOKORO_DISABLE_POSTFILTER"
        
        /// Override default post-filter model path.
        static let postFilterPath = "KOKORO_POSTFILTER_PATH"
        
        /// Values that evaluate to "true" for boolean environment variables.
        static let trueBoolValues: Set<String> = ["1", "true", "yes"]
    }
    
    /// File extensions for model formats.
    enum FileExtensions {
        /// Compiled Core ML model extension.
        static let compiledModel = "mlmodelc"
    }
    
    /// Default file paths used by the system.
    enum Paths {
        /// Default location for post-filter model.
        static let defaultPostFilter = "coreml/KokoroPostFilter.mlpackage"
    }
    
    /// Core ML model input tensor names (must match model contract).
    enum ModelInputs {
        /// ASR token embedding tensor name.
        static let asr = "asr"
        
        /// F0 curve tensor name.
        static let f0Curve = "f0_curve"
        
        /// Noise control tensor name.
        static let noise = "n"
        
        /// Speaker embedding tensor name.
        static let speaker = "s"
    }
    
    /// Post-filter model configuration.
    enum PostFilter {
        /// Input tensor name for post-filter model.
        static let inputTensorName = "audio_in"
    }
    
    /// Default tensor shapes for the 5-second model.
    enum DefaultShapes {
        /// ASR token embedding shape: (batch, tokens, height, sequence).
        static let asr = [1, 512, 1, 200]
        
        /// F0 curve shape: (batch, channels, height, time_frames).
        static let f0 = [1, 1, 1, 400]
        
        /// Noise control shape: (batch, channels, height, time_frames).
        static let n = [1, 1, 1, 400]
        
        /// Speaker embedding shape: (batch, embedding_dim).
        static let s = [1, 128]
    }
}
