import Foundation
import CoreML
import AVFoundation

/// Public API surface for Kokoro TTS synthesis.
///
/// Phase 3 goals addressed here:
/// - Add 15s & 30s buckets
/// - Dynamic bucket selection based on estimated duration
/// - Simple entrypoint returning audio samples for HAR path
public enum KokoroTTS {

    /// Supported synthesis buckets (fixed frame windows tuned for ANE)
    public enum Bucket: Int, CaseIterable {
        case s5 = 5
        case s15 = 15
        case s30 = 30

        /// Returns the mlpackage file name for this bucket.
        var modelResourceName: String {
            switch self {
            case .s5: return "KokoroDecoder_HAR_5s.mlpackage"
            case .s15: return "KokoroDecoder_HAR_15s.mlpackage"
            case .s30: return "KokoroDecoder_HAR_30s.mlpackage"
            }
        }
    }

    /// Minimal error set for API consumers.
    public enum Error: Swift.Error { case modelNotFound(String); case predictionFailed(String) }

    /// Simple phonemizer placeholder.
    /// Replace with real Swift phonemizer in a follow-up edit; for now, we
    /// expose an internal hook so the executable can continue using the Phase 2 JSON path.
    public struct Phonemizer {
        public init() {}
        public func tokenize(_ text: String, voice: String) -> [Int] {
            // TODO: Replace with real phoneme-to-id mapping
            // Temporary: map ASCII to pseudo IDs for API completeness
            return text.unicodeScalars.map { Int($0.value % 96) }
        }
    }

    /// Synthesizes audio for input tensors using HAR decoder path.
    /// Automatically picks the appropriate bucket based on input time length.
    public static func synthesizeWithHAR(
        asr: MLMultiArray,
        f0: MLMultiArray,
        n: MLMultiArray,
        s: MLMultiArray,
        harSpec: MLMultiArray,
        harPhase: MLMultiArray
    ) throws -> [Float] {
        let frames = asr.shape.last?.intValue ?? 0
        let bucket = pickBucket(fromFrames: frames)
        let model = try loadModel(resourceName: bucket.modelResourceName)
        let out = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
            "har_spec": MLFeatureValue(multiArray: harSpec),
            "har_phase": MLFeatureValue(multiArray: harPhase),
            "asr": MLFeatureValue(multiArray: asr),
            "f0_curve": MLFeatureValue(multiArray: f0),
            "n": MLFeatureValue(multiArray: n),
            "s": MLFeatureValue(multiArray: s),
        ]))
        guard let outName = model.modelDescription.outputDescriptionsByName.keys.first,
              let arr = out.featureValue(for: outName)?.multiArrayValue else {
            throw Error.predictionFailed("Missing output tensor")
        }
        if ProcessInfo.processInfo.environment["LOG_DEBUG"] == "1" {
            let shapeStr = arr.shape.map { $0.stringValue }.joined(separator: "x")
            print("HAR output shape=\(shapeStr)")
        }
        // Detect output kind. Some Decoder_HAR variants emit waveform [1,1,T];
        // others emit latent features [C,T] or [1,C,T]. Handle both.
        let shape = arr.shape.map { $0.intValue }
        if shape == [1, 1, shape.last ?? 0] || (shape.count == 1) {
            // Waveform path
            var samples = mlMultiArrayToVector(arr)
            applyPostprocessing(&samples, sampleRate: 24000)
            return samples
        }
        if shape.count == 3, shape.first == 1, shape[1] > 1 {
            // Latent path: [1, C, T]
            let channels = shape[1]
            let framesT = shape[2]
            // If requested, delegate reconstruction to Python parity script for exactness/speed
            if ProcessInfo.processInfo.environment["PY_RECON"] == "1" {
                // Write latent as CSV (each line = channel, columns = time)
                var lines: [String] = []; lines.reserveCapacity(channels)
                for c in 0..<channels {
                    var row = [String](); row.reserveCapacity(framesT)
                    for t in 0..<framesT { row.append(String(arr[[0, NSNumber(value: c), NSNumber(value: t)]].floatValue)) }
                    lines.append(row.joined(separator: ","))
                }
                let tmpDir = FileManager.default.temporaryDirectory
                let latentURL = tmpDir.appendingPathComponent("kokoro_latent_\(UUID().uuidString).csv")
                try lines.joined(separator: "\n").write(to: latentURL, atomically: true, encoding: .utf8)
                let outWav = tmpDir.appendingPathComponent("kokoro_out_\(UUID().uuidString).wav")
                // Call tools/reconstruct_from_latent.py
                let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                let script = cwd.appendingPathComponent("tools/reconstruct_from_latent.py").path
                let proc = Process(); proc.launchPath = "/usr/bin/env"
                proc.arguments = ["python3", script, "--latent", latentURL.path, "--out_wav", outWav.path, "--n_fft", "20", "--hop", "5", "--sr", "24000"]
                let pipe = Pipe(); proc.standardOutput = pipe; proc.standardError = pipe
                proc.launch(); proc.waitUntilExit()
                guard proc.terminationStatus == 0 else {
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    let log = String(data: data, encoding: .utf8) ?? ""
                    throw Error.predictionFailed("Python iSTFT failed: \(log)")
                }
                // Read WAV back into samples
                let data = try Data(contentsOf: outWav)
                guard data.count > 44 else { return [] }
                let pcm = data.dropFirst(44)
                let samplesI16: [Int16] = pcm.withUnsafeBytes { raw in
                    let buf = raw.bindMemory(to: Int16.self); return Array(buf)
                }
                var sOut = samplesI16.map { Float($0) / 32767.0 }
                applyPostprocessing(&sOut, sampleRate: 24000)
                return sOut
            }
            // Otherwise do Swift reconstruction
            var x: [[Float]] = Array(repeating: Array(repeating: 0, count: framesT), count: channels)
            for c in 0..<channels {
                for t in 0..<framesT { x[c][t] = arr[[0, NSNumber(value: c), NSNumber(value: t)]].floatValue }
            }
            var sOut = reconstructWaveformFromDecoderHAROutput(xChannelsByTime: x)
            applyPostprocessing(&sOut, sampleRate: 24000)
            return sOut
        }
        if shape.count == 2 {
            // Latent path: [C, T]
            let channels = shape[0]
            let framesT = shape[1]
            var x: [[Float]] = Array(repeating: Array(repeating: 0, count: framesT), count: channels)
            for c in 0..<channels {
                for t in 0..<framesT { x[c][t] = arr[[NSNumber(value: c), NSNumber(value: t)]].floatValue }
            }
            var sOut = reconstructWaveformFromDecoderHAROutput(xChannelsByTime: x)
            applyPostprocessing(&sOut, sampleRate: 24000)
            return sOut
        }

        // Fallback: interpret as flat waveform
        var samples = mlMultiArrayToVector(arr)
        applyPostprocessing(&samples, sampleRate: 24000)
        return samples
    }

    /// Selects bucket based on input length (frames at ~40 fps).
    public static func pickBucket(fromFrames frames: Int) -> Bucket {
        // Heuristic thresholds: 5s ~ 200 frames; 15s ~ 600 frames
        if frames <= 200 { return .s5 }
        if frames <= 600 { return .s15 }
        return .s30
    }

    /// Attempts to locate and load the CoreML model.
    private static func loadModel(resourceName: String) throws -> MLModel {
        // 1) Expect resources in the executable bundle resources directory
        let execURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
        let bundleURL = execURL
            .appendingPathComponent("KokoroPhase2_KokoroPhase2.resources")
            .appendingPathComponent(resourceName)
        let candidateURLs: [URL] = [
            bundleURL,
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Swift/KokoroPhase2/Resources/")
                .appendingPathComponent(resourceName)
        ]
        guard let found = candidateURLs.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw Error.modelNotFound(resourceName)
        }
        let compiled = try MLModel.compileModel(at: found)
        let config = MLModelConfiguration()
        if ProcessInfo.processInfo.environment["CPU_ONLY"] == "1" {
            config.computeUnits = .cpuOnly
        } else if ProcessInfo.processInfo.environment["GPU_ONLY"] == "1" {
            config.computeUnits = .cpuAndGPU
        } else {
            config.computeUnits = .all
        }
        return try MLModel(contentsOf: compiled, configuration: config)
    }

    /// Converts any MLMultiArray into a flat [Float], preserving element order.
    private static func mlMultiArrayToVector(_ arr: MLMultiArray) -> [Float] {
        var out = [Float](repeating: 0, count: arr.count)
        for i in 0..<arr.count { out[i] = arr[i].floatValue }
        return out
    }

    /// Optional postprocessing to reduce residual static and DC.
    /// Controlled by env vars:
    /// - DC_REMOVE=1: subtract mean of last window (default 100 ms)
    /// - DC_WINDOW_MS: override window length (default 100)
    /// - TAIL_FADE_MS: apply linear fade over last N ms (default 30)
    private static func applyPostprocessing(_ samples: inout [Float], sampleRate: Int) {
        guard !samples.isEmpty else { return }
        let env = ProcessInfo.processInfo.environment
        // DC removal on tail window
        if env["DC_REMOVE"] == "1" {
            let winMs = Int(env["DC_WINDOW_MS"] ?? "100") ?? 100
            let win = max(1, min(samples.count, Int((Float(winMs)/1000.0) * Float(sampleRate))))
            if win > 0 {
                var mean: Double = 0
                for i in 0..<win { mean += Double(samples[samples.count - 1 - i]) }
                mean /= Double(win)
                let m = Float(mean)
                for i in 0..<samples.count { samples[i] -= m }
            }
        }
        // Tail fade
        let fadeMs = Int(env["TAIL_FADE_MS"] ?? "30") ?? 30
        let fade = max(0, min(samples.count, Int((Float(fadeMs)/1000.0) * Float(sampleRate))))
        if fade > 0 {
            for i in 0..<fade {
                let scale = Float(fade - i) / Float(fade)
                samples[samples.count - 1 - i] *= scale
            }
        }
    }

    /// Reconstructs waveform from Decoder_HAR latent output.
    /// Expects x with shape [C, T], where C = nFFT/2+1 magnitude + nFFT/2+1 phase channels.
    private static func reconstructWaveformFromDecoderHAROutput(
        xChannelsByTime: [[Float]],
        nFFT: Int = 20,
        hop: Int = 5,
        center: Bool = true
    ) -> [Float] {
        let freqBins = nFFT / 2 + 1
        let channels = xChannelsByTime.count
        let frames = xChannelsByTime.first?.count ?? 0
        guard channels >= freqBins * 2 else { return [] }
        var mag = Array(repeating: [Float](repeating: 0, count: frames), count: freqBins)
        var pha = Array(repeating: [Float](repeating: 0, count: frames), count: freqBins)
        for k in 0..<freqBins {
            let specChan = xChannelsByTime[k]
            let phaseChan = xChannelsByTime[freqBins + k]
            for t in 0..<frames {
                mag[k][t] = expf(specChan[t])
                pha[k][t] = sinf(phaseChan[t])
            }
        }
        // Hann (periodic) window
        func hannPeriodic(_ n: Int) -> [Float] {
            var w = [Float](repeating: 0, count: n)
            if n <= 1 { if n == 1 { w[0] = 1 } ; return w }
            for i in 0..<n { w[i] = 0.5 - 0.5 * cos(2.0 * .pi * Float(i) / Float(n)) }
            return w
        }
        let window = hannPeriodic(nFFT)
        let padLen = nFFT / 2
        var cosTable = Array(repeating: [Float](repeating: 0, count: nFFT), count: freqBins)
        var sinTable = Array(repeating: [Float](repeating: 0, count: nFFT), count: freqBins)
        for k in 0..<freqBins {
            for n in 0..<nFFT {
                let angle = 2.0 * .pi * Float(k * n) / Float(nFFT)
                cosTable[k][n] = cos(angle)
                sinTable[k][n] = sin(angle)
            }
        }
        let totalLen = frames * hop + (center ? 2 * padLen : 0) + nFFT
        var y = [Float](repeating: 0, count: totalLen)
        let scale: Float = 1.0 / Float(nFFT)
        for t in 0..<frames {
            var frame = [Float](repeating: 0, count: nFFT)
            for k in 0..<freqBins {
                let realk = mag[k][t] * cos(pha[k][t])
                let imagk = mag[k][t] * sin(pha[k][t])
                for n in 0..<nFFT {
                    frame[n] += (realk * cosTable[k][n] - imagk * sinTable[k][n])
                }
            }
            for n in 0..<nFFT { frame[n] *= window[n] * scale }
            let base = (center ? padLen : 0) + t * hop
            for n in 0..<nFFT {
                let idx = base + n
                if idx < y.count { y[idx] += frame[n] }
            }
        }
        if center && y.count > 2 * padLen {
            return Array(y[padLen..<(y.count - padLen)])
        } else {
            return y
        }
    }
}
