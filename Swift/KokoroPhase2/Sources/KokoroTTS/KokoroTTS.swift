import Foundation
import CoreML
import AVFoundation
import Accelerate

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
        // Streaming gate: if requested, or if HAR time exceeds ANE limit (~16,384),
        // switch to 3s windowed decoding with overlap-add using the 3s bucket.
        let env = ProcessInfo.processInfo.environment
        let harTime = harSpec.shape.last?.intValue ?? 0
        if env["STREAM_HAR"] == "1" || harTime > 16384 {
            let winSec = max(1, Int(env["STREAM_WINDOW_SEC"] ?? "3") ?? 3)
            let overlapFrac = min(0.9, max(0.0, Float(env["STREAM_OVERLAP_FRAC"] ?? "0.5") ?? 0.5))
            return try synthesizeHARStreaming(
                asr: asr, f0: f0, n: n, s: s, harSpec: harSpec, harPhase: harPhase,
                windowSeconds: winSec,
                overlapFraction: overlapFrac
            )
        }

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
                // Call tools/reconstruct_from_latent.py (resolve from multiple roots)
                let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                let scriptCandidates = [
                    cwd.appendingPathComponent("tools/reconstruct_from_latent.py").path,
                    cwd.appendingPathComponent("../tools/reconstruct_from_latent.py").standardized.path,
                    cwd.appendingPathComponent("../../tools/reconstruct_from_latent.py").standardized.path
                ]
                guard let script = scriptCandidates.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
                    throw Error.predictionFailed("reconstruct_from_latent.py not found")
                }
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

    /// Windowed streaming synthesis that keeps HAR time under ANE limits.
    /// Uses a short bucket model (default 3s) and overlap-add stitching.
    private static func synthesizeHARStreaming(
        asr: MLMultiArray,
        f0: MLMultiArray,
        n: MLMultiArray,
        s: MLMultiArray,
        harSpec: MLMultiArray,
        harPhase: MLMultiArray,
        windowSeconds: Int = 3,
        overlapFraction: Float = 0.5,
        sampleRate: Int = 24000,
        asrFps: Int = 40,
        f0Fps: Int = 80,
        harHopSamples: Int = 5
    ) throws -> [Float] {
        let env = ProcessInfo.processInfo.environment
        // Load streaming bucket (prefer waveform-emitting variant if present)
        let waveformModelName = "KokoroDecoder_HAR_WAV_\(windowSeconds)s.mlpackage"
        let latentModelName = "KokoroDecoder_HAR_\(windowSeconds)s.mlpackage"
        let model: MLModel
        let isWaveformPreferred: Bool
        if let m = try? loadModel(resourceName: waveformModelName) {
            model = m; isWaveformPreferred = true
        } else {
            model = try loadModel(resourceName: latentModelName); isWaveformPreferred = false
        }

        // Flatten inputs for fast slicing
        func flatten(_ a: MLMultiArray) -> [Float] {
            var out = [Float](repeating: 0, count: a.count)
            for i in 0..<a.count { out[i] = a[i].floatValue }
            return out
        }
        let asrShape = asr.shape.map { $0.intValue }    // [1,512,1,T_asr]
        let f0Shape  = f0.shape.map { $0.intValue }     // [1,1,1,T_f0]
        let nShape   = n.shape.map { $0.intValue }      // [1,1,1,T_f0]
        let sShape   = s.shape.map { $0.intValue }      // [1,128]
        let hsShape  = harSpec.shape.map { $0.intValue } // [1,C,1,T_har]
        let hpShape  = harPhase.shape.map { $0.intValue }
        let totalAsrT = asrShape.last ?? 0
        let totalF0T  = f0Shape.last ?? 0
        let totalHarT = hsShape.last ?? 0
        let harC = hsShape.count >= 2 ? hsShape[1] : 0

        let asrFlat = flatten(asr)            // [512*T_asr]
        let f0Flat  = flatten(f0)             // [T_f0]
        let nFlat   = flatten(n)              // [T_f0]
        let hsFlat  = flatten(harSpec)        // [C*T_har]
        let hpFlat  = flatten(harPhase)       // [C*T_har]

        // Derive per-window sizes from model input constraints if available
        var asrWin = max(1, windowSeconds * asrFps)          // e.g., 3s*40 = 120
        var f0Win  = max(1, windowSeconds * f0Fps)           // e.g., 3s*80 = 240
        // Map F0 frames to HAR frames: ratio ≈ (sr/harHop)/f0Fps, plus constant offset from STFT (+1)
        let harFramesPerSec = sampleRate / harHopSamples      // 24000/5 = 4800
        let ratioHarToF0 = max(1, harFramesPerSec / f0Fps)    // 4800/80 = 60
        // Default guess; will be overridden by model constraints if present
        var harWinT = ratioHarToF0 * f0Win + max(0, totalHarT - ratioHarToF0 * max(1, totalF0T))
        // Override from model input constraint shapes (fixed buckets)
        if let harSpecDesc = model.modelDescription.inputDescriptionsByName["har_spec"],
           let mc = harSpecDesc.multiArrayConstraint,
           let shape = mc.shape, shape.count >= 4 {
            let tReq = shape.last?.intValue ?? harWinT
            let f0Desc = model.modelDescription.inputDescriptionsByName["f0_curve"]?.multiArrayConstraint?.shape
            let asrDesc = model.modelDescription.inputDescriptionsByName["asr"]?.multiArrayConstraint?.shape
            if let f0Shape = f0Desc, f0Shape.count >= 4 { f0Win = f0Shape.last?.intValue ?? f0Win }
            if let asrShapeReq = asrDesc, asrShapeReq.count >= 4 { asrWin = asrShapeReq.last?.intValue ?? asrWin }
            harWinT = tReq
        }

        // Strides (50% overlap)
        let asrStride = max(1, Int(Float(asrWin) * (1 - overlapFraction))) // 60
        let f0Stride  = max(1, Int(Float(f0Win)  * (1 - overlapFraction))) // 120
        let harStride = ratioHarToF0 * f0Stride                               // 60*120 = 7200

        // Helpers to slice last dimension
        func sliceASR(_ flat: [Float], totalT: Int, startT: Int, winT: Int) -> [Float] {
            let channels = 512
            var out = [Float](repeating: 0, count: channels * winT)
            for c in 0..<channels {
                let srcBase = c * totalT
                let dstBase = c * winT
                let copyT = min(winT, max(0, totalT - startT))
                if copyT > 0 {
                    for t in 0..<copyT { out[dstBase + t] = flat[srcBase + startT + t] }
                }
            }
            return out
        }
        func slice1D(_ flat: [Float], totalT: Int, startT: Int, winT: Int) -> [Float] {
            let copyT = min(winT, max(0, totalT - startT))
            var out = [Float](repeating: 0, count: winT)
            if copyT > 0 {
                for t in 0..<copyT { out[t] = flat[startT + t] }
            }
            return out
        }
        func sliceHAR(_ flat: [Float], channels: Int, totalT: Int, startT: Int, winT: Int) -> [Float] {
            // Flattened as [C*T]
            var out = [Float](repeating: 0, count: channels * winT)
            let copyT = min(winT, max(0, totalT - startT))
            for c in 0..<channels {
                let srcBase = c * totalT
                let dstBase = c * winT
                if copyT > 0 {
                    for t in 0..<copyT { out[dstBase + t] = flat[srcBase + startT + t] }
                }
            }
            return out
        }
        func makeMLMultiArray(shape: [Int], data: [Float]) throws -> MLMultiArray {
            let total = shape.reduce(1, *)
            let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            data.withUnsafeBytes { src in
                memcpy(arr.dataPointer, src.baseAddress!, min(total, data.count) * MemoryLayout<Float>.size)
            }
            return arr
        }

        // Overlap-add buffers
        let samplesPerFrame = sampleRate / asrFps                 // 24000 / 40 = 600
        let chunkSamples = asrWin * samplesPerFrame               // e.g., 120*600 = 72000
        let strideSamples = asrStride * samplesPerFrame           // 60*600 = 36000
        func hann(_ n: Int) -> [Float] {
            if n <= 0 { return [] }
            var w = [Float](repeating: 0, count: n)
            if n == 1 { w[0] = 1; return w }
            for i in 0..<n { w[i] = 0.5 - 0.5 * cos(2.0 * .pi * Float(i) / Float(n)) }
            return w
        }
        var winWeight = hann(chunkSamples)
        if overlapFraction <= 0 { winWeight = [Float](repeating: 1, count: chunkSamples) }

        func numWindows(total: Int, win: Int, stride: Int) -> Int {
            if total <= 0 { return 0 }
            if total <= win { return 1 }
            return Int(ceil(Double(total - win) / Double(stride))) + 1
        }
        let nWin = numWindows(total: totalAsrT, win: asrWin, stride: asrStride)
        let totalOutSamples = max(chunkSamples, (nWin - 1) * strideSamples + chunkSamples)
        var accAudio = [Float](repeating: 0, count: totalOutSamples)
        var accWeight = [Float](repeating: 0, count: totalOutSamples)

        // Pre-create style vector MLMultiArray (shared across windows)
        let sMA = s

        // Prepare batch providers if requested, else iterative. Default ON for speed.
        let useBatch = (env["BATCH_STREAM"] ?? "1") != "0"
        let logTiming = env["TIMING"] == "1"
        // Warm-up: run one tiny window to trigger GPU pipeline compilation once
        let tWarmStart = CFAbsoluteTimeGetCurrent()
        do {
            let warmAsr = try makeMLMultiArray(shape: [1, 512, 1, min(8, asrWin)], data: [Float](repeating: 0, count: 512 * min(8, asrWin)))
            let warmF0  = try makeMLMultiArray(shape: [1, 1, 1, min(16, f0Win)], data: [Float](repeating: 0, count: min(16, f0Win)))
            let warmN   = try makeMLMultiArray(shape: [1, 1, 1, min(16, f0Win)], data: [Float](repeating: 0, count: min(16, f0Win)))
            let warmHS  = try makeMLMultiArray(shape: [1, harC, 1, min(64, harWinT)], data: [Float](repeating: 0, count: harC * min(64, harWinT)))
            let warmHP  = try makeMLMultiArray(shape: [1, harC, 1, min(64, harWinT)], data: [Float](repeating: 0, count: harC * min(64, harWinT)))
            _ = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
                "har_spec": MLFeatureValue(multiArray: warmHS),
                "har_phase": MLFeatureValue(multiArray: warmHP),
                "asr": MLFeatureValue(multiArray: warmAsr),
                "f0_curve": MLFeatureValue(multiArray: warmF0),
                "n": MLFeatureValue(multiArray: warmN),
                "s": MLFeatureValue(multiArray: sMA),
            ]))
        } catch { /* ignore warm-up errors */ }
        let tWarm = CFAbsoluteTimeGetCurrent() - tWarmStart
        if logTiming { print(String(format: "HAR stream: warmup=%.3fs", tWarm)) }
        if useBatch {
            let tBuildStart = CFAbsoluteTimeGetCurrent()
            var batchInputs: [MLFeatureProvider] = []
            batchInputs.reserveCapacity(nWin)
            for w in 0..<nWin {
                let asrStart = w * asrStride
                let f0Start  = w * f0Stride
                let harStart = w * harStride
                let asrSlice = sliceASR(asrFlat, totalT: totalAsrT, startT: asrStart, winT: asrWin)
                let f0Slice  = slice1D(f0Flat,  totalT: totalF0T,  startT: f0Start,  winT: f0Win)
                let nSlice   = slice1D(nFlat,   totalT: totalF0T,  startT: f0Start,  winT: f0Win)
            let hsSlice  = sliceHAR(hsFlat, channels: harC, totalT: totalHarT, startT: harStart, winT: harWinT)
            let hpSlice  = sliceHAR(hpFlat, channels: harC, totalT: totalHarT, startT: harStart, winT: harWinT)
                let asrMA = try makeMLMultiArray(shape: [1, 512, 1, asrWin], data: asrSlice)
                let f0MA  = try makeMLMultiArray(shape: [1, 1,   1, f0Win], data: f0Slice)
                let nMA   = try makeMLMultiArray(shape: [1, 1,   1, f0Win], data: nSlice)
                let hsMA  = try makeMLMultiArray(shape: [1, harC, 1, harWinT], data: hsSlice)
                let hpMA  = try makeMLMultiArray(shape: [1, harC, 1, harWinT], data: hpSlice)
                let fp = try MLDictionaryFeatureProvider(dictionary: [
                    "har_spec": MLFeatureValue(multiArray: hsMA),
                    "har_phase": MLFeatureValue(multiArray: hpMA),
                    "asr": MLFeatureValue(multiArray: asrMA),
                    "f0_curve": MLFeatureValue(multiArray: f0MA),
                    "n": MLFeatureValue(multiArray: nMA),
                    "s": MLFeatureValue(multiArray: sMA),
                ])
                batchInputs.append(fp)
            }
            let tBuild = CFAbsoluteTimeGetCurrent() - tBuildStart
            let options = MLPredictionOptions()
            let inputBatch = MLArrayBatchProvider(array: batchInputs)
            let tInferStart = CFAbsoluteTimeGetCurrent()
            let outBatch = try model.predictions(from: inputBatch, options: options)
            let tInfer = CFAbsoluteTimeGetCurrent() - tInferStart
            let tMergeStart = CFAbsoluteTimeGetCurrent()
            for w in 0..<outBatch.count {
                guard let outName = model.modelDescription.outputDescriptionsByName.keys.first,
                      let outArr = outBatch.features(at: w).featureValue(for: outName)?.multiArrayValue else {
                    throw Error.predictionFailed("Missing output tensor (batch stream chunk)")
                }
                let outShape = outArr.shape.map { $0.intValue }
                var chunkWave: [Float] = []
                if outShape.count == 3 && outShape.first == 1 && outShape[1] > 1 {
                    let channels = outShape[1]
                    let framesT = outShape[2]
                    var x: [[Float]] = Array(repeating: [Float](repeating: 0, count: framesT), count: channels)
                    for c in 0..<channels { for t in 0..<framesT { x[c][t] = outArr[[0, NSNumber(value: c), NSNumber(value: t)]].floatValue } }
                    let useFast = (env["FAST_ISTFT"] ?? "1") != "0"
                    chunkWave = useFast ? reconstructWaveformFromDecoderHAROutputFast(xChannelsByTime: x)
                                         : reconstructWaveformFromDecoderHAROutput(xChannelsByTime: x)
                } else {
                    chunkWave = mlMultiArrayToVector(outArr)
                }
                if let gStr = env["ISTFT_GAIN"], let g = Float(gStr), g != 1.0 {
                    vDSP_vsmul(chunkWave, 1, [g], &chunkWave, 1, vDSP_Length(chunkWave.count))
                }
                let startSample = w * strideSamples
                for i in 0..<chunkWave.count {
                    let idx = startSample + i
                    if idx < accAudio.count {
                        let wv = i < winWeight.count ? winWeight[i] : 1.0
                        accAudio[idx] += chunkWave[i] * wv
                        accWeight[idx] += wv
                    }
                }
            }
            let tMerge = CFAbsoluteTimeGetCurrent() - tMergeStart
            if logTiming {
                let avg = outBatch.count > 0 ? tInfer / Double(outBatch.count) : 0
                print(String(format: "HAR stream: nWin=%d build=%.3fs infer=%.3fs (avg/win=%.3fs) merge=%.3fs total=%.3fs",
                             outBatch.count, tBuild, tInfer, avg, tMerge, tWarm + tBuild + tInfer + tMerge))
            }
        } else {
            // Iterative single predictions (fallback)
            for w in 0..<nWin {
                let tWinStart = CFAbsoluteTimeGetCurrent()
                let asrStart = w * asrStride
                let f0Start  = w * f0Stride
                let harStart = w * harStride
                let asrSlice = sliceASR(asrFlat, totalT: totalAsrT, startT: asrStart, winT: asrWin)
                let f0Slice  = slice1D(f0Flat,  totalT: totalF0T,  startT: f0Start,  winT: f0Win)
                let nSlice   = slice1D(nFlat,   totalT: totalF0T,  startT: f0Start,  winT: f0Win)
                let hsSlice  = sliceHAR(hsFlat, channels: harC, totalT: totalHarT, startT: harStart, winT: harWinT)
                let hpSlice  = sliceHAR(hpFlat, channels: harC, totalT: totalHarT, startT: harStart, winT: harWinT)
                let asrMA = try makeMLMultiArray(shape: [1, 512, 1, asrWin], data: asrSlice)
                let f0MA  = try makeMLMultiArray(shape: [1, 1,   1, f0Win], data: f0Slice)
                let nMA   = try makeMLMultiArray(shape: [1, 1,   1, f0Win], data: nSlice)
                let hsMA  = try makeMLMultiArray(shape: [1, harC, 1, harWinT], data: hsSlice)
                let hpMA  = try makeMLMultiArray(shape: [1, harC, 1, harWinT], data: hpSlice)
                let tInferStart = CFAbsoluteTimeGetCurrent()
                let out = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
                    "har_spec": MLFeatureValue(multiArray: hsMA),
                    "har_phase": MLFeatureValue(multiArray: hpMA),
                    "asr": MLFeatureValue(multiArray: asrMA),
                    "f0_curve": MLFeatureValue(multiArray: f0MA),
                    "n": MLFeatureValue(multiArray: nMA),
                    "s": MLFeatureValue(multiArray: sMA),
                ]))
                let tInfer = CFAbsoluteTimeGetCurrent() - tInferStart
                guard let outName = model.modelDescription.outputDescriptionsByName.keys.first,
                      let outArr = out.featureValue(for: outName)?.multiArrayValue else {
                    throw Error.predictionFailed("Missing output tensor (stream chunk)")
                }
                let outShape = outArr.shape.map { $0.intValue }
                var chunkWave: [Float] = []
                if outShape.count == 3 && outShape.first == 1 && outShape[1] > 1 {
                    let channels = outShape[1]
                    let framesT = outShape[2]
                    var x: [[Float]] = Array(repeating: [Float](repeating: 0, count: framesT), count: channels)
                    for c in 0..<channels { for t in 0..<framesT { x[c][t] = outArr[[0, NSNumber(value: c), NSNumber(value: t)]].floatValue } }
                    let useFast = (env["FAST_ISTFT"] ?? "1") != "0"
                    chunkWave = useFast ? reconstructWaveformFromDecoderHAROutputFast(xChannelsByTime: x)
                                         : reconstructWaveformFromDecoderHAROutput(xChannelsByTime: x)
                } else {
                    chunkWave = mlMultiArrayToVector(outArr)
                }
                if let gStr = env["ISTFT_GAIN"], let g = Float(gStr), g != 1.0 {
                    vDSP_vsmul(chunkWave, 1, [g], &chunkWave, 1, vDSP_Length(chunkWave.count))
                }
                let startSample = w * strideSamples
                for i in 0..<chunkWave.count {
                    let idx = startSample + i
                    if idx < accAudio.count {
                        let wv = i < winWeight.count ? winWeight[i] : 1.0
                        accAudio[idx] += chunkWave[i] * wv
                        accWeight[idx] += wv
                    }
                }
                if logTiming {
                    let tWin = CFAbsoluteTimeGetCurrent() - tWinStart
                    print(String(format: "HAR stream win %d: infer=%.3fs total=%.3fs", w, tInfer, tWin))
                }
            }
        }

        // Normalize by accumulated weights
        for i in 0..<accAudio.count {
            let w = accWeight[i]
            if w > 0 { accAudio[i] /= w }
        }

        // Trim to expected duration
        let expectedSamples = max(0, totalAsrT * (sampleRate / asrFps))
        var audio = accAudio
        if expectedSamples > 0 && expectedSamples < audio.count {
            audio = Array(audio[0..<expectedSamples])
        }

        // Postprocess
        applyPostprocessing(&audio, sampleRate: sampleRate)
        return audio
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
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let candidateURLs: [URL] = [
            bundleURL,
            cwd.appendingPathComponent("Resources").appendingPathComponent(resourceName),
            cwd.appendingPathComponent("Swift/KokoroPhase2/Resources").appendingPathComponent(resourceName),
            cwd.appendingPathComponent(resourceName)
        ]
        guard let found = candidateURLs.first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw Error.modelNotFound(resourceName)
        }
        let compiled = try MLModel.compileModel(at: found)
        let config = MLModelConfiguration()
        let env = ProcessInfo.processInfo.environment
        if env["CPU_ONLY"] == "1" {
            config.computeUnits = .cpuOnly
        } else if env["GPU_ONLY"] == "1" {
            config.computeUnits = .cpuAndGPU
        } else {
            // Default to GPU for HAR paths to avoid slow .all heuristics
            config.computeUnits = .cpuAndGPU
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

    /// Accelerated iSTFT reconstruction using vDSP for inner loops.
    /// Functionally equivalent to reconstructWaveformFromDecoderHAROutput but significantly faster.
    private static func reconstructWaveformFromDecoderHAROutputFast(
        xChannelsByTime: [[Float]],
        nFFT: Int = 20,
        hop: Int = 5,
        center: Bool = true
    ) -> [Float] {
        let freqBins = nFFT / 2 + 1
        let channels = xChannelsByTime.count
        let frames = xChannelsByTime.first?.count ?? 0
        if channels < freqBins * 2 || frames == 0 { return [] }

        // Prepare magnitude and phase
        var mag = Array(repeating: [Float](repeating: 0, count: frames), count: freqBins)
        var pha = Array(repeating: [Float](repeating: 0, count: frames), count: freqBins)
        for k in 0..<freqBins {
            let specChan = xChannelsByTime[k]
            let phaseChan = xChannelsByTime[freqBins + k]
            for t in 0..<frames {
                mag[k][t] = expf(specChan[t])
                // Match prior path and Python parity (phase is provided via sin(raw))
                pha[k][t] = sinf(phaseChan[t])
            }
        }

        // Create inverse DFT setup for real output
        guard let idft = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(nFFT), vDSP_DFT_Direction.INVERSE) else { return [] }
        defer { vDSP_DFT_DestroySetup(idft) }

        // Hann window and scaling
        var window = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&window, vDSP_Length(nFFT), Int32(vDSP_HANN_NORM))
        let scale: Float = 1.0 / Float(nFFT)
        let padLen = nFFT / 2

        // Overlap-add buffer
        let totalLen = frames * hop + (center ? 2 * padLen : 0) + nFFT
        var y = [Float](repeating: 0, count: totalLen)

        // Temporary buffers for frequency and time domain
        var real = [Float](repeating: 0, count: nFFT/2)
        var imag = [Float](repeating: 0, count: nFFT/2)
        var timeReal = [Float](repeating: 0, count: nFFT)
        var timeImag = [Float](repeating: 0, count: nFFT)  // remains zero

        for t in 0..<frames {
            // Build complex spectrum 0..N/2-1 and set Nyquist=0
            for k in 0..<(nFFT/2) {
                let m = mag[k][t]
                let ph = pha[k][t]
                real[k] = m * cosf(ph)
                imag[k] = m * sinf(ph)
            }
            // Execute inverse DFT
            vDSP_DFT_Execute(idft, &real, &imag, &timeReal, &timeImag)
            // Window and scale
            vDSP_vmul(timeReal, 1, window, 1, &timeReal, 1, vDSP_Length(nFFT))
            var sc = scale
            vDSP_vsmul(timeReal, 1, &sc, &timeReal, 1, vDSP_Length(nFFT))
            // Overlap-add
            let base = (center ? padLen : 0) + t * hop
            for n in 0..<nFFT {
                let idx = base + n
                if idx < y.count { y[idx] += timeReal[n] }
            }
        }
        if center && y.count > 2 * padLen {
            return Array(y[padLen..<(y.count - padLen)])
        } else {
            return y
        }
    }
}
