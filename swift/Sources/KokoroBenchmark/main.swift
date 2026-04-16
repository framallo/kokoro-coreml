/// Kokoro Swift pipeline benchmark CLI.
///
/// Called by ``scripts/bakeoff_harness.py`` as a subprocess for Config F.
///
/// **Single-shot mode** (original):
///
///     kokoro-bench --models-dir DIR --inputs-dir DIR \
///                  --hnsf-weights FILE --input-key KEY --seed 42 \
///                  [--output metrics.json] [--wav out.wav] [--dump-tensors DIR]
///
/// **Generator isolation mode** (feed previously dumped x_pre/ref_s/har):
///
///     kokoro-bench --models-dir DIR --inputs-dir DIR --hnsf-weights FILE \
///                  --generator-input-dump DIR [--output metrics.json]
///
/// **Batch mode** (persistent subprocess — models compiled once):
///
///     kokoro-bench --models-dir DIR --inputs-dir DIR \
///                  --hnsf-weights FILE --batch
///
///     Then send JSON commands on stdin, one per line:
///       {"input_key":"3s","seed":42,"output":"/tmp/result.json"}
///     The binary prints "READY\n" to stdout after loading weights,
///     and "DONE\n" after each command completes.
///     Close stdin (EOF) to exit.

import Foundation
import CoreML
@_exported import KokoroPipeline

// MARK: - JSON Input

struct BenchInput: Decodable {
    let key: String
    let text: String
    let voice: String
    let speed: Float
    let input_ids: [Int32]
    let attention_mask: [Int32]
    let ref_s: [Float]
    let num_tokens: Int
    /// Canonical audio duration from the bakeoff manifest (seconds).
    /// Computed by the Python pipeline's extract_vocoder_inputs as T_f0 / 80.0.
    let canonical_duration_s: Double?
}

struct HnsfWeights: Decodable {
    let linear_weights: [Float]
    let linear_bias: Float
}

/// JSON command received on stdin in batch mode.
struct BatchCommand: Decodable {
    let input_key: String
    let seed: UInt64?
    let output: String
    let warmup: Bool?
}

// MARK: - Model Cache

/// Two-layer cache: compiled model URLs (.mlmodelc) persist across the whole
/// session so we never pay compilation twice. Loaded MLModel instances are
/// evicted when switching buckets to keep memory footprint low — critical on
/// 24GB machines where keeping all bucket models resident causes the ANE
/// scheduler to fall back to CPU.
class ModelCache {
    let modelsDir: URL
    let config: MLModelConfiguration

    // Layer 1: Compiled URLs (persist forever — compilation is expensive)
    private var compiledDuration: [Int: URL] = [:]
    private var compiledF0n: [Int: URL] = [:]
    private var compiledDecPre: [Int: URL] = [:]
    private var compiledGen: [Int: URL] = [:]

    // Layer 2: Loaded MLModel instances (evicted on bucket switch)
    var durationModels: [Int: MLModel] = [:]  // kept across buckets (small)
    var f0nModels: [Int: MLModel] = [:]
    var decPreModels: [Int: MLModel] = [:]
    var genModels: [Int: MLModel] = [:]

    init(modelsDir: URL, computeUnits: MLComputeUnits = .all) {
        self.modelsDir = modelsDir
        self.config = MLModelConfiguration()
        self.config.computeUnits = computeUnits
        fputs("  Compute units: \(computeUnits == .all ? "all" : computeUnits == .cpuAndNeuralEngine ? "cpuAndNeuralEngine" : computeUnits == .cpuAndGPU ? "cpuAndGPU" : "cpuOnly")\n", stderr)
    }

    // -- Compiled URL helpers (compile once, cache the .mlmodelc path) --

    private func compiledDurationURL(T: Int) throws -> URL {
        if let url = compiledDuration[T] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_duration_t\(T).mlpackage")
        let actualURL = FileManager.default.fileExists(atPath: pkgURL.path)
            ? pkgURL
            : modelsDir.appendingPathComponent("kokoro_duration.mlpackage")
        fputs("  Compiling duration T=\(T)...\n", stderr)
        let compiled = try MLModel.compileModel(at: actualURL)
        compiledDuration[T] = compiled
        return compiled
    }

    private func compiledF0nURL(tFrames: Int) throws -> URL {
        if let url = compiledF0n[tFrames] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_f0ntrain_t\(tFrames).mlpackage")
        fputs("  Compiling f0ntrain tFrames=\(tFrames)...\n", stderr)
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledF0n[tFrames] = compiled
        return compiled
    }

    private func compiledDecPreURL(bucket: Int) throws -> URL {
        if let url = compiledDecPre[bucket] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_decoder_pre_\(bucket)s.mlpackage")
        fputs("  Compiling decoder_pre \(bucket)s...\n", stderr)
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledDecPre[bucket] = compiled
        return compiled
    }

    private func compiledGenURL(bucket: Int) throws -> URL {
        if let url = compiledGen[bucket] { return url }
        let pkgURL = modelsDir.appendingPathComponent("kokoro_decoder_har_post_\(bucket)s.mlpackage")
        fputs("  Compiling generator \(bucket)s...\n", stderr)
        let compiled = try MLModel.compileModel(at: pkgURL)
        compiledGen[bucket] = compiled
        return compiled
    }

    // -- Model accessors (load from cached compiled URL) --

    func durationModel(T: Int) throws -> MLModel {
        if let cached = durationModels[T] { return cached }
        let compiled = try compiledDurationURL(T: T)
        fputs("  Loading duration T=\(T)...\n", stderr)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        durationModels[T] = model
        return model
    }

    func f0nModel(tFrames: Int) throws -> MLModel {
        if let cached = f0nModels[tFrames] { return cached }
        let compiled = try compiledF0nURL(tFrames: tFrames)
        fputs("  Loading f0ntrain tFrames=\(tFrames)...\n", stderr)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        f0nModels[tFrames] = model
        return model
    }

    func decoderPreModel(bucket: Int) throws -> MLModel {
        if let cached = decPreModels[bucket] { return cached }
        let compiled = try compiledDecPreURL(bucket: bucket)
        fputs("  Loading decoder_pre \(bucket)s...\n", stderr)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        decPreModels[bucket] = model
        return model
    }

    func generatorModel(bucket: Int) throws -> MLModel {
        if let cached = genModels[bucket] { return cached }
        let compiled = try compiledGenURL(bucket: bucket)
        fputs("  Loading generator \(bucket)s...\n", stderr)
        let model = try MLModel(contentsOf: compiled, configuration: config)
        genModels[bucket] = model
        return model
    }

    /// Evict loaded MLModel instances for all buckets except the given one.
    /// Duration models are kept (small). Compiled URLs are preserved so
    /// reloading is just `MLModel(contentsOf:)` — no recompilation.
    func evictExcept(bucket: Int, tFrames: Int) {
        let f0nBefore = f0nModels.count
        let decBefore = decPreModels.count
        let genBefore = genModels.count

        f0nModels = f0nModels.filter { $0.key == tFrames }
        decPreModels = decPreModels.filter { $0.key == bucket }
        genModels = genModels.filter { $0.key == bucket }

        let evicted = (f0nBefore - f0nModels.count) + (decBefore - decPreModels.count) + (genBefore - genModels.count)
        if evicted > 0 {
            fputs("  Evicted \(evicted) loaded model(s) from other buckets (compiled URLs retained)\n", stderr)
        }
    }
}

// MARK: - WAV export (Config F listening checks)

/// Writes mono 16-bit PCM WAV at `sampleRate` Hz, peak-normalized like ``examples/example_synthesis.py``.
func writeWavMono16(path: String, samples: [Float], sampleRate: UInt32) throws {
    let n = samples.count
    var peak: Float = 1e-7
    for s in samples {
        let a = abs(s)
        if a > peak { peak = a }
    }
    var pcm = [Int16](repeating: 0, count: n)
    for i in 0..<n {
        let x = max(-1.0, min(1.0, samples[i] / peak))
        pcm[i] = Int16((x * 32767.0).rounded())
    }
    let dataSize = UInt32(n * 2)
    let byteRate = sampleRate * 2

    var d = Data()
    d.append(contentsOf: "RIFF".utf8)
    let riffChunkSize: UInt32 = 36 + dataSize
    withUnsafeBytes(of: riffChunkSize.littleEndian) { d.append(contentsOf: $0) }
    d.append(contentsOf: "WAVE".utf8)
    d.append(contentsOf: "fmt ".utf8)
    let subchunk1Size: UInt32 = 16
    withUnsafeBytes(of: subchunk1Size.littleEndian) { d.append(contentsOf: $0) }
    let audioFormat: UInt16 = 1
    withUnsafeBytes(of: audioFormat.littleEndian) { d.append(contentsOf: $0) }
    let numChannels: UInt16 = 1
    withUnsafeBytes(of: numChannels.littleEndian) { d.append(contentsOf: $0) }
    withUnsafeBytes(of: sampleRate.littleEndian) { d.append(contentsOf: $0) }
    withUnsafeBytes(of: byteRate.littleEndian) { d.append(contentsOf: $0) }
    let blockAlign: UInt16 = 2
    withUnsafeBytes(of: blockAlign.littleEndian) { d.append(contentsOf: $0) }
    let bitsPerSample: UInt16 = 16
    withUnsafeBytes(of: bitsPerSample.littleEndian) { d.append(contentsOf: $0) }
    d.append(contentsOf: "data".utf8)
    withUnsafeBytes(of: dataSize.littleEndian) { d.append(contentsOf: $0) }
    pcm.withUnsafeBytes { d.append(contentsOf: $0) }

    try d.write(to: URL(fileURLWithPath: path))
}

// MARK: - Pipeline run (shared between single-shot and batch)

/// Runs the full timed pipeline for one input. Returns the result dictionary as JSON Data.
func runPipeline(
    benchInput: BenchInput,
    inputKey: String,
    seed: UInt64,
    weights: HnsfWeights,
    cache: ModelCache,
    wavOutputPath: String? = nil,
    tensorDumpPath: String? = nil
) throws -> Data {
    var tensorDump: TensorDumpWriter? = nil
    if let tensorDumpPath {
        tensorDump = try TensorDumpWriter(directory: URL(fileURLWithPath: tensorDumpPath))
    }

    func withTensorDump(_ body: (inout TensorDumpWriter) throws -> Void) throws {
        if var writer = tensorDump {
            try body(&writer)
            tensorDump = writer
        }
    }

    // Pick duration T
    let enumSizes = [32, 64, 128, 256, 512]
    let actualTokens = benchInput.num_tokens
    guard let T = enumSizes.first(where: { $0 >= actualTokens }) else {
        fputs("Token count \(actualTokens) exceeds max enumeration 512\n", stderr)
        throw NSError(domain: "kokoro-bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "Token count exceeds 512"])
    }

    let durModel = try cache.durationModel(T: T)

    // Build duration input
    let idsArray = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
    let maskArray = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
    let refSArray = try MLMultiArray(shape: [1, 256], dataType: .float32)
    let speedArray = try MLMultiArray(shape: [1], dataType: .float32)

    let idsPtr = idsArray.dataPointer.assumingMemoryBound(to: Int32.self)
    let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
    let refSPtr = refSArray.dataPointer.assumingMemoryBound(to: Float.self)

    for j in 0..<min(benchInput.input_ids.count, T) { idsPtr[j] = benchInput.input_ids[j] }
    for j in 0..<min(benchInput.attention_mask.count, T) { maskPtr[j] = benchInput.attention_mask[j] }
    for j in 0..<min(benchInput.ref_s.count, 256) { refSPtr[j] = benchInput.ref_s[j] }
    speedArray[0] = NSNumber(value: benchInput.speed)

    try withTensorDump { writer in
        try writer.writeMLMultiArray(name: "tokens", array: idsArray)
        try writer.writeMLMultiArray(name: "attention_mask", array: maskArray)
        try writer.writeMLMultiArray(name: "ref_s", array: refSArray)
        try writer.writeMLMultiArray(name: "speed", array: speedArray)
    }

    let durInput = try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(multiArray: idsArray),
        "attention_mask": MLFeatureValue(multiArray: maskArray),
        "ref_s": MLFeatureValue(multiArray: refSArray),
        "speed": MLFeatureValue(multiArray: speedArray),
    ])

    // Quick duration prediction to determine bucket
    let durOut = try durModel.prediction(from: durInput)
    let predDurArr = durOut.featureValue(for: "pred_dur")!.multiArrayValue!
    let tokenCount = predDurArr.shape.last!.intValue
    let validTokenCount = min(tokenCount, benchInput.num_tokens)
    let durationFramesForBucket = try readDurationFrames(from: predDurArr, validCount: validTokenCount)
    let totalFrames = durationFramesForBucket.reduce(0, +)
    let canonicalDurForBucket = benchInput.canonical_duration_s ?? (Double(totalFrames * 2) / 80.0)
    let availableBuckets = [3, 7, 10, 15, 30]
    let bucketThreshold = Int(ceil(canonicalDurForBucket))
    let bucketSec = availableBuckets.first(where: { $0 >= bucketThreshold }) ?? availableBuckets.last!

    fputs("  Input: \(inputKey), tokens: \(actualTokens)/\(T), canonical=\(String(format: "%.1f", canonicalDurForBucket))s, bucket: \(bucketSec)s\n", stderr)

    // Evict models from other buckets before loading this bucket's models.
    // On memory-constrained machines (24GB M2 Air), keeping all bucket models
    // resident causes the ANE scheduler to fall back to CPU for large models.
    // tFrames must match the exported F0Ntrain model for this bucket.
    // Canonical source: PipelineConstants.tFramesForBucket in KokoroPipeline.
    // Using a too-small tFrames silently truncates aligned features via zeroPad3D.
    let tFrames: Int = {
        switch bucketSec {
        case 3: return 120    // kokoro_f0ntrain_t120.mlpackage
        case 7: return 280    // kokoro_f0ntrain_t280.mlpackage
        case 10: return 400   // kokoro_f0ntrain_t400.mlpackage
        case 15: return 600   // kokoro_f0ntrain_t600.mlpackage
        case 30: return 1200  // kokoro_f0ntrain_t1200.mlpackage
        default: return 400
        }
    }()
    cache.evictExcept(bucket: bucketSec, tFrames: tFrames)

    // Load bucket-specific models (cached if same bucket as last run)
    let f0nModel = try cache.f0nModel(tFrames: tFrames)
    let decPreModel = try cache.decoderPreModel(bucket: bucketSec)
    let genModel = try cache.generatorModel(bucket: bucketSec)

    let bucketSamples = bucketSec * 24000
    let fullF0Len = Int(round(Double(bucketSamples) / 300.0))

    // Always warm models before the timed block. Even if the harness already
    // sent a warmup command for this bucket, evictExcept may have dropped the
    // MLModel instances, and freshly loaded instances need one prediction to
    // trigger ANE plan compilation. This is cheap (~1ms) if already warm.
    do {
        fputs("  Ensuring models are warm...\n", stderr)
        let _ = try durModel.prediction(from: durInput)
        // Warm F0Ntrain
        let warmEnArr = try makeZeroArray3D(channels: 640, time: tFrames)
        let warmSArr = try makeZeroArray2D(dim: 128)
        let warmF0nIn = try MLDictionaryFeatureProvider(dictionary: [
            "en": MLFeatureValue(multiArray: warmEnArr),
            "s": MLFeatureValue(multiArray: warmSArr),
        ])
        let _ = try f0nModel.prediction(from: warmF0nIn)
        // Warm DecoderPre
        let warmFC = (fullF0Len - 1) / 2 + 1
        let warmAsr = try makeZeroArray3D(channels: 512, time: warmFC)
        let warmF0 = try makeZeroArray3D(channels: 1, time: fullF0Len)
        let warmN = try makeZeroArray3D(channels: 1, time: fullF0Len)
        let warmRefS = try makeZeroArray2D(dim: 256)
        let warmDecIn = try MLDictionaryFeatureProvider(dictionary: [
            "asr": MLFeatureValue(multiArray: warmAsr),
            "f0": MLFeatureValue(multiArray: warmF0),
            "n_input": MLFeatureValue(multiArray: warmN),
            "ref_s": MLFeatureValue(multiArray: warmRefS),
        ])
        let _ = try decPreModel.prediction(from: warmDecIn)
        // Warm GeneratorFromHar
        let genShapesWarm = inputShapes(from: genModel)
        var warmGenInputs: [String: MLFeatureValue] = [:]
        for (name, shape) in genShapesWarm {
            if shape.count == 3 {
                warmGenInputs[name] = MLFeatureValue(multiArray: try makeZeroArray3D(channels: shape[1], time: shape[2]))
            } else if shape.count == 2 {
                warmGenInputs[name] = MLFeatureValue(multiArray: try makeZeroArray2D(dim: shape[1]))
            }
        }
        let _ = try genModel.prediction(from: try MLDictionaryFeatureProvider(dictionary: warmGenInputs))
    }

    // --- Timed run ---
    fputs("  Running timed iteration...\n", stderr)

    // Stage 1: Duration CoreML
    let t0 = CFAbsoluteTimeGetCurrent()
    let durOutput = try durModel.prediction(from: durInput)
    let t1 = CFAbsoluteTimeGetCurrent()
    let tDuration = t1 - t0

    // Extract outputs
    let predDurArray = durOutput.featureValue(for: "pred_dur")!.multiArrayValue!
    let dArray = durOutput.featureValue(for: "d")!.multiArrayValue!
    let tEnArray = durOutput.featureValue(for: "t_en")!.multiArrayValue!

    let tc = predDurArray.shape.last!.intValue
    let validTokens = min(tc, benchInput.num_tokens)
    let predDur = try readDurationFrames(from: predDurArray, validCount: validTokens)
    let frames = predDur.reduce(0, +)

    try withTensorDump { writer in
        try writer.writeMLMultiArray(name: "pred_dur", array: predDurArray)
        try writer.writeInt32Array(
            name: "pred_dur_valid",
            values: predDur.map { Int32($0) },
            shape: [1, predDur.count]
        )
        try writer.writeMLMultiArray(name: "duration_d", array: dArray)
        try writer.writeMLMultiArray(name: "duration_t_en", array: tEnArray)
    }

    // Stage 2: Alignment
    let t2 = CFAbsoluteTimeGetCurrent()
    let alignment = buildAlignmentMatrix(predDur: predDur, traceLength: tc, frameCount: frames)
    let t3 = CFAbsoluteTimeGetCurrent()
    let tAlignment = t3 - t2

    try withTensorDump { writer in
        try writer.writeFloatArray(name: "alignment", values: alignment, shape: [1, tc, frames])
    }

    // Stage 3: Matrix ops
    let t4 = CFAbsoluteTimeGetCurrent()
    let dTransposed = try transpose3D(source: dArray, dim1: 640, dim2: tc)
    let en = try matmul3D(a: dTransposed, b: alignment, M: 640, K: tc, N: frames)
    let asr = try matmul3D(a: tEnArray, b: alignment, M: 512, K: tc, N: frames)
    let t5 = CFAbsoluteTimeGetCurrent()
    let tMatrixOps = t5 - t4

    try withTensorDump { writer in
        try writer.writeMLMultiArray(name: "d_transposed", array: dTransposed)
        try writer.writeMLMultiArray(name: "en", array: en)
        try writer.writeMLMultiArray(name: "asr", array: asr)
    }

    // Stage 4: F0Ntrain
    let t6 = CFAbsoluteTimeGetCurrent()
    let enPadded = try zeroPad3D(source: en, channels: 640, targetTime: tFrames)
    let sArray = try makeZeroArray2D(dim: 128)
    let sP = sArray.dataPointer.assumingMemoryBound(to: Float.self)
    for j in 0..<128 { sP[j] = benchInput.ref_s[128 + j] }

    try withTensorDump { writer in
        try writer.writeMLMultiArray(name: "en_padded", array: enPadded)
        try writer.writeMLMultiArray(name: "s", array: sArray)
    }

    let f0nInput = try MLDictionaryFeatureProvider(dictionary: [
        "en": MLFeatureValue(multiArray: enPadded),
        "s": MLFeatureValue(multiArray: sArray),
    ])
    let f0nOutput = try f0nModel.prediction(from: f0nInput)
    let f0PredArr = f0nOutput.featureValue(for: "F0_pred")!.multiArrayValue!
    let nPredArr = f0nOutput.featureValue(for: "N_pred")!.multiArrayValue!
    let t7 = CFAbsoluteTimeGetCurrent()
    let tF0Ntrain = t7 - t6

    // Extract F0/N. Core ML outputs can be strided, so use the same logical
    // flattening helper as waveform export.
    let f0Curve = floatValues(from: f0PredArr)
    let nCurve = floatValues(from: nPredArr)
    let f0Len = f0Curve.count

    try withTensorDump { writer in
        try writer.writeFloatArray(name: "f0", values: f0Curve, shape: [1, f0Len])
        try writer.writeFloatArray(name: "n", values: nCurve, shape: [1, f0Len])
    }

    // Stage 5: Padding
    let t8 = CFAbsoluteTimeGetCurrent()
    let f0Padded = zeroPad1D(source: f0Curve, targetLength: fullF0Len)
    let nPadded = zeroPad1D(source: nCurve, targetLength: fullF0Len)
    // asr was computed in Stage 3 (reused here, not recomputed)
    let frameCount = (fullF0Len - 1) / 2 + 1
    let asrPadded = try zeroPad3D(source: asr, channels: 512, targetTime: frameCount)
    let t9 = CFAbsoluteTimeGetCurrent()
    let tPadding = t9 - t8

    try withTensorDump { writer in
        try writer.writeFloatArray(name: "f0_padded", values: f0Padded, shape: [1, fullF0Len])
        try writer.writeFloatArray(name: "n_padded", values: nPadded, shape: [1, fullF0Len])
        try writer.writeMLMultiArray(name: "asr_padded", array: asrPadded)
    }

    // Stage 6: DecoderPre CoreML
    let t10 = CFAbsoluteTimeGetCurrent()
    let f0Array3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
    copyInto(array: f0Array3D, from: f0Padded)
    let nArray3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
    copyInto(array: nArray3D, from: nPadded)
    let decRefS = try makeZeroArray2D(dim: 256)
    let decRefSP = decRefS.dataPointer.assumingMemoryBound(to: Float.self)
    for j in 0..<256 { decRefSP[j] = benchInput.ref_s[j] }

    let decPreInput = try MLDictionaryFeatureProvider(dictionary: [
        "asr": MLFeatureValue(multiArray: asrPadded),
        "f0": MLFeatureValue(multiArray: f0Array3D),
        "n_input": MLFeatureValue(multiArray: nArray3D),
        "ref_s": MLFeatureValue(multiArray: decRefS),
    ])
    let decPreOutput = try decPreModel.prediction(from: decPreInput)
    let xPre = decPreOutput.featureValue(for: "x_pre")!.multiArrayValue!
    let t11 = CFAbsoluteTimeGetCurrent()
    let tDecoderPre = t11 - t10

    try withTensorDump { writer in
        try writer.writeMLMultiArray(name: "x_pre", array: xPre)
    }

    // Stage 7: hn-nsf Swift
    let t12 = CFAbsoluteTimeGetCurrent()
    let harFlat: [Float]
    let harFrames: Int
    let harDebug: HarDebugComponents?
    if tensorDumpPath != nil {
        let components = buildHarComponents(
            f0Padded: f0Padded,
            linearWeights: weights.linear_weights,
            linearBias: weights.linear_bias,
            seed: seed
        )
        harFlat = components.har
        harFrames = components.nFrames
        harDebug = components
    } else {
        let built = buildHar(
            f0Padded: f0Padded,
            linearWeights: weights.linear_weights,
            linearBias: weights.linear_bias,
            seed: seed
        )
        harFlat = built.har
        harFrames = built.nFrames
        harDebug = nil
    }
    let t13 = CFAbsoluteTimeGetCurrent()
    let tHnsf = t13 - t12

    try withTensorDump { writer in
        if let harDebug {
            try writer.writeFloatArray(name: "har_source", values: harDebug.harSource, shape: [1, harDebug.harSource.count])
            try writer.writeFloatArray(name: "har_magnitude", values: harDebug.magnitude, shape: [1, 11, harDebug.nFrames])
            try writer.writeFloatArray(name: "har_phase", values: harDebug.phase, shape: [1, 11, harDebug.nFrames])
        }
        try writer.writeFloatArray(name: "har", values: harFlat, shape: [1, 22, harFrames])
    }

    // Stage 8: GeneratorFromHar CoreML
    let t14 = CFAbsoluteTimeGetCurrent()
    let harArray = try makeZeroArray3D(channels: 22, time: harFrames)
    copyInto(array: harArray, from: harFlat)
    let genRefS = try makeZeroArray2D(dim: 256)
    let genRefSP = genRefS.dataPointer.assumingMemoryBound(to: Float.self)
    for j in 0..<256 { genRefSP[j] = benchInput.ref_s[j] }

    let genShapes = inputShapes(from: genModel)
    let xPreExpTime = genShapes["x_pre"]?.last ?? xPre.shape.last!.intValue
    let harExpTime = genShapes["har"]?.last ?? harFrames
    let xPrePadded = try zeroPad3D(source: xPre, channels: xPre.shape[1].intValue, targetTime: xPreExpTime)
    let harPadded = try zeroPad3D(source: harArray, channels: 22, targetTime: harExpTime)

    try withTensorDump { writer in
        try writer.writeMLMultiArray(name: "x_pre_padded", array: xPrePadded)
        try writer.writeMLMultiArray(name: "har_padded", array: harPadded)
    }

    let genInput = try MLDictionaryFeatureProvider(dictionary: [
        "x_pre": MLFeatureValue(multiArray: xPrePadded),
        "ref_s": MLFeatureValue(multiArray: genRefS),
        "har": MLFeatureValue(multiArray: harPadded),
    ])
    let genOutput = try genModel.prediction(from: genInput)
    let t15 = CFAbsoluteTimeGetCurrent()
    let tGenerator = t15 - t14

    // Stage 9: Trim
    let t16 = CFAbsoluteTimeGetCurrent()
    let waveformKey = genOutput.featureNames.contains("waveform") ? "waveform" : genOutput.featureNames.first!
    let waveformArr = genOutput.featureValue(for: waveformKey)!.multiArrayValue!
    let origF0Len = frames * 2
    let targetLen = Int(round(Double(origF0Len) / 80.0 * 24000.0))
    let waveformValues = floatValues(from: waveformArr)
    let trimLen = min(waveformValues.count, targetLen)
    let samples = Array(waveformValues.prefix(trimLen))
    let t17 = CFAbsoluteTimeGetCurrent()
    let tTrim = t17 - t16

    try withTensorDump { writer in
        try writer.writeFloatArray(
            name: "waveform_full",
            values: waveformValues,
            shape: waveformArr.shape.map { $0.intValue }
        )
        try writer.writeFloatArray(name: "waveform", values: samples, shape: [trimLen])
    }

    if let wavPath = wavOutputPath {
        try writeWavMono16(path: wavPath, samples: samples, sampleRate: 24000)
        fputs("  WAV written: \(wavPath)\n", stderr)
    }

    let wallTime = t17 - t0
    let canonicalDur = benchInput.canonical_duration_s ?? (Double(origF0Len) / 80.0)
    let observedDur = Double(trimLen) / 24000.0

    let result: [String: Any] = [
        "config": "f",
        "input_key": inputKey,
        "status": "ok",
        "error": NSNull(),
        "wall_time_s": round(wallTime * 1e6) / 1e6,
        "canonical_audio_duration_s": round(canonicalDur * 1e6) / 1e6,
        "observed_audio_duration_s": round(observedDur * 1e6) / 1e6,
        "predicted_duration_frames": frames,
        "predicted_duration_tokens": predDur.count,
        "rtf_canonical": canonicalDur > 0 ? round((wallTime / canonicalDur) * 1e6) / 1e6 : NSNull(),
        "rtf_observed": observedDur > 0 ? round((wallTime / observedDur) * 1e6) / 1e6 : NSNull(),
        "speed_vs_realtime_canonical": wallTime > 0 ? round((canonicalDur / wallTime) * 100) / 100 : NSNull(),
        "bucket_used": "\(bucketSec)s",
        "t_duration_coreml_s": round(tDuration * 1e6) / 1e6,
        "t_alignment_s": round(tAlignment * 1e6) / 1e6,
        "t_matrix_ops_s": round(tMatrixOps * 1e6) / 1e6,
        "t_f0ntrain_coreml_s": round(tF0Ntrain * 1e6) / 1e6,
        "t_padding_s": round(tPadding * 1e6) / 1e6,
        "t_decoder_pre_coreml_s": round(tDecoderPre * 1e6) / 1e6,
        "t_hnsf_swift_s": round(tHnsf * 1e6) / 1e6,
        "t_coreml_predict_s": round(tGenerator * 1e6) / 1e6,
        "t_trim_s": round(tTrim * 1e6) / 1e6,
        "t_prefix_extract_s": NSNull(),
        "t_decoder_pre_cpu_s": NSNull(),
        "t_har_builder_cpu_s": NSNull(),
        "t_orchestration_s": NSNull(),
    ]

    try withTensorDump { writer in
        try writer.writeManifest(metadata: [
            "producer": "swift",
            "executable": "kokoro-bench",
            "input_key": inputKey,
            "text": benchInput.text,
            "voice": benchInput.voice,
            "speed": benchInput.speed,
            "seed": seed,
            "bucket_seconds": bucketSec,
            "duration_token_length": T,
            "num_tokens": benchInput.num_tokens,
            "natural_frames": frames,
            "canonical_duration_s": benchInput.canonical_duration_s ?? NSNull(),
            "observed_audio_duration_s": observedDur,
            "t_frames": tFrames,
            "full_f0_len": fullF0Len,
            "decoder_frame_count": frameCount,
            "x_pre_expected_time": xPreExpTime,
            "har_expected_time": harExpTime,
            "trim_len": trimLen,
            "hnsf_reference_command": "uv run python scripts/validate_hnsf_swift.py generate",
        ])
    }

    try validateDurationAgreement(inputKey: inputKey, canonical: benchInput.canonical_duration_s, observed: observedDur)

    return try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
}

// MARK: - Single-shot mode

func runSingleShot(modelsDir: String, inputsDir: String, hnsfWeightsPath: String,
                    inputKey: String, seed: UInt64, outputPath: String?, wavPath: String?,
                    tensorDumpPath: String?,
                    warmupCount: Int,
                    computeUnits: MLComputeUnits = .all) throws {
    let weightsData = try Data(contentsOf: URL(fileURLWithPath: hnsfWeightsPath))
    let weights = try JSONDecoder().decode(HnsfWeights.self, from: weightsData)

    let inputPath = URL(fileURLWithPath: inputsDir).appendingPathComponent("\(inputKey).json")
    let inputData = try Data(contentsOf: inputPath)
    let benchInput = try JSONDecoder().decode(BenchInput.self, from: inputData)

    let cache = ModelCache(modelsDir: URL(fileURLWithPath: modelsDir), computeUnits: computeUnits)

    fputs("Loading models...\n", stderr)
    let jsonData = try runPipeline(
        benchInput: benchInput,
        inputKey: inputKey,
        seed: seed,
        weights: weights,
        cache: cache,
        wavOutputPath: wavPath,
        tensorDumpPath: tensorDumpPath
    )

    let jsonString = String(data: jsonData, encoding: .utf8)!
    if let outputPath = outputPath {
        try jsonString.write(toFile: outputPath, atomically: true, encoding: .utf8)
        fputs("Result written to: \(outputPath)\n", stderr)
    } else {
        print(jsonString)
    }
}

func runGeneratorInputDump(modelsDir: String, tensorInputDumpPath: String,
                           outputPath: String?, tensorDumpPath: String?,
                           computeUnits: MLComputeUnits = .all) throws {
    let reader = try TensorDumpReader(directory: URL(fileURLWithPath: tensorInputDumpPath))
    let bucketSec = reader.metadata["bucket_seconds"] as? Int ?? 3
    let trimLen = reader.metadata["trim_len"] as? Int
    let cache = ModelCache(modelsDir: URL(fileURLWithPath: modelsDir), computeUnits: computeUnits)
    let genModel = try cache.generatorModel(bucket: bucketSec)

    let xPre = try reader.readFloatArray(name: "x_pre_padded")
    let refS = try reader.readFloatArray(name: "ref_s")
    let har = try reader.readFloatArray(name: "har_padded")
    let xPreArray = try makeFloatArray(shape: xPre.shape, values: xPre.values)
    let refSArray = try makeFloatArray(shape: refS.shape, values: refS.values)
    let harArray = try makeFloatArray(shape: har.shape, values: har.values)

    let t0 = CFAbsoluteTimeGetCurrent()
    let output = try genModel.prediction(from: try MLDictionaryFeatureProvider(dictionary: [
        "x_pre": MLFeatureValue(multiArray: xPreArray),
        "ref_s": MLFeatureValue(multiArray: refSArray),
        "har": MLFeatureValue(multiArray: harArray),
    ]))
    let t1 = CFAbsoluteTimeGetCurrent()
    let waveformKey = output.featureNames.contains("waveform") ? "waveform" : output.featureNames.first!
    let waveformArray = output.featureValue(for: waveformKey)!.multiArrayValue!
    let outputTrimLen = min(waveformArray.count, trimLen ?? waveformArray.count)

    if let tensorDumpPath {
        var writer = try TensorDumpWriter(directory: URL(fileURLWithPath: tensorDumpPath))
        try writer.writeMLMultiArray(name: "x_pre_padded", array: xPreArray)
        try writer.writeMLMultiArray(name: "ref_s", array: refSArray)
        try writer.writeMLMultiArray(name: "har_padded", array: harArray)
        try writer.writeMLMultiArray(name: "waveform_full", array: waveformArray)
        let waveformValues = floatValues(from: waveformArray)
        let waveform = Array(waveformValues.prefix(outputTrimLen))
        try writer.writeFloatArray(name: "waveform", values: waveform, shape: [outputTrimLen])
        try writer.writeManifest(metadata: [
            "producer": "swift",
            "executable": "kokoro-bench",
            "mode": "generator-input-dump",
            "source_tensor_dump": tensorInputDumpPath,
            "bucket_seconds": bucketSec,
            "trim_len": outputTrimLen,
            "prediction_key": waveformKey,
        ])
    }

    let result: [String: Any] = [
        "config": "f",
        "mode": "generator-input-dump",
        "status": "ok",
        "bucket_used": "\(bucketSec)s",
        "source_tensor_dump": tensorInputDumpPath,
        "observed_audio_duration_s": round((Double(outputTrimLen) / 24000.0) * 1e6) / 1e6,
        "t_coreml_predict_s": round((t1 - t0) * 1e6) / 1e6,
    ]
    let jsonString = String(
        data: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]),
        encoding: .utf8
    )!
    if let outputPath {
        try jsonString.write(toFile: outputPath, atomically: true, encoding: .utf8)
        fputs("Result written to: \(outputPath)\n", stderr)
    } else {
        print(jsonString)
    }
}

// MARK: - Batch mode

func runBatch(modelsDir: String, inputsDir: String, hnsfWeightsPath: String,
              computeUnits: MLComputeUnits = .all) throws {
    let weightsData = try Data(contentsOf: URL(fileURLWithPath: hnsfWeightsPath))
    let weights = try JSONDecoder().decode(HnsfWeights.self, from: weightsData)
    let cache = ModelCache(modelsDir: URL(fileURLWithPath: modelsDir), computeUnits: computeUnits)
    let inputsDirURL = URL(fileURLWithPath: inputsDir)

    fputs("Batch mode ready. Waiting for commands on stdin...\n", stderr)
    // Signal to parent process that we're ready to receive commands
    print("READY")
    fflush(stdout)

    while let line = readLine() {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }

        guard let cmdData = trimmed.data(using: .utf8) else {
            fputs("  Invalid UTF-8 in command\n", stderr)
            print("ERROR")
            fflush(stdout)
            continue
        }

        do {
            let cmd = try JSONDecoder().decode(BatchCommand.self, from: cmdData)
            let seed = cmd.seed ?? 42
            let isWarmup = cmd.warmup ?? false

            // Load input
            let inputPath = inputsDirURL.appendingPathComponent("\(cmd.input_key).json")
            let inputFileData = try Data(contentsOf: inputPath)
            let benchInput = try JSONDecoder().decode(BenchInput.self, from: inputFileData)

            fputs("  Processing: \(cmd.input_key) (warmup=\(isWarmup))\n", stderr)

            let jsonData = try runPipeline(
                benchInput: benchInput,
                inputKey: cmd.input_key,
                seed: seed,
                weights: weights,
                cache: cache
            )

            let jsonString = String(data: jsonData, encoding: .utf8)!
            try jsonString.write(toFile: cmd.output, atomically: true, encoding: .utf8)

            print("DONE")
            fflush(stdout)
        } catch {
            fputs("  Error processing command: \(error)\n", stderr)
            // Write error result to output file if possible
            if let cmdData = trimmed.data(using: .utf8),
               let cmd = try? JSONDecoder().decode(BatchCommand.self, from: cmdData) {
                let errorResult: [String: Any] = [
                    "config": "f",
                    "input_key": cmd.input_key,
                    "status": "swift_error",
                    "error": String(describing: error),
                ]
                if let errData = try? JSONSerialization.data(withJSONObject: errorResult),
                   let errStr = String(data: errData, encoding: .utf8) {
                    try? errStr.write(toFile: cmd.output, atomically: true, encoding: .utf8)
                }
            }
            print("DONE")
            fflush(stdout)
        }
    }

    fputs("Batch mode: stdin closed, exiting.\n", stderr)
}

// MARK: - Main

func main() throws {
    let args = CommandLine.arguments
    var modelsDir: String?
    var inputsDir: String?
    var hnsfWeightsPath: String?
    var inputKey: String?
    var outputPath: String?
    var wavPath: String?
    var tensorDumpPath: String?
    var generatorInputDumpPath: String?
    var warmupCount = 1
    var seed: UInt64 = 42
    var batchMode = false
    var computeUnitsStr = "all"

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--models-dir":
            i += 1; modelsDir = args[i]
        case "--inputs-dir":
            i += 1; inputsDir = args[i]
        case "--hnsf-weights":
            i += 1; hnsfWeightsPath = args[i]
        case "--input-key":
            i += 1; inputKey = args[i]
        case "--warmup":
            i += 1; warmupCount = Int(args[i]) ?? 1
        case "--seed":
            i += 1; seed = UInt64(args[i]) ?? 42
        case "--output":
            i += 1; outputPath = args[i]
        case "--wav":
            i += 1; wavPath = args[i]
        case "--dump-tensors":
            i += 1; tensorDumpPath = args[i]
        case "--generator-input-dump":
            i += 1; generatorInputDumpPath = args[i]
        case "--batch":
            batchMode = true
        case "--compute-units":
            i += 1; computeUnitsStr = args[i]
        default:
            break
        }
        i += 1
    }

    guard let modelsDir = modelsDir,
          let inputsDir = inputsDir,
          let hnsfWeightsPath = hnsfWeightsPath else {
        fputs("Usage: kokoro-bench --models-dir DIR --inputs-dir DIR --hnsf-weights FILE [--input-key KEY | --batch]\n", stderr)
        exit(1)
    }

    // Parse compute units
    let computeUnits: MLComputeUnits
    switch computeUnitsStr.lowercased() {
    case "all": computeUnits = .all
    case "cpuandneuralengine": computeUnits = .cpuAndNeuralEngine
    case "cpuandgpu": computeUnits = .cpuAndGPU
    case "cpuonly": computeUnits = .cpuOnly
    default:
        fputs("Unknown compute units: \(computeUnitsStr). Use: all, cpuAndNeuralEngine, cpuAndGPU, cpuOnly\n", stderr)
        exit(1)
    }

    if batchMode {
        try runBatch(modelsDir: modelsDir, inputsDir: inputsDir, hnsfWeightsPath: hnsfWeightsPath,
                     computeUnits: computeUnits)
    } else if let generatorInputDumpPath {
        try runGeneratorInputDump(
            modelsDir: modelsDir,
            tensorInputDumpPath: generatorInputDumpPath,
            outputPath: outputPath,
            tensorDumpPath: tensorDumpPath,
            computeUnits: computeUnits
        )
    } else {
        guard let inputKey = inputKey else {
            fputs("Error: --input-key required in single-shot mode (or use --batch)\n", stderr)
            exit(1)
        }
        try runSingleShot(modelsDir: modelsDir, inputsDir: inputsDir, hnsfWeightsPath: hnsfWeightsPath,
                          inputKey: inputKey, seed: seed, outputPath: outputPath, wavPath: wavPath,
                          tensorDumpPath: tensorDumpPath,
                          warmupCount: warmupCount,
                          computeUnits: computeUnits)
    }
}

try main()
