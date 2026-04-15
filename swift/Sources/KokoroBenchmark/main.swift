/// Kokoro Swift pipeline benchmark CLI.
///
/// Called by ``scripts/bakeoff_harness.py`` as a subprocess for Config F.
/// Loads all CoreML models once, runs each input with timing, and outputs
/// JSON results to stdout.
///
/// Usage::
///
///     kokoro-bench --models-dir /path/to/coreml \
///                  --inputs-dir /path/to/swift_bench_inputs \
///                  --hnsf-weights /path/to/hnsf_weights.json \
///                  --input-key tiny \
///                  --seed 42
///
/// Output: single JSON object on stdout with timing fields matching the
/// bakeoff results schema.

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

// MARK: - Main

func main() throws {
    // Parse arguments
    let args = CommandLine.arguments
    var modelsDir: String?
    var inputsDir: String?
    var hnsfWeightsPath: String?
    var inputKey: String?
    var outputPath: String?  // Write JSON to file instead of stdout (avoids E5RT noise)
    var warmupCount = 1
    var seed: UInt64 = 42

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
        default:
            break
        }
        i += 1
    }

    guard let modelsDir = modelsDir,
          let inputsDir = inputsDir,
          let hnsfWeightsPath = hnsfWeightsPath,
          let inputKey = inputKey else {
        fputs("Usage: kokoro-bench --models-dir DIR --inputs-dir DIR --hnsf-weights FILE --input-key KEY\n", stderr)
        exit(1)
    }

    // Load hn-nsf weights
    let weightsData = try Data(contentsOf: URL(fileURLWithPath: hnsfWeightsPath))
    let weights = try JSONDecoder().decode(HnsfWeights.self, from: weightsData)

    // Load input
    let inputPath = URL(fileURLWithPath: inputsDir).appendingPathComponent("\(inputKey).json")
    let inputData = try Data(contentsOf: inputPath)
    let benchInput = try JSONDecoder().decode(BenchInput.self, from: inputData)

    // Load CoreML models manually (not through KokoroPipeline init to control timing)
    let modelsURL = URL(fileURLWithPath: modelsDir)
    let config = MLModelConfiguration()
    config.computeUnits = .all

    // Duration model
    fputs("Loading models...\n", stderr)
    let durURL = modelsURL.appendingPathComponent("kokoro_duration.mlpackage")
    let durCompiled = try MLModel.compileModel(at: durURL)
    let durationModel = try MLModel(contentsOf: durCompiled, configuration: config)

    // Determine bucket from input length
    // We need to run duration model first to get pred_dur, then select bucket
    // For warmup and bucket selection, do a quick duration prediction
    let T = 128
    let idsArray = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
    let maskArray = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
    let refSArray = try MLMultiArray(shape: [1, 256], dataType: .float32)
    let speedArray = try MLMultiArray(shape: [1], dataType: .float32)

    let idsPtr = idsArray.dataPointer.assumingMemoryBound(to: Int32.self)
    let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
    let refSPtr = refSArray.dataPointer.assumingMemoryBound(to: Float.self)

    for j in 0..<min(benchInput.input_ids.count, T) {
        idsPtr[j] = benchInput.input_ids[j]
    }
    for j in 0..<min(benchInput.attention_mask.count, T) {
        maskPtr[j] = benchInput.attention_mask[j]
    }
    for j in 0..<min(benchInput.ref_s.count, 256) {
        refSPtr[j] = benchInput.ref_s[j]
    }
    speedArray[0] = NSNumber(value: benchInput.speed)

    let durInput = try MLDictionaryFeatureProvider(dictionary: [
        "input_ids": MLFeatureValue(multiArray: idsArray),
        "attention_mask": MLFeatureValue(multiArray: maskArray),
        "ref_s": MLFeatureValue(multiArray: refSArray),
        "speed": MLFeatureValue(multiArray: speedArray),
    ])

    // Quick duration prediction to determine bucket
    let durOut = try durationModel.prediction(from: durInput)
    let predDurArr = durOut.featureValue(for: "pred_dur")!.multiArrayValue!
    let predDurPtr = predDurArr.dataPointer.assumingMemoryBound(to: Float.self)
    let tokenCount = predDurArr.shape.last!.intValue
    var totalFrames = 0
    for j in 0..<tokenCount {
        totalFrames += max(1, Int(round(predDurPtr[j])))
    }
    // Use canonical duration from manifest for bucket selection.
    // The model's pred_dur includes padding tokens (clamped to min=1), inflating
    // the frame count. The Python pipeline uses T_f0 / 80.0 from actual vocoder inputs.
    let canonicalDurForBucket = benchInput.canonical_duration_s ?? (Double(totalFrames) / 80.0)
    let bucketSec = Int(ceil(canonicalDurForBucket)) <= 3 ? 3 : 10

    fputs("  Input: \(inputKey), tokens: \(benchInput.num_tokens), frames: \(totalFrames), canonical=\(canonicalDurForBucket)s, bucket: \(bucketSec)s\n", stderr)

    // Load bucket-specific models
    let tFrames = bucketSec == 3 ? 120 : 400
    let f0nURL = modelsURL.appendingPathComponent("kokoro_f0ntrain_t\(tFrames).mlpackage")
    let f0nModel = try MLModel(contentsOf: try MLModel.compileModel(at: f0nURL), configuration: config)

    let decPreURL = modelsURL.appendingPathComponent("kokoro_decoder_pre_\(bucketSec)s.mlpackage")
    let decPreModel = try MLModel(contentsOf: try MLModel.compileModel(at: decPreURL), configuration: config)

    let genURL = modelsURL.appendingPathComponent("kokoro_decoder_har_post_\(bucketSec)s.mlpackage")
    let genModel = try MLModel(contentsOf: try MLModel.compileModel(at: genURL), configuration: config)

    // Compute bucket geometry for warmup and timed run
    let bucketSamples = bucketSec * 24000
    let fullF0Len = Int(round(Double(bucketSamples) / 300.0))

    // Warmup all models (not just duration — Generator needs compilation warmup)
    fputs("Warming up (\(warmupCount) calls)...\n", stderr)
    for _ in 0..<warmupCount {
        let _ = try durationModel.prediction(from: durInput)
    }
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

    // --- Timed run ---
    fputs("Running timed iteration...\n", stderr)

    // Stage 1: Duration CoreML
    let t0 = CFAbsoluteTimeGetCurrent()
    let durOutput = try durationModel.prediction(from: durInput)
    let t1 = CFAbsoluteTimeGetCurrent()
    let tDuration = t1 - t0

    // Extract outputs
    let predDurArray = durOutput.featureValue(for: "pred_dur")!.multiArrayValue!
    let dArray = durOutput.featureValue(for: "d")!.multiArrayValue!
    let tEnArray = durOutput.featureValue(for: "t_en")!.multiArrayValue!

    let pdPtr = predDurArray.dataPointer.assumingMemoryBound(to: Float.self)
    let tc = predDurArray.shape.last!.intValue
    var predDur = [Int](repeating: 0, count: tc)
    for j in 0..<tc {
        predDur[j] = max(1, Int(round(pdPtr[j])))
    }
    let frames = predDur.reduce(0, +)
    let secs = Double(frames) / 80.0

    // Stage 2: Alignment
    let t2 = CFAbsoluteTimeGetCurrent()
    let alignment = buildAlignmentMatrix(predDur: predDur, traceLength: tc, frameCount: frames)
    let t3 = CFAbsoluteTimeGetCurrent()
    let tAlignment = t3 - t2

    // Stage 3: Matrix ops
    // Duration model outputs d as (1, tokens, 640) and t_en as (1, tokens, 512).
    // We need: en = d_transposed @ alignment = (1, 640, tokens) @ (tokens, frames) = (1, 640, frames)
    // Since matmul3D expects (1, M, K), we need to transpose d first.
    let t4 = CFAbsoluteTimeGetCurrent()
    // Transpose d: (1, tokens, 640) -> (1, 640, tokens)
    let dTransposed = try transpose3D(source: dArray, dim1: 640, dim2: tc)
    let en = try matmul3D(a: dTransposed, b: alignment, M: 640, K: tc, N: frames)
    // Transpose t_en: (1, tokens, 512) -> (1, 512, tokens)
    let tEnTransposed = try transpose3D(source: tEnArray, dim1: 512, dim2: tc)
    let _ = try matmul3D(a: tEnTransposed, b: alignment, M: 512, K: tc, N: frames)
    let t5 = CFAbsoluteTimeGetCurrent()
    let tMatrixOps = t5 - t4

    // Stage 4: F0Ntrain
    let t6 = CFAbsoluteTimeGetCurrent()
    let enPadded = try zeroPad3D(source: en, channels: 640, targetTime: tFrames)
    let sArray = try makeZeroArray2D(dim: 128)
    let sP = sArray.dataPointer.assumingMemoryBound(to: Float.self)
    for j in 0..<128 { sP[j] = benchInput.ref_s[128 + j] }

    let f0nInput = try MLDictionaryFeatureProvider(dictionary: [
        "en": MLFeatureValue(multiArray: enPadded),
        "s": MLFeatureValue(multiArray: sArray),
    ])
    let f0nOutput = try f0nModel.prediction(from: f0nInput)
    let f0PredArr = f0nOutput.featureValue(for: "F0_pred")!.multiArrayValue!
    let nPredArr = f0nOutput.featureValue(for: "N_pred")!.multiArrayValue!
    let t7 = CFAbsoluteTimeGetCurrent()
    let tF0Ntrain = t7 - t6

    // Extract F0/N
    let f0Len = f0PredArr.count
    let f0P = f0PredArr.dataPointer.assumingMemoryBound(to: Float.self)
    let nP = nPredArr.dataPointer.assumingMemoryBound(to: Float.self)
    var f0Curve = [Float](repeating: 0, count: f0Len)
    var nCurve = [Float](repeating: 0, count: f0Len)
    for j in 0..<f0Len { f0Curve[j] = f0P[j]; nCurve[j] = nP[j] }

    // Stage 5: Padding
    let t8 = CFAbsoluteTimeGetCurrent()
    let f0Padded = zeroPad1D(source: f0Curve, targetLength: fullF0Len)
    let nPadded = zeroPad1D(source: nCurve, targetLength: fullF0Len)

    // ASR padding for DecoderPre (recompute with transposed t_en)
    let asr = try matmul3D(a: tEnTransposed, b: alignment, M: 512, K: tc, N: frames)
    let frameCount = (fullF0Len - 1) / 2 + 1  // Conv1d(k=3,s=2,p=1) output length
    let asrPadded = try zeroPad3D(source: asr, channels: 512, targetTime: frameCount)
    let t9 = CFAbsoluteTimeGetCurrent()
    let tPadding = t9 - t8

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

    // Stage 7: hn-nsf Swift
    let t12 = CFAbsoluteTimeGetCurrent()
    let (harFlat, harFrames) = buildHar(
        f0Padded: f0Padded,
        linearWeights: weights.linear_weights,
        linearBias: weights.linear_bias,
        seed: seed
    )
    let t13 = CFAbsoluteTimeGetCurrent()
    let tHnsf = t13 - t12

    // Stage 8: GeneratorFromHar CoreML
    let t14 = CFAbsoluteTimeGetCurrent()
    let harArray = try makeZeroArray3D(channels: 22, time: harFrames)
    copyInto(array: harArray, from: harFlat)
    let genRefS = try makeZeroArray2D(dim: 256)
    let genRefSP = genRefS.dataPointer.assumingMemoryBound(to: Float.self)
    for j in 0..<256 { genRefSP[j] = benchInput.ref_s[j] }

    // Read expected shapes from model
    let genShapes = inputShapes(from: genModel)
    let xPreExpTime = genShapes["x_pre"]?.last ?? xPre.shape.last!.intValue
    let harExpTime = genShapes["har"]?.last ?? harFrames
    let xPrePadded = try zeroPad3D(source: xPre, channels: xPre.shape[1].intValue, targetTime: xPreExpTime)
    let harPadded = try zeroPad3D(source: harArray, channels: 22, targetTime: harExpTime)

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
    let origF0Len = frames * 2  // F0Ntrain 2x upsample
    let targetLen = Int(round(Double(origF0Len) / 80.0 * 24000.0))
    let trimLen = min(waveformArr.count, targetLen)
    let t17 = CFAbsoluteTimeGetCurrent()
    let tTrim = t17 - t16

    let wallTime = t17 - t0
    // Use canonical duration from manifest (Python pipeline's T_f0 / 80.0) if available.
    // Fall back to model-derived duration (less accurate due to padding token effects).
    let canonicalDur = benchInput.canonical_duration_s ?? (Double(origF0Len) / 80.0)
    let observedDur = Double(trimLen) / 24000.0

    // Output JSON matching bakeoff results schema
    let result: [String: Any] = [
        "config": "f",
        "input_key": inputKey,
        "status": "ok",
        "error": NSNull(),
        "wall_time_s": round(wallTime * 1e6) / 1e6,
        "canonical_audio_duration_s": round(canonicalDur * 1e6) / 1e6,
        "observed_audio_duration_s": round(observedDur * 1e6) / 1e6,
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
        // Legacy fields (set to nil for compatibility)
        "t_prefix_extract_s": NSNull(),
        "t_decoder_pre_cpu_s": NSNull(),
        "t_har_builder_cpu_s": NSNull(),
        "t_orchestration_s": NSNull(),
    ]

    let jsonData = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    let jsonString = String(data: jsonData, encoding: .utf8)!

    if let outputPath = outputPath {
        try jsonString.write(toFile: outputPath, atomically: true, encoding: .utf8)
        fputs("Result written to: \(outputPath)\n", stderr)
    } else {
        print(jsonString)
    }
}

try main()
