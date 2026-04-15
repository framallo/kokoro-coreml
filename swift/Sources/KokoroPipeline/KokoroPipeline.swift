/// Kokoro TTS pipeline orchestrator: chains CoreML models + Swift DSP.
///
/// Replaces the Python ``extract_vocoder_inputs()`` + ``build_decoder_har_post_inputs_np()``
/// + ``decoder_har_post_bucket_impl()`` chain with native Swift + CoreML.
///
/// ## Pipeline stages
///
/// 1. Duration CoreML → pred_dur, d, t_en, s, ref_s
/// 2. Alignment (Swift) → one-hot matrix from pred_dur
/// 3. Matrix ops (Accelerate) → en = d @ alignment, asr = t_en @ alignment
/// 4. F0Ntrain CoreML → F0_pred, N_pred
/// 5. Pad to bucket geometry (Swift)
/// 6. DecoderPre (bridge / pre-computed / CoreML Phase 4)
/// 7. hn-nsf (Swift/Accelerate, Double precision phase) → har
/// 8. GeneratorFromHar CoreML → waveform
/// 9. Trim to natural utterance length
///
/// ## Stage timing
///
/// Every stage is timed with ``ContinuousClock`` and reported in ``SynthesisResult``.

import CoreML
import Foundation
import Accelerate

// MARK: - Configuration

/// Pipeline configuration matching Kokoro model constants.
public enum PipelineConstants {
    /// Audio sample rate (Hz).
    public static let sampleRate: Int = 24000
    /// F0 frame rate (Hz). Converts F0 frame count to seconds.
    public static let f0FrameRate: Double = 80.0
    /// Duration model fixed token length.
    public static let durationTokenLength: Int = 128
    /// Voice embedding total dimension.
    public static let voiceEmbeddingDim: Int = 256
    /// Voice style dimension (ref_s[:, 128:]).
    public static let styleDim: Int = 128
    /// Voice baseline dimension (ref_s[:, :128]).
    public static let baselineDim: Int = 128
    /// Hidden dimension from duration encoder output.
    public static let hiddenDim: Int = 640
    /// Text encoder output dimension.
    public static let textEncoderDim: Int = 512
}

// MARK: - Stage Timing

/// Timing breakdown for a single synthesis call.
public struct StageTimings {
    public var durationCoreML: Double = 0
    public var alignment: Double = 0
    public var matrixOps: Double = 0
    public var f0ntrainCoreML: Double = 0
    public var padding: Double = 0
    public var decoderPre: Double = 0
    public var hnsfSwift: Double = 0
    public var generatorCoreML: Double = 0
    public var trim: Double = 0

    /// Total pipeline wall time.
    public var total: Double {
        durationCoreML + alignment + matrixOps + f0ntrainCoreML +
        padding + decoderPre + hnsfSwift + generatorCoreML + trim
    }

    /// Pre-decoder overhead (everything before GeneratorFromHar predict).
    public var preDecoder: Double {
        total - generatorCoreML - trim
    }
}

/// Result of a synthesis call including audio and timing.
public struct SynthesisResult {
    /// Raw audio waveform at 24 kHz, trimmed to natural utterance length.
    public let audio: [Float]
    /// Per-stage timing breakdown.
    public let timings: StageTimings
    /// Selected bucket in seconds.
    public let bucketSeconds: Int
    /// Audio duration in seconds (from F0 frame count).
    public let audioDurationSeconds: Double
}

// MARK: - Pipeline

/// Main TTS pipeline orchestrator.
///
/// Loads CoreML models and provides ``synthesize()`` for text-to-audio.
/// DecoderPre is currently a bridge (pre-computed tensors loaded from disk).
public class KokoroPipeline {
    private let durationModel: MLModel
    private let f0ntrainModels: [Int: MLModel]  // keyed by T_frames
    private let generatorModels: [Int: MLModel]  // keyed by bucket seconds

    /// Learned weights from SourceModuleHnNSF.l_linear.
    private let linearWeights: [Float]
    private let linearBias: Float

    /// Available bucket durations in seconds.
    private let availableBuckets: [Int]

    /// Pre-computed DecoderPre outputs for bridge mode.
    /// Keyed by a string identifier (e.g., input name or hash).
    /// Phase 4 replaces this with a CoreML model.
    public var precomputedDecoderPre: [String: MLMultiArray] = [:]

    /// Load all models from a directory.
    ///
    /// Expected files:
    /// - ``kokoro_duration.mlpackage``
    /// - ``kokoro_f0ntrain_t{T}.mlpackage`` for each bucket's T_frames
    /// - ``kokoro_decoder_har_post_{N}s.mlpackage`` for each bucket
    /// - ``linear_weights.json`` (SourceModuleHnNSF learned weights)
    public init(
        modelsDirectory: URL,
        buckets: [Int] = [3, 10],
        linearWeights: [Float],
        linearBias: Float
    ) throws {
        // Duration model
        let durURL = modelsDirectory.appendingPathComponent("kokoro_duration.mlpackage")
        let durConfig = MLModelConfiguration()
        durConfig.computeUnits = .all
        self.durationModel = try MLModel(contentsOf: MLModel.compileModel(at: durURL), configuration: durConfig)

        // F0Ntrain models (one per bucket's T_frames)
        var f0Models: [Int: MLModel] = [:]
        // T_frames values: 3s -> 120, 10s -> 400 (from bucket geometry)
        let tFramesMap: [Int: Int] = [3: 120, 10: 400, 45: 1800]
        for sec in buckets {
            if let t = tFramesMap[sec] {
                let url = modelsDirectory.appendingPathComponent("kokoro_f0ntrain_t\(t).mlpackage")
                if FileManager.default.fileExists(atPath: url.path) {
                    let config = MLModelConfiguration()
                    config.computeUnits = .all
                    let compiled = try MLModel.compileModel(at: url)
                    f0Models[t] = try MLModel(contentsOf: compiled, configuration: config)
                }
            }
        }
        self.f0ntrainModels = f0Models

        // Generator (HAR-post) models
        var genModels: [Int: MLModel] = [:]
        for sec in buckets {
            let url = modelsDirectory.appendingPathComponent("kokoro_decoder_har_post_\(sec)s.mlpackage")
            if FileManager.default.fileExists(atPath: url.path) {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                let compiled = try MLModel.compileModel(at: url)
                genModels[sec] = try MLModel(contentsOf: compiled, configuration: config)
            }
        }
        self.generatorModels = genModels
        self.availableBuckets = Array(genModels.keys.sorted())

        self.linearWeights = linearWeights
        self.linearBias = linearBias
    }

    /// Synthesize audio from pre-tokenized input.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs, padded/truncated to 128. Shape semantics: (128,).
    ///   - attentionMask: Mask for actual tokens (1) vs padding (0). Shape: (128,).
    ///   - refS: Voice embedding, shape (256,).
    ///   - speed: Speech rate multiplier.
    ///   - decoderPreKey: Key to look up pre-computed DecoderPre output (bridge mode).
    /// - Returns: SynthesisResult with audio and timing breakdown.
    public func synthesize(
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float = 1.0,
        decoderPreKey: String? = nil
    ) throws -> SynthesisResult {
        var timings = StageTimings()

        // ---- Stage 1: Duration CoreML ----
        let t0 = CFAbsoluteTimeGetCurrent()
        let durInput = try buildDurationInput(inputIds: inputIds, attentionMask: attentionMask, refS: refS, speed: speed)
        let durOutput = try durationModel.prediction(from: durInput)
        let t1 = CFAbsoluteTimeGetCurrent()
        timings.durationCoreML = t1 - t0

        // Extract duration model outputs
        let predDurArray = durOutput.featureValue(for: "pred_dur")!.multiArrayValue!
        let dArray = durOutput.featureValue(for: "d")!.multiArrayValue!
        let tEnArray = durOutput.featureValue(for: "t_en")!.multiArrayValue!

        // Parse pred_dur into integer array
        let predDurPtr = predDurArray.dataPointer.assumingMemoryBound(to: Float.self)
        let tokenCount = predDurArray.shape.last!.intValue
        var predDur = [Int](repeating: 0, count: tokenCount)
        for i in 0..<tokenCount {
            predDur[i] = max(1, Int(round(predDurPtr[i])))
        }
        let totalFrames = predDur.reduce(0, +)
        let totalSeconds = Double(totalFrames) / PipelineConstants.f0FrameRate

        // ---- Stage 2: Alignment ----
        let t2 = CFAbsoluteTimeGetCurrent()
        let alignment = buildAlignmentMatrix(
            predDur: predDur,
            traceLength: tokenCount,
            frameCount: totalFrames
        )
        let t3 = CFAbsoluteTimeGetCurrent()
        timings.alignment = t3 - t2

        // ---- Stage 3: Matrix ops (en = d @ alignment, asr = t_en @ alignment) ----
        let t4 = CFAbsoluteTimeGetCurrent()
        let en = try matmul3D(
            a: dArray,
            b: alignment,
            M: PipelineConstants.hiddenDim,
            K: tokenCount,
            N: totalFrames
        )
        // asr is needed for DecoderPre input (Phase 4). Currently unused in bridge mode.
        let _ = try matmul3D(
            a: tEnArray,
            b: alignment,
            M: PipelineConstants.textEncoderDim,
            K: tokenCount,
            N: totalFrames
        )
        let t5 = CFAbsoluteTimeGetCurrent()
        timings.matrixOps = t5 - t4

        // ---- Stage 4: F0Ntrain CoreML ----
        let t6 = CFAbsoluteTimeGetCurrent()
        // Select bucket
        guard let bucketSec = selectBucket(totalSeconds: totalSeconds, availableBuckets: availableBuckets) else {
            throw PipelineError.noBucketAvailable
        }
        let tFramesMap: [Int: Int] = [3: 120, 10: 400, 45: 1800]
        guard let tFrames = tFramesMap[bucketSec],
              let f0nModel = f0ntrainModels[tFrames] else {
            throw PipelineError.modelNotLoaded("f0ntrain_t\(tFramesMap[bucketSec] ?? 0)")
        }

        // Pad en to F0Ntrain's expected T dimension
        let enPadded = try zeroPad3D(source: en, channels: PipelineConstants.hiddenDim, targetTime: tFrames)

        // Style embedding: s = ref_s[128:]
        let sArray = try makeZeroArray2D(dim: PipelineConstants.styleDim)
        let sPtr = sArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<PipelineConstants.styleDim {
            sPtr[i] = refS[PipelineConstants.baselineDim + i]
        }

        let f0nInput = try MLDictionaryFeatureProvider(dictionary: [
            "en": MLFeatureValue(multiArray: enPadded),
            "s": MLFeatureValue(multiArray: sArray),
        ])
        let f0nOutput = try f0nModel.prediction(from: f0nInput)
        let f0PredArray = f0nOutput.featureValue(for: "F0_pred")!.multiArrayValue!
        let nPredArray = f0nOutput.featureValue(for: "N_pred")!.multiArrayValue!
        let t7 = CFAbsoluteTimeGetCurrent()
        timings.f0ntrainCoreML = t7 - t6

        // Extract F0/N as flat arrays
        let f0Len = f0PredArray.count
        let f0Ptr = f0PredArray.dataPointer.assumingMemoryBound(to: Float.self)
        let nPtr = nPredArray.dataPointer.assumingMemoryBound(to: Float.self)
        var f0Curve = [Float](repeating: 0, count: f0Len)
        var nCurve = [Float](repeating: 0, count: f0Len)
        for i in 0..<f0Len {
            f0Curve[i] = f0Ptr[i]
            nCurve[i] = nPtr[i]
        }

        // ---- Stage 5: Padding to bucket geometry ----
        let t8 = CFAbsoluteTimeGetCurrent()
        let bucketSamples = bucketSec * PipelineConstants.sampleRate
        let fullF0Len = Int(round(Double(bucketSamples) / Double(HarmonicConstants.upsampleScale)))
        // Pad F0 and N to full_f0_len
        let f0Padded = zeroPad1D(source: f0Curve, targetLength: fullF0Len)
        let t9 = CFAbsoluteTimeGetCurrent()
        timings.padding = t9 - t8

        // ---- Stage 6: DecoderPre (bridge) ----
        let t10 = CFAbsoluteTimeGetCurrent()
        guard let xPre = precomputedDecoderPre[decoderPreKey ?? "default"] else {
            throw PipelineError.decoderPreNotAvailable
        }
        let t11 = CFAbsoluteTimeGetCurrent()
        timings.decoderPre = t11 - t10

        // ---- Stage 7: hn-nsf (Swift/Accelerate) ----
        let t12 = CFAbsoluteTimeGetCurrent()
        let (harFlat, harFrames) = buildHar(
            f0Padded: f0Padded,
            linearWeights: linearWeights,
            linearBias: linearBias,
            seed: 42 // Deterministic for benchmarks
        )
        let t13 = CFAbsoluteTimeGetCurrent()
        timings.hnsfSwift = t13 - t12

        // ---- Stage 8: GeneratorFromHar CoreML ----
        let t14 = CFAbsoluteTimeGetCurrent()
        guard let genModel = generatorModels[bucketSec] else {
            throw PipelineError.modelNotLoaded("decoder_har_post_\(bucketSec)s")
        }

        // Build har MLMultiArray
        let harArray = try makeZeroArray3D(channels: HarmonicConstants.harChannels, time: harFrames)
        copyInto(array: harArray, from: harFlat)

        // ref_s as MLMultiArray
        let refSArray = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
        let refSPtr = refSArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<PipelineConstants.voiceEmbeddingDim {
            refSPtr[i] = refS[i]
        }

        // Read expected shapes from model
        let genShapes = inputShapes(from: genModel)
        let xPreExpectedTime = genShapes["x_pre"]?.last ?? xPre.shape.last!.intValue
        let harExpectedTime = genShapes["har"]?.last ?? harFrames

        // Pad x_pre and har to model-expected dimensions if needed
        let xPrePadded = try zeroPad3D(
            source: xPre,
            channels: xPre.shape[1].intValue,
            targetTime: xPreExpectedTime
        )
        let harPadded = try zeroPad3D(
            source: harArray,
            channels: HarmonicConstants.harChannels,
            targetTime: harExpectedTime
        )

        let genInput = try MLDictionaryFeatureProvider(dictionary: [
            "x_pre": MLFeatureValue(multiArray: xPrePadded),
            "ref_s": MLFeatureValue(multiArray: refSArray),
            "har": MLFeatureValue(multiArray: harPadded),
        ])
        let genOutput = try genModel.prediction(from: genInput)
        let t15 = CFAbsoluteTimeGetCurrent()
        timings.generatorCoreML = t15 - t14

        // ---- Stage 9: Trim ----
        let t16 = CFAbsoluteTimeGetCurrent()
        let waveformKey = genOutput.featureNames.contains("waveform") ? "waveform" : genOutput.featureNames.first!
        let waveformArray = genOutput.featureValue(for: waveformKey)!.multiArrayValue!
        let waveformPtr = waveformArray.dataPointer.assumingMemoryBound(to: Float.self)
        let waveformLen = waveformArray.count

        // Trim to natural utterance length: T_f0 / 80.0 * 24000
        // T_f0 is the ORIGINAL (unpadded) F0 length = totalFrames * 2 (from F0Ntrain 2x upsample)
        let originalF0Len = totalFrames * 2  // F0Ntrain doubles the frame count
        let targetLen = Int(round(Double(originalF0Len) / PipelineConstants.f0FrameRate * Double(PipelineConstants.sampleRate)))
        let trimLen = min(waveformLen, targetLen)

        var audio = [Float](repeating: 0, count: trimLen)
        for i in 0..<trimLen {
            audio[i] = waveformPtr[i]
        }
        let t17 = CFAbsoluteTimeGetCurrent()
        timings.trim = t17 - t16

        return SynthesisResult(
            audio: audio,
            timings: timings,
            bucketSeconds: bucketSec,
            audioDurationSeconds: Double(originalF0Len) / PipelineConstants.f0FrameRate
        )
    }

    // MARK: - Private Helpers

    private func buildDurationInput(
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float
    ) throws -> MLDictionaryFeatureProvider {
        let T = PipelineConstants.durationTokenLength

        let idsArray = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let refSArray = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
        let speedArray = try MLMultiArray(shape: [1], dataType: .float32)

        let idsPtr = idsArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        let refSPtr = refSArray.dataPointer.assumingMemoryBound(to: Float.self)

        for i in 0..<min(inputIds.count, T) {
            idsPtr[i] = inputIds[i]
        }
        for i in 0..<min(attentionMask.count, T) {
            maskPtr[i] = attentionMask[i]
        }
        for i in 0..<min(refS.count, PipelineConstants.voiceEmbeddingDim) {
            refSPtr[i] = refS[i]
        }
        speedArray[0] = NSNumber(value: speed)

        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: idsArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
            "ref_s": MLFeatureValue(multiArray: refSArray),
            "speed": MLFeatureValue(multiArray: speedArray),
        ])
    }
}

// MARK: - Errors

public enum PipelineError: Error, LocalizedError {
    case noBucketAvailable
    case modelNotLoaded(String)
    case decoderPreNotAvailable

    public var errorDescription: String? {
        switch self {
        case .noBucketAvailable:
            return "No bucket available for the requested duration"
        case .modelNotLoaded(let name):
            return "Model not loaded: \(name)"
        case .decoderPreNotAvailable:
            return "DecoderPre output not available. Pre-compute with scripts/decoder_pre_bridge.py or implement Phase 4 (CoreML DecoderPre)."
        }
    }
}
