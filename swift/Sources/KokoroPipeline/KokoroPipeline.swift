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
    /// Legacy duration model fixed token length.
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

    /// F0Ntrain input T dimension for each bucket (seconds → T_frames).
    /// Derived from bucket geometry: full_f0_len = bucket_sec * 24000 / 300,
    /// then F0Ntrain's 2× upsample means T_frames = full_f0_len / 2.
    /// Single source of truth — used by both init() and synthesize().
    public static let tFramesForBucket: [Int: Int] = [
        3: 120, 7: 280, 10: 400, 15: 600, 30: 1200, 45: 1800,
    ]

    /// Duration model enumerated token sizes. Caller pads to nearest.
    public static let durationTokenSizes: [Int] = [32, 64, 128, 256, 512]
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
///
/// ## Model loading
///
/// ``init(modelsDirectory:buckets:linearWeights:linearBias:)`` calls
/// ``MLModel.compileModel`` synchronously. On first run this can take
/// hundreds of milliseconds per model. For app integration, call init
/// on a background thread or use pre-compiled ``.mlmodelc`` bundles.
public class KokoroPipeline {
    private let durationModels: [Int: MLModel] // keyed by token length
    private let availableDurationTokenSizes: [Int]
    private let f0ntrainModels: [Int: MLModel]  // keyed by T_frames
    private let decoderPreModels: [Int: MLModel] // keyed by bucket seconds
    private let generatorModels: [Int: MLModel]  // keyed by bucket seconds

    /// Learned weights from SourceModuleHnNSF.l_linear.
    private let linearWeights: [Float]
    private let linearBias: Float

    /// Available bucket durations in seconds.
    private let availableBuckets: [Int]

    /// Load all models from a directory.
    ///
    /// Expected files:
    /// - ``kokoro_duration_t{T}.mlpackage`` for each token size, or legacy ``kokoro_duration.mlpackage``
    /// - ``kokoro_f0ntrain_t{T}.mlpackage`` for each bucket's T_frames
    /// - ``kokoro_decoder_pre_{N}s.mlpackage`` for each bucket
    /// - ``kokoro_decoder_har_post_{N}s.mlpackage`` for each bucket
    ///
    /// Note: ``MLModel.compileModel`` is called synchronously. For app
    /// integration, call init on a background queue or use pre-compiled
    /// ``.mlmodelc`` bundles to avoid blocking the main thread.
    public init(
        modelsDirectory: URL,
        buckets: [Int] = [3, 7, 10, 15, 30],
        linearWeights: [Float],
        linearBias: Float
    ) throws {
        // Duration models. Prefer enumerated T-specific packages and keep the
        // legacy 128-token package as a compatibility fallback.
        var durModels: [Int: MLModel] = [:]
        for tokenSize in PipelineConstants.durationTokenSizes {
            let specificURL = modelsDirectory.appendingPathComponent("kokoro_duration_t\(tokenSize).mlpackage")
            let legacyURL = modelsDirectory.appendingPathComponent("kokoro_duration.mlpackage")
            let url: URL?
            if FileManager.default.fileExists(atPath: specificURL.path) {
                url = specificURL
            } else if tokenSize == PipelineConstants.durationTokenLength,
                      FileManager.default.fileExists(atPath: legacyURL.path) {
                url = legacyURL
            } else {
                url = nil
            }

            if let url {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                durModels[tokenSize] = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
            }
        }
        guard !durModels.isEmpty else {
            throw PipelineError.modelNotLoaded("duration")
        }
        self.durationModels = durModels
        self.availableDurationTokenSizes = Array(durModels.keys.sorted())

        // F0Ntrain models (one per bucket's T_frames)
        var f0Models: [Int: MLModel] = [:]
        for sec in buckets {
            if let t = PipelineConstants.tFramesForBucket[sec] {
                let url = modelsDirectory.appendingPathComponent("kokoro_f0ntrain_t\(t).mlpackage")
                if FileManager.default.fileExists(atPath: url.path) {
                    let config = MLModelConfiguration()
                    config.computeUnits = .all
                    f0Models[t] = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
                }
            }
        }
        self.f0ntrainModels = f0Models

        // DecoderPre models (Phase 4: CoreML, no longer bridge)
        var decPreModels: [Int: MLModel] = [:]
        for sec in buckets {
            let url = modelsDirectory.appendingPathComponent("kokoro_decoder_pre_\(sec)s.mlpackage")
            if FileManager.default.fileExists(atPath: url.path) {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                decPreModels[sec] = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
            }
        }
        self.decoderPreModels = decPreModels

        // Generator (HAR-post) models
        var genModels: [Int: MLModel] = [:]
        for sec in buckets {
            let url = modelsDirectory.appendingPathComponent("kokoro_decoder_har_post_\(sec)s.mlpackage")
            if FileManager.default.fileExists(atPath: url.path) {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                genModels[sec] = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
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
    ///   - inputIds: Token IDs, optionally padded to one of ``PipelineConstants.durationTokenSizes``.
    ///   - attentionMask: Mask for actual tokens (1) vs padding (0).
    ///   - refS: Voice embedding, shape (256,).
    ///   - speed: Speech rate multiplier.
    /// - Returns: SynthesisResult with audio and timing breakdown.
    public func synthesize(
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float = 1.0
    ) throws -> SynthesisResult {
        var timings = StageTimings()

        // ---- Stage 1: Duration CoreML ----
        let t0 = CFAbsoluteTimeGetCurrent()
        let durationTokenLength = try selectDurationTokenLength(
            inputIds: inputIds,
            attentionMask: attentionMask,
            availableSizes: availableDurationTokenSizes
        )
        guard let durationModel = durationModels[durationTokenLength] else {
            throw PipelineError.modelNotLoaded("duration_t\(durationTokenLength)")
        }
        let durInput = try buildDurationInput(
            inputIds: inputIds,
            attentionMask: attentionMask,
            refS: refS,
            speed: speed,
            tokenLength: durationTokenLength
        )
        let durOutput = try durationModel.prediction(from: durInput)
        let t1 = CFAbsoluteTimeGetCurrent()
        timings.durationCoreML = t1 - t0

        // Extract duration model outputs.
        // Note: Duration model also outputs "ref_s_out" (a passthrough of the input ref_s,
        // renamed to avoid CoreML aliasing — see export_duration.py:111). We use the caller's
        // refS directly instead, since ref_s_out is just ref_s + zeros_like(ref_s).
        let predDurArray = durOutput.featureValue(for: "pred_dur")!.multiArrayValue!
        let dArray = durOutput.featureValue(for: "d")!.multiArrayValue!
        let tEnArray = durOutput.featureValue(for: "t_en")!.multiArrayValue!

        // Parse only non-padding token durations. The static Core ML model
        // returns a full T-sized integer tensor; padded positions must not
        // contribute one frame each.
        let tokenCount = predDurArray.shape.last!.intValue
        let validTokenCount = min(tokenCount, attentionMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) })
        let predDur = try readDurationFrames(from: predDurArray, validCount: validTokenCount)
        let totalFrames = predDur.reduce(0, +)
        let totalSeconds = Double(totalFrames * 2) / PipelineConstants.f0FrameRate

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
        // Duration model outputs d as (1, tokens, 640); matmul3D expects (1, M=640, K=tokens).
        let dTransposed = try transpose3D(
            source: dArray, dim1: PipelineConstants.hiddenDim, dim2: tokenCount
        )
        let en = try matmul3D(
            a: dTransposed,
            b: alignment,
            M: PipelineConstants.hiddenDim,
            K: tokenCount,
            N: totalFrames
        )
        // asr is used by DecoderPre CoreML: the aligned text features.
        let asr = try matmul3D(
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
        guard let tFrames = PipelineConstants.tFramesForBucket[bucketSec],
              let f0nModel = f0ntrainModels[tFrames] else {
            throw PipelineError.modelNotLoaded("f0ntrain_t\(PipelineConstants.tFramesForBucket[bucketSec] ?? 0)")
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

        // Extract F0/N as flat arrays. Core ML outputs can be strided just like
        // waveform outputs, so flatten through indexed access rather than raw
        // pointer traversal.
        let f0Curve = floatValues(from: f0PredArray)
        let nCurve = floatValues(from: nPredArray)

        // ---- Stage 5: Padding to bucket geometry ----
        let t8 = CFAbsoluteTimeGetCurrent()
        let bucketSamples = bucketSec * PipelineConstants.sampleRate
        let fullF0Len = Int(round(Double(bucketSamples) / Double(HarmonicConstants.upsampleScale)))
        // Pad F0 and N to full_f0_len
        let f0Padded = zeroPad1D(source: f0Curve, targetLength: fullF0Len)
        let t9 = CFAbsoluteTimeGetCurrent()
        timings.padding = t9 - t8

        // ---- Stage 6: DecoderPre CoreML ----
        let t10 = CFAbsoluteTimeGetCurrent()
        guard let decPreModel = decoderPreModels[bucketSec] else {
            throw PipelineError.modelNotLoaded("decoder_pre_\(bucketSec)s")
        }

        // Build DecoderPre inputs: asr (padded), f0 (as 3D), n (as 3D), ref_s
        let asrPadded = try zeroPad3D(source: asr, channels: PipelineConstants.textEncoderDim, targetTime: decPreFrameCount(bucketSec: bucketSec))

        // F0 and N as (1, 1, full_f0_len) for DecoderPre Conv1d input
        let f0Array3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
        copyInto(array: f0Array3D, from: f0Padded)
        let nPadded = zeroPad1D(source: nCurve, targetLength: fullF0Len)
        let nArray3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
        copyInto(array: nArray3D, from: nPadded)

        let refSArray = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
        let refSPtr = refSArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<PipelineConstants.voiceEmbeddingDim {
            refSPtr[i] = refS[i]
        }

        let decPreInput = try MLDictionaryFeatureProvider(dictionary: [
            "asr": MLFeatureValue(multiArray: asrPadded),
            "f0": MLFeatureValue(multiArray: f0Array3D),
            "n_input": MLFeatureValue(multiArray: nArray3D),
            "ref_s": MLFeatureValue(multiArray: refSArray),
        ])
        let decPreOutput = try decPreModel.prediction(from: decPreInput)
        let xPre = decPreOutput.featureValue(for: "x_pre")!.multiArrayValue!
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

        // ref_s as MLMultiArray (reuse for GeneratorFromHar)
        let genRefSArray = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
        let genRefSPtr = genRefSArray.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<PipelineConstants.voiceEmbeddingDim {
            genRefSPtr[i] = refS[i]
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
            "ref_s": MLFeatureValue(multiArray: genRefSArray),
            "har": MLFeatureValue(multiArray: harPadded),
        ])
        let genOutput = try genModel.prediction(from: genInput)
        let t15 = CFAbsoluteTimeGetCurrent()
        timings.generatorCoreML = t15 - t14

        // ---- Stage 9: Trim ----
        let t16 = CFAbsoluteTimeGetCurrent()
        let waveformKey = genOutput.featureNames.contains("waveform") ? "waveform" : genOutput.featureNames.first!
        let waveformArray = genOutput.featureValue(for: waveformKey)!.multiArrayValue!
        let waveformValues = floatValues(from: waveformArray)

        // Trim to natural utterance length: T_f0 / 80.0 * 24000
        // T_f0 is the ORIGINAL (unpadded) F0 length = totalFrames * 2 (from F0Ntrain 2x upsample)
        let originalF0Len = totalFrames * 2  // F0Ntrain doubles the frame count
        let targetLen = Int(round(Double(originalF0Len) / PipelineConstants.f0FrameRate * Double(PipelineConstants.sampleRate)))
        let trimLen = min(waveformValues.count, targetLen)

        let audio = Array(waveformValues.prefix(trimLen))
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

    /// Compute the ASR frame count for DecoderPre's expected input shape.
    ///
    /// Matches ``conv1d_output_length_from_module(full_f0_len, dec.F0_conv)``
    /// where F0_conv has kernel=3, stride=2, padding=1.
    /// Formula: ``(full_f0_len + 2*padding - kernel) / stride + 1 = (L + 1) / 2``
    private func decPreFrameCount(bucketSec: Int) -> Int {
        let bucketSamples = bucketSec * PipelineConstants.sampleRate
        let fullF0Len = Int(round(Double(bucketSamples) / Double(HarmonicConstants.upsampleScale)))
        // Conv1d(k=3, s=2, p=1): output = (input + 2*1 - 3) / 2 + 1 = (input - 1) / 2 + 1
        return (fullF0Len - 1) / 2 + 1
    }

    private func buildDurationInput(
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float,
        tokenLength: Int
    ) throws -> MLDictionaryFeatureProvider {
        let T = tokenLength

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

    private func selectDurationTokenLength(
        inputIds: [Int32],
        attentionMask: [Int32],
        availableSizes: [Int]
    ) throws -> Int {
        let maskedTokenCount = attentionMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        let requestedTokenCount = maskedTokenCount > 0 ? maskedTokenCount : inputIds.count

        for size in availableSizes.sorted() where requestedTokenCount <= size {
            return size
        }

        throw PipelineError.inputTooLong(
            tokens: requestedTokenCount,
            maxTokens: availableSizes.max() ?? 0
        )
    }
}

// MARK: - Errors

public enum PipelineError: Error, LocalizedError {
    case noBucketAvailable
    case modelNotLoaded(String)
    case inputTooLong(tokens: Int, maxTokens: Int)

    public var errorDescription: String? {
        switch self {
        case .noBucketAvailable:
            return "No bucket available for the requested duration"
        case .modelNotLoaded(let name):
            return "Model not loaded: \(name)"
        case .inputTooLong(let tokens, let maxTokens):
            return "Input has \(tokens) tokens, but the largest loaded duration model supports \(maxTokens)"
        }
    }
}
