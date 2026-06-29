import Foundation
import KokoroPipeline

/// In-memory mono PCM audio returned by the public SDK.
public struct KokoroAudio: Equatable, Sendable {
    /// Mono floating-point PCM samples in the range produced by KokoroPipeline.
    public let samples: [Float]

    /// Sample rate in Hz.
    public let sampleRate: Int

    /// Audio duration in seconds.
    public var durationSeconds: Double {
        samples.isEmpty ? 0 : Double(samples.count) / Double(sampleRate)
    }

    /// Creates an audio value.
    ///
    /// - Parameters:
    ///   - samples: Mono floating-point PCM samples.
    ///   - sampleRate: Sample rate in Hz.
    public init(samples: [Float], sampleRate: Int = PipelineConstants.sampleRate) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}
