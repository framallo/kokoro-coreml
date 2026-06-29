import Foundation
import KokoroPipeline

/// High-level SDK facade for local Kokoro TTS.
///
/// Phase 1 intentionally exposes package identity and boundary constants only.
/// The raw-text synthesis entry point lands after the SDK has checked runtime
/// assets, Swift prep parity, resource discovery, and model provider plumbing.
public actor KokoroTTS {
    /// SDK module name used in diagnostics and probe output.
    public static let sdkName = "KokoroTTS"

    /// The low-level token cap inherited from ``KokoroPipeline``.
    public static let maxCallerChunkTokens = PipelineConstants.maxCallerChunkTokens

    /// Creates an empty facade placeholder.
    ///
    /// Called by early package consumers that want to verify linkage without
    /// depending on the unfinished raw-text synthesis API.
    public init() {}
}
