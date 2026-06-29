import Foundation
import MisakiSwift

/// MisakiSwift-backed English phonemizer used by the raw-text SDK.
///
/// This matches the Gist iOS app pattern: use the mattmireles MisakiSwift fork
/// for on-device English G2P and keep it out of the lower-floor
/// ``KokoroPipeline`` package.
public final class KokoroMisakiPhonemizer: KokoroPhonemizer {
    /// Underlying Misaki English grapheme-to-phoneme engine.
    private let g2p: EnglishG2P

    /// Whether this phonemizer uses the British English Misaki path.
    public let british: Bool

    /// Creates a MisakiSwift-backed phonemizer.
    ///
    /// - Parameter british: When true, asks MisakiSwift for British English
    ///   phonemes. The default is the U.S. English path used by `af_*` voices.
    public init(british: Bool = false) {
        self.british = british
        self.g2p = EnglishG2P(british: british)
    }

    /// Converts raw text to Kokoro-compatible phonemes with MisakiSwift.
    ///
    /// - Parameter text: Raw English text.
    /// - Returns: Non-empty phonemes and their UTF-16 length.
    public func phonemize(_ text: String) throws -> KokoroPhonemeResult {
        let phonemes = g2p.phonemize(text: text).0
        guard !phonemes.isEmpty else {
            throw KokoroPhonemizerError.emptyOutput
        }
        return KokoroPhonemeResult(phonemes: phonemes)
    }
}
