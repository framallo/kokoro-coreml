import Foundation

/// Stable Kokoro voice identifier used by the public SDK.
public struct KokoroVoiceID: RawRepresentable, Equatable, Hashable, Sendable, ExpressibleByStringLiteral {
    /// Raw voice identifier, for example `af_heart`.
    public let rawValue: String

    /// Default friendly American female voice used by Kokoro examples.
    public static let afHeart = KokoroVoiceID("af_heart")

    /// Gist iOS default voice.
    public static let afBella = KokoroVoiceID("af_bella")

    /// Gist iOS male voice used for counter-argument layers.
    public static let amMichael = KokoroVoiceID("am_michael")

    /// Voices bundled by the starter Gist-compatible profile.
    public static let starterVoices: [KokoroVoiceID] = [.afBella, .amMichael, .afHeart]

    /// Creates a voice identifier from its raw string.
    ///
    /// - Parameter rawValue: Kokoro voice ID such as `af_heart`.
    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// Creates a voice identifier from a string literal.
    ///
    /// - Parameter value: Kokoro voice ID such as `af_heart`.
    public init(stringLiteral value: StringLiteralType) {
        self.rawValue = value
    }

    /// Creates a voice identifier from its raw string.
    ///
    /// - Parameter value: Kokoro voice ID such as `af_heart`.
    public init(_ value: String) {
        self.rawValue = value
    }

    /// Misaki language selector used by the existing JS bridge.
    public var languageCode: String {
        rawValue.hasPrefix("b") ? "b" : "a"
    }

    /// Whether this voice asks for the British English Misaki path.
    public var usesBritishEnglish: Bool {
        languageCode == "b"
    }
}
