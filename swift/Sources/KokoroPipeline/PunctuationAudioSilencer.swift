import Foundation

private let kokoroSilentPunctuationTokenIds = Set<Int32>([
    1,  // ;
    2,  // :
    3,  // ,
    4,  // .
    5,  // !
    6,  // ?
    9,  // —
    10, // …
    11, // "
    12, // (
    13, // )
    14, // “
    15, // ”
])
private let kokoroWhitespaceTokenId = Int32(16)

/// Suppress Core ML decoder transients on punctuation-owned duration spans.
///
/// Kokoro's duration model assigns real time to punctuation tokens so pauses
/// survive synthesis. The PyTorch reference renders those spans as near-silence,
/// but the split Core ML decoder path can leave short impulses there. This
/// post-process keeps the predicted timing intact while fading punctuation spans
/// and adjacent punctuation-owned whitespace down to silence in the final
/// waveform.
public func suppressPunctuationTokenAudio(
    _ audio: [Float],
    inputIds: [Int32],
    predDur: [Int],
    samplesPerDurationFrame: Int,
    fadeSamples: Int = 120
) -> [Float] {
    guard !audio.isEmpty, samplesPerDurationFrame > 0 else { return audio }
    let tokenCount = min(inputIds.count, predDur.count)
    guard tokenCount > 0 else { return audio }

    var result = audio
    var frameStart = 0
    for tokenIndex in 0..<tokenCount {
        let durationFrames = max(0, predDur[tokenIndex])
        defer { frameStart += durationFrames }

        guard durationFrames > 0,
              shouldSuppressPunctuationSpan(inputIds: inputIds, tokenIndex: tokenIndex) else {
            continue
        }

        let rawStart = frameStart * samplesPerDurationFrame
        let rawEnd = (frameStart + durationFrames) * samplesPerDurationFrame
        fadeToSilence(
            audio: &result,
            rawStart: rawStart,
            rawEnd: rawEnd,
            fadeSamples: fadeSamples
        )
    }
    return result
}

private func shouldSuppressPunctuationSpan(inputIds: [Int32], tokenIndex: Int) -> Bool {
    let tokenId = inputIds[tokenIndex]
    if kokoroSilentPunctuationTokenIds.contains(tokenId) {
        return true
    }
    guard tokenId == kokoroWhitespaceTokenId else {
        return false
    }

    let previousIsPunctuation = tokenIndex > 0
        && kokoroSilentPunctuationTokenIds.contains(inputIds[tokenIndex - 1])
    let nextIsPunctuation = tokenIndex + 1 < inputIds.count
        && kokoroSilentPunctuationTokenIds.contains(inputIds[tokenIndex + 1])
    return previousIsPunctuation || nextIsPunctuation
}

private func fadeToSilence(
    audio: inout [Float],
    rawStart: Int,
    rawEnd: Int,
    fadeSamples: Int
) {
    let clampedRawStart = max(0, min(rawStart, audio.count))
    let clampedRawEnd = max(clampedRawStart, min(rawEnd, audio.count))
    guard clampedRawStart < clampedRawEnd else { return }

    let fade = max(0, fadeSamples)
    let fadeStart = max(0, clampedRawStart - fade)
    let fadeEnd = min(audio.count, clampedRawEnd + fade)

    if fadeStart < clampedRawStart {
        let count = clampedRawStart - fadeStart
        for index in fadeStart..<clampedRawStart {
            let progress = Float(index - fadeStart) / Float(max(1, count))
            audio[index] *= max(0, 1 - progress)
        }
    }

    for index in clampedRawStart..<clampedRawEnd {
        audio[index] = 0
    }

    if clampedRawEnd < fadeEnd {
        let count = fadeEnd - clampedRawEnd
        for index in clampedRawEnd..<fadeEnd {
            let progress = Float(index - clampedRawEnd + 1) / Float(max(1, count))
            audio[index] *= min(1, progress)
        }
    }
}
