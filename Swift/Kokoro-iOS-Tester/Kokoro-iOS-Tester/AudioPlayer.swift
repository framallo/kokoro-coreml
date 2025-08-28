import Foundation
import AVFoundation

final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var currentFormat: AVAudioFormat?
    private var isEngineStarted = false
    private let mixer: AVAudioMixerNode

    init() {
        mixer = engine.mainMixerNode
        engine.attach(player)
    }

    private func ensureSession(sampleRate: Double) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default, options: [])
        try session.setPreferredSampleRate(sampleRate)
        try session.setActive(true, options: [])
    }

    private func ensureEngine(sampleRate: Double) throws -> AVAudioFormat {
        // Desired mono playback format
        let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        // Reconnect player → mixer with exact buffer format to satisfy channelCount precondition
        engine.disconnectNodeOutput(player)
        engine.connect(player, to: mixer, format: format)
        if !isEngineStarted {
            engine.prepare()
            try engine.start()
            isEngineStarted = true
        }
        currentFormat = format
        return format
    }

    func play(samples: [Float], sampleRate: Double) {
        do {
            try ensureSession(sampleRate: sampleRate)
            let format = try ensureEngine(sampleRate: sampleRate)

            let frameCount = AVAudioFrameCount(samples.count)
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return }
            buffer.frameLength = frameCount
            let dst = buffer.floatChannelData![0]
            samples.withUnsafeBufferPointer { ptr in
                dst.update(from: ptr.baseAddress!, count: Int(frameCount))
            }

            if !player.isPlaying { player.play() }
            player.scheduleBuffer(buffer, at: nil, options: [])
        } catch {
            print("AudioPlayer error: \(error)")
        }
    }
}
