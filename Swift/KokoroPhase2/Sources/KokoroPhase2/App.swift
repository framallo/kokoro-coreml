import Foundation
import CoreML
import AVFoundation

struct VocoderInputs: Decodable {
    struct Meta: Decodable { let text: String; let voice: String; let sample_rate: Int }
    let meta: Meta
    let asr_shape: [Int]
    let f0_shape: [Int]
    let n_shape: [Int]
    let s_shape: [Int]
    let asr: [Float]
    let f0: [Float]
    let n: [Float]
    let s: [Float]
}

func locateResource(named name: String) -> URL {
    // 1) SwiftPM bundled resources path (when resources are embedded)
    let execURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let bundleURL = execURL.appendingPathComponent("KokoroPhase2_KokoroPhase2.resources").appendingPathComponent(name)
    if FileManager.default.fileExists(atPath: bundleURL.path) { return bundleURL }
    // 2) Project-relative path when running from repo root
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let projectURL = cwd.appendingPathComponent("Swift/KokoroPhase2/Resources/")
    let fileURL = projectURL.appendingPathComponent(name)
    return fileURL
}

func makeMLMultiArray(shape: [Int], data: [Float]) throws -> MLMultiArray {
    let total = shape.reduce(1, *)
    precondition(data.count == total, "data count \(data.count) != total shape \(total)")
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
    ptr.initialize(from: data, count: total)
    return array
}

func saveWAV(_ samples: [Float], sampleRate: Double, url: URL) throws {
    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    let frameCount = AVAudioFrameCount(samples.count)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
    buffer.frameLength = frameCount
    samples.withUnsafeBufferPointer { src in
        buffer.floatChannelData!.pointee.update(from: src.baseAddress!, count: Int(frameCount))
    }
    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
}

func playWAV(from url: URL) throws {
    let player = try AVAudioPlayer(contentsOf: url)
    player.prepareToPlay()
    player.play()
    RunLoop.current.run(until: Date().addingTimeInterval(player.duration))
}

@main
struct App {
    static func main() throws {
        // Load JSON inputs prepared by Phase 1
        let inputsURL = locateResource(named: "inputs_vocoder.json")
        let data = try Data(contentsOf: inputsURL)
        let vocoderInputs = try JSONDecoder().decode(VocoderInputs.self, from: data)

        // Load CoreML model from resources (use vocoder windowed model for Phase 2 MVP)
        let rawURL = locateResource(named: "KokoroVocoder.mlpackage")
        // Compile .mlpackage to .mlmodelc before loading
        let modelURL = try MLModel.compileModel(at: rawURL)
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model = try MLModel(contentsOf: modelURL, configuration: config)

        // Prepare inputs
        // Pad inputs to match vocoder window shapes (asr_len=200, f0_len=400)
        func padTail(_ data: [Float], from: Int, to: Int) -> [Float] {
            if from >= to { return Array(data.prefix(to)) }
            var out = data
            out.append(contentsOf: [Float](repeating: 0, count: to - from))
            return out
        }
        let asrTarget = [1,512,1,200]
        let f0Target = [1,1,1,400]
        let nTarget = [1,1,1,400]
        let asr = try makeMLMultiArray(shape: asrTarget, data: padTail(vocoderInputs.asr, from: vocoderInputs.asr.count, to: asrTarget.reduce(1,*)))
        let f0  = try makeMLMultiArray(shape: f0Target, data: padTail(vocoderInputs.f0, from: vocoderInputs.f0.count, to: f0Target.reduce(1,*)))
        let n   = try makeMLMultiArray(shape: nTarget, data: padTail(vocoderInputs.n, from: vocoderInputs.n.count, to: nTarget.reduce(1,*)))
        let s   = try makeMLMultiArray(shape: vocoderInputs.s_shape, data: vocoderInputs.s)

        let start = CFAbsoluteTimeGetCurrent()
        let out = try model.prediction(dict: [
            "asr": MLFeatureValue(multiArray: asr),
            "f0_curve": MLFeatureValue(multiArray: f0),
            "n": MLFeatureValue(multiArray: n),
            "s": MLFeatureValue(multiArray: s),
        ])
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Extract audio
        guard let outName = model.modelDescription.outputDescriptionsByName.keys.first,
              let waveformArray = out.featureValue(for: outName)?.multiArrayValue else {
            throw NSError(domain: "kokoro.phase2", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing waveform output"])
        }
        let count = waveformArray.count
        var audio = [Float](repeating: 0, count: count)
        for i in 0..<count { audio[i] = waveformArray[i].floatValue }

        // Save WAV
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let outDir = cwd.appendingPathComponent("outputs/phase2", isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let wavURL = outDir.appendingPathComponent("phase2_\(ts).wav")
        try saveWAV(audio, sampleRate: Double(vocoderInputs.meta.sample_rate), url: wavURL)

        print("Saved: \(wavURL.path)")
        print(String(format: "CoreML elapsed: %.3f s", elapsed))
        try playWAV(from: wavURL)
    }
}
