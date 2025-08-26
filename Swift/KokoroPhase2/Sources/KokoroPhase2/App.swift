import Foundation
import CoreML
import AVFoundation
import Accelerate
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

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

// MARK: - Mel Spectrogram Generation (n_mels=80, n_fft=1024, hop=300, fmin=0, fmax=12000)

func hzToMel(_ hz: Float) -> Float { return 2595.0 * log10(1.0 + hz / 700.0) }
func melToHz(_ mel: Float) -> Float { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

func buildMelFilterBank(sampleRate: Int, nFFT: Int, nMels: Int, fmin: Float, fmax: Float) -> [[Float]] {
    let nyquist = Float(sampleRate) / 2.0
    let fmaxClamped = min(fmax, nyquist)
    let melMin = hzToMel(fmin)
    let melMax = hzToMel(fmaxClamped)
    let melPoints = (0..<(nMels + 2)).map { i in
        return melMin + (melMax - melMin) * Float(i) / Float(nMels + 1)
    }
    let hzPoints = melPoints.map { melToHz($0) }
    let bin = hzPoints.map { Int(round(($0 / Float(sampleRate)) * Float(nFFT))) }
    var fb = Array(repeating: Array(repeating: Float(0), count: nFFT/2 + 1), count: nMels)
    if nFFT/2 >= 1 {
        for m in 1...nMels {
            let f_m_minus = bin[m-1]
            let f_m = bin[m]
            let f_m_plus = bin[m+1]
            if f_m_minus < f_m {
                for k in max(0, f_m_minus)..<min(f_m, nFFT/2 + 1) {
                    fb[m-1][k] = Float(k - f_m_minus) / Float(max(1, f_m - f_m_minus))
                }
            }
            if f_m < f_m_plus {
                for k in max(f_m, 0)..<min(f_m_plus, nFFT/2 + 1) {
                    fb[m-1][k] = Float(f_m_plus - k) / Float(max(1, f_m_plus - f_m))
                }
            }
        }
    }
    return fb
}

func hannWindow(_ n: Int) -> [Float] {
    var window = [Float](repeating: 0, count: n)
    vDSP_hann_window(&window, vDSP_Length(n), Int32(vDSP_HANN_NORM))
    return window
}

func melSpectrogram(audio: [Float], sampleRate: Int, nFFT: Int = 1024, hop: Int = 300, nMels: Int = 80, fmin: Float = 0, fmax: Float = 12000) -> [[Float]] {
    let frameLen = nFFT
    let hopLen = hop
    if audio.isEmpty { return [] }
    let numFrames = max(1, (audio.count - frameLen) / hopLen + 1)
    let window = hannWindow(frameLen)
    // Precompute FFT setup
    let log2n = vDSP_Length(log2(Float(nFFT)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [] }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    // Mel filter bank
    let melFB = buildMelFilterBank(sampleRate: sampleRate, nFFT: nFFT, nMels: nMels, fmin: fmin, fmax: fmax)

    var mel = Array(repeating: [Float](repeating: 0, count: numFrames), count: nMels) // shape: [nMels][T]

    var frame = [Float](repeating: 0, count: frameLen)
    var windowed = [Float](repeating: 0, count: frameLen)
    var realp = [Float](repeating: 0, count: frameLen/2)
    var imagp = [Float](repeating: 0, count: frameLen/2)

    for t in 0..<numFrames {
        let start = t * hopLen
        // Zero-pad if needed
        let end = min(start + frameLen, audio.count)
        if end - start < frameLen {
            for i in 0..<frameLen { frame[i] = 0 }
        }
        if start < audio.count {
            let count = end - start
            for i in 0..<count { frame[i] = audio[start + i] }
            if count < frameLen { for i in count..<frameLen { frame[i] = 0 } }
        }
        vDSP_vmul(frame, 1, window, 1, &windowed, 1, vDSP_Length(frameLen))
        // Convert to split complex
        realp.withUnsafeMutableBufferPointer { rbuf in
            imagp.withUnsafeMutableBufferPointer { ibuf in
                var split = DSPSplitComplex(realp: rbuf.baseAddress!, imagp: ibuf.baseAddress!)
                windowed.withUnsafeBufferPointer { ptr in
                    ptr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: frameLen/2) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(frameLen/2))
                    }
                }
                vDSP_fft_zip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                var mag = [Float](repeating: 0, count: frameLen/2 + 1)
                // Compute magnitude squared for bins 0..nFFT/2
                mag[0] = split.realp[0] * split.realp[0] + split.imagp[0] * split.imagp[0]
                for k in 1..<(frameLen/2) {
                    let r = split.realp[k]
                    let i = split.imagp[k]
                    mag[k] = r*r + i*i
                }
                mag[frameLen/2] = 0 // Nyquist (imag part stored elsewhere); safe to zero
                // Apply mel filters
                for m in 0..<nMels {
                    var sum: Float = 0
                    let filt = melFB[m]
                    vDSP_dotpr(mag, 1, filt, 1, &sum, vDSP_Length(frameLen/2 + 1))
                    mel[m][t] = sum
                }
            }
        }
    }
    return mel
}

func saveMelCSV(mel: [[Float]], url: URL) throws {
    // mel: [nMels][T] -> write rows as frames (T rows, 80 columns)
    let nMels = mel.count
    let T = mel.first?.count ?? 0
    var lines: [String] = []
    for t in 0..<T {
        var row = [String]()
        row.reserveCapacity(nMels)
        for m in 0..<nMels { row.append(String(mel[m][t])) }
        lines.append(row.joined(separator: ","))
    }
    try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
}

func saveMelPNG(mel: [[Float]], url: URL) throws {
    let nMels = mel.count
    let T = mel.first?.count ?? 0
    if nMels == 0 || T == 0 { return }
    // Normalize to 0..1
    var minVal: Float = .greatestFiniteMagnitude
    var maxVal: Float = -.greatestFiniteMagnitude
    for m in 0..<nMels { for t in 0..<T { let v = mel[m][t]; if v < minVal { minVal = v }; if v > maxVal { maxVal = v } } }
    let range = max(maxVal - minVal, 1e-6)
    var pixels = [UInt8](repeating: 0, count: T * nMels)
    // Map mel[m][t] to grayscale (invert for typical spectrogram look)
    for t in 0..<T {
        for m in 0..<nMels {
            let norm = (mel[m][t] - minVal) / range
            let val = UInt8(max(0, min(255, Int((1.0 - norm) * 255.0))))
            pixels[(nMels - 1 - m) * T + t] = val // flip mel axis so low at bottom
        }
    }
    let colorSpace = CGColorSpaceCreateDeviceGray()
    let bytesPerRow = T
    guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return }
    guard let cgImage = CGImage(width: T, height: nMels, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return }
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, cgImage, nil)
    CGImageDestinationFinalize(dest)
}

func saveMatrixCSV(rows: Int, cols: Int, value: (_ r: Int, _ c: Int) -> Float, url: URL) throws {
    var lines: [String] = []
    lines.reserveCapacity(rows)
    for r in 0..<rows {
        var rowVals = [String]()
        rowVals.reserveCapacity(cols)
        for c in 0..<cols { rowVals.append(String(value(r, c))) }
        lines.append(rowVals.joined(separator: ","))
    }
    try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
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

        // Save artifacts to outputs/[timestamp]/[timestamp].{wav,csv,png,json}
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let outDir = cwd.appendingPathComponent("outputs/\(ts)", isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let wavURL = outDir.appendingPathComponent("\(ts).wav")
        try saveWAV(audio, sampleRate: Double(vocoderInputs.meta.sample_rate), url: wavURL)

        print("Saved: \(wavURL.path)")
        print(String(format: "CoreML elapsed: %.3f s", elapsed))
        try playWAV(from: wavURL)

        // Save mel spectrogram (CSV + PNG)
        let mel = melSpectrogram(audio: audio, sampleRate: vocoderInputs.meta.sample_rate, nFFT: 1024, hop: 300, nMels: 80, fmin: 0, fmax: 12000)
        let melCSV = outDir.appendingPathComponent("\(ts).csv")
        try saveMatrixCSV(rows: mel.count, cols: mel.first?.count ?? 0, value: { r, c in mel[r][c] }, url: melCSV)
        let pngURL = outDir.appendingPathComponent("\(ts).png")
        try saveMelPNG(mel: mel, url: pngURL)

        // Save metadata.json
        let meta: [String: Any] = [
            "input_text": vocoderInputs.meta.text,
            "model": "KokoroVocoder.mlpackage",
            "sample_rate": vocoderInputs.meta.sample_rate,
            "mel_params": [
                "n_mels": 80,
                "hop_length": 300,
                "n_fft": 1024,
                "fmin": 0,
                "fmax": 12000
            ],
            "latency_seconds": [
                "coreml_inference": elapsed
            ]
        ]
        var metaAug = meta
        metaAug["artifacts"] = [
            "wav": wavURL.lastPathComponent,
            "csv": melCSV.lastPathComponent,
            "png": pngURL.lastPathComponent
        ]
        let metaData = try JSONSerialization.data(withJSONObject: metaAug, options: [.prettyPrinted])
        let metaURL = outDir.appendingPathComponent("\(ts).json")
        try metaData.write(to: metaURL)
    }
}
