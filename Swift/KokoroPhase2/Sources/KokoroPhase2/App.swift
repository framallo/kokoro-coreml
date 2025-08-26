import Foundation
import CoreML
import AVFoundation
import Accelerate
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// Keep a global strong reference to the audio player to prevent premature deallocation
private var persistentAudioPlayer: AVAudioPlayer?

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
    // Optional HAR features for Decoder_HAR model
    let har_spec_shape: [Int]?
    let har_phase_shape: [Int]?
    let har_spec: [Float]?
    let har_phase: [Float]?
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
    let totalElementCount = shape.reduce(1, *)
    precondition(data.count == totalElementCount, "data count \(data.count) != total shape \(totalElementCount)")
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    // Copy bytes directly to avoid initialize/assign semantics on potentially initialized memory
    data.withUnsafeBytes { srcBytes in
        let destRaw = array.dataPointer
        memcpy(destRaw, srcBytes.baseAddress!, totalElementCount * MemoryLayout<Float>.size)
    }
    return array
}

/// Pads or truncates a flattened 4D tensor (row-major) along the last dimension.
/// srcShape and dstShape are [B, C, H, T] where T is time.
func padOrTruncate4DLastDim(flat: [Float], srcShape: [Int], dstShape: [Int]) -> [Float] {
    precondition(srcShape.count == 4 && dstShape.count == 4, "Shapes must be 4D")
    let (srcB, srcC, srcH, srcT) = (srcShape[0], srcShape[1], srcShape[2], srcShape[3])
    let (dstB, dstC, dstH, dstT) = (dstShape[0], dstShape[1], dstShape[2], dstShape[3])
    precondition(srcB == dstB && srcC <= dstC && srcH <= dstH && srcB == 1, "Unsupported broadcast")
    var out = [Float](repeating: 0, count: dstB * dstC * dstH * dstT)
    let copyT = min(srcT, dstT)
    // Only B=1 supported in this app
    for c in 0..<min(srcC, dstC) {
        for h in 0..<min(srcH, dstH) {
            let srcOffset = ((0 * srcC + c) * srcH + h) * srcT
            let dstOffset = ((0 * dstC + c) * dstH + h) * dstT
            flat.withUnsafeBufferPointer { srcBuf in
                out.withUnsafeMutableBufferPointer { dstBuf in
                    for t in 0..<copyT {
                        dstBuf[dstOffset + t] = srcBuf[srcOffset + t]
                    }
                }
            }
        }
    }
    return out
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
    persistentAudioPlayer = try AVAudioPlayer(contentsOf: url)
    guard let player = persistentAudioPlayer else { return }
    player.numberOfLoops = 0
    player.prepareToPlay()
    player.play()
    // Drive the runloop until playback finishes, keeping strong reference alive
    while player.isPlaying {
        _ = RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.1))
    }
    player.stop()
    persistentAudioPlayer = nil
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

    // Mel filter bank (bins 0..nFFT/2)
    let melFB = buildMelFilterBank(sampleRate: sampleRate, nFFT: nFFT, nMels: nMels, fmin: fmin, fmax: fmax)

    // Real-to-complex DFT setup
    guard let dft = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(frameLen), vDSP_DFT_Direction.FORWARD) else { return [] }
    defer { vDSP_DFT_DestroySetup(dft) }

    var mel = Array(repeating: [Float](repeating: 0, count: numFrames), count: nMels) // shape: [nMels][T]

    var frame = [Float](repeating: 0, count: frameLen)
    var windowed = [Float](repeating: 0, count: frameLen)
    var realIn = [Float](repeating: 0, count: frameLen)
    var imagIn = [Float](repeating: 0, count: frameLen)
    var realOut = [Float](repeating: 0, count: frameLen/2)
    var imagOut = [Float](repeating: 0, count: frameLen/2)

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

        // Execute real-to-complex DFT
        // Copy to inputs; imagIn is zeroed already
        for i in 0..<frameLen { realIn[i] = windowed[i] }
        vDSP_DFT_Execute(dft, &realIn, &imagIn, &realOut, &imagOut)

        // Build magnitude-squared spectrum 0..nFFT/2
        var mag = [Float](repeating: 0, count: frameLen/2 + 1)
        // bins 0..(N/2-1)
        for k in 0..<(frameLen/2) {
            let r = realOut[k]
            let i = imagOut[k]
            mag[k] = r*r + i*i
        }
        // Nyquist
        mag[frameLen/2] = 0

        // Apply mel filters
        for m in 0..<nMels {
            var sum: Float = 0
            let filt = melFB[m]
            vDSP_dotpr(mag, 1, filt, 1, &sum, vDSP_Length(frameLen/2 + 1))
            mel[m][t] = sum
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

// MARK: - Decoder_HAR inverse iSTFT (parity with Python CustomSTFT)

func hannPeriodic(_ n: Int) -> [Float] {
    // Match PyTorch torch.hann_window(win_length, periodic=True)
    var w = [Float](repeating: 0, count: n)
    if n <= 1 { if n == 1 { w[0] = 1 } ; return w }
    for i in 0..<n { w[i] = 0.5 - 0.5 * cos(2.0 * .pi * Float(i) / Float(n)) }
    return w
}

func reconstructWaveformFromDecoderHAROutput(xChannelsByTime: [[Float]], nFFT: Int = 20, hop: Int = 5, center: Bool = true) -> [Float] {
    // x has shape [C, T], where C = nFFT + 2 (per Kokoro iSTFTNet). Split into spec and phase.
    let freqBins = nFFT / 2 + 1
    let channels = xChannelsByTime.count
    let frames = xChannelsByTime.first?.count ?? 0
    guard channels >= freqBins * 2 else { return [] }
    // Build spec and phase [freqBins][frames]
    var mag = Array(repeating: [Float](repeating: 0, count: frames), count: freqBins)
    var pha = Array(repeating: [Float](repeating: 0, count: frames), count: freqBins)
    for k in 0..<freqBins {
        let specChan = xChannelsByTime[k]
        let phaseChan = xChannelsByTime[freqBins + k]
        for t in 0..<frames {
            mag[k][t] = expf(specChan[t])
            pha[k][t] = sinf(phaseChan[t])
        }
    }
    let window = hannPeriodic(nFFT)
    let padLen = nFFT / 2
    // Precompute cos/sin(k,n)
    var cosTable = Array(repeating: [Float](repeating: 0, count: nFFT), count: freqBins)
    var sinTable = Array(repeating: [Float](repeating: 0, count: nFFT), count: freqBins)
    for k in 0..<freqBins {
        for n in 0..<nFFT {
            let angle = 2.0 * .pi * Float(k * n) / Float(nFFT)
            cosTable[k][n] = cos(angle)
            sinTable[k][n] = sin(angle)
        }
    }
    // Overlap-add buffer (include center pad on both ends)
    let totalLen = frames * hop + (center ? 2 * padLen : 0) + nFFT
    var y = [Float](repeating: 0, count: totalLen)
    let scale: Float = 1.0 / Float(nFFT)
    for t in 0..<frames {
        var frame = [Float](repeating: 0, count: nFFT)
        for k in 0..<freqBins {
            // real = mag * cos(phase); imag = mag * sin(phase)
            let realk = mag[k][t] * cos(pha[k][t])
            let imagk = mag[k][t] * sin(pha[k][t])
            // Accumulate IDFT contribution for each time sample n
            for n in 0..<nFFT {
                // inverse: real*cos - imag*sin
                frame[n] += (realk * cosTable[k][n] - imagk * sinTable[k][n])
            }
        }
        // Apply window and scale
        for n in 0..<nFFT { frame[n] *= window[n] * scale }
        // Write to output with stride hop and optional center pad
        let base = (center ? padLen : 0) + t * hop
        for n in 0..<nFFT {
            let idx = base + n
            if idx < y.count { y[idx] += frame[n] }
        }
    }
    // Remove center padding
    if center && y.count > 2 * padLen {
        return Array(y[padLen..<(y.count - padLen)])
    } else {
        return y
    }
}

@main
struct App {
    static func main() throws {
        // Special mode: compute mel CSV from an existing WAV file using the same HTK mel + linear power
        if let melWavPath = ProcessInfo.processInfo.environment["MEL_WAV_PATH"], !melWavPath.isEmpty {
            let wavURL = URL(fileURLWithPath: melWavPath)
            // Simple 16-bit PCM mono reader (assumes standard 44-byte header)
            let data = try Data(contentsOf: wavURL)
            guard data.count > 44 else { throw NSError(domain: "kokoro.mel", code: -10, userInfo: [NSLocalizedDescriptionKey: "WAV too short"])}
            // Parse sample rate (bytes 24..27 little-endian)
            let sr: Int = data.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: UInt8.self).baseAddress!
                let b0 = Int(p[24]), b1 = Int(p[25]), b2 = Int(p[26]), b3 = Int(p[27])
                return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            }
            // Read PCM int16 samples from offset 44
            let pcm = data.dropFirst(44)
            let samplesI16: [Int16] = pcm.withUnsafeBytes { raw in
                let buf = raw.bindMemory(to: Int16.self)
                return Array(buf)
            }
            let audio = samplesI16.map { Float($0) / 32767.0 }
            let mel = melSpectrogram(audio: audio, sampleRate: sr, nFFT: 1024, hop: 300, nMels: 80, fmin: 0, fmax: 12000)
            let outCSV = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("outputs/golden/golden2.csv")
            try FileManager.default.createDirectory(at: outCSV.deletingLastPathComponent(), withIntermediateDirectories: true)
            try saveMatrixCSV(rows: mel.count, cols: mel.first?.count ?? 0, value: { r, c in mel[r][c] }, url: outCSV)
            print("Saved: \(outCSV.path)")
            return
        }
        // Load JSON inputs prepared by Phase 1
        let inputsURL = locateResource(named: "inputs_vocoder.json")
        let data = try Data(contentsOf: inputsURL)
        let vocoderInputs = try JSONDecoder().decode(VocoderInputs.self, from: data)

        // Prefer exact parity path if HAR features present and explicitly enabled
        let harSpecPresent = (vocoderInputs.har_spec != nil && vocoderInputs.har_phase != nil && vocoderInputs.har_spec_shape != nil && vocoderInputs.har_phase_shape != nil)
        let preferHAR = ProcessInfo.processInfo.environment["USE_DECODER_HAR"] == "1"
        var model: MLModel
        var modelNameUsed = ""
        if harSpecPresent && preferHAR {
            let harURL = locateResource(named: "KokoroDecoder_HAR_5s.mlpackage")
            if FileManager.default.fileExists(atPath: harURL.path) {
                let compiled = try MLModel.compileModel(at: harURL)
                let config = MLModelConfiguration(); config.computeUnits = .all
                model = try MLModel(contentsOf: compiled, configuration: config)
                modelNameUsed = "KokoroDecoder_HAR_5s.mlpackage"
            } else {
                // Fallback to vocoder
                let rawURL = locateResource(named: "KokoroVocoder.mlpackage")
                let compiled = try MLModel.compileModel(at: rawURL)
                let config = MLModelConfiguration(); config.computeUnits = .all
                model = try MLModel(contentsOf: compiled, configuration: config)
                modelNameUsed = "KokoroVocoder.mlpackage"
            }
        } else {
            let rawURL = locateResource(named: "KokoroVocoder.mlpackage")
            let compiled = try MLModel.compileModel(at: rawURL)
            let config = MLModelConfiguration(); config.computeUnits = .all
            model = try MLModel(contentsOf: compiled, configuration: config)
            modelNameUsed = "KokoroVocoder.mlpackage"
        }

        if preferHAR && harSpecPresent && modelNameUsed.contains("Decoder_HAR") {
            // Direct HAR decoding path (exact parity with golden)
            guard let hsShape = vocoderInputs.har_spec_shape, let hpShape = vocoderInputs.har_phase_shape,
                  let hs = vocoderInputs.har_spec, let hp = vocoderInputs.har_phase else {
                throw NSError(domain: "kokoro.phase2", code: -3, userInfo: [NSLocalizedDescriptionKey: "HAR features missing despite presence flag"])
            }
            var hsData = hs
            var hpData = hp
            // Models expect fixed shapes like (1, 11, 1, 24001). Pad/truncate last dim if needed.
            let hsFixed = [hsShape[0], hsShape[1], hsShape[2], max(hsShape[3], 24001)]
            let hpFixed = [hpShape[0], hpShape[1], hpShape[2], max(hpShape[3], 24001)]
            if hsShape != hsFixed { hsData = padOrTruncate4DLastDim(flat: hs, srcShape: hsShape, dstShape: hsFixed) }
            if hpShape != hpFixed { hpData = padOrTruncate4DLastDim(flat: hp, srcShape: hpShape, dstShape: hpFixed) }
            let hsMA = try makeMLMultiArray(shape: hsFixed, data: hsData)
            let hpMA = try makeMLMultiArray(shape: hpFixed, data: hpData)

            // Also supply required conditioning inputs
            let asrMA = try makeMLMultiArray(shape: vocoderInputs.asr_shape, data: vocoderInputs.asr)
            let f0MA  = try makeMLMultiArray(shape: vocoderInputs.f0_shape, data: vocoderInputs.f0)
            let nMA   = try makeMLMultiArray(shape: vocoderInputs.n_shape, data: vocoderInputs.n)
            let sMA   = try makeMLMultiArray(shape: vocoderInputs.s_shape, data: vocoderInputs.s)
            let start = CFAbsoluteTimeGetCurrent()
            let out = try model.prediction(dict: [
                "har_spec": MLFeatureValue(multiArray: hsMA),
                "har_phase": MLFeatureValue(multiArray: hpMA),
                "asr": MLFeatureValue(multiArray: asrMA),
                "f0_curve": MLFeatureValue(multiArray: f0MA),
                "n": MLFeatureValue(multiArray: nMA),
                "s": MLFeatureValue(multiArray: sMA),
            ])
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            // Two possible outputs: some Decoder_HAR variants output latent x (C x T), others output waveform
            let outName = model.modelDescription.outputDescriptionsByName.keys.first!
            var audio: [Float]
            if let arr = out.featureValue(for: outName)?.multiArrayValue, arr.shape.count == 2 {
                // Shape assumed [C, T]
                let C = arr.shape[0].intValue
                let T = arr.shape[1].intValue
                let s0 = arr.strides[0].intValue
                let s1 = arr.strides[1].intValue
                var x = Array(repeating: Array(repeating: Float(0), count: T), count: C)
                for c in 0..<C {
                    for t in 0..<T {
                        let idx = c * s0 + t * s1
                        x[c][t] = arr[idx].floatValue
                    }
                }
                // Hybrid parity: save latent and call Python iSTFT to reconstruct
                let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
                let outDir = cwd.appendingPathComponent("outputs/\(ts)", isDirectory: true)
                try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
                let latentCSV = outDir.appendingPathComponent("\(ts)_latent.csv")
                var lines: [String] = []
                lines.reserveCapacity(T)
                for t in 0..<T {
                    var row = [String](); row.reserveCapacity(C)
                    for c in 0..<C { row.append(String(x[c][t])) }
                    lines.append(row.joined(separator: ","))
                }
                try lines.joined(separator: "\n").write(to: latentCSV, atomically: true, encoding: .utf8)
                let outWavHybrid = outDir.appendingPathComponent("\(ts).wav")
                let py = "/usr/bin/env"
                let script = cwd.appendingPathComponent("tools/reconstruct_from_latent.py").path
                let proc = Process()
                proc.launchPath = py
                proc.arguments = ["python3", script, "--latent", latentCSV.path, "--out_wav", outWavHybrid.path, "--n_fft", "20", "--hop", "5", "--sr", "24000"]
                let pipe = Pipe(); proc.standardOutput = pipe; proc.standardError = pipe
                proc.launch(); proc.waitUntilExit()
                if proc.terminationStatus != 0 {
                    let data = pipe.fileHandleForReading.readDataToEndOfFile()
                    let log = String(data: data, encoding: .utf8) ?? ""
                    throw NSError(domain: "kokoro.phase2", code: -4, userInfo: [NSLocalizedDescriptionKey: "Python iSTFT failed", "log": log])
                }
                // Load WAV back for playback and mel export
                // Simple WAV reader (16-bit PCM mono)
                let wavData = try Data(contentsOf: outWavHybrid)
                // Skip 44-byte header
                let pcm = wavData.dropFirst(44)
                let samples = pcm.withUnsafeBytes { raw -> [Int16] in
                    let ptr = raw.bindMemory(to: Int16.self)
                    return Array(ptr)
                }
                audio = samples.map { Float($0) / 32767.0 }
            } else if let waveformArray = out.featureValue(for: outName)?.multiArrayValue {
                let count = waveformArray.count
                var tmp = [Float](repeating: 0, count: count)
                for i in 0..<count { tmp[i] = waveformArray[i].floatValue }
                audio = tmp
            } else {
                throw NSError(domain: "kokoro.phase2", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing waveform output"])
            }

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

            let mel = melSpectrogram(audio: audio, sampleRate: vocoderInputs.meta.sample_rate, nFFT: 1024, hop: 300, nMels: 80, fmin: 0, fmax: 12000)
            let melCSV = outDir.appendingPathComponent("\(ts).csv")
            try saveMatrixCSV(rows: mel.count, cols: mel.first?.count ?? 0, value: { r, c in mel[r][c] }, url: melCSV)
            let pngURL = outDir.appendingPathComponent("\(ts).png")
            try saveMelPNG(mel: mel, url: pngURL)

            let meta: [String: Any] = [
                "input_text": vocoderInputs.meta.text,
                "model": modelNameUsed,
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
            return
        }

        // Prepare streaming synthesis (windowed vocoder with overlap-add)
        let asrLenTotal = vocoderInputs.asr_shape.last ?? 0
        let f0LenTotal = vocoderInputs.f0_shape.last ?? 0
        let windowAsr = 200
        let windowF0 = 400
        let strideAsr = windowAsr / 4   // 75% overlap
        let strideF0 = windowF0 / 4
        let samplesPerFrame = 600       // 24kHz / 40fps

        func sliceASR(_ flat: [Float], totalT: Int, startT: Int, winT: Int) -> [Float] {
            let channels = 512
            var out = [Float](repeating: 0, count: channels * winT)
            for c in 0..<channels {
                let base = c * totalT
                for t in 0..<winT {
                    let tt = startT + t
                    if tt < totalT { out[c * winT + t] = flat[base + tt] }
                }
            }
            return out
        }
        func slice1D(_ flat: [Float], totalT: Int, startT: Int, winT: Int) -> [Float] {
            var out = [Float](repeating: 0, count: winT)
            let end = min(totalT, startT + winT)
            if startT < end { Array(flat[startT..<end]).withUnsafeBufferPointer { buf in
                for i in 0..<(end - startT) { out[i] = buf[i] }
            } }
            return out
        }
        let s = try makeMLMultiArray(shape: vocoderInputs.s_shape, data: vocoderInputs.s)

        // Determine number of windows
        func numWindows(total: Int, window: Int, stride: Int) -> Int {
            if total <= 0 { return 0 }
            if total <= window { return 1 }
            return Int(ceil(Double(total - window) / Double(stride))) + 1
        }
        let nWin = numWindows(total: asrLenTotal, window: windowAsr, stride: strideAsr)

        // Prepare output buffers with weight normalization
        let chunkSamples = windowAsr * samplesPerFrame
        let strideSamples = strideAsr * samplesPerFrame
        let totalSamples = max(chunkSamples, (nWin - 1) * strideSamples + chunkSamples)
        var accAudio = [Float](repeating: 0, count: totalSamples)
        var accWeight = [Float](repeating: 0, count: totalSamples)
        let windowWeight = hannWindow(chunkSamples)

        let start = CFAbsoluteTimeGetCurrent()
        for w in 0..<nWin {
            let asrStart = w * strideAsr
            let f0Start = w * strideF0
            let asrSlice = sliceASR(vocoderInputs.asr, totalT: asrLenTotal, startT: asrStart, winT: windowAsr)
            let f0Slice = slice1D(vocoderInputs.f0, totalT: f0LenTotal, startT: f0Start, winT: windowF0)
            let nSlice  = slice1D(vocoderInputs.n,  totalT: f0LenTotal, startT: f0Start, winT: windowF0)

            let asrMA = try makeMLMultiArray(shape: [1,512,1,windowAsr], data: asrSlice)
            let f0MA  = try makeMLMultiArray(shape: [1,1,1,windowF0], data: f0Slice)
            let nMA   = try makeMLMultiArray(shape: [1,1,1,windowF0], data: nSlice)
            let out = try model.prediction(dict: [
                "asr": MLFeatureValue(multiArray: asrMA),
                "f0_curve": MLFeatureValue(multiArray: f0MA),
                "n": MLFeatureValue(multiArray: nMA),
                "s": MLFeatureValue(multiArray: s),
            ])
            guard let outName = model.modelDescription.outputDescriptionsByName.keys.first,
                  let chunkArray = out.featureValue(for: outName)?.multiArrayValue else {
                throw NSError(domain: "kokoro.phase2", code: -2, userInfo: [NSLocalizedDescriptionKey: "Missing waveform output (chunk)"])
            }
            let outCount = chunkArray.count
            var tmp = [Float](repeating: 0, count: outCount)
            for i in 0..<outCount { tmp[i] = chunkArray[i].floatValue }

            let startSample = w * strideSamples
            // Accumulate with Hann window; normalize by weights afterwards
            for i in 0..<outCount {
                let idx = startSample + i
                if idx < totalSamples {
                    let wv = i < windowWeight.count ? windowWeight[i] : 1.0
                    accAudio[idx] += tmp[i] * wv
                    accWeight[idx] += wv
                }
            }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Normalize overlap weights
        var audio = accAudio
        for i in 0..<audio.count {
            let w = accWeight[i]
            if w > 0 { audio[i] /= w }
        }

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
