import Foundation
import CoreML
import AVFoundation
import Accelerate
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import KokoroTTS

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
    // Optional stats for parity
    struct StatsTensor: Decodable {
        let shape: [Int]
        let mean: Float
        let std: Float
        let min: Float
        let max: Float
        struct PerChannel: Decodable { let mean: [Float]?; let std: [Float]? }
        let per_channel: PerChannel?
    }
    struct Stats: Decodable { let asr: StatsTensor?; let f0: StatsTensor?; let n: StatsTensor?; let s: StatsTensor? }
    let stats: Stats?
}

func locateResource(named name: String) -> URL {
    // 1) SwiftPM bundled resources path (when resources are embedded)
    let execURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
    let bundleURL = execURL.appendingPathComponent("KokoroPhase2_KokoroPhase2.resources").appendingPathComponent(name)
    if FileManager.default.fileExists(atPath: bundleURL.path) { return bundleURL }
    // 2) Common locations when running from repo root or from Swift/KokoroPhase2/
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let candidates: [URL] = [
        cwd.appendingPathComponent("Resources").appendingPathComponent(name),
        cwd.appendingPathComponent("Swift/KokoroPhase2/Resources").appendingPathComponent(name),
        cwd.appendingPathComponent(name)
    ]
    if let found = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
        return found
    }
    // Fallback to expected repo path
    return cwd.appendingPathComponent("Swift/KokoroPhase2/Resources").appendingPathComponent(name)
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
            // Convert to log scale to match Python tooling expectations (prevents 'chipmunk' audio)
            mel[m][t] = logf(max(sum, 1e-6))
        }
    }
    return mel
}
/// Estimates effective non-zero frame length of ASR features by scanning from the end.
/// ASR is expected to be flattened [channels * totalT] in channel-major order.
func effectiveNonzeroFrames(asrFlat: [Float], channels: Int, totalT: Int, threshold: Float = 1e-7) -> Int {
    guard channels > 0, totalT > 0 else { return 0 }
    // Scan backwards for any sample exceeding threshold
    var t = totalT - 1
    while t >= 0 {
        var found = false
        let baseT = t
        // Iterate channels; break early when found
        for c in 0..<channels {
            let v = asrFlat[c * totalT + baseT]
            if v > threshold || v < -threshold { found = true; break }
        }
        if found { return t + 1 }
        t -= 1
    }
    return 0
}

// Trim trailing silence/noise by RMS threshold with fadeout
func trimTailByRMS(_ audio: [Float], sampleRate: Int, windowMs: Float = 10.0, threshold: Float = 0.003, fadeMs: Float = 20.0) -> [Float] {
    guard !audio.isEmpty else { return audio }
    let win = max(1, Int((windowMs / 1000.0) * Float(sampleRate)))
    let fade = max(1, Int((fadeMs / 1000.0) * Float(sampleRate)))
    let count = audio.count
    var lastIdx: Int = count - 1
    // Scan windows from the end towards start
    var start = count - win
    while start >= 0 {
        var se: Double = 0
        for i in 0..<win {
            let v = Double(audio[start + i])
            se += v * v
        }
        let rms = sqrt(se / Double(win))
        if Float(rms) > threshold { lastIdx = min(count - 1, start + win - 1); break }
        start -= win
    }
    // Keep up to lastIdx + fade
    let cut = min(count, lastIdx + fade)
    var out = Array(audio[0..<cut])
    // Apply fade on the tail
    let fadeLen = min(fade, out.count)
    if fadeLen > 0 {
        for i in 0..<fadeLen {
            let scale = Float(fadeLen - i) / Float(fadeLen)
            out[out.count - 1 - i] *= scale
        }
    }
    return out
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

// MARK: - Mel CSV Load & Compare vs Golden

/// Loads a mel CSV where each row is a frame and each column is a mel bin.
/// Returns matrix shaped as [nMels][T].
func loadMelCSVTranspose(url: URL) throws -> [[Float]] {
    let text = try String(contentsOf: url)
    let newlineSet = CharacterSet.newlines
    // Split on newlines robustly
    let lines = text.components(separatedBy: newlineSet).filter { !$0.isEmpty }
    if lines.isEmpty { return [] }
    // Parse rows -> [[Float]] frame-major
    var frames: [[Float]] = []
    frames.reserveCapacity(lines.count)
    for line in lines {
        let parts = line.split(separator: ",", omittingEmptySubsequences: false)
        var row: [Float] = []
        row.reserveCapacity(parts.count)
        for p in parts {
            if let v = Float(p.trimmingCharacters(in: CharacterSet.whitespaces)) {
                row.append(v)
            } else {
                row.append(0)
            }
        }
        frames.append(row)
    }
    // Transpose to [nMels][T]
    let T = frames.count
    let nMels = frames.first?.count ?? 0
    var mel = Array(repeating: Array(repeating: Float(0), count: T), count: nMels)
    for t in 0..<T {
        let row = frames[t]
        for m in 0..<min(nMels, row.count) {
            mel[m][t] = row[m]
        }
    }
    return mel
}

func computeMSE(_ a: [[Float]], _ b: [[Float]], shift: Int = 0) -> (mse: Float, usedFrames: Int) {
    let nMels = min(a.count, b.count)
    guard nMels > 0 else { return (0, 0) }
    let Ta = a.first?.count ?? 0
    let Tb = b.first?.count ?? 0
    var startA = 0, startB = 0
    if shift >= 0 { startA = 0; startB = shift } else { startA = -shift; startB = 0 }
    let T = max(0, min(Ta - startA, Tb - startB))
    if T == 0 { return (Float.greatestFiniteMagnitude, 0) }
    var se: Double = 0
    var count: Int = 0
    for m in 0..<nMels {
        for t in 0..<T {
            let da = a[m][startA + t]
            let db = b[m][startB + t]
            let d = Double(da - db)
            se += d * d
            count += 1
        }
    }
    return (Float(se / Double(count)), T)
}

func compareAgainstGolden(currentMel: [[Float]], cwd: URL) {
    let goldenCSVEnv = ProcessInfo.processInfo.environment["GOLDEN_CSV_PATH"]
    let goldenCSVURL = goldenCSVEnv.flatMap { URL(fileURLWithPath: $0) } ?? cwd.appendingPathComponent("outputs/golden/golden.csv")
    let goldenWAVEnv = ProcessInfo.processInfo.environment["GOLDEN_WAV_PATH"]
    let goldenWAVURL = goldenWAVEnv.flatMap { URL(fileURLWithPath: $0) } ?? cwd.appendingPathComponent("outputs/golden/golden.wav")
    // 1) Compare with CSV if available (legacy)
    if FileManager.default.fileExists(atPath: goldenCSVURL.path) {
        do {
            let goldenMel = try loadMelCSVTranspose(url: goldenCSVURL)
            let raw = computeMSE(currentMel, goldenMel, shift: 0)
            var best = raw; var bestShift = 0
            for s in -6...6 { let r = computeMSE(currentMel, goldenMel, shift: s); if r.mse < best.mse { best = r; bestShift = s } }
            print(String(format: "CSV Mel MSE raw=%.6f (T=%d), best=%.6f @shift=%d (T=%d)", raw.mse, raw.usedFrames, best.mse, bestShift, best.usedFrames))
            // Banded analysis to localize error
            reportBandMSE(currentMel, goldenMel, label: "CSV")
        } catch {
            print("⚠️  Failed CSV compare: \(error)")
        }
    } else {
        print("⚠️  Golden CSV not found at: \(goldenCSVURL.path)")
    }
    // 2) Compare with WAV mel computed using the same Swift mel
    if FileManager.default.fileExists(atPath: goldenWAVURL.path) {
        do {
            let (goldenAudio, sr) = try loadWAVMono16(url: goldenWAVURL)
            let goldenMelSwift = melSpectrogram(audio: goldenAudio, sampleRate: sr, nFFT: 1024, hop: 300, nMels: 80, fmin: 0, fmax: 12000)
            let raw = computeMSE(currentMel, goldenMelSwift, shift: 0)
            var best = raw; var bestShift = 0
            for s in -6...6 { let r = computeMSE(currentMel, goldenMelSwift, shift: s); if r.mse < best.mse { best = r; bestShift = s } }
            print(String(format: "WAV Mel MSE raw=%.6f (T=%d), best=%.6f @shift=%d (T=%d)", raw.mse, raw.usedFrames, best.mse, bestShift, best.usedFrames))
            reportBandMSE(currentMel, goldenMelSwift, label: "WAV")
        } catch {
            print("⚠️  Failed WAV compare: \(error)")
        }
    } else {
        print("⚠️  Golden WAV not found at: \(goldenWAVURL.path)")
    }
}

/// Loads 16-bit PCM mono WAV, returns normalized float samples and sample rate.
func loadWAVMono16(url: URL) throws -> ([Float], Int) {
    let data = try Data(contentsOf: url)
    guard data.count > 44 else { throw NSError(domain: "kokoro.wav", code: -10, userInfo: [NSLocalizedDescriptionKey: "WAV too short"]) }
    let sr: Int = data.withUnsafeBytes { raw in
        let p = raw.bindMemory(to: UInt8.self).baseAddress!
        let b0 = Int(p[24]), b1 = Int(p[25]), b2 = Int(p[26]), b3 = Int(p[27])
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }
    let pcm = data.dropFirst(44)
    let samplesI16: [Int16] = pcm.withUnsafeBytes { raw in
        let buf = raw.bindMemory(to: Int16.self)
        return Array(buf)
    }
    let audio = samplesI16.map { Float($0) / 32767.0 }
    return (audio, sr)
}

// MARK: - Audio-level comparison

func computeAudioMSE(_ a: [Float], _ b: [Float], maxShiftSamples: Int = 2400) -> (mse: Float, shift: Int, used: Int) {
    if a.isEmpty || b.isEmpty { return (Float.greatestFiniteMagnitude, 0, 0) }
    func mseAtShift(_ s: Int) -> (Float, Int) {
        let startA = max(0, -s)
        let startB = max(0, s)
        let count = min(a.count - startA, b.count - startB)
        if count <= 0 { return (Float.greatestFiniteMagnitude, 0) }
        var se: Double = 0
        for i in 0..<count { let d = Double(a[startA + i] - b[startB + i]); se += d*d }
        return (Float(se / Double(count)), count)
    }
    var best = mseAtShift(0)
    var bestShift = 0
    let step = max(1, maxShiftSamples / 30)
    var s = -maxShiftSamples
    while s <= maxShiftSamples {
        let (m, used) = mseAtShift(s)
        if m < best.0 { best = (m, used); bestShift = s }
        s += step
    }
    return (best.0, bestShift, best.1)
}

/// Shifts audio in time by `shift` samples while preserving length.
/// Positive shift pads zeros at start and trims the tail; negative drops from start and pads zeros at end.
func shiftAudio(_ audio: [Float], shift: Int) -> [Float] {
    guard !audio.isEmpty, shift != 0 else { return audio }
    let count = audio.count
    if shift > 0 {
        let pad = min(shift, count)
        var out = [Float](repeating: 0, count: pad)
        if count > pad { out += audio[0..<(count - pad)] }
        return out
    } else {
        let drop = min(-shift, count)
        var out = Array(audio[drop..<count])
        out += [Float](repeating: 0, count: drop)
        return out
    }
}

private func reportBandMSE(_ a: [[Float]], _ b: [[Float]], label: String) {
    let nMels = min(a.count, b.count)
    guard nMels >= 80 else { return }
    // Low: 0..19, Mid: 20..49, High: 50..79
    func mseRange(_ r: Range<Int>) -> Float {
        var se: Double = 0
        var count = 0
        let T = min(a.first?.count ?? 0, b.first?.count ?? 0)
        for m in r { for t in 0..<T { let d = Double(a[m][t] - b[m][t]); se += d*d; count += 1 } }
        return count > 0 ? Float(se / Double(count)) : 0
    }
    let low = mseRange(0..<20), mid = mseRange(20..<50), high = mseRange(50..<80)
    print(String(format: "%@ band MSE: low=%.4f mid=%.4f high=%.4f", label, low, mid, high))
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

func saveVectorCSV(_ v: [Float], url: URL) throws {
    let s = v.map { String($0) }.joined(separator: ",")
    try s.write(to: url, atomically: true, encoding: .utf8)
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
            // Direct HAR decoding path using KokoroTTS library with dynamic bucket selection
            guard let hsShape = vocoderInputs.har_spec_shape, let hpShape = vocoderInputs.har_phase_shape,
                  let hs = vocoderInputs.har_spec, let hp = vocoderInputs.har_phase else {
                throw NSError(domain: "kokoro.phase2", code: -3, userInfo: [NSLocalizedDescriptionKey: "HAR features missing despite presence flag"])
            }
            var hsData = hs
            var hpData = hp
            // Pad/truncate last dimension to a safe minimum window; bucket logic in library chooses final model
            let hsFixed = [hsShape[0], hsShape[1], hsShape[2], max(hsShape[3], 24001)]
            let hpFixed = [hpShape[0], hpShape[1], hpShape[2], max(hpShape[3], 24001)]
            if hsShape != hsFixed { hsData = padOrTruncate4DLastDim(flat: hs, srcShape: hsShape, dstShape: hsFixed) }
            if hpShape != hpFixed { hpData = padOrTruncate4DLastDim(flat: hp, srcShape: hpShape, dstShape: hpFixed) }

            let hsMA = try makeMLMultiArray(shape: hsFixed, data: hsData)
            let hpMA = try makeMLMultiArray(shape: hpFixed, data: hpData)
            let asrMA = try makeMLMultiArray(shape: vocoderInputs.asr_shape, data: vocoderInputs.asr)
            let f0MA  = try makeMLMultiArray(shape: vocoderInputs.f0_shape, data: vocoderInputs.f0)
            let nMA   = try makeMLMultiArray(shape: vocoderInputs.n_shape, data: vocoderInputs.n)
            let sMA   = try makeMLMultiArray(shape: vocoderInputs.s_shape, data: vocoderInputs.s)

            let start = CFAbsoluteTimeGetCurrent()
            let audio = try KokoroTTS.synthesizeWithHAR(asr: asrMA, f0: f0MA, n: nMA, s: sMA, harSpec: hsMA, harPhase: hpMA)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

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
            if ProcessInfo.processInfo.environment["COMPARE_GOLDEN"] == "1" {
                compareAgainstGolden(currentMel: mel, cwd: cwd)
                let goldenWAV = cwd.appendingPathComponent("outputs/golden/golden.wav")
                if FileManager.default.fileExists(atPath: goldenWAV.path) {
                    let (gold, sr) = try loadWAVMono16(url: goldenWAV)
                    let (m, shift, used) = computeAudioMSE(audio, gold, maxShiftSamples: Int(0.2 * Double(sr)))
                    print(String(format: "Audio MSE=%.6f @shift=%d (samples used=%d)", m, shift, used))
                }
            }

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
        // Allow runtime override for stride denominator; default to 2 (50% overlap) for Hann COLA
        let strideDenASR: Int = Int(ProcessInfo.processInfo.environment["ASR_STRIDE_FRAC"] ?? ProcessInfo.processInfo.environment["ASR_STRIDE_DENOM"] ?? "2") ?? 2
        let strideDenF0: Int = Int(ProcessInfo.processInfo.environment["F0_STRIDE_FRAC"] ?? ProcessInfo.processInfo.environment["F0_STRIDE_DENOM"] ?? "2") ?? 2
        let strideAsr = max(1, windowAsr / max(1, strideDenASR))
        let strideF0 = max(1, windowF0 / max(1, strideDenF0))
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
        func zscoreASR(_ slice: inout [Float], winT: Int, stats: VocoderInputs.StatsTensor) {
            guard let mu = stats.per_channel?.mean, let sigma = stats.per_channel?.std, mu.count >= 512, sigma.count >= 512 else { return }
            for c in 0..<512 {
                let m = mu[c]
                let s = max(sigma[c], 1e-6)
                let off = c * winT
                for t in 0..<winT { slice[off + t] = (slice[off + t] - m) / s }
            }
        }
        func zscore1D(_ slice: inout [Float], mean: Float, std: Float) {
            let s = max(std, 1e-6)
            for i in 0..<slice.count { slice[i] = (slice[i] - mean) / s }
        }
        let s = try makeMLMultiArray(shape: vocoderInputs.s_shape, data: vocoderInputs.s)

        // Optional: print stats for parity check
        if let st = vocoderInputs.stats {
            func p(_ name: String, _ t: VocoderInputs.StatsTensor?) { if let t = t { print(String(format: "%@ stats: mean=%.6f std=%.6f min=%.6f max=%.6f", name, t.mean, t.std, t.min, t.max)) } }
            p("asr", st.asr); p("f0", st.f0); p("n", st.n); p("s", st.s)
        }

        // Determine number of windows
        func numWindows(total: Int, window: Int, stride: Int) -> Int {
            if total <= 0 { return 0 }
            if total <= window { return 1 }
            return Int(ceil(Double(total - window) / Double(stride))) + 1
        }
        // Use full declared ASR length for synthesis windows
        let synthesisT = asrLenTotal
        let nWin = numWindows(total: synthesisT, window: windowAsr, stride: strideAsr)

        // Prepare output buffers with weight normalization
        let chunkSamples = windowAsr * samplesPerFrame
        let strideSamples = strideAsr * samplesPerFrame
        let totalSamples = max(chunkSamples, (nWin - 1) * strideSamples + chunkSamples)
        var accAudio = [Float](repeating: 0, count: totalSamples)
        var accWeight = [Float](repeating: 0, count: totalSamples)
        var windowWeight = hannWindow(chunkSamples)
        if ProcessInfo.processInfo.environment["RECT_WINDOW"] == "1" || nWin == 1 {
            windowWeight = [Float](repeating: 1.0, count: chunkSamples)
        }

        let start = CFAbsoluteTimeGetCurrent()
        for w in 0..<nWin {
            let asrStart = w * strideAsr
            let f0Start = w * strideF0
            var asrSlice = sliceASR(vocoderInputs.asr, totalT: asrLenTotal, startT: asrStart, winT: windowAsr)
            var f0Slice = slice1D(vocoderInputs.f0, totalT: f0LenTotal, startT: f0Start, winT: windowF0)
            var nSlice  = slice1D(vocoderInputs.n,  totalT: f0LenTotal, startT: f0Start, winT: windowF0)

            // Optional normalization toggles (env-gated)
            let env = ProcessInfo.processInfo.environment
            if env["ASR_ZSCORE_CH"] == "1", let st = vocoderInputs.stats?.asr { zscoreASR(&asrSlice, winT: windowAsr, stats: st) }
            if env["F0_ZSCORE"] == "1", let st = vocoderInputs.stats?.f0 { zscore1D(&f0Slice, mean: st.mean, std: st.std) }
            if env["N_ZSCORE"] == "1", let st = vocoderInputs.stats?.n  { zscore1D(&nSlice,  mean: st.mean, std: st.std) }

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
        // Trim to expected duration (frames * samplesPerFrame)
        let expectedSamples = max(0, asrLenTotal * samplesPerFrame)
        if expectedSamples > 0 && expectedSamples < audio.count {
            audio = Array(audio[0..<expectedSamples])
        }
        // Trim by RMS tail and fade-out
        audio = trimTailByRMS(audio, sampleRate: vocoderInputs.meta.sample_rate)

        // Save artifacts to outputs/[timestamp]/[timestamp].{wav,csv,png,json}
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let ts = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let outDir = cwd.appendingPathComponent("outputs/\(ts)", isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let wavURL = outDir.appendingPathComponent("\(ts).wav")
        try saveWAV(audio, sampleRate: Double(vocoderInputs.meta.sample_rate), url: wavURL)

        print("Saved: \(wavURL.path)")
        print(String(format: "CoreML elapsed: %.3f s", elapsed))
        // Optional time alignment to golden for A/B (does not affect saved WAV unless SAVE_ALIGNED=1)
        if ProcessInfo.processInfo.environment["COMPARE_GOLDEN"] == "1" {
            let goldenWAV = cwd.appendingPathComponent("outputs/golden/golden.wav")
            if FileManager.default.fileExists(atPath: goldenWAV.path) {
                let (gold, sr) = try loadWAVMono16(url: goldenWAV)
                let (_, shift, _) = computeAudioMSE(audio, gold, maxShiftSamples: Int(0.02 * Double(sr)))
                if shift != 0 {
                    let aligned = shiftAudio(audio, shift: shift)
                    if ProcessInfo.processInfo.environment["SAVE_ALIGNED"] == "1" {
                        let alignedURL = outDir.appendingPathComponent("\(ts)_aligned.wav")
                        try saveWAV(aligned, sampleRate: Double(vocoderInputs.meta.sample_rate), url: alignedURL)
                        print("Saved aligned: \(alignedURL.path) @shift=\(shift)")
                    }
                }
            }
        }
        try playWAV(from: wavURL)

        // Save mel spectrogram (CSV + PNG)
        let mel = melSpectrogram(audio: audio, sampleRate: vocoderInputs.meta.sample_rate, nFFT: 1024, hop: 300, nMels: 80, fmin: 0, fmax: 12000)
        let melCSV = outDir.appendingPathComponent("\(ts).csv")
        try saveMatrixCSV(rows: mel.count, cols: mel.first?.count ?? 0, value: { r, c in mel[r][c] }, url: melCSV)
        let pngURL = outDir.appendingPathComponent("\(ts).png")
        try saveMelPNG(mel: mel, url: pngURL)
        if ProcessInfo.processInfo.environment["COMPARE_GOLDEN"] == "1" {
            compareAgainstGolden(currentMel: mel, cwd: cwd)
            let goldenWAV = cwd.appendingPathComponent("outputs/golden/golden.wav")
            if FileManager.default.fileExists(atPath: goldenWAV.path) {
                let (gold, sr) = try loadWAVMono16(url: goldenWAV)
                let (m, shift, used) = computeAudioMSE(audio, gold, maxShiftSamples: Int(0.2 * Double(sr)))
                print(String(format: "Audio MSE=%.6f @shift=%d (samples used=%d)", m, shift, used))
            }
        }

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
