import Foundation
import CoreML
import KokoroPhase2

struct Fixture: Codable {
    let asr: [Float]
    let f0_curve: [Float]
    let n: [Float]
    let s: [Float]
    let shapes: [String:[Int]]
    let text: String
    let voice: String
    let har_spec: [Float]?
    let har_phase: [Float]?
}

func makeArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: shape as [NSNumber], dataType: .float32)
    let ptr = UnsafeMutableBufferPointer(start: arr.dataPointer.assumingMemoryBound(to: Float.self), count: arr.count)
    guard ptr.count == values.count else { throw KokoroPhase2Error.shapeMismatch("values.count != MLMultiArray.count") }
    values.withUnsafeBufferPointer { src in
        ptr.baseAddress!.update(from: src.baseAddress!, count: src.count)
    }
    return arr
}

func rmsDBFS(_ x: [Float]) -> Double {
    if x.isEmpty { return -120.0 }
    var sum: Double = 0
    for v in x { sum += Double(v) * Double(v) }
    let mean = sum / Double(x.count)
    let rms = sqrt(max(mean, 1e-12))
    return 20.0 * log10(rms)
}

func applyTargetDBFS(_ x: inout [Float], targetDBFS: Double) {
    let cur = rmsDBFS(x)
    let delta = targetDBFS - cur
    let gain = pow(10.0, delta / 20.0)
    let g = Float(gain)
    for i in 0..<x.count { x[i] *= g }
}

func loadFixture(url: URL) throws -> Fixture {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(Fixture.self, from: data)
}

func now() -> Double { CFAbsoluteTimeGetCurrent() }

func isoTimestamp() -> String {
    let df = ISO8601DateFormatter()
    return df.string(from: Date())
}

func makeRunDir(base: URL) throws -> URL {
    let formatter = DateFormatter(); formatter.dateFormat = "yyyyMMdd_HHmmss"
    let dir = base.appendingPathComponent("phase2_" + formatter.string(from: Date()))
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

func main() throws {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        print("usage: kokoro-phase2-cli <fixture.json> <mlpackage_path> [out_dir]")
        return
    }
    let fixtureURL = URL(fileURLWithPath: args[1])
    let modelURL = URL(fileURLWithPath: args[2])
    let outBase = args.count >= 4 ? URL(fileURLWithPath: args[3]) : URL(fileURLWithPath: "outputs/local")
    let runDir = try makeRunDir(base: outBase)
    let outWav = runDir.appendingPathComponent("output.wav")

    let fixture = try loadFixture(url: fixtureURL)

    let t0 = now()
    let runner = try DecoderOnly5sRunner(mlpackageURL: modelURL)

    let asr = try makeArray(fixture.asr, shape: fixture.shapes["asr"] ?? [1,512,1,200])
    let f0 = try makeArray(fixture.f0_curve, shape: fixture.shapes["f0_curve"] ?? [1,1,1,400])
    let n = try makeArray(fixture.n, shape: fixture.shapes["n"] ?? [1,1,1,400])
    let s = try makeArray(fixture.s, shape: fixture.shapes["s"] ?? [1,128])
    let maybeHarSpec: MLMultiArray? = {
        if let arr = fixture.har_spec, let shp = fixture.shapes["har_spec"] { return try? makeArray(arr, shape: shp) }
        return nil
    }()
    let maybeHarPhase: MLMultiArray? = {
        if let arr = fixture.har_phase, let shp = fixture.shapes["har_phase"] { return try? makeArray(arr, shape: shp) }
        return nil
    }()

    let t1 = now()
    var audio: [Float]
    var sr: Int = 24000
    do {
        (audio, sr) = try runner.predict(asr: asr, f0: f0, n: n, s: s)
    } catch {
        // If the model requires HAR inputs, retry with them (when available)
        if (maybeHarSpec != nil || maybeHarPhase != nil) {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "asr": MLFeatureValue(multiArray: asr),
                "f0_curve": MLFeatureValue(multiArray: f0),
                "n": MLFeatureValue(multiArray: n),
                "s": MLFeatureValue(multiArray: s),
                "har_spec": maybeHarSpec.map(MLFeatureValue.init(multiArray:)),
                "har_phase": maybeHarPhase.map(MLFeatureValue.init(multiArray:)),
            ].compactMapValues { $0 })
            let out = try runner.rawModel.prediction(from: provider)
            if let firstOut = out.featureNames.first, let arr = out.featureValue(for: firstOut)?.multiArrayValue {
                if let harSpecShape = fixture.shapes["har_spec"], let f0Shape = fixture.shapes["f0_curve"], let arrShape = arr.shape as? [NSNumber], arrShape.count == 3 {
                    // Shapes
                    // har_spec: [1,C,1,T] → C
                    let channelsIn = harSpecShape[1]
                    // model output: [1,Cout,Frames]
                    let cOut = arrShape[1].intValue
                    let frames = arrShape[2].intValue
                    // Derive nFFT and hop from output and f0 length
                    let nFFT = max(4, cOut - 2)
                    let f0Len = f0Shape.last ?? 400
                    let seconds = Double(f0Len) / 80.0
                    let targetSamples = Int(seconds * 24000.0)
                    // With center padding, frames ≈ targetSamples / hop + 1 ⇒ hop ≈ targetSamples / (frames-1)
                    let hop = max(1, Int(round(Double(targetSamples) / Double(max(1, frames - 1)))))
                    let har = HarPostProcessor(nFFT: nFFT, hop: hop, winLength: nFFT)
                    audio = try har.inverseFromNetworkOutput(arr, channels: cOut, frames: frames)
                } else {
                    let floats = try DecoderOnly5sRunner.flattenFloatArrayStatic(arr)
                    audio = floats
                }
            } else {
                throw KokoroPhase2Error.predictionFailed("No output from HAR model")
            }
        } else {
            throw error
        }
    }
    let t2 = now()
    try FileManager.default.createDirectory(at: outWav.deletingLastPathComponent(), withIntermediateDirectories: true)
    // Optional gain calibration to target dBFS
    var audioOut = audio
    if let t = ProcessInfo.processInfo.environment["KOKORO_TARGET_DBFS"], let target = Double(t) {
        applyTargetDBFS(&audioOut, targetDBFS: target)
    }
    try WAV.writePCM16(fileURL: outWav, samples: audioOut, sampleRate: sr)
    // Optional: dump inputs for parity checks
    if ProcessInfo.processInfo.environment["KOKORO_DUMP_INPUTS"] == "1" {
        func writeCSV(_ path: URL, _ rows: [[Float]]) throws {
            let text = rows.map { row in row.map { String($0) }.joined(separator: ",") }.joined(separator: "\n")
            try text.data(using: .utf8)!.write(to: path)
        }
        // asr: [1,512,1,200] → 512x200
        do {
            var rows: [[Float]] = []
            let flat = fixture.asr
            let cols = (fixture.shapes["asr"] ?? [1,512,1,200]).last ?? 200
            let chans = (fixture.shapes["asr"] ?? [1,512,1,200])[1]
            for c in 0..<(chans) {
                let start = c * cols
                let end = start + cols
                rows.append(Array(flat[start..<end]))
            }
            try writeCSV(runDir.appendingPathComponent("asr.csv"), rows)
        }
        // f0_curve: [1,1,1,400]
        do {
            let row = fixture.f0_curve
            try writeCSV(runDir.appendingPathComponent("f0_curve.csv"), [row])
        }
        // n: [1,1,1,400]
        do {
            let row = fixture.n
            try writeCSV(runDir.appendingPathComponent("n.csv"), [row])
        }
        // s: [1,128]
        do {
            let row = fixture.s
            try writeCSV(runDir.appendingPathComponent("s.csv"), [row])
        }
    }
    // Write metadata
    let metadata: [String: Any] = [
        "input_text": fixture.text,
        "voice": fixture.voice,
        "bucket_seconds": 5,
        "sample_rate": sr,
        "model": modelURL.lastPathComponent,
        "timestamp": isoTimestamp(),
        "latency": [
            "load_s": t1 - t0,
            "coreml_s": t2 - t1,
            "total_s": t2 - t0,
        ],
        "levels": [
            "dbfs_raw": rmsDBFS(audio),
            "dbfs_written": rmsDBFS(audioOut),
        ]
    ]
    let metaURL = runDir.appendingPathComponent("metadata.json")
    let metaData = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
    try metaData.write(to: metaURL)
    print("wrote: \(outWav.path)")
    print("meta:  \(metaURL.path)")
}

try main()
