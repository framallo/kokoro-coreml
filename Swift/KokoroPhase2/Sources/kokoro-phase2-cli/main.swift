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

    let t1 = now()
    let (audio, sr) = try runner.predict(asr: asr, f0: f0, n: n, s: s)
    let t2 = now()
    try FileManager.default.createDirectory(at: outWav.deletingLastPathComponent(), withIntermediateDirectories: true)
    try WAV.writePCM16(fileURL: outWav, samples: audio, sampleRate: sr)
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
        ]
    ]
    let metaURL = runDir.appendingPathComponent("metadata.json")
    let metaData = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
    try metaData.write(to: metaURL)
    print("wrote: \(outWav.path)")
    print("meta:  \(metaURL.path)")
}

try main()
