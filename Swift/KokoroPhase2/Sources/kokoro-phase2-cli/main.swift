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
    _ = values.withUnsafeBufferPointer { src -> Void in
        ptr.assign(from: src.baseAddress!, count: src.count)
    }
    return arr
}

func loadFixture(url: URL) throws -> Fixture {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(Fixture.self, from: data)
}

func main() throws {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        print("usage: kokoro-phase2-cli <fixture.json> <mlpackage_path> [out.wav]")
        return
    }
    let fixtureURL = URL(fileURLWithPath: args[1])
    let modelURL = URL(fileURLWithPath: args[2])
    let outWav = args.count >= 4 ? URL(fileURLWithPath: args[3]) : URL(fileURLWithPath: "outputs/local/phase2/out.wav")

    let fixture = try loadFixture(url: fixtureURL)

    let runner = try DecoderOnly5sRunner(mlpackageURL: modelURL)

    let asr = try makeArray(fixture.asr, shape: fixture.shapes["asr"] ?? [1,512,1,200])
    let f0 = try makeArray(fixture.f0_curve, shape: fixture.shapes["f0_curve"] ?? [1,1,1,400])
    let n = try makeArray(fixture.n, shape: fixture.shapes["n"] ?? [1,1,1,400])
    let s = try makeArray(fixture.s, shape: fixture.shapes["s"] ?? [1,128])

    let (audio, sr) = try runner.predict(asr: asr, f0: f0, n: n, s: s)
    try FileManager.default.createDirectory(at: outWav.deletingLastPathComponent(), withIntermediateDirectories: true)
    try WAV.writePCM16(fileURL: outWav, samples: audio, sampleRate: sr)
    print("wrote: \(outWav.path)")
}

try main()
