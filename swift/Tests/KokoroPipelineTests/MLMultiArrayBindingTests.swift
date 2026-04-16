import CoreML
import XCTest
@testable import KokoroPipeline

final class MLMultiArrayBindingTests: XCTestCase {

    func testReadDurationFramesFromInt32PredDur() throws {
        let arr = try MLMultiArray(shape: [1, 5], dataType: .int32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        [13, 1, 2, 0, -4].enumerated().forEach { idx, value in
            ptr[idx] = Int32(value)
        }

        XCTAssertEqual(try readDurationFrames(from: arr, validCount: 3), [13, 1, 2])
        XCTAssertEqual(try readDurationFrames(from: arr), [13, 1, 2, 1, 1])
    }

    func testReadDurationFramesFromFloatPredDur() throws {
        let arr = try MLMultiArray(shape: [1, 4], dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        [12.6, 0.2, -3.0, 2.4].enumerated().forEach { idx, value in
            ptr[idx] = Float(value)
        }

        XCTAssertEqual(try readDurationFrames(from: arr), [13, 1, 1, 2])
    }

    func testValidateDurationAgreementRejectsHalfLengthAudio() throws {
        XCTAssertThrowsError(
            try validateDurationAgreement(inputKey: "15s", canonical: 13.9, observed: 6.4)
        )
    }

    func testValidateDurationAgreementAllowsCloseAudio() throws {
        XCTAssertNoThrow(
            try validateDurationAgreement(inputKey: "15s", canonical: 13.9, observed: 13.7)
        )
    }

    func testTensorDumpWriterWritesManifestAndPayloads() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokoro-tensor-dump-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        var writer = try TensorDumpWriter(directory: dir)
        try writer.writeInt32Array(name: "tokens", values: [1, 2, 3], shape: [1, 3])
        try writer.writeFloatArray(name: "waveform", values: [0.0, 0.25, -0.5], shape: [3])
        try writer.writeManifest(metadata: ["producer": "test"])

        let manifestURL = dir.appendingPathComponent("tensor_manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONSerialization.jsonObject(with: manifestData) as? [String: Any]
        let tensors = manifest?["tensors"] as? [[String: Any]]

        XCTAssertEqual(manifest?["schema_version"] as? Int, 1)
        XCTAssertEqual((manifest?["metadata"] as? [String: Any])?["producer"] as? String, "test")
        XCTAssertEqual(tensors?.count, 2)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokens.i32").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("waveform.f32").path))
    }
}
