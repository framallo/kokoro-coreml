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
}
