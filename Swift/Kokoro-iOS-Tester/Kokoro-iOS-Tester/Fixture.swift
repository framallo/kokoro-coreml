import Foundation
import CoreML

struct Fixture: Codable {
    let asr: [Float]
    let f0_curve: [Float]
    let n: [Float]
    let s: [Float]
    let shapes: [String: [Int]]
    let text: String
    let voice: String
    let har_spec: [Float]?
    let har_phase: [Float]?
}

enum FixtureLoader {
    static func loadFixture(named name: String, in bundle: Bundle = .main) throws -> Fixture {
        // Try top-level first, then within our bundled "Resources" folder reference
        let url = bundle.url(forResource: name, withExtension: "json") ??
                  bundle.url(forResource: name, withExtension: "json", subdirectory: "Resources")
        guard let url else {
            throw NSError(domain: "Fixture", code: 1, userInfo: [NSLocalizedDescriptionKey: "Fixture \(name) not found in bundle"])
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Fixture.self, from: data)
    }
}

enum ArrayFactory {
    static func makeArray(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let ptr = UnsafeMutableBufferPointer(start: arr.dataPointer.assumingMemoryBound(to: Float.self), count: arr.count)
        guard ptr.count == values.count else {
            throw NSError(domain: "Fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "values.count != MLMultiArray.count (\(values.count) != \(ptr.count))"])
        }
        values.withUnsafeBufferPointer { src in
            ptr.baseAddress!.update(from: src.baseAddress!, count: src.count)
        }
        return arr
    }
}
