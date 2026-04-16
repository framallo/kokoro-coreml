/// Language-neutral tensor dump writer for audio parity debugging.
///
/// Writes ``tensor_manifest.json`` plus little-endian raw ``.f32`` / ``.i32``
/// payloads. Python loads the same format in
/// ``scripts/audio_parity_tensor_io.py``.

import CoreML
import Foundation

public struct TensorDumpWriter {
    public let directory: URL
    private var records: [[String: Any]] = []

    public init(directory: URL) throws {
        self.directory = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    public mutating func writeFloatArray(name: String, values: [Float], shape: [Int]) throws {
        try validateCount(name: name, count: values.count, shape: shape)
        let filename = "\(safeTensorName(name)).f32"
        var data = Data(capacity: values.count * MemoryLayout<UInt32>.size)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        try data.write(to: directory.appendingPathComponent(filename))
        records.append([
            "name": name,
            "dtype": "float32",
            "shape": shape,
            "path": filename,
            "summary": floatSummary(values),
        ])
    }

    public mutating func writeInt32Array(name: String, values: [Int32], shape: [Int]) throws {
        try validateCount(name: name, count: values.count, shape: shape)
        let filename = "\(safeTensorName(name)).i32"
        var data = Data(capacity: values.count * MemoryLayout<Int32>.size)
        for value in values {
            var little = value.littleEndian
            withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
        }
        try data.write(to: directory.appendingPathComponent(filename))
        records.append([
            "name": name,
            "dtype": "int32",
            "shape": shape,
            "path": filename,
            "summary": intSummary(values),
        ])
    }

    public mutating func writeMLMultiArray(name: String, array: MLMultiArray) throws {
        let shape = array.shape.map { $0.intValue }
        switch array.dataType {
        case .int32:
            var values = [Int32]()
            values.reserveCapacity(array.count)
            for offset in 0..<array.count {
                values.append(array[multiIndex(offset: offset, shape: shape)].int32Value)
            }
            try writeInt32Array(name: name, values: values, shape: shape)
        default:
            var values = [Float]()
            values.reserveCapacity(array.count)
            for offset in 0..<array.count {
                values.append(array[multiIndex(offset: offset, shape: shape)].floatValue)
            }
            try writeFloatArray(name: name, values: values, shape: shape)
        }
    }

    public mutating func writeManifest(metadata: [String: Any]) throws {
        let manifest: [String: Any] = [
            "schema_version": 1,
            "metadata": metadata,
            "tensors": records,
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("tensor_manifest.json"))
    }

    private func validateCount(name: String, count: Int, shape: [Int]) throws {
        let expected = shape.reduce(1, *)
        guard count == expected else {
            throw TensorDumpError.invalidShape(name: name, count: count, expected: expected)
        }
    }
}

public enum TensorDumpError: Error, LocalizedError {
    case invalidShape(name: String, count: Int, expected: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidShape(let name, let count, let expected):
            return "Tensor \(name) has \(count) values, expected \(expected) from shape"
        }
    }
}

private func safeTensorName(_ name: String) -> String {
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
    let scalars = name.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" }
    let safe = String(scalars).trimmingCharacters(in: CharacterSet(charactersIn: "._"))
    return safe.isEmpty ? "tensor" : safe
}

private func multiIndex(offset: Int, shape: [Int]) -> [NSNumber] {
    guard !shape.isEmpty else { return [] }
    var remainder = offset
    var result = [Int](repeating: 0, count: shape.count)
    for dimIndex in stride(from: shape.count - 1, through: 0, by: -1) {
        let dim = max(1, shape[dimIndex])
        result[dimIndex] = remainder % dim
        remainder /= dim
    }
    return result.map { NSNumber(value: $0) }
}

private func floatSummary(_ values: [Float]) -> [String: Any] {
    guard !values.isEmpty else { return ["count": 0] }
    let finite = values.filter { $0.isFinite }
    var summary: [String: Any] = [
        "count": values.count,
        "finite_count": finite.count,
        "nan_count": values.filter { $0.isNaN }.count,
        "inf_count": values.filter { $0.isInfinite }.count,
    ]
    guard !finite.isEmpty else { return summary }
    let minValue = finite.min() ?? 0
    let maxValue = finite.max() ?? 0
    let sum = finite.reduce(0.0) { $0 + Double($1) }
    let l2 = sqrt(finite.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(finite.count))
    summary["min"] = minValue
    summary["max"] = maxValue
    summary["mean"] = sum / Double(finite.count)
    summary["l2"] = l2
    return summary
}

private func intSummary(_ values: [Int32]) -> [String: Any] {
    guard !values.isEmpty else { return ["count": 0] }
    let minValue = values.min() ?? 0
    let maxValue = values.max() ?? 0
    let sum = values.reduce(0) { $0 + Int64($1) }
    return [
        "count": values.count,
        "min": minValue,
        "max": maxValue,
        "mean": Double(sum) / Double(values.count),
    ]
}
