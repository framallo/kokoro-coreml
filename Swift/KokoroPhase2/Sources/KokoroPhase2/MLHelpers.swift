import CoreML

extension MLModel {
    func prediction(dict: [String: MLFeatureValue]) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        return try self.prediction(from: provider)
    }
}

enum MLHelpers {
    static func toMLMultiArray(_ array: [Float], shape: [Int]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let ml = try MLMultiArray(shape: nsShape, dataType: .float32)
        let count = shape.reduce(1, *)
        precondition(array.count == count, "array.count != shape product")
        array.withUnsafeBytes { src in
            memcpy(ml.dataPointer, src.baseAddress!, count * MemoryLayout<Float>.size)
        }
        return ml
    }
}
