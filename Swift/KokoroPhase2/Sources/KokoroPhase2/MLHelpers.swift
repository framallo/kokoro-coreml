import CoreML

extension MLModel {
    func prediction(dict: [String: MLFeatureValue]) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        return try self.prediction(from: provider)
    }
}
