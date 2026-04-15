/// Helpers for CoreML MLMultiArray construction, matrix operations,
/// and zero-padding to bucket geometry.
///
/// These replace the numpy array operations in the Python pipeline
/// (``build_decoder_har_post_inputs_np`` padding, ``d @ alignment``
/// matrix multiply, etc.) with native Swift + Accelerate.

import CoreML
import Accelerate

// MARK: - MLMultiArray Construction

/// Create a 3D MLMultiArray of shape (1, channels, time) filled with zeros.
///
/// Matches the numpy pattern: ``np.zeros((1, C, T), dtype=np.float32)``
public func makeZeroArray3D(channels: Int, time: Int) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: [1, NSNumber(value: channels), NSNumber(value: time)], dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    memset(ptr, 0, channels * time * MemoryLayout<Float>.size)
    return arr
}

/// Create a 2D MLMultiArray of shape (1, dim) filled with zeros.
public func makeZeroArray2D(dim: Int) throws -> MLMultiArray {
    let arr = try MLMultiArray(shape: [1, NSNumber(value: dim)], dataType: .float32)
    let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
    memset(ptr, 0, dim * MemoryLayout<Float>.size)
    return arr
}

/// Copy a flat Float array into a pre-allocated MLMultiArray, zero-padding if needed.
///
/// The source data is copied starting from index 0. If the source is shorter
/// than the MLMultiArray, the remainder stays zero. If longer, it's truncated.
public func copyInto(array: MLMultiArray, from source: [Float]) {
    let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
    let count = min(source.count, array.count)
    _ = source.withUnsafeBufferPointer { srcBuf in
        memcpy(ptr, srcBuf.baseAddress!, count * MemoryLayout<Float>.size)
    }
}

// MARK: - Matrix Multiply

/// Batch matrix multiply: C = A @ B for 3D tensors.
///
/// A: (1, M, K), B: (1, K, N) → C: (1, M, N)
///
/// Uses cblas_sgemm from Accelerate for the inner (M, K) × (K, N) → (M, N) multiply.
/// Row-major layout (matching MLMultiArray default for 3D float32).
///
/// This replaces: ``en = d.transpose(-1, -2) @ pred_aln_trg``
/// and ``asr = t_en @ pred_aln_trg``
public func matmul3D(
    a: MLMultiArray, // (1, M, K)
    b: [Float],      // flat (K, N) in row-major
    M: Int, K: Int, N: Int
) throws -> MLMultiArray {
    let result = try makeZeroArray3D(channels: M, time: N)

    // Ensure A is contiguous (CoreML outputs may have non-trivial strides).
    let aStrides = a.strides.map { $0.intValue }
    let aIsContiguous = aStrides.count >= 3 && aStrides[2] == 1 && aStrides[1] == K

    let aContiguous: MLMultiArray
    if aIsContiguous {
        aContiguous = a
    } else {
        // Copy to contiguous layout
        aContiguous = try makeZeroArray3D(channels: M, time: K)
        let dstPtr = aContiguous.dataPointer.assumingMemoryBound(to: Float.self)
        for m in 0..<M {
            for k in 0..<K {
                dstPtr[m * K + k] = a[[0, m, k] as [NSNumber]].floatValue
            }
        }
    }

    let aPtr = aContiguous.dataPointer.assumingMemoryBound(to: Float.self)
    let cPtr = result.dataPointer.assumingMemoryBound(to: Float.self)

    // A is (1, M, K) row-major → pointer to M*K matrix
    // B is flat (K, N) row-major
    // C is (1, M, N) row-major → pointer to M*N matrix
    b.withUnsafeBufferPointer { bBuf in
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(M), Int32(N), Int32(K),
            1.0,       // alpha
            aPtr, Int32(K),
            bBuf.baseAddress!, Int32(N),
            0.0,       // beta
            cPtr, Int32(N)
        )
    }
    return result
}

// MARK: - Zero-Pad to Bucket Geometry

/// Zero-pad a 3D array (1, C, T_src) to (1, C, T_target).
///
/// Matches: ``asr_pad = np.zeros((1, 512, frame_count)); asr_pad[:,:,:t] = asr[:,:,:t]``
///
/// Uses subscript access (not raw pointer arithmetic) to handle MLMultiArray
/// strides correctly. CoreML model outputs may have non-trivial strides.
public func zeroPad3D(source: MLMultiArray, channels: Int, targetTime: Int) throws -> MLMultiArray {
    let result = try makeZeroArray3D(channels: channels, time: targetTime)
    let srcTime = source.shape[2].intValue
    let copyTime = min(srcTime, targetTime)

    // Check if source has standard contiguous strides (batch=C*T, channel=T, time=1)
    let srcStrides = source.strides.map { $0.intValue }
    let isContiguous = srcStrides.count >= 3 && srcStrides[2] == 1 && srcStrides[1] == srcTime

    if isContiguous {
        // Fast path: memcpy per channel
        let srcPtr = source.dataPointer.assumingMemoryBound(to: Float.self)
        let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<channels {
            let srcOffset = c * srcTime
            let dstOffset = c * targetTime
            memcpy(dstPtr + dstOffset, srcPtr + srcOffset, copyTime * MemoryLayout<Float>.size)
        }
    } else {
        // Safe path: subscript access for non-contiguous layouts
        let dstPtr = result.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<channels {
            for t in 0..<copyTime {
                dstPtr[c * targetTime + t] = source[[0, c, t] as [NSNumber]].floatValue
            }
        }
    }
    return result
}

/// Zero-pad a 1D array (1, T_src) to (1, T_target).
///
/// Matches: ``f0_pad = np.zeros((1, full_f0_len)); f0_pad[:,:t] = f0[:,:t]``
public func zeroPad1D(source: [Float], targetLength: Int) -> [Float] {
    var result = [Float](repeating: 0, count: targetLength)
    let copyLen = min(source.count, targetLength)
    for i in 0..<copyLen {
        result[i] = source[i]
    }
    return result
}

// MARK: - Shape Introspection

/// Extract input shapes from a CoreML model spec.
///
/// Returns a dictionary mapping input name → shape array.
public func inputShapes(from model: MLModel) -> [String: [Int]] {
    var result: [String: [Int]] = [:]
    let desc = model.modelDescription
    for (name, feature) in desc.inputDescriptionsByName {
        if let constraint = feature.multiArrayConstraint {
            result[name] = constraint.shape.map { $0.intValue }
        }
    }
    return result
}
