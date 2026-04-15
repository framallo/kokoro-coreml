import XCTest
@testable import KokoroPipeline

/// Timing benchmarks for hn-nsf Swift implementation.
///
/// These measure the actual latency of the Swift DSP path on this machine.
/// Run with: swift test -c release (for representative numbers).
final class BenchmarkTests: XCTestCase {

    // Learned weights from SourceModuleHnNSF.l_linear (Kokoro-82M)
    let linearWeights: [Float] = [
        -0.08154187, -0.18519667, -0.18263398, -0.17837206, -0.09873895,
         0.08264039,  0.08743999, -0.39068547, -0.54774433
    ]
    let linearBias: Float = -0.02945026

    func testHnsfTiming3sBucket() {
        // 3s bucket: full_f0_len = 240 (3 * 24000 / 300)
        let f0 = [Float](repeating: 200.0, count: 240)
        measure {
            let _ = buildHar(
                f0Padded: f0,
                linearWeights: linearWeights,
                linearBias: linearBias,
                seed: 42
            )
        }
    }

    func testHnsfTiming10sBucket() {
        // 10s bucket: full_f0_len = 800 (10 * 24000 / 300)
        let f0 = [Float](repeating: 200.0, count: 800)
        measure {
            let _ = buildHar(
                f0Padded: f0,
                linearWeights: linearWeights,
                linearBias: linearBias,
                seed: 42
            )
        }
    }
}
