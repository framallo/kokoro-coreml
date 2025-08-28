import Foundation
import Darwin
import Combine
import CoreML
import AVFoundation
import KokoroPhase2

final class InferenceViewModel: ObservableObject {
    @Published var statusText: String = "Ready"
    @Published var lastInferenceMs: Double?
    @Published var isRunning: Bool = false

    private var runner: DecoderOnly5sRunner?
    private let audio = AudioPlayer()

    init() {
        // Warm-up at launch
        DispatchQueue.global(qos: .userInitiated).async {
            self.statusOnMain("Warming up…")
            do {
                try self.ensureRunnerLoaded()
                try self.runWarmUp()
                self.statusOnMain("Ready")
            } catch {
                self.statusOnMain("Warm-up failed: \(error.localizedDescription)")
            }
        }
    }

    func runTest() {
        guard !isRunning else { return }
        isRunning = true
        statusText = "Running test…"
        DispatchQueue.global(qos: .userInitiated).async {
            defer { DispatchQueue.main.async { self.isRunning = false } }
            do {
                try self.ensureRunnerLoaded()
                // Load main 5s fixture from bundle
                let fixture = try FixtureLoader.loadFixture(named: "fixture_har_5s")

                let asr = try ArrayFactory.makeArray(fixture.asr, shape: fixture.shapes["asr"] ?? [1,512,1,200])
                let f0  = try ArrayFactory.makeArray(fixture.f0_curve, shape: fixture.shapes["f0_curve"] ?? [1,1,1,400])
                let n   = try ArrayFactory.makeArray(fixture.n, shape: fixture.shapes["n"] ?? [1,1,1,400])
                let s   = try ArrayFactory.makeArray(fixture.s, shape: fixture.shapes["s"] ?? [1,128])

                let t0 = CFAbsoluteTimeGetCurrent()
                let (audio, sr) = try self.runner!.predict(asr: asr, f0: f0, n: n, s: s)
                let t1 = CFAbsoluteTimeGetCurrent()
                let ms = (t1 - t0) * 1000.0

                DispatchQueue.main.async {
                    self.lastInferenceMs = ms
                    self.statusText = "Done"
                }

                // Play as soon as ready (24 kHz mono)
                self.audio.play(samples: audio, sampleRate: Double(sr))
            } catch {
                DispatchQueue.main.async {
                    self.statusText = "Error: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Private

    private func ensureRunnerLoaded() throws {
        if runner != nil { return }
        // Prefer compiled model in bundle, fall back to raw package
        let modelURL =
            Bundle.main.url(forResource: "kokoro_decoder_only_5s", withExtension: "mlmodelc") ??
            Bundle.main.url(forResource: "kokoro_decoder_only_5s", withExtension: "mlpackage") ??
            Bundle.main.url(forResource: "kokoro_decoder_only_5s", withExtension: "mlpackage", subdirectory: "Resources")
        guard let modelURL else {
            throw NSError(domain: "Inference", code: 10, userInfo: [NSLocalizedDescriptionKey: "kokoro_decoder_only_5s.mlpackage not found in app bundle. Add it to the target."])
        }
        // Force ANE usage for speed testing
        setenv("KOKORO_COMPUTE_UNITS", "all", 1)
        runner = try DecoderOnly5sRunner(mlpackageURL: modelURL)
    }

    private func runWarmUp() throws {
        // Prefer a bundled tiny fixture; fall back to zeros with expected shapes
        if let _ = Bundle.main.url(forResource: "fixture_hi", withExtension: "json"),
           let fx = try? FixtureLoader.loadFixture(named: "fixture_hi") {
            let asr = try ArrayFactory.makeArray(fx.asr, shape: fx.shapes["asr"] ?? [1,512,1,200])
            let f0  = try ArrayFactory.makeArray(fx.f0_curve, shape: fx.shapes["f0_curve"] ?? [1,1,1,400])
            let n   = try ArrayFactory.makeArray(fx.n, shape: fx.shapes["n"] ?? [1,1,1,400])
            let s   = try ArrayFactory.makeArray(fx.s, shape: fx.shapes["s"] ?? [1,128])
            _ = try runner!.predict(asr: asr, f0: f0, n: n, s: s)
            return
        }
        // Fallback zeros
        let zerosASR = [Float](repeating: 0, count: 1*512*1*200)
        let zerosF0  = [Float](repeating: 0, count: 400)
        let zerosN   = [Float](repeating: 0, count: 400)
        let zerosS   = [Float](repeating: 0, count: 128)
        let asr = try ArrayFactory.makeArray(zerosASR, shape: [1,512,1,200])
        let f0  = try ArrayFactory.makeArray(zerosF0, shape: [1,1,1,400])
        let n   = try ArrayFactory.makeArray(zerosN, shape: [1,1,1,400])
        let s   = try ArrayFactory.makeArray(zerosS, shape: [1,128])
        _ = try runner!.predict(asr: asr, f0: f0, n: n, s: s)
    }

    private func statusOnMain(_ text: String) {
        DispatchQueue.main.async { self.statusText = text }
    }
}
