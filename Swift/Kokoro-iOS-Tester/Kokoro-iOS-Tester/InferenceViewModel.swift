import Foundation
import Darwin
import Combine
import CoreML
import AVFoundation
import KokoroPhase2

/// Main view model for Kokoro TTS inference testing and performance measurement.
///
/// This view model orchestrates the complete TTS inference pipeline, handling model loading,
/// performance measurement, and audio playback for the iOS testing application. It provides
/// a SwiftUI-compatible interface for running TTS inference tests on device.
///
/// **Architecture:**
/// - Manages Core ML model lifecycle (loading, warm-up, inference)
/// - Provides real-time performance metrics (inference timing)
/// - Handles asynchronous operations with proper UI state management
/// - Integrates with audio playback system for immediate result audition
///
/// **Performance Optimization:**
/// - Implements warm-up procedure to initialize ANE/GPU compute units
/// - Uses background queues for model operations to maintain UI responsiveness
/// - Pre-loads test fixtures to minimize I/O during timing measurements
///
/// **Cross-Module Dependencies:**
/// - KokoroPhase2.DecoderOnly5sRunner: Core ML inference engine
/// - AudioPlayer: Real-time audio playback for generated speech
/// - FixtureLoader/ArrayFactory: Test data loading and tensor creation
final class InferenceViewModel: ObservableObject {
    /// Current status message displayed to user (e.g., "Ready", "Running test…", "Done")
    @Published var statusText: String = "Ready"
    
    /// Last measured inference time in milliseconds for performance tracking
    @Published var lastInferenceMs: Double?
    
    /// Whether inference operation is currently running (disables UI controls)
    @Published var isRunning: Bool = false

    /// Core ML inference runner, lazily initialized during warm-up
    private var runner: DecoderOnly5sRunner?
    
    /// Audio playback system for immediate audition of generated speech
    private let audio = AudioPlayer()

    /// Initialize view model with automatic warm-up procedure.
    ///
    /// The initializer immediately begins the warm-up process on a background queue
    /// to prepare the Core ML model and compute units for inference. This ensures
    /// the first user-initiated test runs at full performance without cold-start delays.
    ///
    /// Warm-up Process:
    /// 1. Load Core ML model from app bundle
    /// 2. Configure compute units (ANE/GPU/CPU based on availability)
    /// 3. Run dummy inference to initialize all computation paths
    /// 4. Update UI status to indicate readiness
    ///
    /// Called by:
    /// - SwiftUI view system during view model instantiation
    /// - App launch sequence when inference view becomes active
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

    // MARK: - Private Implementation
    
    /// Model loading constants for Core ML bundle management
    private enum ModelConstants {
        /// Core ML model resource name in app bundle
        static let modelResourceName = "kokoro_decoder_only_5s"
        
        /// Preferred model extension (compiled format for faster loading)
        static let compiledExtension = "mlmodelc"
        
        /// Fallback model extension (source format)
        static let packageExtension = "mlpackage"
        
        /// Bundle subdirectory for resources (fallback location)
        static let resourcesSubdirectory = "Resources"
        
        /// Environment variable for compute unit configuration
        static let computeUnitsEnvVar = "KOKORO_COMPUTE_UNITS"
        
        /// Compute units setting for maximum performance (ANE + GPU + CPU)
        static let allComputeUnits = "all"
    }

    /// Ensure Core ML inference runner is loaded and ready for use.
    ///
    /// This method handles the complete model loading pipeline with intelligent fallback
    /// between different model formats and bundle locations. It prioritizes compiled
    /// models for faster loading while supporting source packages as fallback.
    ///
    /// **Model Loading Strategy:**
    /// 1. Search for compiled .mlmodelc in main bundle (fastest loading)
    /// 2. Fall back to .mlpackage in main bundle (slower, but more compatible)
    /// 3. Final fallback to Resources subdirectory (development/testing)
    ///
    /// **Compute Unit Configuration:**
    /// - Forces "all" compute units via environment variable
    /// - Enables ANE (Apple Neural Engine) for maximum inference speed
    /// - Used specifically for performance testing and benchmarking
    ///
    /// **Cross-Module Dependencies:**
    /// - KokoroPhase2.DecoderOnly5sRunner: Model wrapper and inference engine
    /// - Bundle.main: iOS app bundle resource management
    /// - Environment variables: Runtime compute unit configuration
    ///
    /// **Called by:**
    /// - init(): During warm-up procedure to prepare model for testing
    /// - runTest(): Ensures model is ready before inference (safety check)
    ///
    /// **Throws:**
    /// - NSError: If model bundle is missing or cannot be loaded
    private func ensureRunnerLoaded() throws {
        if runner != nil { return }
        
        // Search for model in bundle with prioritized fallback chain
        let modelURL =
            Bundle.main.url(forResource: ModelConstants.modelResourceName, withExtension: ModelConstants.compiledExtension) ??
            Bundle.main.url(forResource: ModelConstants.modelResourceName, withExtension: ModelConstants.packageExtension) ??
            Bundle.main.url(forResource: ModelConstants.modelResourceName, withExtension: ModelConstants.packageExtension, subdirectory: ModelConstants.resourcesSubdirectory)
        
        guard let modelURL else {
            throw NSError(domain: "Inference", code: 10, userInfo: [NSLocalizedDescriptionKey: "\(ModelConstants.modelResourceName).\(ModelConstants.packageExtension) not found in app bundle. Add it to the target."])
        }
        
        // Configure for maximum performance testing
        setenv(ModelConstants.computeUnitsEnvVar, ModelConstants.allComputeUnits, 1)
        runner = try DecoderOnly5sRunner(mlpackageURL: modelURL)
    }

    /// Tensor shape constants for model input validation and warm-up
    private enum TensorShapes {
        /// ASR token embeddings: (batch, tokens, height, sequence)
        static let asr = [1, 512, 1, 200]
        
        /// F0 curve (pitch): (batch, channels, height, time_frames)
        static let f0Curve = [1, 1, 1, 400]
        
        /// Noise control: (batch, channels, height, time_frames)
        static let noise = [1, 1, 1, 400]
        
        /// Speaker embedding: (batch, embedding_dim)
        static let speaker = [1, 128]
        
        /// Total elements for zero-filled fallback tensors
        static let asrElements = 1 * 512 * 1 * 200    // 102,400 elements
        static let f0Elements = 400                     // 400 elements  
        static let noiseElements = 400                  // 400 elements
        static let speakerElements = 128                // 128 elements
    }
    
    /// Fixture loading constants
    private enum FixtureConstants {
        /// Small test fixture for warm-up inference (minimal computational load)
        static let warmupFixtureName = "fixture_hi"
        
        /// JSON file extension for fixture files
        static let fixtureExtension = "json"
    }

    /// Execute warm-up inference to initialize Core ML compute units and memory paths.
    ///
    /// This method performs a complete inference pass using either bundled test fixtures
    /// or zero-filled tensors to "prime" the Core ML model and underlying compute units.
    /// Warm-up is essential for accurate performance measurement as it eliminates
    /// cold-start overhead from timing results.
    ///
    /// **Warm-up Strategy:**
    /// 1. Attempt to load lightweight fixture from bundle (preferred for realism)
    /// 2. Fall back to zero-filled tensors if fixture unavailable (guaranteed to work)
    /// 3. Execute full inference pipeline to initialize all computation paths
    /// 4. Discard results (warm-up only, not for evaluation)
    ///
    /// **Performance Benefits:**
    /// - Initializes ANE/GPU memory allocations and kernel compilation
    /// - Pre-loads Core ML model weights into active memory
    /// - Establishes optimal memory layout for subsequent inferences
    /// - Eliminates JIT compilation overhead from timing measurements
    ///
    /// **Cross-Module Dependencies:**
    /// - FixtureLoader: Loads pre-defined test inputs from JSON files
    /// - ArrayFactory: Converts raw data to MLMultiArray format
    /// - DecoderOnly5sRunner: Executes Core ML inference
    ///
    /// **Called by:**
    /// - init(): During view model initialization to prepare for user testing
    ///
    /// **Throws:**
    /// - Core ML errors: If model inference fails during warm-up
    /// - Array conversion errors: If tensor creation fails
    private func runWarmUp() throws {
        // Strategy 1: Use realistic fixture data if available
        if let _ = Bundle.main.url(forResource: FixtureConstants.warmupFixtureName, withExtension: FixtureConstants.fixtureExtension),
           let fx = try? FixtureLoader.loadFixture(named: FixtureConstants.warmupFixtureName) {
            let asr = try ArrayFactory.makeArray(fx.asr, shape: fx.shapes["asr"] ?? TensorShapes.asr)
            let f0  = try ArrayFactory.makeArray(fx.f0_curve, shape: fx.shapes["f0_curve"] ?? TensorShapes.f0Curve)
            let n   = try ArrayFactory.makeArray(fx.n, shape: fx.shapes["n"] ?? TensorShapes.noise)
            let s   = try ArrayFactory.makeArray(fx.s, shape: fx.shapes["s"] ?? TensorShapes.speaker)
            _ = try runner!.predict(asr: asr, f0: f0, n: n, s: s)
            return
        }
        
        // Strategy 2: Fallback to zero-filled tensors (guaranteed to work)
        let zerosASR = [Float](repeating: 0, count: TensorShapes.asrElements)
        let zerosF0  = [Float](repeating: 0, count: TensorShapes.f0Elements)
        let zerosN   = [Float](repeating: 0, count: TensorShapes.noiseElements)
        let zerosS   = [Float](repeating: 0, count: TensorShapes.speakerElements)
        
        let asr = try ArrayFactory.makeArray(zerosASR, shape: TensorShapes.asr)
        let f0  = try ArrayFactory.makeArray(zerosF0, shape: TensorShapes.f0Curve)
        let n   = try ArrayFactory.makeArray(zerosN, shape: TensorShapes.noise)
        let s   = try ArrayFactory.makeArray(zerosS, shape: TensorShapes.speaker)
        _ = try runner!.predict(asr: asr, f0: f0, n: n, s: s)
    }

    /// Update status text on main thread for UI consistency.
    ///
    /// This utility method ensures all status updates are properly dispatched to the main
    /// thread, maintaining SwiftUI's threading requirements and preventing UI update warnings.
    ///
    /// **Threading Safety:**
    /// - Always dispatches to DispatchQueue.main regardless of calling thread
    /// - Prevents "UI updates on background thread" warnings
    /// - Ensures immediate UI responsiveness for status changes
    ///
    /// **Called by:**
    /// - init(): During warm-up progress updates
    /// - runTest(): During inference progress updates
    /// - Error handling paths: For displaying error messages
    ///
    /// **Parameters:**
    /// - text: Status message to display in UI
    private func statusOnMain(_ text: String) {
        DispatchQueue.main.async { self.statusText = text }
    }
}
