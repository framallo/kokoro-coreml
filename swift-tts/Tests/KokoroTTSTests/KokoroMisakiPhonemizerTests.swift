import XCTest
@testable import KokoroTTS

final class KokoroMisakiPhonemizerTests: XCTestCase {
    /// Environment variable required before tests invoke Misaki/MLX at runtime.
    private static let runtimeTestsFlag = "KOKORO_RUN_MISAKI_RUNTIME_TESTS"

    /// Skips Misaki runtime tests unless the caller provided an app-style MLX environment.
    ///
    /// SwiftPM shell tests compile MisakiSwift but do not build or expose the
    /// `mlx-swift_Cmlx.bundle` shader resources required by MLX. Phase 1 uses
    /// the xcodebuild-built probe for runtime proof and keeps these tests as an
    /// explicit opt-in guard for developers running inside that environment.
    private func skipUnlessRuntimeProbeEnabled() throws {
        guard ProcessInfo.processInfo.environment[Self.runtimeTestsFlag] == "1" else {
            throw XCTSkip("Set \(Self.runtimeTestsFlag)=1 only when MLX shader resources are available.")
        }
    }

    /// Verifies that the default U.S. English Misaki path returns usable phonemes.
    func testUSPhonemizerReturnsNonEmptyPhonemes() throws {
        try skipUnlessRuntimeProbeEnabled()
        let phonemizer = KokoroMisakiPhonemizer()

        let result = try phonemizer.phonemize("Hello world.")

        XCTAssertFalse(result.phonemes.isEmpty)
        XCTAssertEqual(result.utf16Count, result.phonemes.utf16.count)
    }

    /// Verifies that the British English path is wired and produces phonemes.
    func testBritishPhonemizerReturnsNonEmptyPhonemes() throws {
        try skipUnlessRuntimeProbeEnabled()
        let phonemizer = KokoroMisakiPhonemizer(british: true)

        let result = try phonemizer.phonemize("I live in Reading.")

        XCTAssertFalse(result.phonemes.isEmpty)
        XCTAssertEqual(result.utf16Count, result.phonemes.utf16.count)
    }

    /// Verifies the raw-text SDK package can see the low-level pipeline package.
    func testFacadeExposesPipelineTokenBoundary() {
        XCTAssertEqual(KokoroTTS.maxCallerChunkTokens, 450)
    }

    /// Verifies the SDK default diagnostics do not expose raw caller payloads.
    func testDiagnosticsDefaultIsPrivacySafe() {
        let policy = KokoroDiagnosticsPolicy.privacySafeDefault

        XCTAssertFalse(policy.includesRawText)
        XCTAssertFalse(policy.includesPhonemes)
        XCTAssertFalse(policy.persistsRawPayloads)
    }

    /// Verifies explicit debug diagnostics may expose text without persistence.
    func testDiagnosticsDebugPayloadsStillDisablePersistence() {
        let policy = KokoroDiagnosticsPolicy.interactiveDebugPayloads

        XCTAssertTrue(policy.includesRawText)
        XCTAssertTrue(policy.includesPhonemes)
        XCTAssertFalse(policy.persistsRawPayloads)
    }
}
