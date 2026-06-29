import Foundation
import KokoroTTS

/// Runs raw-text local synthesis against a generated SDK bundle.
func runSmoke() async throws {
    var args = Array(CommandLine.arguments.dropFirst())
    let cpuOnly = args.contains("--cpu-only")
    args.removeAll { $0 == "--cpu-only" }
    guard let bundlePath = args.first else {
        FileHandle.standardError.write(Data("usage: kokoro-sdk-smoke [--cpu-only] <bundle-root> [text]\n".utf8))
        throw ExitCode.failure
    }
    let text = args.dropFirst().first ?? "Hello world."
    let tts = try await KokoroTTS.load(
        resources: .directory(URL(fileURLWithPath: bundlePath)),
        computePolicy: cpuOnly ? .cpuOnly : .gistDefault
    )
    let audio = try await tts.synthesize(text, voice: .afHeart)
    print("samples=\(audio.samples.count) sampleRate=\(audio.sampleRate) duration=\(audio.durationSeconds)")
}

/// Process exit sentinel for invalid CLI usage.
enum ExitCode: Error {
    /// Invalid command-line invocation.
    case failure
}

try await runSmoke()
