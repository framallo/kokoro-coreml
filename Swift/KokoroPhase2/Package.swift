// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KokoroPhase2",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        // Library exposing a simple TTS API for Phase 3
        .library(name: "KokoroTTS", targets: ["KokoroTTS"]),
        // Minimal executable that consumes the library
        .executable(name: "KokoroPhase2", targets: ["KokoroPhase2"]),
    ],
    dependencies: [],
    targets: [
        // Core library with public API and synthesis pipeline
        .target(
            name: "KokoroTTS",
            path: "Sources/KokoroTTS"
        ),
        .executableTarget(
            name: "KokoroPhase2",
            dependencies: ["KokoroTTS"],
            resources: []
        )
    ]
)
