// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "KokoroPhase2",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(name: "KokoroPhase2", targets: ["KokoroPhase2"]),
        .executable(name: "kokoro-phase2-cli", targets: ["kokoro-phase2-cli"])
    ],
    targets: [
        .target(
            name: "KokoroPhase2",
            dependencies: []
        ),
        .executableTarget(
            name: "kokoro-phase2-cli",
            dependencies: ["KokoroPhase2"]
        )
    ]
)
