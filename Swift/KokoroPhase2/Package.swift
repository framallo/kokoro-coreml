// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KokoroPhase2",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "KokoroPhase2", targets: ["KokoroPhase2"]),
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "KokoroPhase2",
            resources: []
        )
    ]
)
