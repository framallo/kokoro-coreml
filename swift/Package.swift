// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift Package Manager required to build this package.

import PackageDescription

let package = Package(
    name: "KokoroPipeline",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "KokoroPipeline", targets: ["KokoroPipeline"]),
    ],
    targets: [
        .target(
            name: "KokoroPipeline",
            path: "Sources/KokoroPipeline"
        ),
        .testTarget(
            name: "KokoroPipelineTests",
            dependencies: ["KokoroPipeline"],
            path: "Tests/KokoroPipelineTests"
        ),
    ]
)
