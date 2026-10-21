// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Reelpack",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Reelpack", targets: ["App"]),
        .executable(name: "reelpack-cli", targets: ["CLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "App",
            dependencies: ["Shared"],
            path: "App"
        ),
        .executableTarget(
            name: "CLI",
            dependencies: [
                "Shared",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "CLI"
        ),
        .target(
            name: "Shared",
            path: "Shared"
        ),
        .testTarget(
            name: "SharedTests",
            dependencies: ["Shared"],
            path: "Tests/SharedTests"
        ),
    ]
)
