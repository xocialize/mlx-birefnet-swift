// swift-tools-version: 6.2
// mlx-birefnet-swift — BiRefNet foreground matting on Swift/MLX, wrapped as an MLXEngine `matting`
// ModelPackage (contract 1.5.0). The `BiRefNet` core is vendored from mnmly/mlx-swift-BiRefNet (MIT,
// Hiroaki Yamane) — see NOTICE; the `MLXBiRefNet` wrapper adds the conformant `BiRefNetPackage`.
import PackageDescription

let mlxCore: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
    .product(name: "MLXRandom", package: "mlx-swift"),
]

let package = Package(
    name: "mlx-birefnet-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "BiRefNet", targets: ["BiRefNet"]),               // vendored MLX core
        .library(name: "MLXBiRefNet", targets: ["MLXBiRefNet"]),         // engine-consumable wrapper
        .executable(name: "birefnet-convert", targets: ["BiRefNetConvert"]), // PyTorch/HF → MLX weight converter
        .executable(name: "birefnet-smoke", targets: ["BiRefNetSmoke"]),     // real-forward gate over BiRefNetPackage
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),  // Hub weight download
        // Remote engine (0.10.0 supersets the 1.5.0 matting contract). MUST be the remote URL — a
        // local .package(path:) collides with the same package identity arriving by URL from the
        // sibling wrappers in a consuming app's graph, jamming engine resolution. (APP-VALIDATION
        // BRIDGE-026 / mlxengine-forge.)
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.10.0"),
    ],
    targets: [
        // Vendored core — kept in Swift 5 language mode to avoid churn on the donor's concurrency.
        .target(
            name: "BiRefNet",
            dependencies: mlxCore,
            path: "Sources/BiRefNet",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        // Engine wrapper — new code; the conformant `matting` ModelPackage.
        .target(
            name: "MLXBiRefNet",
            dependencies: [
                "BiRefNet",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/MLXBiRefNet"
        ),
        .executableTarget(
            name: "BiRefNetConvert",
            dependencies: ["BiRefNet", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/Convert",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "BiRefNetSmoke",
            dependencies: [
                "MLXBiRefNet",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Sources/Smoke"
        ),
        .testTarget(
            name: "MLXBiRefNetTests",
            dependencies: ["MLXBiRefNet"],
            path: "Tests/MLXBiRefNetTests"
        ),
    ]
)
