// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "RemixFlowKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "RemixFlowKit", targets: ["RemixFlowKit"]),
    ],
    dependencies: [
        // Apple's MLX for Swift (Metal-accelerated arrays + NN + quantization).
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.18.0"),
    ],
    targets: [
        .target(
            name: "RemixFlowKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
    ]
)
