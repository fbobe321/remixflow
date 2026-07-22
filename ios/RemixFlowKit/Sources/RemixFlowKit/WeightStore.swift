import Foundation
import MLX

/// Loads ACE-Step v1.5 weights (safetensors) into MLXArrays and optionally
/// applies 4-bit group quantization to the linear matrices.
///
/// The measured quality (see `mobile/mlx/quantize/`) is: **4-bit, group size 32
/// → 0.9936 audio mel-correlation vs fp16**. The VAE is kept fp16 (quality-
/// critical, small). SDEdit's re-anchoring hides per-step quant error.
public final class WeightStore {
    public private(set) var weights: [String: MLXArray] = [:]

    public init(directory: URL) throws {
        // Each component lives in its own subfolder of the HF snapshot:
        // transformer/, vae/, text_encoder/, condition_encoder/.
        for sub in ["transformer", "vae", "text_encoder", "condition_encoder"] {
            let dir = directory.appendingPathComponent(sub)
            guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else { continue }
            for f in files where f.pathExtension == "safetensors" {
                let arrays = try MLX.loadArrays(url: f)
                for (k, v) in arrays { weights["\(sub)/\(k)"] = v.asType(.float16) }
            }
        }
    }

    public subscript(_ key: String) -> MLXArray { weights[key]! }
    public func has(_ key: String) -> Bool { weights[key] != nil }

    /// Quantize-in-place the 2D linear weights of a component to `bits`/`groupSize`.
    /// Keeps norms (1D), convs (3D), and embeddings unquantized. Returns dequantized
    /// fp16 for the reference build; swap for MLX packed quantized matmul to save RAM.
    public func quantizeLinears(prefix: String, bits: Int = 4, groupSize: Int = 32) {
        for (k, v) in weights where k.hasPrefix(prefix) && k.hasSuffix(".weight")
            && v.ndim == 2 && v.dim(1) % groupSize == 0 && !k.contains("norm") {
            weights[k] = Self.quantizeDequantize(v, bits: bits, groupSize: groupSize)
        }
    }

    /// Group-affine quantize→dequantize matching the measured scheme (MLX's k-bit
    /// matmul is numerically equivalent to this). Replace with packed
    /// `MLX.quantized(...)` + `quantizedMatmul` for the memory win on-device.
    static func quantizeDequantize(_ w: MLXArray, bits: Int, groupSize: Int) -> MLXArray {
        let shape = w.shape
        let flat = w.reshaped([-1, groupSize]).asType(.float32)
        let lo = MLX.min(flat, axis: 1, keepDims: true)
        let hi = MLX.max(flat, axis: 1, keepDims: true)
        let n = Float((1 << bits) - 1)
        var scale = (hi - lo) / n
        scale = MLX.where(scale .== 0, MLXArray(Float(1)), scale)
        let q = MLX.clip(MLX.round((flat - lo) / scale), min: 0, max: n)
        return (q * scale + lo).reshaped(shape).asType(.float16)
    }
}
