import Foundation
import MLX

/// Shared ops used across the ported ACE-Step components. These mirror the
/// framework-free NumPy spec in `mobile/mlx/**` (which is validated bit-close to
/// PyTorch), so each Swift model is a mechanical translation of proven code.
enum Ops {

    /// RMSNorm over the last axis.  y = x * rsqrt(mean(x^2) + eps) * weight
    static func rmsNorm(_ x: MLXArray, _ weight: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let v = MLX.mean(x * x, axis: -1, keepDims: true)
        return x * MLX.rsqrt(v + eps) * weight
    }

    /// Dense linear.  weight is stored [out, in]; y = x @ weightᵀ (+ bias)
    static func linear(_ x: MLXArray, _ weight: MLXArray, bias: MLXArray? = nil) -> MLXArray {
        let y = x.matmul(weight.transposed())
        return bias.map { y + $0 } ?? y
    }

    static func silu(_ x: MLXArray) -> MLXArray { x * MLX.sigmoid(x) }

    /// Snake activation (Oobleck VAE): x + (1/(β+1e-9)) * sin(αx)^2, log-scale α,β.
    static func snake(_ x: MLXArray, alphaRaw: MLXArray, betaRaw: MLXArray) -> MLXArray {
        let a = MLX.exp(alphaRaw.reshaped([-1]))
        let b = MLX.exp(betaRaw.reshaped([-1]))
        return x + (1.0 / (b + 1e-9)) * MLX.pow(MLX.sin(a * x), 2)
    }

    /// RoPE frequencies (Qwen3 "cat-halves" layout): cos/sin are [seq, headDim],
    /// first & second halves identical.
    static func ropeFreqs(seq: Int, headDim: Int, theta: Float) -> (MLXArray, MLXArray) {
        let idx = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
        let freqs = 1.0 / MLX.pow(theta, idx / Float(headDim))
        let pos = MLXArray((0..<seq).map { Float($0) }).reshaped([seq, 1])
        let ang = pos * freqs.reshaped([1, headDim / 2])
        let cos = MLX.concatenated([MLX.cos(ang), MLX.cos(ang)], axis: -1)
        let sin = MLX.concatenated([MLX.sin(ang), MLX.sin(ang)], axis: -1)
        return (cos, sin)
    }

    /// Apply RoPE to x [B, L, H, D] with cos/sin [L, D] (rotate-half on last dim).
    static func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let c = cos.reshaped([1, cos.dim(0), 1, d])
        let s = sin.reshaped([1, sin.dim(0), 1, d])
        let x1 = x[.ellipsis, 0 ..< (d / 2)]
        let x2 = x[.ellipsis, (d / 2) ..< d]
        let rot = MLX.concatenated([-x2, x1], axis: -1)
        return x * c + rot * s
    }

    /// Scaled dot-product attention with an optional additive mask. GQA is handled
    /// by the caller repeating K/V heads. q,k,v: [B, H, L, D].
    static func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                          scale: Float, mask: MLXArray? = nil) -> MLXArray {
        var scores = q.matmul(k.transposed(0, 1, 3, 2)) * scale
        if let mask { scores = scores + mask }
        return MLX.softmax(scores, axis: -1).matmul(v)
    }
}
