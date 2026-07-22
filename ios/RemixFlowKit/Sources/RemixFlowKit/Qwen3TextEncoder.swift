import Foundation
import MLX

/// Qwen3-0.6B text encoder → last_hidden_state. Translated from the parity-proven
/// `mobile/mlx/textenc/qwen3_mlx.py` (3.5e-6 vs PyTorch).
public final class Qwen3TextEncoder {
    private let w: [String: MLXArray]
    let heads = 16, kv = 8, headDim = 128, nLayers = 28
    let theta: Float = 1_000_000, eps: Float = 1e-6

    public init(weights: [String: MLXArray]) { self.w = weights }

    private func attn(_ hn: MLXArray, _ i: Int, _ cos: MLXArray, _ sin: MLXArray, _ mask: MLXArray) -> MLXArray {
        let p = "layers.\(i).self_attn"
        let B = hn.dim(0), L = hn.dim(1)
        var q = Ops.linear(hn, w["\(p).q_proj.weight"]!).reshaped([B, L, heads, headDim])
        var k = Ops.linear(hn, w["\(p).k_proj.weight"]!).reshaped([B, L, kv, headDim])
        var v = Ops.linear(hn, w["\(p).v_proj.weight"]!).reshaped([B, L, kv, headDim])
        q = Ops.rmsNorm(q, w["\(p).q_norm.weight"]!, eps: eps)
        k = Ops.rmsNorm(k, w["\(p).k_norm.weight"]!, eps: eps)
        q = Ops.applyRope(q, cos: cos, sin: sin); k = Ops.applyRope(k, cos: cos, sin: sin)
        let rep = heads / kv
        k = MLX.repeated(k, count: rep, axis: 2); v = MLX.repeated(v, count: rep, axis: 2)
        let o = Ops.attention(q.transposed(0, 2, 1, 3), k.transposed(0, 2, 1, 3), v.transposed(0, 2, 1, 3),
                              scale: powf(Float(headDim), -0.5), mask: mask)
        return Ops.linear(o.transposed(0, 2, 1, 3).reshaped([B, L, heads * headDim]), w["\(p).o_proj.weight"]!)
    }

    /// `ids`: Int32 token ids [B, L]. Returns last_hidden_state [B, L, 1024].
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var h = w["embed_tokens.weight"]![ids]                 // gather rows
        let B = h.dim(0), L = h.dim(1)
        let (cos, sin) = Ops.ropeFreqs(seq: L, headDim: headDim, theta: theta)
        let idx = MLXArray((0..<L).map { Float($0) })
        let causal = MLX.where(idx.reshaped([L, 1]) .>= idx.reshaped([1, L]),
                               MLXArray(Float(0)), MLXArray(Float(-1e30))).reshaped([1, 1, L, L])
        _ = B
        for i in 0..<nLayers {
            let hn = Ops.rmsNorm(h, w["layers.\(i).input_layernorm.weight"]!, eps: eps)
            h = h + attn(hn, i, cos, sin, causal)
            let hn2 = Ops.rmsNorm(h, w["layers.\(i).post_attention_layernorm.weight"]!, eps: eps)
            let ff = Ops.linear(Ops.silu(Ops.linear(hn2, w["layers.\(i).mlp.gate_proj.weight"]!))
                * Ops.linear(hn2, w["layers.\(i).mlp.up_proj.weight"]!), w["layers.\(i).mlp.down_proj.weight"]!)
            h = h + ff
        }
        return Ops.rmsNorm(h, w["norm.weight"]!, eps: eps)
    }

    /// Just the token embedding lookup (used for lyrics, which skip the transformer).
    public func embed(_ ids: MLXArray) -> MLXArray { w["embed_tokens.weight"]![ids] }
}
