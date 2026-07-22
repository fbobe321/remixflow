import Foundation
import MLX

/// ACE-Step v1.5 Diffusion Transformer (4.17 B). Translated from the parity-
/// proven `mobile/mlx/dit/dit_mlx.py` (8.8e-6 vs PyTorch). Its linear weights are
/// the ones quantized to 4-bit (WeightStore.quantizeLinears("transformer/…")).
public final class DiT {
    private let w: [String: MLXArray]
    let hidden = 2560, heads = 32, kv = 8, headDim = 128
    let patch = 2, nLayers = 32, slidingWindow = 128
    let theta: Float = 1_000_000, eps: Float = 1e-6
    // layer i: sliding if (i+1) odd, else full — matches the checkpoint config.
    private func isSliding(_ i: Int) -> Bool { (i + 1) % 2 == 1 }

    public init(transformerWeights: [String: MLXArray]) { self.w = transformerWeights }

    private func timestepEmbedding(_ t: MLXArray, _ p: String) -> (MLXArray, MLXArray) {
        // sinusoid(t*1000, dim 256, flip_sin_to_cos) → linear_1 → silu → linear_2
        let half = 128
        let expo = MLXArray((0..<half).map { -logf(10000) * Float($0) / Float(half) })
        let args = (t * 1000).reshaped([-1, 1]) * MLX.exp(expo).reshaped([1, half])
        let tf = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        var e = Ops.linear(tf, w["\(p).linear_1.weight"]!, bias: w["\(p).linear_1.bias"])
        e = Ops.silu(e)
        e = Ops.linear(e, w["\(p).linear_2.weight"]!, bias: w["\(p).linear_2.bias"])
        let proj = Ops.linear(Ops.silu(e), w["\(p).time_proj.weight"]!, bias: w["\(p).time_proj.bias"])
        return (e, proj.reshaped([proj.dim(0), 6, hidden]))
    }

    private func attn(_ x: MLXArray, _ kvIn: MLXArray, _ p: String,
                      _ cos: MLXArray, _ sin: MLXArray, _ mask: MLXArray?, cross: Bool) -> MLXArray {
        let B = x.dim(0), L = x.dim(1), Lk = kvIn.dim(1)
        var q = Ops.linear(x, w["\(p).to_q.weight"]!).reshaped([B, L, heads, headDim])
        var k = Ops.linear(kvIn, w["\(p).to_k.weight"]!).reshaped([B, Lk, kv, headDim])
        var v = Ops.linear(kvIn, w["\(p).to_v.weight"]!).reshaped([B, Lk, kv, headDim])
        q = Ops.rmsNorm(q, w["\(p).norm_q.weight"]!, eps: eps)
        k = Ops.rmsNorm(k, w["\(p).norm_k.weight"]!, eps: eps)
        if !cross { q = Ops.applyRope(q, cos: cos, sin: sin); k = Ops.applyRope(k, cos: cos, sin: sin) }
        let rep = heads / kv
        k = MLX.repeated(k, count: rep, axis: 2); v = MLX.repeated(v, count: rep, axis: 2)
        let qh = q.transposed(0, 2, 1, 3), kh = k.transposed(0, 2, 1, 3), vh = v.transposed(0, 2, 1, 3)
        let o = Ops.attention(qh, kh, vh, scale: powf(Float(headDim), -0.5), mask: mask)
        let flat = o.transposed(0, 2, 1, 3).reshaped([B, L, heads * headDim])
        return Ops.linear(flat, w["\(p).to_out.0.weight"]!)
    }

    private func block(_ x0: MLXArray, _ i: Int, _ proj: MLXArray, _ enc: MLXArray,
                       _ cos: MLXArray, _ sin: MLXArray, _ mask: MLXArray?) -> MLXArray {
        var x = x0
        let pfx = "layers.\(i)"
        let six = w["\(pfx).scale_shift_table"]! + proj              // [B,6,H]
        func part(_ j: Int) -> MLXArray { six[0..., j ..< (j + 1), 0...] }
        let (sh, sc, g) = (part(0), part(1), part(2))
        let (csh, csc, cg) = (part(3), part(4), part(5))
        var nh = Ops.rmsNorm(x, w["\(pfx).self_attn_norm.weight"]!, eps: eps) * (1 + sc) + sh
        x = x + attn(nh, nh, "\(pfx).self_attn", cos, sin, mask, cross: false) * g
        nh = Ops.rmsNorm(x, w["\(pfx).cross_attn_norm.weight"]!, eps: eps)
        x = x + attn(nh, enc, "\(pfx).cross_attn", cos, sin, nil, cross: true)
        nh = Ops.rmsNorm(x, w["\(pfx).mlp_norm.weight"]!, eps: eps) * (1 + csc) + csh
        let ff = Ops.linear(Ops.silu(Ops.linear(nh, w["\(pfx).mlp.gate_proj.weight"]!))
            * Ops.linear(nh, w["\(pfx).mlp.up_proj.weight"]!), w["\(pfx).mlp.down_proj.weight"]!)
        return x + ff * cg
    }

    private func slidingMask(_ seq: Int) -> MLXArray {
        let idx = MLXArray((0..<seq).map { Float($0) })
        let diff = idx.reshaped([seq, 1]) - idx.reshaped([1, seq])
        return MLX.where(MLX.abs(diff) .<= Float(slidingWindow), MLXArray(Float(0)), MLXArray(Float(-1e30)))
            .reshaped([1, 1, seq, seq])
    }

    /// Predict the flow-matching velocity for noised latents `hidden` at `t`.
    public func callAsFunction(hidden: MLXArray, context: MLXArray, enc encIn: MLXArray,
                               t: MLXArray, tR: MLXArray) -> MLXArray {
        let (tembT, projT) = timestepEmbedding(t, "time_embed")
        let (tembR, projR) = timestepEmbedding(t - tR, "time_embed_r")
        let temb = tembT + tembR, proj = projT + projR
        var x = MLX.concatenated([context, hidden], axis: -1)       // [B,T,192]
        let orig = x.dim(1)
        if x.dim(1) % patch != 0 { x = MLX.padded(x, widths: [.init((0, 0)), .init((0, patch - x.dim(1) % patch)), .init((0, 0))]) }
        let B = x.dim(0), C = x.dim(2), seq = x.dim(1) / patch
        let patches = x.reshaped([B, seq, patch, C]).transposed(0, 1, 3, 2).reshaped([B, seq, C * patch])
        x = patches.matmul(w["proj_in_conv.weight"]!.reshaped([hidden, C * patch]).transposed()) + w["proj_in_conv.bias"]!
        let enc = Ops.linear(encIn, w["condition_embedder.weight"]!, bias: w["condition_embedder.bias"])
        let (cos, sin) = Ops.ropeFreqs(seq: seq, headDim: headDim, theta: theta)
        let smask = slidingMask(seq)
        for i in 0..<nLayers {
            x = block(x, i, proj, enc, cos, sin, isSliding(i) ? smask : nil)
        }
        let ss = w["scale_shift_table"]! + temb.reshaped([temb.dim(0), 1, hidden])
        let sh = ss[0..., 0 ..< 1, 0...], sc = ss[0..., 1 ..< 2, 0...]
        x = Ops.rmsNorm(x, w["norm_out.weight"]!, eps: eps) * (1 + sc) + sh
        let pw = w["proj_out_conv.weight"]!                          // [H, acoustic, patch]
        let acoustic = pw.dim(1)
        var outs: [MLXArray] = []
        for j in 0..<patch { outs.append(x.matmul(pw[0..., 0..., j]) + w["proj_out_conv.bias"]!) }
        let y = MLX.stacked(outs, axis: 2).reshaped([B, seq * patch, acoustic])
        return y[0..., 0 ..< orig, 0...]
    }
}
