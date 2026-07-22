import Foundation
import MLX

/// AceStepConditionEncoder — fuses text + lyric + timbre into the DiT's cross-
/// attention sequence. Full translation of the parity-proven
/// `mobile/mlx/condenc/condenc_mlx.py` (3.4e-7 vs PyTorch, mask exact).
///
/// Encoder block = input_ln → GQA(QK-RMSNorm + RoPE) → residual → post_ln →
/// SwiGLU → residual. Lyric encoder is bidirectional (padding + sliding masks);
/// timbre encoder pools the first token per segment and unpacks by an order mask;
/// the two-stage pack sorts valid tokens to the left with a unique key.
public final class ConditionEncoder {
    private let w: [String: MLXArray]
    let hidden = 2048, heads = 16, kv = 8, headDim = 128, slidingWindow = 128
    let nLyric = 8, nTimbre = 4
    let theta: Float = 1_000_000, eps: Float = 1e-6

    public init(weights: [String: MLXArray]) { self.w = weights }

    // MARK: - shared block

    private func attn(_ x: MLXArray, _ p: String, _ cos: MLXArray, _ sin: MLXArray, _ mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var q = Ops.linear(x, w["\(p).to_q.weight"]!).reshaped([B, L, heads, headDim])
        var k = Ops.linear(x, w["\(p).to_k.weight"]!).reshaped([B, L, kv, headDim])
        var v = Ops.linear(x, w["\(p).to_v.weight"]!).reshaped([B, L, kv, headDim])
        q = Ops.rmsNorm(q, w["\(p).norm_q.weight"]!, eps: eps)
        k = Ops.rmsNorm(k, w["\(p).norm_k.weight"]!, eps: eps)
        q = Ops.applyRope(q, cos: cos, sin: sin); k = Ops.applyRope(k, cos: cos, sin: sin)
        let rep = heads / kv
        k = MLX.repeated(k, count: rep, axis: 2); v = MLX.repeated(v, count: rep, axis: 2)
        let o = Ops.attention(q.transposed(0, 2, 1, 3), k.transposed(0, 2, 1, 3), v.transposed(0, 2, 1, 3),
                              scale: powf(Float(headDim), -0.5), mask: mask)
        return Ops.linear(o.transposed(0, 2, 1, 3).reshaped([B, L, heads * headDim]), w["\(p).to_out.0.weight"]!)
    }

    private func layer(_ x0: MLXArray, _ p: String, _ cos: MLXArray, _ sin: MLXArray, _ mask: MLXArray?) -> MLXArray {
        var x = x0
        let xn = Ops.rmsNorm(x, w["\(p).input_layernorm.weight"]!, eps: eps)
        x = x + attn(xn, "\(p).self_attn", cos, sin, mask)
        let xn2 = Ops.rmsNorm(x, w["\(p).post_attention_layernorm.weight"]!, eps: eps)
        let ff = Ops.linear(Ops.silu(Ops.linear(xn2, w["\(p).mlp.gate_proj.weight"]!))
            * Ops.linear(xn2, w["\(p).mlp.up_proj.weight"]!), w["\(p).mlp.down_proj.weight"]!)
        return x + ff * 1  // encoder blocks have no gate
    }

    // MARK: - masks

    private func padMask(_ attnMask: MLXArray) -> MLXArray {   // [B,seq] → [B,1,1,seq] additive
        MLX.where(attnMask.reshaped([attnMask.dim(0), 1, 1, attnMask.dim(1)]) .> 0,
                  MLXArray(Float(0)), MLXArray(Float(-1e30)))
    }
    private func slidingMask(_ seq: Int) -> MLXArray {
        let idx = MLXArray((0..<seq).map { Float($0) })
        let diff = idx.reshaped([seq, 1]) - idx.reshaped([1, seq])
        return MLX.where(MLX.abs(diff) .<= Float(slidingWindow), MLXArray(Float(0)), MLXArray(Float(-1e30)))
            .reshaped([1, 1, seq, seq])
    }

    // MARK: - lyric & timbre encoders

    private func lyricEncoder(_ lyric: MLXArray, _ lyricMask: MLXArray) -> MLXArray {
        let p = "lyric_encoder"
        var x = Ops.linear(lyric, w["\(p).embed_tokens.weight"]!, bias: w["\(p).embed_tokens.bias"])
        let seq = x.dim(1)
        let (cos, sin) = Ops.ropeFreqs(seq: seq, headDim: headDim, theta: theta)
        let pad = padMask(lyricMask)
        let full = pad
        let band = MLX.minimum(slidingMask(seq), MLXArray(Float(0))) + pad
        for i in 0..<nLyric {
            x = layer(x, "\(p).layers.\(i)", cos, sin, (i + 1) % 2 == 1 ? band : full)
        }
        return Ops.rmsNorm(x, w["\(p).norm.weight"]!, eps: eps)
    }

    /// refer: [Nseg, St, 64]; order: batch id per segment (host).
    private func timbreEncoder(_ refer: MLXArray, _ order: [Int]) -> (MLXArray, MLXArray) {
        let p = "timbre_encoder"
        var x = Ops.linear(refer, w["\(p).embed_tokens.weight"]!, bias: w["\(p).embed_tokens.bias"])
        let seq = x.dim(1)
        let (cos, sin) = Ops.ropeFreqs(seq: seq, headDim: headDim, theta: theta)
        let band = slidingMask(seq)
        for i in 0..<nTimbre {
            x = layer(x, "\(p).layers.\(i)", cos, sin, (i + 1) % 2 == 1 ? band : nil)
        }
        x = Ops.rmsNorm(x, w["\(p).norm.weight"]!, eps: eps)
        let pooled = x[0..., 0, 0...]                     // [Nseg, hidden]
        return unpack(pooled, order)
    }

    /// Reorganize [N, d] packed embeddings into [B, maxCount, d] by `order`, + a
    /// [B, maxCount] presence mask. Layout computed host-side (order is small).
    private func unpack(_ pooled: MLXArray, _ order: [Int]) -> (MLXArray, MLXArray) {
        let N = order.count
        let B = (order.max() ?? -1) + 1
        var counts = [Int](repeating: 0, count: B)
        for o in order { counts[o] += 1 }
        let mc = counts.max() ?? 1
        var src = [Int32](repeating: 0, count: B * mc)
        var maskH = [Float](repeating: 0, count: B * mc)
        var seen = [Int](repeating: 0, count: B)
        for i in 0..<N {
            let b = order[i]; let slot = seen[b]
            src[b * mc + slot] = Int32(i); maskH[b * mc + slot] = 1; seen[b] += 1
        }
        let gathered = pooled[MLXArray(src)].reshaped([B, mc, pooled.dim(1)])
        let mask = MLXArray(maskH).reshaped([B, mc])
        return (gathered * mask.reshaped([B, mc, 1]), mask)
    }

    // MARK: - packing (valid tokens first, unique sort key)

    private func pack(_ h1: MLXArray, _ h2: MLXArray, _ m1: MLXArray, _ m2: MLXArray) -> (MLXArray, MLXArray) {
        let hc = MLX.concatenated([h1, h2], axis: 1)
        let mc = MLX.concatenated([m1, m2], axis: 1)
        let B = hc.dim(0), L = hc.dim(1), D = hc.dim(2)
        let pos = MLXArray((0..<L).map { Float($0) }).reshaped([1, L])
        let key = mc.asType(.float32) * Float(L) - pos
        let idx = MLX.argSort(-key, axis: 1)                        // unique key → stable-independent
        let hl = MLX.takeAlong(hc, idx.reshaped([B, L, 1]).broadcast(to: [B, L, D]), axis: 1)
        let lengths = MLX.sum(mc, axis: 1)                          // [B]
        let ar = MLXArray((0..<L).map { Float($0) }).reshaped([1, L])
        let nm = (ar .< lengths.reshaped([B, 1])).asType(.int32)
        return (hl, nm)
    }

    /// Full conditioning. Returns (encoder_hidden_states [B,S,2048], mask [B,S]).
    /// text/lyric: [B,L,1024]; masks [B,L]; refer: [Nseg,St,64]; order host [Int].
    public func callAsFunction(text: MLXArray, textMask: MLXArray,
                               lyric: MLXArray, lyricMask: MLXArray,
                               refer: MLXArray, order: [Int]) -> (MLXArray, MLXArray) {
        let textP = Ops.linear(text, w["text_projector.weight"]!)
        let lyricE = lyricEncoder(lyric, lyricMask)
        let (timbreU, timbreM) = timbreEncoder(refer, order)
        var (hs, m) = pack(lyricE, timbreU, lyricMask, timbreM)
        (hs, m) = pack(hs, textP, m, textMask)
        return (hs, m)
    }

    // MARK: - neutral fallback (used until the tokenizer/prompt path is wired)

    public func neutral(seqLatent T: Int, textEncoder: Qwen3TextEncoder) -> (MLXArray, MLXArray) {
        let enc = w["null_condition_emb"] ?? MLX.zeros([1, 1, hidden])
        let context = MLX.concatenated([MLX.zeros([1, T, 64]), MLX.ones([1, T, 64])], axis: -1)
        return (enc, context)
    }
}
