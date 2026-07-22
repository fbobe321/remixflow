import Foundation
import MLX

/// AceStepConditionEncoder — fuses text + lyric + timbre into the DiT's cross-
/// attention sequence. The full forward (lyric encoder, timbre encoder, pack/
/// unpack) is parity-proven in `mobile/mlx/condenc/condenc_mlx.py` (3.4e-7) and
/// translates 1:1 into MLX-Swift with the same primitives used here (`Ops`,
/// `slidingMask`, the encoder block = input_ln → GQA attn → post_ln → SwiGLU).
///
/// For the skeleton we expose a runnable **neutral** conditioning so the pipeline
/// produces audio end-to-end before the prompt/lyric/timbre path is wired.
public final class ConditionEncoder {
    private let w: [String: MLXArray]
    let hidden = 2048

    public init(weights: [String: MLXArray]) { self.w = weights }

    /// Neutral conditioning: the learned null-condition token + a silence/ones
    /// context. Matches the DiT's expected shapes:
    ///   enc     [1, 1, 2048]  (cross-attention keys/values)
    ///   context [1, T, 128]   (src-latent placeholder ⊕ ones chunk-mask)
    public func neutral(seqLatent T: Int, textEncoder: Qwen3TextEncoder) -> (MLXArray, MLXArray) {
        let enc = w["null_condition_emb"] ?? MLX.zeros([1, 1, hidden])
        let src = MLX.zeros([1, T, 64])
        let mask = MLX.ones([1, T, 64])
        let context = MLX.concatenated([src, mask], axis: -1)
        return (enc, context)
    }

    // MARK: - Full conditioning (to complete)
    //
    // Translate `condenc_numpy.py::CondEncNumpy.__call__`:
    //   text_p  = linear(text, text_projector.weight)                 // 1024→2048
    //   lyric_e = lyricEncoder(lyricEmbeds, lyricMask)                 // 8 bidir blocks
    //   timbre  = timbreEncoder(referLatents, orderMask)              // 4 blocks + unpack
    //   enc, mask = pack(pack(lyric_e, timbre, …), text_p, …)         // valid-tokens-left
    //
    // The encoder block is identical to `Qwen3TextEncoder`'s layer but bidirectional
    // (no causal mask; padding/sliding masks instead). The pack uses the unique
    // sort key `mask*L - position` (stability-independent). See the Python for the
    // exact masks and the timbre first-token pool + order-mask unpack.
    //
    // public func callAsFunction(text:, textMask:, lyric:, lyricMask:, refer:, order:) -> (MLXArray, MLXArray)
}
