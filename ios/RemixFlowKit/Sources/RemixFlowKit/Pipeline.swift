import Foundation
import MLX

/// Steering parameters (subset mirroring the server's controls).
public struct Steering: Sendable {
    public var variationAmount: Float = 0.35
    public var prompt: String = "faithful cover, same style and mood"
    public var lyrics: String = ""
    public init() {}
}

/// A generation backend. Mirrors the server's `Generator` protocol so the app can
/// start on a mock/remote backend and swap in the on-device MLX pipeline.
public protocol RemixGenerator {
    /// audio: [channels, samples] at 48 kHz. Returns a variation, same layout.
    func generate(audio: MLXArray, steering: Steering, seed: UInt64) -> MLXArray
}

/// SDEdit flow-matching schedule + loop. Translated from `mobile/mlx/flow/sdedit.py`
/// (validated 1e-7 vs the diffusers scheduler).
public enum SDEdit {
    public static func schedule(strength: Float, steps: Int = 8, shift: Float = 3) -> [Float] {
        (0..<steps).map { i -> Float in
            let base = strength + (0 - strength) * Float(i) / Float(steps)   // linspace[strength,0)[:-1]
            return shift * base / (1 + (shift - 1) * base)
        }
    }

    /// x0 = source latents; noise the trajectory to σ0 then Euler-integrate to 0.
    public static func run(srcLatents: MLXArray, strength: Float, seed: UInt64,
                           velocity: (MLXArray, Float) -> MLXArray) -> MLXArray {
        let sched = schedule(strength: strength)
        let sigma0 = sched.first ?? strength
        let noise = MLXRandom.normal(srcLatents.shape, key: MLXRandom.key(seed))
        var x = (1 - sigma0) * srcLatents + sigma0 * noise
        for (i, sigma) in sched.enumerated() {
            let v = velocity(x, sigma)
            let next = i + 1 < sched.count ? sched[i + 1] : 0
            x = x + (next - sigma) * v
        }
        return x
    }
}

/// The on-device ACE-Step SDEdit pipeline: VAE-encode → condition → denoise → decode.
public final class ACEStepPipeline: RemixGenerator {
    let vae: OobleckVAE
    let dit: DiT
    let textEncoder: Qwen3TextEncoder
    let conditionEncoder: ConditionEncoder
    let tokenizer: RFTokenizer?

    /// `tokenizer` enables the full text/lyric/timbre conditioning; pass nil to use
    /// the neutral fallback.
    public init(store: WeightStore, tokenizer: RFTokenizer? = nil) {
        func sub(_ p: String) -> [String: MLXArray] {
            Dictionary(uniqueKeysWithValues: store.weights
                .filter { $0.key.hasPrefix("\(p)/") }
                .map { (String($0.key.dropFirst(p.count + 1)), $0.value) })
        }
        vae = OobleckVAE(vaeWeights: sub("vae"))
        dit = DiT(transformerWeights: sub("transformer"))
        textEncoder = Qwen3TextEncoder(weights: sub("text_encoder"))
        conditionEncoder = ConditionEncoder(weights: sub("condition_encoder"))
        self.tokenizer = tokenizer
    }

    /// Cross-attention conditioning from prompt + lyrics + the source as timbre.
    /// Prompt → Qwen3 last_hidden_state; lyrics → Qwen3 embedding only; timbre →
    /// the source's VAE latents as one reference segment. (Server splits 3×10 s;
    /// one segment is a reasonable on-device simplification.)
    private func conditioning(prompt: String, lyrics: String, sourceMean: MLXArray) -> MLXArray {
        guard let tokenizer else {
            return conditionEncoder.neutral(seqLatent: sourceMean.dim(1), textEncoder: textEncoder).0
        }
        let textHidden = textEncoder(tokenizer.ids(prompt))                 // [1, Lt, 1024]
        let textMask = MLX.ones([1, textHidden.dim(1)])
        let lyricHidden = textEncoder.embed(tokenizer.ids(lyrics))          // [1, Ll, 1024] (embed only)
        let lyricMask = MLX.ones([1, lyricHidden.dim(1)])
        let (enc, _) = conditionEncoder(text: textHidden, textMask: textMask,
                                        lyric: lyricHidden, lyricMask: lyricMask,
                                        refer: sourceMean, order: [0])       // 1 timbre segment
        return enc
    }

    public func generate(audio: MLXArray, steering: Steering, seed: UInt64) -> MLXArray {
        let x = audio.transposed().reshaped([1, audio.dim(1), 2])           // [2,N] → [1,N,2]
        let params = vae.encode(x)                                          // [1, T, 128]
        let mean = params[0..., 0..., 0 ..< 64]                             // latent mean [1, T, 64]

        let enc = conditioning(prompt: steering.prompt, lyrics: steering.lyrics, sourceMean: mean)
        // SDEdit context is neutral: the noised source enters via `latents`, not context.
        let T = mean.dim(1)
        let context = MLX.concatenated([MLX.zeros([1, T, 64]), MLX.ones([1, T, 64])], axis: -1)

        let out = SDEdit.run(srcLatents: mean, strength: steering.variationAmount, seed: seed) { xt, sigma in
            let t = MLXArray([sigma])
            return dit(hidden: xt, context: context, enc: enc, t: t, tR: t)
        }
        return vae.decode(out)[0].transposed()                             // [1,L,2] → [2,L]
    }
}
