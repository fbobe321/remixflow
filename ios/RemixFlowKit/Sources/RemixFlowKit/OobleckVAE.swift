import Foundation
import MLX

/// Oobleck VAE — encode (waveform → latent) and decode (latent → 48 kHz stereo).
/// Translated from the parity-proven `mobile/mlx/vae/{encoder,vae_decoder}_mlx.py`
/// (6.1e-6 / 3.9e-6 vs PyTorch). Channels-last [N, L, C]; conv weights transposed
/// to MLX layout [Cout, K, Cin] at load. Kept fp16 (not quantized).
public final class OobleckVAE {
    private let w: [String: MLXArray]

    /// `weights` are the `vae/…` entries from the WeightStore. Conv weights are
    /// transposed here from PyTorch [Cout, Cin, K] → MLX [Cout, K, Cin].
    public init(vaeWeights: [String: MLXArray]) {
        var out: [String: MLXArray] = [:]
        for (k, v) in vaeWeights {
            if k.hasSuffix(".weight"), k.contains("conv_t1") {
                out[k] = v.transposed(1, 2, 0)      // convT [Cin,Cout,K] → [Cout,K,Cin]
            } else if k.hasSuffix(".weight"), k.contains("conv") {
                out[k] = v.transposed(0, 2, 1)      // conv  [Cout,Cin,K] → [Cout,K,Cin]
            } else {
                out[k] = v
            }
        }
        self.w = out
    }

    private func sn(_ x: MLXArray, _ p: String) -> MLXArray {
        Ops.snake(x, alphaRaw: w["\(p).alpha"]!, betaRaw: w["\(p).beta"]!)
    }
    private func cv(_ x: MLXArray, _ p: String, stride: Int = 1, pad: Int = 0, dil: Int = 1, bias: Bool = true) -> MLXArray {
        let out = MLX.conv1d(x, w["\(p).weight"]!, stride: stride, padding: pad, dilation: dil)
        if bias, let b = w["\(p).bias"] { return out + b }
        return out
    }
    private func convT(_ x: MLXArray, _ p: String, stride: Int, pad: Int) -> MLXArray {
        var out = MLX.convTransposed1d(x, w["\(p).weight"]!, stride: stride, padding: 0)
        if let b = w["\(p).bias"] { out = out + b }
        if pad > 0 { out = out[0..., pad ..< (out.dim(1) - pad), 0...] }  // full transpose then crop
        return out
    }
    private func res(_ x: MLXArray, _ p: String, dil: Int) -> MLXArray {
        var o = sn(x, "\(p).snake1")
        o = cv(o, "\(p).conv1", pad: ((7 - 1) * dil) / 2, dil: dil)
        o = sn(o, "\(p).snake2")
        o = cv(o, "\(p).conv2", pad: 0)
        let c = (x.dim(1) - o.dim(1)) / 2
        let xr = c > 0 ? x[0..., c ..< (x.dim(1) - c), 0...] : x
        return xr + o
    }

    // MARK: Decoder (latent [N,T,64] → waveform [N,L,2])
    public func decode(_ z: MLXArray) -> MLXArray {
        let strides = (0..<blockCount("block")).map { w["block.\($0).conv_t1.weight"]!.dim(1) / 2 }
        var x = cv(z, "conv1", pad: 3)
        for (i, s) in strides.enumerated() {
            x = sn(x, "block.\(i).snake1")
            x = convT(x, "block.\(i).conv_t1", stride: s, pad: Int(ceil(Double(s) / 2)))
            x = res(x, "block.\(i).res_unit1", dil: 1)
            x = res(x, "block.\(i).res_unit2", dil: 3)
            x = res(x, "block.\(i).res_unit3", dil: 9)
        }
        x = sn(x, "snake1")
        return cv(x, "conv2", pad: 3, bias: false)
    }

    // MARK: Encoder (waveform [N,L,2] → params [N,T,128]); mean = params[..., :64]
    public func encode(_ audio: MLXArray) -> MLXArray {
        let strides = (0..<blockCount("block")).map { i -> Int in
            // encoder block downsample conv is `block.i.conv1` (k = 2·stride)
            w["block.\(i).conv1.weight"]!.dim(1) / 2
        }
        var x = cv(audio, "conv1", pad: 3)
        for (i, s) in strides.enumerated() {
            x = res(x, "block.\(i).res_unit1", dil: 1)
            x = res(x, "block.\(i).res_unit2", dil: 3)
            x = res(x, "block.\(i).res_unit3", dil: 9)
            x = sn(x, "block.\(i).snake1")
            x = cv(x, "block.\(i).conv1", stride: s, pad: Int(ceil(Double(s) / 2)))
        }
        x = sn(x, "snake1")
        return cv(x, "conv2", pad: 1)
    }

    private func blockCount(_ prefix: String) -> Int {
        var n = 0
        while w["\(prefix).\(n).conv_t1.weight"] != nil || w["\(prefix).\(n).conv1.weight"] != nil { n += 1 }
        return n
    }
}
