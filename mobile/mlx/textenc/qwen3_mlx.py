"""MLX port of the Qwen3 text encoder (run on Apple Silicon).

1:1 translation of qwen3_numpy.py (validated to 3.5e-6 vs PyTorch). Pure matmul +
elementwise, so no layout gymnastics. Weights load from the HF snapshot; linear
layers can be 4-bit quantized with mlx.nn.quantize in a follow-up.

    python qwen3_mlx.py /path/to/hf/snapshot   # -> QWEN3_MLX_PARITY_PASS
"""
from __future__ import annotations

import glob
import json
import os
import sys

import mlx.core as mx
import numpy as np

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def load_weights(snap_dir):
    import torch
    from safetensors.torch import load_file
    w = {}
    for f in sorted(glob.glob(os.path.join(snap_dir, "text_encoder", "*.safetensors"))):
        for k, v in load_file(f).items():
            w[k] = mx.array(v.to(torch.float32).numpy())
    return w


def rms_norm(x, weight, eps):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def linear(x, w):
    return x @ w.T


def rope_freqs(seq, hd, theta):
    freqs = 1.0 / (theta ** (mx.arange(0, hd, 2) / hd))
    ang = mx.arange(seq).reshape(-1, 1) * freqs.reshape(1, -1)
    cos = mx.concatenate([mx.cos(ang), mx.cos(ang)], axis=-1)
    sin = mx.concatenate([mx.sin(ang), mx.sin(ang)], axis=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * c + mx.concatenate([-x2, x1], axis=-1) * s


class Qwen3MLX:
    def __init__(self, snap_dir, config_path):
        self.w = load_weights(snap_dir)
        c = json.load(open(config_path))
        self.heads, self.kv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.theta, self.eps, self.n = c["rope_theta"], c["rms_norm_eps"], c["num_hidden_layers"]

    def _attn(self, hn, i, cos, sin, mask):
        w = self.w
        p = f"layers.{i}.self_attn"
        B, L, _ = hn.shape
        q = linear(hn, w[f"{p}.q_proj.weight"]).reshape(B, L, self.heads, self.hd)
        k = linear(hn, w[f"{p}.k_proj.weight"]).reshape(B, L, self.kv, self.hd)
        v = linear(hn, w[f"{p}.v_proj.weight"]).reshape(B, L, self.kv, self.hd)
        q = rms_norm(q, w[f"{p}.q_norm.weight"], self.eps)
        k = rms_norm(k, w[f"{p}.k_norm.weight"], self.eps)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        rep = self.heads // self.kv
        k = mx.repeat(k, rep, axis=2); v = mx.repeat(v, rep, axis=2)
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5) + mask
        out = (mx.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.hd)
        return linear(out, w[f"{p}.o_proj.weight"])

    def __call__(self, ids):
        w = self.w
        h = w["embed_tokens.weight"][ids]
        B, L, _ = h.shape
        cos, sin = rope_freqs(L, self.hd, self.theta)
        idx = mx.arange(L)
        causal = mx.where(idx[:, None] >= idx[None, :], mx.array(0.0), mx.array(-1e30))[None, None]
        for i in range(self.n):
            hn = rms_norm(h, w[f"layers.{i}.input_layernorm.weight"], self.eps)
            h = h + self._attn(hn, i, cos, sin, causal)
            hn = rms_norm(h, w[f"layers.{i}.post_attention_layernorm.weight"], self.eps)
            ff = linear(mx.silu(linear(hn, w[f"layers.{i}.mlp.gate_proj.weight"])) *
                        linear(hn, w[f"layers.{i}.mlp.up_proj.weight"]), w[f"layers.{i}.mlp.down_proj.weight"])
            h = h + ff
        return rms_norm(h, w["norm.weight"], self.eps)


def main():
    snap = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ACESTEP_SNAP", "")
    ids = np.load(os.path.join(FIX, "textenc_input.npy"))
    ref = np.load(os.path.join(FIX, "textenc_output.npy"))
    m = Qwen3MLX(snap, os.path.join(FIX, "textenc_config.json"))
    out = np.array(m(mx.array(ids)))
    err = np.abs(out - ref); denom = float(np.abs(ref).max())
    print(f"rel max err {err.max()/denom:.3e} | corr {np.corrcoef(out.flatten(), ref.flatten())[0,1]:.10f}")
    print("QWEN3_MLX_PARITY_PASS" if err.max() / denom < 2e-3 else "FAIL")


if __name__ == "__main__":
    main()
