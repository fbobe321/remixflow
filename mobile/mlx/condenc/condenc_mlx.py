"""MLX port of AceStepConditionEncoder (run on Apple Silicon).

1:1 translation of condenc_numpy.py (validated to 3.4e-7 vs PyTorch). Reuses the
GQA+QK-norm+RoPE+SwiGLU block; the pack uses a unique sort key (no dependence on
argsort stability); the timbre unpack computes its layout host-side from `order`.

    python condenc_mlx.py /path/to/hf/snapshot   # -> CONDENC_MLX_PARITY_PASS
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
    for f in sorted(glob.glob(os.path.join(snap_dir, "condition_encoder", "*.safetensors"))):
        for k, v in load_file(f).items():
            w[k] = mx.array(v.to(torch.float32).numpy())
    return w


def rms_norm(x, wt, eps): return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * wt
def lin(x, w, b=None): return (x @ w.T) + b if b is not None else x @ w.T


def rope_freqs(seq, hd, theta):
    fr = 1.0 / (theta ** (mx.arange(0, hd, 2) / hd))
    ang = mx.arange(seq).reshape(-1, 1) * fr.reshape(1, -1)
    return mx.concatenate([mx.cos(ang), mx.cos(ang)], -1), mx.concatenate([mx.sin(ang), mx.sin(ang)], -1)


def apply_rope(x, cos, sin):
    d = x.shape[-1]; c, s = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * c + mx.concatenate([-x2, x1], -1) * s


class CondEncMLX:
    def __init__(self, snap_dir, config_path):
        self.w = load_weights(snap_dir)
        c = json.load(open(config_path))
        self.heads, self.kv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.theta, self.eps, self.sw = c["rope_theta"], c["rms_norm_eps"], c["sliding_window"]
        self.n_lyric, self.n_timbre = c["num_lyric_encoder_hidden_layers"], c["num_timbre_encoder_hidden_layers"]

    def _attn(self, x, p, cos, sin, mask):
        w = self.w; B, L, _ = x.shape
        q = lin(x, w[f"{p}.to_q.weight"]).reshape(B, L, self.heads, self.hd)
        k = lin(x, w[f"{p}.to_k.weight"]).reshape(B, L, self.kv, self.hd)
        v = lin(x, w[f"{p}.to_v.weight"]).reshape(B, L, self.kv, self.hd)
        q = rms_norm(q, w[f"{p}.norm_q.weight"], self.eps); k = rms_norm(k, w[f"{p}.norm_k.weight"], self.eps)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        rep = self.heads // self.kv
        k = mx.repeat(k, rep, 2); v = mx.repeat(v, rep, 2)
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        sc = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5)
        if mask is not None: sc = sc + mask
        o = (mx.softmax(sc, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.hd)
        return lin(o, w[f"{p}.to_out.0.weight"])

    def _layer(self, x, p, cos, sin, mask):
        w = self.w
        xn = rms_norm(x, w[f"{p}.input_layernorm.weight"], self.eps)
        x = x + self._attn(xn, f"{p}.self_attn", cos, sin, mask)
        xn = rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
        ff = lin(mx.silu(lin(xn, w[f"{p}.mlp.gate_proj.weight"])) * lin(xn, w[f"{p}.mlp.up_proj.weight"]),
                 w[f"{p}.mlp.down_proj.weight"])
        return x + ff

    def _pad_mask(self, attn_mask):
        return mx.where(attn_mask[:, None, None, :] > 0, mx.array(0.0), mx.array(-1e30))

    def _sliding_mask(self, seq):
        d = mx.arange(seq)[:, None] - mx.arange(seq)[None, :]
        return mx.where(mx.abs(d) <= self.sw, mx.array(0.0), mx.array(-1e30))[None, None]

    def _lyric_encoder(self, lyric, lyric_mask):
        w = self.w; pfx = "lyric_encoder"
        x = lin(lyric, w[f"{pfx}.embed_tokens.weight"], w[f"{pfx}.embed_tokens.bias"])
        seq = x.shape[1]
        cos, sin = rope_freqs(seq, self.hd, self.theta)
        pad = self._pad_mask(lyric_mask)
        full = pad
        band = mx.minimum(self._sliding_mask(seq), mx.array(0.0)) + pad
        for i in range(self.n_lyric):
            x = self._layer(x, f"{pfx}.layers.{i}", cos, sin, band if (i + 1) % 2 else full)
        return rms_norm(x, w[f"{pfx}.norm.weight"], self.eps)

    def _timbre_encoder(self, refer, order):
        w = self.w; pfx = "timbre_encoder"
        x = lin(refer, w[f"{pfx}.embed_tokens.weight"], w[f"{pfx}.embed_tokens.bias"])
        seq = x.shape[1]
        cos, sin = rope_freqs(seq, self.hd, self.theta)
        band = self._sliding_mask(seq)
        for i in range(self.n_timbre):
            x = self._layer(x, f"{pfx}.layers.{i}", cos, sin, band if (i + 1) % 2 else None)
        x = rms_norm(x, w[f"{pfx}.norm.weight"], self.eps)
        pooled = x[:, 0, :]
        return self._unpack(pooled, np.asarray(order))

    @staticmethod
    def _unpack(pooled, order):
        # order is small host data -> compute layout with numpy, gather in MLX.
        N = order.shape[0]
        B = int(order.max()) + 1
        counts = np.bincount(order, minlength=B)
        mc = int(counts.max())
        src = np.zeros((B, mc), dtype=np.int64)   # source row per (b, slot)
        m = np.zeros((B, mc), dtype=np.int64)
        seen = np.zeros(B, dtype=int)
        for i in range(N):
            b = int(order[i]); src[b, seen[b]] = i; m[b, seen[b]] = 1; seen[b] += 1
        gathered = pooled[mx.array(src.reshape(-1))].reshape(B, mc, pooled.shape[-1])
        gathered = gathered * mx.array(m.reshape(B, mc, 1).astype(np.float32))
        return gathered, mx.array(m)

    @staticmethod
    def _pack(h1, h2, m1, m2):
        hc = mx.concatenate([h1, h2], 1); mc = mx.concatenate([m1, m2], 1)
        B, L, D = hc.shape
        key = mc * L - mx.arange(L).reshape(1, L)
        idx = mx.argsort(-key, axis=1)                 # unique key -> stability-independent
        hl = mx.take_along_axis(hc, idx[..., None], axis=1)
        lengths = mc.sum(1)
        nm = (mx.arange(L).reshape(1, L) < lengths.reshape(B, 1)).astype(mx.int64)
        return hl, nm

    def __call__(self, text, text_mask, lyric, lyric_mask, refer, order):
        text_p = lin(text, self.w["text_projector.weight"])
        lyric_e = self._lyric_encoder(lyric, lyric_mask)
        timbre_u, timbre_m = self._timbre_encoder(refer, order)
        hs, m = self._pack(lyric_e, timbre_u, lyric_mask, timbre_m)
        hs, m = self._pack(hs, text_p, m, text_mask)
        return hs, m


def main():
    snap = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ACESTEP_SNAP", "")
    d = np.load(os.path.join(FIX, "condenc_input.npz"))
    ref = np.load(os.path.join(FIX, "condenc_output.npy"))
    rm = np.load(os.path.join(FIX, "condenc_output_mask.npy"))
    m = CondEncMLX(snap, os.path.join(FIX, "condenc_config.json"))
    a = lambda k: mx.array(d[k].astype(np.float32))
    out, mask = m(a("text"), mx.array(d["text_mask"]), a("lyric"), mx.array(d["lyric_mask"]),
                  a("refer"), d["order"])
    out = np.array(out); mask = np.array(mask)
    err = np.abs(out - ref); denom = float(np.abs(ref).max())
    print(f"rel max {err.max()/denom:.3e} | mask match {np.array_equal(mask, rm)} | corr {np.corrcoef(out.flatten(), ref.flatten())[0,1]:.10f}")
    print("CONDENC_MLX_PARITY_PASS" if err.max() / denom < 2e-3 and np.array_equal(mask, rm) else "FAIL")


if __name__ == "__main__":
    main()
