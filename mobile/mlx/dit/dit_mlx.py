"""MLX port of the ACE-Step DiT (run on Apple Silicon).

A 1:1 translation of dit_numpy.py (validated to 8.8e-6 vs PyTorch). Loads weights
straight from the HF snapshot transformer/*.safetensors. Linear layers can be
4-bit quantized with mlx.nn.quantize in a follow-up; this file is the fp16/fp32
correctness reference.

Run on a Mac:
    pip install mlx numpy safetensors torch    # torch only to cast bf16 -> fp32
    python dit_mlx.py /path/to/hf/snapshot     # prints parity vs ../fixtures/dit_output.npy
"""
from __future__ import annotations

import glob
import json
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "..", "fixtures")


def load_weights(snap_dir):
    # bf16 -> fp32 via torch, then to mx.
    import torch
    from safetensors.torch import load_file
    w = {}
    for f in sorted(glob.glob(os.path.join(snap_dir, "transformer", "*.safetensors"))):
        for k, v in load_file(f).items():
            w[k] = mx.array(v.to(torch.float32).numpy())
    return w


def rms_norm(x, weight, eps):
    v = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(v + eps) * weight


def linear(x, w, b=None):
    y = x @ w.T
    return y + b if b is not None else y


def sinusoid(t, dim=256, max_period=10000.0):
    half = dim // 2
    emb = mx.exp(-np.log(max_period) * mx.arange(half) / half)
    args = t.reshape(-1, 1) * emb.reshape(1, -1)
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def timestep_embedding(t, w, p, hidden):
    tf = sinusoid(t * 1000.0)
    temb = linear(tf, w[f"{p}.linear_1.weight"], w[f"{p}.linear_1.bias"])
    temb = mx.silu(temb)
    temb = linear(temb, w[f"{p}.linear_2.weight"], w[f"{p}.linear_2.bias"])
    proj = linear(mx.silu(temb), w[f"{p}.time_proj.weight"], w[f"{p}.time_proj.bias"])
    return temb, proj.reshape(proj.shape[0], 6, hidden)


def rope_freqs(seq, hd, theta):
    freqs = 1.0 / (theta ** (mx.arange(0, hd, 2) / hd))
    ang = mx.arange(seq).reshape(-1, 1) * freqs.reshape(1, -1)
    cos = mx.concatenate([mx.cos(ang), mx.cos(ang)], axis=-1)
    sin = mx.concatenate([mx.sin(ang), mx.sin(ang)], axis=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    x_rot = mx.concatenate([-x2, x1], axis=-1)
    return x * c + x_rot * s


class DiTMLX:
    def __init__(self, snap_dir, config_path):
        self.w = load_weights(snap_dir)
        c = json.load(open(config_path))
        self.H, self.heads, self.kv, self.hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.patch, self.theta, self.eps, self.sw = c["patch_size"], c["rope_theta"], c["rms_norm_eps"], c["sliding_window"]
        self.layer_types, self.n = c["layer_types"], c["num_hidden_layers"]

    def _attn(self, x, kv_in, p, cos, sin, mask, is_cross):
        w = self.w
        B, L, _ = x.shape
        Lk = kv_in.shape[1]
        q = linear(x, w[f"{p}.to_q.weight"]).reshape(B, L, self.heads, self.hd)
        k = linear(kv_in, w[f"{p}.to_k.weight"]).reshape(B, Lk, self.kv, self.hd)
        v = linear(kv_in, w[f"{p}.to_v.weight"]).reshape(B, Lk, self.kv, self.hd)
        q = rms_norm(q, w[f"{p}.norm_q.weight"], self.eps)
        k = rms_norm(k, w[f"{p}.norm_k.weight"], self.eps)
        if not is_cross:
            q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        rep = self.heads // self.kv
        k = mx.repeat(k, rep, axis=2); v = mx.repeat(v, rep, axis=2)
        q = q.transpose(0, 2, 1, 3); k = k.transpose(0, 2, 1, 3); v = v.transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5)
        if mask is not None:
            scores = scores + mask
        attn = mx.softmax(scores, axis=-1) @ v
        out = attn.transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.hd)
        return linear(out, w[f"{p}.to_out.0.weight"])

    def _block(self, x, i, proj, enc, cos, sin, mask):
        w = self.w
        pfx = f"layers.{i}"
        sixt = w[f"{pfx}.scale_shift_table"] + proj
        sh, sc, g, csh, csc, cg = [sixt[:, j:j + 1, :] for j in range(6)]
        nh = rms_norm(x, w[f"{pfx}.self_attn_norm.weight"], self.eps) * (1 + sc) + sh
        x = x + self._attn(nh, nh, f"{pfx}.self_attn", cos, sin, mask, False) * g
        nh = rms_norm(x, w[f"{pfx}.cross_attn_norm.weight"], self.eps)
        x = x + self._attn(nh, enc, f"{pfx}.cross_attn", cos, sin, None, True)
        nh = rms_norm(x, w[f"{pfx}.mlp_norm.weight"], self.eps) * (1 + csc) + csh
        ff = linear(mx.silu(linear(nh, w[f"{pfx}.mlp.gate_proj.weight"])) *
                    linear(nh, w[f"{pfx}.mlp.up_proj.weight"]), w[f"{pfx}.mlp.down_proj.weight"])
        return x + ff * cg

    def _sliding_mask(self, seq):
        idx = mx.arange(seq)
        diff = idx[:, None] - idx[None, :]
        keep = mx.abs(diff) <= self.sw
        return mx.where(keep, mx.array(0.0), mx.array(-1e30))[None, None]

    def __call__(self, hidden, context, enc, t, t_r):
        w = self.w
        temb_t, proj_t = timestep_embedding(t, w, "time_embed", self.H)
        temb_r, proj_r = timestep_embedding(t - t_r, w, "time_embed_r", self.H)
        temb = temb_t + temb_r
        proj = proj_t + proj_r
        x = mx.concatenate([context, hidden], axis=-1)
        orig = x.shape[1]
        if x.shape[1] % self.patch != 0:
            pad = self.patch - x.shape[1] % self.patch
            x = mx.pad(x, [(0, 0), (0, pad), (0, 0)])
        B, Lp, C = x.shape
        seq = Lp // self.patch
        # Conv1d(k=patch, stride=patch) == reshape non-overlapping patches @ weight
        patches = x.reshape(B, seq, self.patch, C).transpose(0, 1, 3, 2).reshape(B, seq, C * self.patch)
        cwm = w["proj_in_conv.weight"].reshape(self.H, C * self.patch)
        x = patches @ cwm.T + w["proj_in_conv.bias"]
        enc = linear(enc, w["condition_embedder.weight"], w["condition_embedder.bias"])
        cos, sin = rope_freqs(seq, self.hd, self.theta)
        smask = self._sliding_mask(seq)
        for i in range(self.n):
            mask = smask if self.layer_types[i] == "sliding_attention" else None
            x = self._block(x, i, proj, enc, cos, sin, mask)
        ss = w["scale_shift_table"] + temb[:, None, :]
        sh, sc = ss[:, 0:1, :], ss[:, 1:2, :]
        x = rms_norm(x, w["norm_out.weight"], self.eps) * (1 + sc) + sh
        pw = w["proj_out_conv.weight"]  # [H, acoustic, patch]
        acoustic = pw.shape[1]
        outs = [x @ pw[:, :, j] + w["proj_out_conv.bias"] for j in range(self.patch)]
        y = mx.stack(outs, axis=2).reshape(B, seq * self.patch, acoustic)
        return y[:, :orig, :]


def main():
    snap = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ACESTEP_SNAP", "")
    d = np.load(os.path.join(FIX, "dit_input.npz"))
    ref = np.load(os.path.join(FIX, "dit_output.npy"))
    m = DiTMLX(snap, os.path.join(FIX, "dit_config.json"))
    a = lambda k: mx.array(d[k].astype(np.float32))
    out = np.array(m(a("hidden"), a("context"), a("enc"), a("t"), a("t_r")))
    err = np.abs(out - ref); denom = float(np.abs(ref).max())
    print("shapes:", out.shape, ref.shape)
    print(f"rel max err {err.max()/denom:.3e} | corr {np.corrcoef(out.flatten(), ref.flatten())[0,1]:.10f}")
    print("DIT_MLX_PARITY_PASS" if err.max() / denom < 2e-3 else "FAIL")


if __name__ == "__main__":
    main()
