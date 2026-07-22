"""Framework-free NumPy reimplementation of the ACE-Step DiT (the *spec*).

Reproduces `AceStepTransformer1DModel.forward` exactly so the MLX port can be a
mechanical translation. Loads weights straight from the HF snapshot's
transformer/*.safetensors (no re-export).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def load_weights(snap_dir):
    # Weights are bf16 (no numpy dtype) — load via torch, cast to fp64.
    import torch
    from safetensors.torch import load_file
    w = {}
    for f in sorted(glob.glob(os.path.join(snap_dir, "transformer", "*.safetensors"))):
        for k, v in load_file(f).items():
            w[k] = v.to(torch.float64).numpy()
    return w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rms_norm(x, weight, eps=1e-6):
    # normalize over last dim
    v = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(v + eps) * weight


def linear(x, w, b=None):
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def sinusoid(t, dim=256, max_period=10000.0):
    # get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)
    half = dim // 2
    exponent = -np.log(max_period) * np.arange(half) / half
    emb = np.exp(exponent)                      # [half]
    args = t[:, None] * emb[None, :]            # [B, half]
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1)  # [B, dim]


def timestep_embedding(t, w, prefix, hidden):
    # AceStepTimestepEmbedding: sinusoid(t*1000) -> linear_1 -> silu -> linear_2 -> temb
    #   timestep_proj = time_proj(silu(temb)).reshape(B, 6, hidden)
    tf = sinusoid(t * 1000.0, dim=256)
    temb = linear(tf, w[f"{prefix}.linear_1.weight"], w[f"{prefix}.linear_1.bias"])
    temb = silu(temb)
    temb = linear(temb, w[f"{prefix}.linear_2.weight"], w[f"{prefix}.linear_2.bias"])
    proj = linear(silu(temb), w[f"{prefix}.time_proj.weight"], w[f"{prefix}.time_proj.bias"])
    proj = proj.reshape(proj.shape[0], 6, hidden)
    return temb, proj


def rope_freqs(seq_len, head_dim, theta):
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))  # [D/2]
    pos = np.arange(seq_len)
    ang = np.outer(pos, freqs)                                       # [L, D/2]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1)        # [L, D]
    sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    # x: [B, L, H, D]; cos/sin: [L, D]. unbind_dim=-2 => split last dim into halves.
    d = x.shape[-1]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    x_rot = np.concatenate([-x2, x1], axis=-1)
    return x * c + x_rot * s


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class DiTNumpy:
    def __init__(self, snap_dir, config_path):
        self.w = load_weights(snap_dir)
        self.cfg = json.load(open(config_path))
        self.H = self.cfg["hidden_size"]
        self.heads = self.cfg["num_attention_heads"]
        self.kv = self.cfg["num_key_value_heads"]
        self.hd = self.cfg["head_dim"]
        self.patch = self.cfg["patch_size"]
        self.theta = self.cfg["rope_theta"]
        self.eps = self.cfg["rms_norm_eps"]
        self.sw = self.cfg["sliding_window"]
        self.layer_types = self.cfg["layer_types"]
        self.n = self.cfg["num_hidden_layers"]

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
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        # GQA: repeat kv heads to match q heads
        rep = self.heads // self.kv
        k = np.repeat(k, rep, axis=2)
        v = np.repeat(v, rep, axis=2)
        # [B, H, L, D]
        q = q.transpose(0, 2, 1, 3); k = k.transpose(0, 2, 1, 3); v = v.transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5)  # [B,H,L,Lk]
        if mask is not None:
            scores = scores + mask  # [1,1,L,Lk] additive
        attn = softmax(scores, -1) @ v                              # [B,H,L,D]
        out = attn.transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.hd)
        return linear(out, w[f"{p}.to_out.0.weight"])

    def _block(self, x, i, proj, enc, cos, sin, mask):
        w = self.w
        pfx = f"layers.{i}"
        sixt = (w[f"{pfx}.scale_shift_table"] + proj)  # [1,6,H] + [B,6,H]
        sh, sc, g, csh, csc, cg = [sixt[:, j:j + 1, :] for j in range(6)]
        # self-attn (AdaLN)
        nh = rms_norm(x, w[f"{pfx}.self_attn_norm.weight"], self.eps) * (1 + sc) + sh
        x = x + self._attn(nh, nh, f"{pfx}.self_attn", cos, sin, mask, False) * g
        # cross-attn (no gate)
        nh = rms_norm(x, w[f"{pfx}.cross_attn_norm.weight"], self.eps)
        x = x + self._attn(nh, enc, f"{pfx}.cross_attn", cos, sin, None, True)
        # mlp (AdaLN, SwiGLU)
        nh = rms_norm(x, w[f"{pfx}.mlp_norm.weight"], self.eps) * (1 + csc) + csh
        ff = linear(silu(linear(nh, w[f"{pfx}.mlp.gate_proj.weight"])) *
                    linear(nh, w[f"{pfx}.mlp.up_proj.weight"]), w[f"{pfx}.mlp.down_proj.weight"])
        x = x + ff * cg
        return x

    def _sliding_mask(self, seq):
        idx = np.arange(seq)
        diff = idx[:, None] - idx[None, :]
        keep = np.abs(diff) <= self.sw            # is_causal=False, both sides
        m = np.where(keep, 0.0, -1e30)
        return m[None, None]                       # [1,1,seq,seq]

    def __call__(self, hidden, context, enc, t, t_r):
        w = self.w
        # dual timestep
        temb_t, proj_t = timestep_embedding(t, w, "time_embed", self.H)
        temb_r, proj_r = timestep_embedding(t - t_r, w, "time_embed_r", self.H)
        temb = temb_t + temb_r
        proj = proj_t + proj_r
        # patchify
        x = np.concatenate([context, hidden], axis=-1)   # [B,T,192]
        orig = x.shape[1]
        if x.shape[1] % self.patch != 0:
            pad = self.patch - x.shape[1] % self.patch
            x = np.pad(x, ((0, 0), (0, pad), (0, 0)))
        # Conv1d(in,H,k=patch,stride=patch) == reshape patches then linear
        xt = x.transpose(0, 2, 1)                        # [B,192,T]
        cw = w["proj_in_conv.weight"]                    # [H,192,patch]
        cb = w["proj_in_conv.bias"]
        B, C, Lp = xt.shape
        seq = Lp // self.patch
        patches = xt.reshape(B, C, seq, self.patch).transpose(0, 2, 1, 3).reshape(B, seq, C * self.patch)
        cwm = cw.reshape(self.H, C * self.patch)
        x = patches @ cwm.T + cb                         # [B, seq, H]
        enc = linear(enc, w["condition_embedder.weight"], w["condition_embedder.bias"])
        cos, sin = rope_freqs(seq, self.hd, self.theta)
        smask = self._sliding_mask(seq)
        for i in range(self.n):
            mask = smask if self.layer_types[i] == "sliding_attention" else None
            x = self._block(x, i, proj, enc, cos, sin, mask)
        # output norm + AdaLN + depatchify
        sh, sc = (w["scale_shift_table"] + temb[:, None, :])[:, 0:1, :], (w["scale_shift_table"] + temb[:, None, :])[:, 1:2, :]
        x = rms_norm(x, w["norm_out.weight"], self.eps) * (1 + sc) + sh
        # ConvTranspose1d(H, acoustic, k=patch, stride=patch): each token -> patch samples
        pw = w["proj_out_conv.weight"]                   # [H, acoustic, patch]
        pb = w["proj_out_conv.bias"]
        acoustic = pw.shape[1]
        # y[:, t*patch + j, :] = x[:, t, :] @ pw[:, :, j] + pb
        outs = []
        for j in range(self.patch):
            outs.append(x @ pw[:, :, j] + pb)            # [B, seq, acoustic]
        y = np.stack(outs, axis=2).reshape(B, seq * self.patch, acoustic)
        return y[:, :orig, :]
