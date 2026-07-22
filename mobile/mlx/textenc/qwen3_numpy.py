"""Framework-free NumPy reimplementation of the Qwen3 text encoder (the spec).

Reproduces `transformers.Qwen3Model.forward` -> last_hidden_state. Qwen3 is the
same primitives as the ACE-Step DiT self-attention (GQA + QK-RMSNorm + RoPE +
SwiGLU) plus token embedding, causal masking, and a final RMSNorm.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def load_weights(snap_dir):
    import torch
    from safetensors.torch import load_file
    w = {}
    for f in sorted(glob.glob(os.path.join(snap_dir, "text_encoder", "*.safetensors"))):
        for k, v in load_file(f).items():
            w[k] = v.to(torch.float64).numpy()
    return w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rms_norm(x, weight, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def linear(x, w):
    return x @ w.T


def rope_freqs(seq, hd, theta):
    freqs = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(np.arange(seq), freqs)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * c + np.concatenate([-x2, x1], axis=-1) * s


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class Qwen3Numpy:
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
        q = rms_norm(q, w[f"{p}.q_norm.weight"], self.eps)   # QK-norm over head_dim
        k = rms_norm(k, w[f"{p}.k_norm.weight"], self.eps)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        rep = self.heads // self.kv
        k = np.repeat(k, rep, axis=2); v = np.repeat(v, rep, axis=2)
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5) + mask
        out = (softmax(scores, -1) @ v).transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.hd)
        return linear(out, w[f"{p}.o_proj.weight"])

    def __call__(self, ids):
        w = self.w
        h = w["embed_tokens.weight"][ids]           # [B, L, hidden]
        B, L, _ = h.shape
        cos, sin = rope_freqs(L, self.hd, self.theta)
        causal = np.where(np.arange(L)[:, None] >= np.arange(L)[None, :], 0.0, -1e30)[None, None]
        for i in range(self.n):
            hn = rms_norm(h, w[f"layers.{i}.input_layernorm.weight"], self.eps)
            h = h + self._attn(hn, i, cos, sin, causal)
            hn = rms_norm(h, w[f"layers.{i}.post_attention_layernorm.weight"], self.eps)
            ff = linear(silu(linear(hn, w[f"layers.{i}.mlp.gate_proj.weight"])) *
                        linear(hn, w[f"layers.{i}.mlp.up_proj.weight"]), w[f"layers.{i}.mlp.down_proj.weight"])
            h = h + ff
        return rms_norm(h, w["norm.weight"], self.eps)
