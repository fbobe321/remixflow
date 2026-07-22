"""Framework-free NumPy reimplementation of AceStepConditionEncoder (the spec).

Fuses text + lyric + timbre into the packed cross-attention sequence:
  text_projector(text)                      Linear 1024->2048 (no bias)
  lyric_encoder(lyric, lyric_mask)          embed + 8 pre-LN blocks (bidirectional)
  timbre_encoder(refer, order)              embed + 4 blocks, pool[:,0], unpack
  pack(pack(lyric, timbre), text)           valid-tokens-left packing
Blocks are AceStep GQA + QK-RMSNorm + RoPE + SwiGLU (same primitives as the DiT).
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
    for f in sorted(glob.glob(os.path.join(snap_dir, "condition_encoder", "*.safetensors"))):
        for k, v in load_file(f).items():
            w[k] = v.to(torch.float64).numpy()
    return w


def silu(x): return x / (1.0 + np.exp(-x))
def rms_norm(x, wt, eps=1e-6): return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * wt
def lin(x, w, b=None): return x @ w.T + b if b is not None else x @ w.T


def rope_freqs(seq, hd, theta):
    fr = 1.0 / (theta ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(np.arange(seq), fr)
    return np.concatenate([np.cos(ang), np.cos(ang)], -1), np.concatenate([np.sin(ang), np.sin(ang)], -1)


def apply_rope(x, cos, sin):
    d = x.shape[-1]; c, s = cos[None, :, None, :], sin[None, :, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * c + np.concatenate([-x2, x1], -1) * s


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True); e = np.exp(x); return e / e.sum(axis, keepdims=True)


class CondEncNumpy:
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
        k = np.repeat(k, rep, 2); v = np.repeat(v, rep, 2)
        q, k, v = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        sc = (q @ k.transpose(0, 1, 3, 2)) * (self.hd ** -0.5)
        if mask is not None: sc = sc + mask
        o = (softmax(sc, -1) @ v).transpose(0, 2, 1, 3).reshape(B, L, self.heads * self.hd)
        return lin(o, w[f"{p}.to_out.0.weight"])

    def _layer(self, x, p, cos, sin, mask):
        w = self.w
        xn = rms_norm(x, w[f"{p}.input_layernorm.weight"], self.eps)
        x = x + self._attn(xn, f"{p}.self_attn", cos, sin, mask)
        xn = rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], self.eps)
        ff = lin(silu(lin(xn, w[f"{p}.mlp.gate_proj.weight"])) * lin(xn, w[f"{p}.mlp.up_proj.weight"]),
                 w[f"{p}.mlp.down_proj.weight"])
        return x + ff

    def _pad_mask(self, attn_mask):  # [B,seq] -> [B,1,1,seq] additive (col masked)
        return np.where(attn_mask[:, None, None, :] > 0, 0.0, -1e30)

    def _sliding_mask(self, seq):
        d = np.arange(seq)[:, None] - np.arange(seq)[None, :]
        return np.where(np.abs(d) <= self.sw, 0.0, -1e30)[None, None]

    def _lyric_encoder(self, lyric, lyric_mask):
        w = self.w; pfx = "lyric_encoder"
        x = lin(lyric, w[f"{pfx}.embed_tokens.weight"], w[f"{pfx}.embed_tokens.bias"])
        seq = x.shape[1]
        cos, sin = rope_freqs(seq, self.hd, self.theta)
        pad = self._pad_mask(lyric_mask)
        full = pad                                   # bidirectional + padding
        band = np.minimum(self._sliding_mask(seq), 0.0) + pad  # banded + padding
        for i in range(self.n_lyric):
            mask = band if (i + 1) % 2 else full     # sliding on odd (i+1), full on even
            x = self._layer(x, f"{pfx}.layers.{i}", cos, sin, mask)
        return rms_norm(x, w[f"{pfx}.norm.weight"], self.eps)

    def _timbre_encoder(self, refer, order):
        w = self.w; pfx = "timbre_encoder"
        x = lin(refer, w[f"{pfx}.embed_tokens.weight"], w[f"{pfx}.embed_tokens.bias"])  # [Nseg,seq,H]
        seq = x.shape[1]
        cos, sin = rope_freqs(seq, self.hd, self.theta)
        band = self._sliding_mask(seq)
        for i in range(self.n_timbre):
            mask = band if (i + 1) % 2 else None     # full layers: no mask
            x = self._layer(x, f"{pfx}.layers.{i}", cos, sin, mask)
        x = rms_norm(x, w[f"{pfx}.norm.weight"], self.eps)
        pooled = x[:, 0, :]                          # [Nseg, H]
        return self._unpack(pooled, order)

    @staticmethod
    def _unpack(packed, order):
        N, d = packed.shape
        B = int(order.max()) + 1
        counts = np.bincount(order, minlength=B)
        mc = int(counts.max())
        out = np.zeros((B, mc, d), dtype=packed.dtype)
        m = np.zeros((B, mc), dtype=np.int64)
        seen = np.zeros(B, dtype=int)
        for i in range(N):
            b = int(order[i]); out[b, seen[b]] = packed[i]; m[b, seen[b]] = 1; seen[b] += 1
        return out, m

    @staticmethod
    def _pack(h1, h2, m1, m2):
        hc = np.concatenate([h1, h2], 1); mc = np.concatenate([m1, m2], 1)
        B, L, D = hc.shape
        # Unique sort key = mask*L - position -> valid first, ties by position asc
        # (== torch stable descending). Unique => no dependence on sort stability.
        key = mc * L - np.arange(L)[None, :]
        idx = np.argsort(-key, axis=1)
        hl = np.take_along_axis(hc, idx[..., None], axis=1)
        lengths = mc.sum(1)
        nm = (np.arange(L)[None, :] < lengths[:, None]).astype(np.int64)
        return hl, nm

    def __call__(self, text, text_mask, lyric, lyric_mask, refer, order):
        text_p = lin(text, self.w["text_projector.weight"])
        lyric_e = self._lyric_encoder(lyric, lyric_mask)
        timbre_u, timbre_m = self._timbre_encoder(refer, order)
        hs, m = self._pack(lyric_e, timbre_u, lyric_mask, timbre_m)
        hs, m = self._pack(hs, text_p, m, text_mask)
        return hs, m
