"""Framework-independent NumPy reimplementation of the Oobleck VAE decoder.

This is the *spec*: it reads the folded weights + config and reproduces the
diffusers `AutoencoderOobleck` decoder forward with explicit conv/transpose-conv
(no torch). Once this matches PyTorch (validate_numpy.py), the MLX port is a
mechanical translation of these same ops.
"""
from __future__ import annotations

import json
import os

import numpy as np
from safetensors.numpy import load_file


def snake(x, alpha_raw, beta_raw):
    # logscale: alpha=exp(a), beta=exp(b); y = x + (1/(beta+1e-9)) * sin(alpha*x)^2
    # params are [1, C, 1]; x here is [C, L] -> reshape params to [C, 1].
    a = np.exp(alpha_raw).reshape(-1, 1)
    b = np.exp(beta_raw).reshape(-1, 1)
    return x + (1.0 / (b + 1e-9)) * np.sin(a * x) ** 2


def conv1d(x, w, b, stride=1, pad=0, dilation=1):
    """x:[Cin,L]  w:[Cout,Cin,K]  b:[Cout] or None -> [Cout,Lout] (cross-correlation)."""
    Cin, L = x.shape
    Cout, _, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    Lp = xp.shape[1]
    Lout = (Lp - dilation * (K - 1) - 1) // stride + 1
    # im2col: [Cin, K, Lout]
    idx = (np.arange(Lout)[:, None] * stride + np.arange(K)[None, :] * dilation)  # [Lout,K]
    cols = xp[:, idx]                       # [Cin, Lout, K]
    cols = cols.transpose(0, 2, 1).reshape(Cin * K, Lout)  # [Cin*K, Lout]
    wm = w.transpose(0, 1, 2).reshape(Cout, Cin * K)       # [Cout, Cin*K]
    out = wm @ cols                                        # [Cout, Lout]
    if b is not None:
        out += b[:, None]
    return out


def conv_transpose1d(x, w, b, stride=1, pad=0):
    """x:[Cin,L]  w:[Cin,Cout,K]  -> [Cout, (L-1)*stride - 2*pad + K]."""
    Cin, L = x.shape
    _, Cout, K = w.shape
    full = (L - 1) * stride + K
    out = np.zeros((Cout, full), dtype=np.float64)
    # out[o, i*stride + k] += sum_c w[c,o,k] * x[c,i]
    for k in range(K):
        contrib = np.einsum("co,ci->oi", w[:, :, k], x)  # [Cout, L]
        pos = np.arange(L) * stride + k
        out[:, pos] += contrib
    if b is not None:
        out += b[:, None]
    if pad > 0:
        out = out[:, pad:full - pad]
    return out


class OobleckDecoderNumpy:
    def __init__(self, weights_path, config_path):
        self.w = load_file(weights_path)
        self.cfg = json.load(open(config_path))
        # Decoder strides upsample in REVERSE of the encoder's downsampling; derive
        # each block's stride from its transposed-conv kernel (K = 2*stride).
        self.n_blocks = sum(1 for k in self.w if k.endswith(".conv_t1.weight"))
        self.strides = [self.w[f"block.{i}.conv_t1.weight"].shape[2] // 2
                        for i in range(self.n_blocks)]

    def _snake(self, x, prefix):
        return snake(x, self.w[f"{prefix}.alpha"].astype(np.float64),
                     self.w[f"{prefix}.beta"].astype(np.float64))

    def _conv(self, x, prefix, stride=1, pad=0, dilation=1, bias=True):
        w = self.w[f"{prefix}.weight"].astype(np.float64)
        b = self.w.get(f"{prefix}.bias")
        b = b.astype(np.float64) if (bias and b is not None) else None
        return conv1d(x, w, b, stride, pad, dilation)

    def _convT(self, x, prefix, stride, pad):
        w = self.w[f"{prefix}.weight"].astype(np.float64)
        b = self.w.get(f"{prefix}.bias")
        b = b.astype(np.float64) if b is not None else None
        return conv_transpose1d(x, w, b, stride, pad)

    def _res_unit(self, x, prefix, dilation):
        out = self._snake(x, f"{prefix}.snake1")
        out = self._conv(out, f"{prefix}.conv1", pad=((7 - 1) * dilation) // 2, dilation=dilation)
        out = self._snake(out, f"{prefix}.snake2")
        out = self._conv(out, f"{prefix}.conv2", pad=0)  # k=1
        crop = (x.shape[-1] - out.shape[-1]) // 2
        xr = x[:, crop:x.shape[-1] - crop] if crop > 0 else x
        return xr + out

    def _block(self, x, i, stride):
        import math
        x = self._snake(x, f"block.{i}.snake1")
        x = self._convT(x, f"block.{i}.conv_t1", stride=stride, pad=math.ceil(stride / 2))
        x = self._res_unit(x, f"block.{i}.res_unit1", 1)
        x = self._res_unit(x, f"block.{i}.res_unit2", 3)
        x = self._res_unit(x, f"block.{i}.res_unit3", 9)
        return x

    def __call__(self, z):
        x = z.astype(np.float64)[0]  # [64, T]
        x = self._conv(x, "conv1", pad=3)  # 64 -> 2048, k7
        for i, stride in enumerate(self.strides):
            x = self._block(x, i, stride)
        x = self._snake(x, "snake1")
        x = self._conv(x, "conv2", pad=3, bias=False)  # 128 -> 2, k7
        return x[None]  # [1, 2, L]
