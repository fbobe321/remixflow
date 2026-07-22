"""De-risk the MLX port WITHOUT a Mac: emulate MLX's channels-last (N,L,C) layout
and (C_out, K, C_in) weight layout in NumPy, applying the exact same weight
transposes as vae_decoder_mlx.py, and check parity vs the PyTorch reference.

If this passes, the only remaining risk in the MLX file is MLX's own conv op
correctness (well-tested upstream) — the layout/transpose logic is proven.
"""
import json
import math
import os

import numpy as np
from safetensors.numpy import load_file

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def conv1d_nlc(x, w, b, stride=1, pad=0, dilation=1):
    # x:[N,L,Cin]  w:[Cout,K,Cin]  (MLX layout)
    N, L, Cin = x.shape
    Cout, K, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    Lp = xp.shape[1]
    Lout = (Lp - dilation * (K - 1) - 1) // stride + 1
    idx = np.arange(Lout)[:, None] * stride + np.arange(K)[None, :] * dilation  # [Lout,K]
    cols = xp[:, idx, :]                       # [N, Lout, K, Cin]
    cols = cols.reshape(N, Lout, K * Cin)
    wm = w.reshape(Cout, K * Cin)              # [Cout, K*Cin]
    out = cols @ wm.T                          # [N, Lout, Cout]
    if b is not None:
        out = out + b
    return out


def conv_transpose1d_nlc(x, w, b, stride=1, pad=0):
    # x:[N,L,Cin]  w:[Cout,K,Cin] (MLX layout). Full transpose then crop.
    N, L, Cin = x.shape
    Cout, K, _ = w.shape
    full = (L - 1) * stride + K
    out = np.zeros((N, full, Cout), dtype=np.float64)
    for k in range(K):
        contrib = x @ w[:, k, :].T            # [N, L, Cout]
        pos = np.arange(L) * stride + k
        out[:, pos, :] += contrib
    if b is not None:
        out = out + b
    if pad > 0:
        out = out[:, pad:full - pad, :]
    return out


def snake(x, a_raw, b_raw):
    a = np.exp(a_raw).reshape(-1)
    b = np.exp(b_raw).reshape(-1)
    return x + (1.0 / (b + 1e-9)) * np.sin(a * x) ** 2


class Dec:
    def __init__(self, path):
        raw = load_file(path)
        self.w = {}
        for k, v in raw.items():
            v = v.astype(np.float64)
            if k.endswith(".weight") and "conv_t1" in k:
                self.w[k] = v.transpose(1, 2, 0)   # [Cin,Cout,K] -> [Cout,K,Cin]
            elif k.endswith(".weight") and "conv" in k:
                self.w[k] = v.transpose(0, 2, 1)   # [Cout,Cin,K] -> [Cout,K,Cin]
            else:
                self.w[k] = v
        self.nb = sum(1 for k in raw if k.endswith(".conv_t1.weight"))
        self.strides = [raw[f"block.{i}.conv_t1.weight"].shape[2] // 2 for i in range(self.nb)]

    def sn(self, x, p): return snake(x, self.w[f"{p}.alpha"], self.w[f"{p}.beta"])

    def cv(self, x, p, pad=0, dil=1, bias=True):
        return conv1d_nlc(x, self.w[f"{p}.weight"], self.w.get(f"{p}.bias") if bias else None, 1, pad, dil)

    def cvt(self, x, p, s, pad):
        return conv_transpose1d_nlc(x, self.w[f"{p}.weight"], self.w.get(f"{p}.bias"), s, pad)

    def ru(self, x, p, d):
        o = self.sn(x, f"{p}.snake1"); o = self.cv(o, f"{p}.conv1", pad=((7 - 1) * d) // 2, dil=d)
        o = self.sn(o, f"{p}.snake2"); o = self.cv(o, f"{p}.conv2", pad=0)
        c = (x.shape[1] - o.shape[1]) // 2
        xr = x[:, c:x.shape[1] - c, :] if c > 0 else x
        return xr + o

    def blk(self, x, i, s):
        x = self.sn(x, f"block.{i}.snake1")
        x = self.cvt(x, f"block.{i}.conv_t1", s, math.ceil(s / 2))
        for ru, d in (("res_unit1", 1), ("res_unit2", 3), ("res_unit3", 9)):
            x = self.ru(x, f"block.{i}.{ru}", d)
        return x

    def __call__(self, z_nlc):
        x = self.cv(z_nlc, "conv1", pad=3)
        for i, s in enumerate(self.strides):
            x = self.blk(x, i, s)
        x = self.sn(x, "snake1")
        return self.cv(x, "conv2", pad=3, bias=False)


if __name__ == "__main__":
    d = Dec(os.path.join(FIX, "vae_decoder.safetensors"))
    print("strides:", d.strides)
    z = np.load(os.path.join(FIX, "parity_input.npy")).transpose(0, 2, 1)  # NLC
    ref = np.load(os.path.join(FIX, "parity_output.npy"))
    out = d(z).transpose(0, 2, 1)  # back to NCL
    err = np.abs(out - ref); denom = np.abs(ref).max()
    print("shapes:", out.shape, ref.shape)
    print(f"rel max err: {err.max()/denom:.3e} | corr: {np.corrcoef(out.flatten(), ref.flatten())[0,1]:.10f}")
    print("MLX_LAYOUT_PARITY_PASS" if err.max() / denom < 1e-3 else "FAIL")
