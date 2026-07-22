"""MLX Oobleck VAE encoder (waveform -> latent params). Mirror of the decoder
MLX port (channels-last, conv weights transposed to (Cout,K,Cin)). 1:1 with
encoder_numpy (validated 3.9e-6 vs PyTorch).

    python encoder_mlx.py    # -> VAE_ENC_MLX_PARITY_PASS
"""
from __future__ import annotations

import json
import math
import os

import mlx.core as mx
import numpy as np

FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def snake(x, a_raw, b_raw):
    a = mx.exp(a_raw.reshape(-1)); b = mx.exp(b_raw.reshape(-1))
    return x + (1.0 / (b + 1e-9)) * mx.sin(a * x) ** 2


def conv1d(x, w, b, stride=1, pad=0, dilation=1):
    out = mx.conv1d(x, w, stride=stride, padding=pad, dilation=dilation)
    return out + b if b is not None else out


class OobleckEncoderMLX:
    def __init__(self, weights_path):
        raw = mx.load(weights_path)
        self.w = {}
        for k, v in raw.items():
            if k.endswith(".weight") and ".conv" in k or k in ("conv1.weight", "conv2.weight"):
                self.w[k] = v.transpose(0, 2, 1)  # [Cout,Cin,K] -> [Cout,K,Cin]
            else:
                self.w[k] = v
        self.nb = sum(1 for k in raw if k.startswith("block.") and k.endswith(".conv1.weight") and k.count(".") == 3)
        self.strides = [int(raw[f"block.{i}.conv1.weight"].shape[2]) // 2 for i in range(self.nb)]

    def _sn(self, x, p): return snake(x, self.w[f"{p}.alpha"], self.w[f"{p}.beta"])

    def _cv(self, x, p, stride=1, pad=0, dil=1, bias=True):
        b = self.w.get(f"{p}.bias") if bias else None
        return conv1d(x, self.w[f"{p}.weight"], b, stride, pad, dil)

    def _res(self, x, p, dil):
        o = self._sn(x, f"{p}.snake1")
        o = self._cv(o, f"{p}.conv1", pad=((7 - 1) * dil) // 2, dil=dil)
        o = self._sn(o, f"{p}.snake2")
        o = self._cv(o, f"{p}.conv2", pad=0)
        c = (x.shape[1] - o.shape[1]) // 2
        return (x[:, c:x.shape[1] - c, :] if c > 0 else x) + o

    def _block(self, x, i, stride):
        x = self._res(x, f"block.{i}.res_unit1", 1)
        x = self._res(x, f"block.{i}.res_unit2", 3)
        x = self._res(x, f"block.{i}.res_unit3", 9)
        x = self._sn(x, f"block.{i}.snake1")
        return self._cv(x, f"block.{i}.conv1", stride=stride, pad=math.ceil(stride / 2))

    def __call__(self, x_nlc):  # [N, L, 2]
        x = self._cv(x_nlc, "conv1", pad=3)
        for i, s in enumerate(self.strides):
            x = self._block(x, i, s)
        x = self._sn(x, "snake1")
        return self._cv(x, "conv2", pad=1)  # [N, T, 128]


def main():
    e = OobleckEncoderMLX(os.path.join(FIX, "vae_encoder.safetensors"))
    x = np.load(os.path.join(FIX, "vae_enc_input.npy"))       # [1,2,N] NCL
    ref = np.load(os.path.join(FIX, "vae_enc_params.npy"))    # [1,128,T] NCL
    out = np.array(e(mx.array(np.ascontiguousarray(x.transpose(0, 2, 1))))).transpose(0, 2, 1)
    err = np.abs(out - ref); denom = float(np.abs(ref).max())
    print(f"rel max {err.max()/denom:.3e} | corr {np.corrcoef(out.flatten(), ref.flatten())[0,1]:.10f}")
    print("VAE_ENC_MLX_PARITY_PASS" if err.max() / denom < 1e-3 else "FAIL")


if __name__ == "__main__":
    main()
