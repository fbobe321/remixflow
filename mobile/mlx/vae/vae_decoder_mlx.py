"""MLX port of the ACE-Step Oobleck VAE decoder (run on Apple Silicon).

A 1:1 translation of the NumPy spec in ``oobleck_numpy.py`` (which is validated
to 6e-6 vs PyTorch). MLX uses **channels-last** (N, L, C) and conv weights
``(C_out, K, C_in)``, so weights are transposed from the PyTorch layout at load.
Transposed-convs are done with padding=0 then cropped manually, to avoid relying
on MLX's transpose-padding convention.

Run on a Mac:
    pip install mlx numpy safetensors
    python vae_decoder_mlx.py            # prints parity vs ../fixtures/parity_output.npy

Optionally quantize the linear/conv weights to 4-bit with mlx.nn.quantize in a
follow-up; the decoder is kept fp16 in the recommended build (quality-critical).
"""
from __future__ import annotations

import json
import math
import os

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "..", "fixtures")


def snake(x, alpha_raw, beta_raw):
    # x: [N, L, C]; params stored [1, C, 1] -> use as [C] on the last axis.
    a = mx.exp(alpha_raw.reshape(-1))
    b = mx.exp(beta_raw.reshape(-1))
    return x + (1.0 / (b + 1e-9)) * mx.sin(a * x) ** 2


def conv1d(x, w, b, stride=1, pad=0, dilation=1):
    # x: [N, L, Cin]; w(mlx): [Cout, K, Cin]
    out = mx.conv1d(x, w, stride=stride, padding=pad, dilation=dilation)
    if b is not None:
        out = out + b
    return out


def conv_transpose1d_crop(x, w, b, stride, pad):
    # x: [N, L, Cin]; w(mlx): [Cout, K, Cin]. Full transpose (pad=0), then crop.
    out = mx.conv_transpose1d(x, w, stride=stride, padding=0)
    if b is not None:
        out = out + b
    if pad > 0:
        out = out[:, pad:out.shape[1] - pad, :]
    return out


class OobleckDecoderMLX:
    def __init__(self, weights_path):
        raw = mx.load(weights_path)  # torch-layout weights as mx arrays
        self.w = {}
        for k, v in raw.items():
            if k.endswith(".weight") and "conv_t1" in k:
                # torch convT [Cin, Cout, K] -> mlx [Cout, K, Cin]
                self.w[k] = v.transpose(1, 2, 0)
            elif k.endswith(".weight") and (".conv" in k or k in ("conv1.weight", "conv2.weight")):
                # torch conv [Cout, Cin, K] -> mlx [Cout, K, Cin]
                self.w[k] = v.transpose(0, 2, 1)
            else:
                self.w[k] = v
        self.n_blocks = sum(1 for k in raw if k.endswith(".conv_t1.weight"))
        self.strides = [int(raw[f"block.{i}.conv_t1.weight"].shape[2]) // 2
                        for i in range(self.n_blocks)]

    def _snake(self, x, p):
        return snake(x, self.w[f"{p}.alpha"], self.w[f"{p}.beta"])

    def _conv(self, x, p, stride=1, pad=0, dilation=1, bias=True):
        b = self.w.get(f"{p}.bias") if bias else None
        return conv1d(x, self.w[f"{p}.weight"], b, stride, pad, dilation)

    def _convT(self, x, p, stride, pad):
        return conv_transpose1d_crop(x, self.w[f"{p}.weight"], self.w.get(f"{p}.bias"), stride, pad)

    def _res_unit(self, x, p, dilation):
        out = self._snake(x, f"{p}.snake1")
        out = self._conv(out, f"{p}.conv1", pad=((7 - 1) * dilation) // 2, dilation=dilation)
        out = self._snake(out, f"{p}.snake2")
        out = self._conv(out, f"{p}.conv2", pad=0)
        crop = (x.shape[1] - out.shape[1]) // 2
        xr = x[:, crop:x.shape[1] - crop, :] if crop > 0 else x
        return xr + out

    def _block(self, x, i, stride):
        x = self._snake(x, f"block.{i}.snake1")
        x = self._convT(x, f"block.{i}.conv_t1", stride=stride, pad=math.ceil(stride / 2))
        x = self._res_unit(x, f"block.{i}.res_unit1", 1)
        x = self._res_unit(x, f"block.{i}.res_unit2", 3)
        x = self._res_unit(x, f"block.{i}.res_unit3", 9)
        return x

    def __call__(self, z_nlc):
        # z_nlc: [N, T, 64]
        x = self._conv(z_nlc, "conv1", pad=3)
        for i, stride in enumerate(self.strides):
            x = self._block(x, i, stride)
        x = self._snake(x, "snake1")
        x = self._conv(x, "conv2", pad=3, bias=False)
        return x  # [N, L, 2]


def main():
    dec = OobleckDecoderMLX(os.path.join(FIX, "vae_decoder.safetensors"))
    print("derived strides:", dec.strides)
    z = np.load(os.path.join(FIX, "parity_input.npy"))          # [1, 64, T] (NCL)
    ref = np.load(os.path.join(FIX, "parity_output.npy"))       # [1, 2, L]  (NCL)
    z_nlc = mx.array(np.ascontiguousarray(z.transpose(0, 2, 1)))  # -> [1, T, 64]
    out = np.array(dec(z_nlc))                                  # [1, L, 2]
    out = out.transpose(0, 2, 1)                                # -> [1, 2, L]
    err = np.abs(out - ref)
    denom = float(np.abs(ref).max())
    print("shapes:", out.shape, "vs", ref.shape)
    print(f"max abs err: {err.max():.3e} | rel max: {err.max()/denom:.3e}")
    print(f"correlation: {np.corrcoef(out.flatten(), ref.flatten())[0,1]:.10f}")
    print("PARITY_PASS" if err.max() / denom < 1e-3 else "PARITY_FAIL")


if __name__ == "__main__":
    main()
