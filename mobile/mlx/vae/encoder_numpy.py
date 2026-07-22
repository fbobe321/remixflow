"""NumPy Oobleck VAE encoder (waveform -> latent params). Mirror of the decoder;
reuses its conv/snake primitives. Output [B,128,T] = concat(mean, scale); the
latent mean is params[:, :64]. For SDEdit use the mean (or mean+softplus(scale)*eps)."""
from __future__ import annotations

import json
import math
import os

import numpy as np
from safetensors.numpy import load_file

from oobleck_numpy import conv1d, snake  # validated primitives


class OobleckEncoderNumpy:
    def __init__(self, weights_path, config_path):
        self.w = load_file(weights_path)
        self.cfg = json.load(open(config_path))
        # block.N.conv1.weight (3 dots); res-unit convs (block.N.res_unitK.conv1.weight) have 4.
        self.nb = sum(1 for k in self.w if k.startswith("block.") and k.endswith(".conv1.weight") and k.count(".") == 3)
        # per-block downsample stride from its conv1 kernel (k = 2*stride)
        self.strides = [self.w[f"block.{i}.conv1.weight"].shape[2] // 2 for i in range(self.nb)]

    def _sn(self, x, p):
        return snake(x, self.w[f"{p}.alpha"].astype(np.float64), self.w[f"{p}.beta"].astype(np.float64))

    def _cv(self, x, p, stride=1, pad=0, dil=1, bias=True):
        b = self.w.get(f"{p}.bias")
        b = b.astype(np.float64) if (bias and b is not None) else None
        return conv1d(x, self.w[f"{p}.weight"].astype(np.float64), b, stride, pad, dil)

    def _res(self, x, p, dil):
        o = self._sn(x, f"{p}.snake1")
        o = self._cv(o, f"{p}.conv1", pad=((7 - 1) * dil) // 2, dil=dil)
        o = self._sn(o, f"{p}.snake2")
        o = self._cv(o, f"{p}.conv2", pad=0)
        c = (x.shape[-1] - o.shape[-1]) // 2
        return (x[:, c:x.shape[-1] - c] if c > 0 else x) + o

    def _block(self, x, i, stride):
        x = self._res(x, f"block.{i}.res_unit1", 1)
        x = self._res(x, f"block.{i}.res_unit2", 3)
        x = self._res(x, f"block.{i}.res_unit3", 9)
        x = self._sn(x, f"block.{i}.snake1")
        return self._cv(x, f"block.{i}.conv1", stride=stride, pad=math.ceil(stride / 2))

    def __call__(self, x):
        x = x.astype(np.float64)[0]                 # [2, N]
        x = self._cv(x, "conv1", pad=3)             # 2 -> 128, k7
        for i, s in enumerate(self.strides):
            x = self._block(x, i, s)
        x = self._sn(x, "snake1")
        x = self._cv(x, "conv2", pad=1)             # 2048 -> 128, k3
        return x[None]                              # [1, 128, T] (mean|scale)
