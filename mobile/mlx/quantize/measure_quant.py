"""Measure the quality hit of MLX-style group-affine quantization on the DiT,
without a Mac. MLX's k-bit matmul == matmul with dequantized weights, so we
quantize+dequantize the DiT's linear weights in NumPy and re-run the (parity-
proven) forward, comparing to the fp32 reference.

Reports per bit-width: relative error, correlation, and SNR (dB) of the DiT
velocity output vs fp32 — the signal that drives audio quality.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dit"))
from dit_numpy import DiTNumpy  # noqa: E402

SNAP = sys.argv[1]
FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def load_fp32(snap):
    import torch
    from safetensors.torch import load_file
    w = {}
    for f in sorted(glob.glob(os.path.join(snap, "transformer", "*.safetensors"))):
        for k, v in load_file(f).items():
            w[k] = v.to(torch.float32).numpy()
    return w


def quant_dequant(w, bits, gs=64):
    """MLX-style per-group affine quantize+dequantize along the last axis."""
    out_shape = w.shape
    flat = w.reshape(-1, gs)
    wmin = flat.min(1, keepdims=True)
    wmax = flat.max(1, keepdims=True)
    n = (1 << bits) - 1
    scale = (wmax - wmin) / n
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round((flat - wmin) / scale), 0, n)
    dq = q * scale + wmin
    return dq.reshape(out_shape).astype(np.float32)


def is_quantizable(k, v, gs=64):
    # 2D matrices (Linear) with in%gs==0. Skip norms (1D), convs (3D),
    # scale_shift_table, embeddings handled separately.
    return (k.endswith(".weight") and v.ndim == 2 and v.shape[1] % gs == 0
            and "norm" not in k)


class DiTFromWeights(DiTNumpy):
    def __init__(self, weights, config_path):
        self.w = weights
        c = json.load(open(config_path))
        self.H, self.heads, self.kv, self.hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.patch, self.theta, self.eps, self.sw = c["patch_size"], c["rope_theta"], c["rms_norm_eps"], c["sliding_window"]
        self.layer_types, self.n = c["layer_types"], c["num_hidden_layers"]


def run(weights):
    m = DiTFromWeights(weights, os.path.join(FIX, "dit_config.json"))
    d = np.load(os.path.join(FIX, "dit_input.npz"))
    return m(d["hidden"], d["context"], d["enc"], d["t"], d["t_r"])


def stats(out, ref, label):
    err = out - ref
    rel = np.abs(err).max() / np.abs(ref).max()
    corr = np.corrcoef(out.flatten(), ref.flatten())[0, 1]
    snr = 10 * np.log10((ref ** 2).mean() / (err ** 2).mean())
    print(f"  {label:14} rel-max {rel:.2e} | corr {corr:.7f} | SNR {snr:5.1f} dB")
    return snr


print("loading fp32 DiT weights…", flush=True)
base = load_fp32(SNAP)
ref = np.load(os.path.join(FIX, "dit_output.npy"))
nq = sum(1 for k, v in base.items() if is_quantizable(k, v))
tot = sum(1 for k in base if k.endswith(".weight"))
print(f"quantizable linear matrices: {nq}/{tot} weight tensors", flush=True)

print("\nDiT velocity output vs fp32 reference:")
stats(run(base).astype(np.float32), ref, "fp32 (baseline)")
for bits in (8, 4, 3):
    qw = {k: (quant_dequant(v, bits) if is_quantizable(k, v) else v) for k, v in base.items()}
    stats(run(qw).astype(np.float32), ref, f"{bits}-bit")
print("\n(4-bit is the target; higher SNR = closer to fp32. Audio quality tracks this.)")
