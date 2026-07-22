"""Sweep mixed-precision quantization configs on the DiT to find a ~4-bit-sized
recipe that recovers quality. Measures velocity-output SNR vs fp32 + the average
bits/weight (size)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dit"))
from dit_numpy import DiTNumpy  # noqa

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


def qd(w, bits, gs):
    s = w.shape
    flat = w.reshape(-1, gs)
    lo, hi = flat.min(1, keepdims=True), flat.max(1, keepdims=True)
    n = (1 << bits) - 1
    sc = np.where((hi - lo) == 0, 1.0, (hi - lo) / n)
    q = np.clip(np.round((flat - lo) / sc), 0, n)
    return (q * sc + lo).reshape(s).astype(np.float32)


def is_mat(k, v, gs=32):
    return k.endswith(".weight") and v.ndim == 2 and v.shape[1] % gs == 0 and "norm" not in k


class DiTW(DiTNumpy):
    def __init__(self, weights, cp):
        self.w = weights
        c = json.load(open(cp))
        self.H, self.heads, self.kv, self.hd = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.patch, self.theta, self.eps, self.sw = c["patch_size"], c["rope_theta"], c["rms_norm_eps"], c["sliding_window"]
        self.layer_types, self.n = c["layer_types"], c["num_hidden_layers"]


base = None
ref = np.load(os.path.join(FIX, "dit_output.npy"))
inp = np.load(os.path.join(FIX, "dit_input.npz"))


def snr(out):
    err = out.astype(np.float32) - ref
    return 10 * np.log10((ref ** 2).mean() / (err ** 2).mean())


def bits_for(k):
    """Return (bits, gs) per weight for a config, or None to keep fp16."""
    return None  # overridden per config


def build(cfg):
    """cfg: fn(key) -> (bits, gs) or None (keep fp16)."""
    total_bits, total_n = 0, 0
    w = {}
    for k, v in base.items():
        if is_mat(k, v):
            bg = cfg(k)
            if bg is None:
                w[k] = v; total_bits += 16 * v.size
            else:
                b, gs = bg
                w[k] = qd(v, b, gs)
                # affine store: b bits/weight + (scale+bias) 16-bit per group
                total_bits += (b + 32.0 / gs) * v.size
            total_n += v.size
        else:
            w[k] = v
    avg = total_bits / total_n
    return w, avg


def run(cfg, label):
    w, avg = build(cfg)
    out = DiTW(w, os.path.join(FIX, "dit_config.json"))(inp["hidden"], inp["context"], inp["enc"], inp["t"], inp["t_r"])
    print(f"  {label:34} avg {avg:4.1f} bits | SNR {snr(out):5.1f} dB", flush=True)


print("loading fp32 DiT weights…", flush=True)
base = load_fp32(SNAP)

print("\nMixed-precision sweep (DiT velocity SNR vs fp32):")
run(lambda k: (4, 64), "4-bit gs64 (naive)")
run(lambda k: (4, 32), "4-bit gs32")
run(lambda k: (8, 64) if "down_proj" in k else (4, 32), "4-bit gs32, down_proj@8")
run(lambda k: (8, 64) if ("self_attn" in k or "cross_attn" in k) else (4, 32), "attn@8, mlp@4 gs32")
run(lambda k: (8, 64) if ("down_proj" in k or "self_attn" in k or "cross_attn" in k) else (4, 32),
    "attn@8 + down@8, gate/up@4")
run(lambda k: (6, 32), "6-bit gs32")
run(lambda k: (8, 64), "8-bit gs64 (ceiling)")
print("\nGoal: SNR high enough that audio is preserved, avg bits near 4-5.")
