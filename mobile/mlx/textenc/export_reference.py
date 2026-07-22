"""Capture a ground-truth Qwen3 text-encoder forward (last_hidden_state) for
parity testing the NumPy/MLX ports. Weights load from the HF snapshot.

Saves: textenc_input.npy (input_ids), textenc_output.npy (last_hidden_state).
"""
import json
import os

import numpy as np
import torch

SNAP = "/home/bobef/.cache/huggingface/hub/models--ACE-Step--acestep-v15-xl-turbo-diffusers/snapshots/200ba991ae448051e14b0183157e35c2d27c9fb0"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(OUT, exist_ok=True)

from transformers import AutoModel

print("loading Qwen3 text encoder (fp32, cpu)…", flush=True)
m = AutoModel.from_pretrained(os.path.join(SNAP, "text_encoder"), torch_dtype=torch.float32)
m.eval()
cfg = m.config

L = 24
rng = np.random.default_rng(0)
ids = rng.integers(0, cfg.vocab_size, size=(1, L)).astype(np.int64)

with torch.no_grad():
    out = m(input_ids=torch.tensor(ids)).last_hidden_state  # [1, L, hidden]
print("output:", tuple(out.shape), "mean %.4f std %.4f" % (out.mean(), out.std()), flush=True)

np.save(os.path.join(OUT, "textenc_input.npy"), ids)
np.save(os.path.join(OUT, "textenc_output.npy"), out.numpy())
raw = json.load(open(os.path.join(SNAP, "text_encoder", "config.json")))
json.dump({k: raw[k] for k in
           ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "rope_theta", "rms_norm_eps", "vocab_size"]},
          open(os.path.join(OUT, "textenc_config.json"), "w"), indent=2)
print("DONE", flush=True)
