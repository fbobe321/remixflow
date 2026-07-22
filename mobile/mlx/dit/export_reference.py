"""Capture a ground-truth forward of the ACE-Step DiT (standalone) with seeded
synthetic inputs, for parity testing the NumPy/MLX ports.

Weights are NOT re-exported — the ports load them straight from the HF snapshot
(transformer/*.safetensors). Only the small parity fixture is saved:
  dit_input.npz   hidden_states, context_latents, encoder_hidden_states, timestep, timestep_r
  dit_output.npy  the DiT output (velocity) [1, T, 64]  (ground truth)

Sequence length is chosen > sliding_window(128) after patchify so the banded
sliding-attention mask is actually exercised.
"""
import json
import os

import numpy as np
import torch

SNAP = "/home/bobef/.cache/huggingface/hub/models--ACE-Step--acestep-v15-xl-turbo-diffusers/snapshots/200ba991ae448051e14b0183157e35c2d27c9fb0"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(OUT, exist_ok=True)

from diffusers import AceStepTransformer1DModel

print("loading DiT (fp32, cpu) — 4.17B params, ~17 GB RAM…", flush=True)
m = AceStepTransformer1DModel.from_pretrained(os.path.join(SNAP, "transformer"), torch_dtype=torch.float32)
m.eval()
cfg = m.config

T = 300  # -> seq 150 after patchify (> sliding_window 128)
acoustic = cfg.audio_acoustic_hidden_dim          # 64
context_dim = cfg.in_channels - acoustic          # 128
enc_dim = cfg.encoder_hidden_size                 # 2048
L_enc = 32

g = torch.Generator().manual_seed(0)
hidden = torch.randn(1, T, acoustic, generator=g, dtype=torch.float32)
context = torch.randn(1, T, context_dim, generator=g, dtype=torch.float32)
enc = torch.randn(1, L_enc, enc_dim, generator=g, dtype=torch.float32)
t = torch.tensor([0.7], dtype=torch.float32)
t_r = torch.tensor([0.7], dtype=torch.float32)

print("running forward…", flush=True)
with torch.no_grad():
    out = m(hidden_states=hidden, timestep=t, timestep_r=t_r,
            encoder_hidden_states=enc, context_latents=context, return_dict=False)[0]
print("output:", tuple(out.shape), flush=True)

np.savez(os.path.join(OUT, "dit_input.npz"),
         hidden=hidden.numpy(), context=context.numpy(), enc=enc.numpy(),
         t=t.numpy(), t_r=t_r.numpy())
np.save(os.path.join(OUT, "dit_output.npy"), out.numpy())
json.dump({k: getattr(cfg, k) for k in
           ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "in_channels", "audio_acoustic_hidden_dim",
            "patch_size", "rope_theta", "rms_norm_eps", "sliding_window", "encoder_hidden_size",
            "layer_types"]},
          open(os.path.join(OUT, "dit_config.json"), "w"), indent=2)
print("saved fixture. output stats: mean %.4f std %.4f" % (out.mean(), out.std()), flush=True)
print("DONE", flush=True)
