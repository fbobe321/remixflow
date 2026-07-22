"""Export the ACE-Step Oobleck VAE **decoder** to a portable form + a parity
fixture. Runs against PyTorch (CPU is fine — the VAE is 169M params).

Produces, under ../fixtures/:
  vae_decoder.safetensors   weight-norm-FOLDED decoder weights (plain conv + snake)
  vae_decoder_config.json   strides / channels / dims
  parity_input.npy          a latent  [1, 64, T]
  parity_output.npy         PyTorch decoded waveform [1, 2, T*1920]  (ground truth)

The MLX port (run on a Mac) loads the safetensors + parity_input and must match
parity_output within tolerance. The NumPy port (validate_numpy.py) does the same
here so the math is proven before touching MLX.
"""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

SNAP = "/home/bobef/.cache/huggingface/hub/models--ACE-Step--acestep-v15-xl-turbo-diffusers/snapshots/200ba991ae448051e14b0183157e35c2d27c9fb0"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(OUT, exist_ok=True)

from diffusers import AutoencoderOobleck

print("loading VAE (fp32, cpu)…", flush=True)
vae = AutoencoderOobleck.from_pretrained(os.path.join(SNAP, "vae"), torch_dtype=torch.float32)
vae.eval()
dec = vae.decoder

# A deterministic latent: ~4s of audio (100 latent frames @ 25 fps).
torch.manual_seed(0)
T = 100
z = torch.randn(1, vae.config.decoder_input_channels, T, dtype=torch.float32)

with torch.no_grad():
    ref = dec(z)  # ground-truth decode, weight-norm active
print("decoded:", tuple(z.shape), "->", tuple(ref.shape), flush=True)

np.save(os.path.join(OUT, "parity_input.npy"), z.numpy())
np.save(os.path.join(OUT, "parity_output.npy"), ref.numpy())

# Fold weight_norm: replace weight_g/weight_v with the effective `weight`.
folded = {}
for name, m in dec.named_modules():
    if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)  # now m.weight is effective
        folded[f"{name}.weight"] = m.weight.detach().contiguous()
        if m.bias is not None:
            folded[f"{name}.bias"] = m.bias.detach().contiguous()
    # Snake1d params (logscale): store raw alpha/beta (exp applied at runtime).
    if m.__class__.__name__ == "Snake1d":
        folded[f"{name}.alpha"] = m.alpha.detach().contiguous()
        folded[f"{name}.beta"] = m.beta.detach().contiguous()

save_file(folded, os.path.join(OUT, "vae_decoder.safetensors"))

cfg = {
    "decoder_channels": vae.config.decoder_channels,
    "decoder_input_channels": vae.config.decoder_input_channels,
    "audio_channels": vae.config.audio_channels,
    "channel_multiples": list(vae.config.channel_multiples),
    "upsampling_ratios": list(vae.config.downsampling_ratios),  # decoder uses these
    "snake_logscale": True,
    "total_upsample": int(np.prod(vae.config.downsampling_ratios)),
}
json.dump(cfg, open(os.path.join(OUT, "vae_decoder_config.json"), "w"), indent=2)
print("exported", len(folded), "tensors; config:", cfg, flush=True)
print("DONE", flush=True)
