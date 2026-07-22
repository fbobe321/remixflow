"""Export the Oobleck VAE **encoder** (waveform -> latent params) + a parity
fixture. The encoder is needed for SDEdit (source waveform -> latents)."""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

SNAP = "/home/bobef/.cache/huggingface/hub/models--ACE-Step--acestep-v15-xl-turbo-diffusers/snapshots/200ba991ae448051e14b0183157e35c2d27c9fb0"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures")

from diffusers import AutoencoderOobleck

print("loading VAE…", flush=True)
vae = AutoencoderOobleck.from_pretrained(os.path.join(SNAP, "vae"), torch_dtype=torch.float32).eval()
enc = vae.encoder

torch.manual_seed(0)
x = torch.randn(1, vae.config.audio_channels, 96000, dtype=torch.float32)  # 2s @ 48k
with torch.no_grad():
    params = enc(x)                      # [1, 128, T]  (mean|scale before chunk)
    dist = vae.encode(x).latent_dist
    mean = dist.mean                     # [1, 64, T]
print("encoder params:", tuple(params.shape), "mean:", tuple(mean.shape), flush=True)

np.save(os.path.join(OUT, "vae_enc_input.npy"), x.numpy())
np.save(os.path.join(OUT, "vae_enc_params.npy"), params.numpy())
np.save(os.path.join(OUT, "vae_enc_mean.npy"), mean.numpy())

folded = {}
for name, m in enc.named_modules():
    if isinstance(m, torch.nn.Conv1d):
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
        folded[f"{name}.weight"] = m.weight.detach().contiguous()
        if m.bias is not None:
            folded[f"{name}.bias"] = m.bias.detach().contiguous()
    if m.__class__.__name__ == "Snake1d":
        folded[f"{name}.alpha"] = m.alpha.detach().contiguous()
        folded[f"{name}.beta"] = m.beta.detach().contiguous()
save_file(folded, os.path.join(OUT, "vae_encoder.safetensors"))
json.dump({"audio_channels": vae.config.audio_channels,
           "encoder_hidden_size": vae.config.encoder_hidden_size,
           "downsampling_ratios": list(vae.config.downsampling_ratios),
           "decoder_input_channels": vae.config.decoder_input_channels},
          open(os.path.join(OUT, "vae_encoder_config.json"), "w"), indent=2)
print("exported", len(folded), "tensors", flush=True)
print("DONE", flush=True)
