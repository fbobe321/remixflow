"""Capture a clean fp32 reference of the AceStepConditionEncoder on synthetic
(shape-correct) inputs. Includes lyric padding so the pack/sort path is exercised
and 3 timbre segments so the unpack path is exercised.

Saves condenc_input.npz (all inputs) and condenc_output.npy (encoder_hidden_states).
"""
import json
import os

import numpy as np
import torch

SNAP = "/home/bobef/.cache/huggingface/hub/models--ACE-Step--acestep-v15-xl-turbo-diffusers/snapshots/200ba991ae448051e14b0183157e35c2d27c9fb0"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(OUT, exist_ok=True)

from diffusers.pipelines.ace_step.modeling_ace_step import AceStepConditionEncoder

print("loading condition encoder (fp32, cpu)…", flush=True)
ce = AceStepConditionEncoder.from_pretrained(os.path.join(SNAP, "condition_encoder"), torch_dtype=torch.float32)
ce.eval()
cfg = ce.config
text_dim = cfg.text_hidden_dim      # 1024
timbre_dim = cfg.timbre_hidden_dim  # 64

g = torch.Generator().manual_seed(0)
Lt, Ll, seg_len, n_seg = 20, 30, 250, 3
text = torch.randn(1, Lt, text_dim, generator=g)
text_mask = torch.ones(1, Lt, dtype=torch.long)
lyric = torch.randn(1, Ll, text_dim, generator=g)
lyric_mask = torch.ones(1, Ll, dtype=torch.long)
lyric_mask[:, -5:] = 0                                   # padding -> exercise pack/sort
refer = torch.randn(n_seg, seg_len, timbre_dim, generator=g)  # 3 timbre segments
order = torch.zeros(n_seg, dtype=torch.long)            # all belong to batch 0

print("running forward…", flush=True)
with torch.no_grad():
    hs, mask = ce(
        text_hidden_states=text, text_attention_mask=text_mask,
        lyric_hidden_states=lyric, lyric_attention_mask=lyric_mask,
        refer_audio_acoustic_hidden_states_packed=refer, refer_audio_order_mask=order,
    )
print("output:", tuple(hs.shape), "mask sum", int(mask.sum()), flush=True)

np.savez(os.path.join(OUT, "condenc_input.npz"),
         text=text.numpy(), text_mask=text_mask.numpy(),
         lyric=lyric.numpy(), lyric_mask=lyric_mask.numpy(),
         refer=refer.numpy(), order=order.numpy())
np.save(os.path.join(OUT, "condenc_output.npy"), hs.numpy())
np.save(os.path.join(OUT, "condenc_output_mask.npy"), mask.numpy())
raw = json.load(open(os.path.join(SNAP, "condition_encoder", "config.json")))
json.dump(raw, open(os.path.join(OUT, "condenc_config.json"), "w"), indent=2)
print("DONE", flush=True)
