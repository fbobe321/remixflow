"""End-to-end audio quality test of DiT quantization: run the REAL ACE-Step
pipeline with fp16 weights vs quantized-dequantized weights, and compare the
decoded audio (this is the definitive quality measure, not single-step SNR).

Generates the same SDEdit variation with each precision and reports
mel-spectrogram correlation vs the fp16 baseline. Saves wavs to the scratchpad.
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/data3/remixflow/backend")
from remixflow.audio.io import Clip, load, save  # noqa
from remixflow.generation.acestep import AceStepGenerator  # noqa
from remixflow.models import Steering  # noqa

OUT = "/tmp/claude-1000/-data3-remixflow/eebcdae0-d882-407a-89c2-f62d2eb7fb47/scratchpad"
SRC = "/data3/remixflow/I_Will_Never_Fall.mp3"


def qd_torch(w, bits, gs):
    s = w.shape
    flat = w.reshape(-1, gs).float()
    lo = flat.min(1, keepdim=True).values
    hi = flat.max(1, keepdim=True).values
    n = (1 << bits) - 1
    sc = (hi - lo) / n
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    q = torch.clamp(torch.round((flat - lo) / sc), 0, n)
    return (q * sc + lo).reshape(s).to(w.dtype)


def quantize_transformer(tr, bits, gs, backup):
    n = 0
    with torch.no_grad():
        for name, p in tr.named_parameters():
            if name.endswith(".weight") and p.ndim == 2 and p.shape[1] % gs == 0 and "norm" not in name:
                p.copy_(qd_torch(backup[name].to(p.device), bits, gs))
                n += 1
    return n


def mel(y, sr=44100):
    import librosa
    return librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64))


def melcorr(a, b):
    ma, mb = mel(a), mel(b)
    k = min(ma.shape[1], mb.shape[1])
    return float(np.corrcoef(ma[:, :k].flatten(), mb[:, :k].flatten())[0, 1])


g = AceStepGenerator()
pipe = g._pipeline()
tr = pipe.transformer
backup = {k: v.detach().clone().cpu() for k, v in tr.named_parameters()}  # fp16/bf16 originals
print("backed up transformer params", flush=True)

clip = load(SRC)
n = int(20 * clip.sample_rate)
seg = Clip(samples=clip.samples[:, :n], sample_rate=clip.sample_rate)
steer = Steering(controls={"variation_amount": 0.5, "rock": 0.5, "brightness": 0.4}, locks=[])

def gen(tag):
    out = g.generate(seg, steer, seed=42).clip
    y = out.to_mono()
    sf.write(f"{OUT}/quant_{tag}.wav", out.samples.T, out.sample_rate, subtype="PCM_16")
    return y

print("generating fp16 baseline…", flush=True)
base = gen("fp16")

results = {}
for bits, gs, tag in [(8, 64, "8bit"), (4, 32, "4bit_gs32"), (4, 64, "4bit_gs64")]:
    nq = quantize_transformer(tr, bits, gs, backup)
    print(f"generating {tag} ({nq} matrices quantized)…", flush=True)
    y = gen(tag)
    results[tag] = melcorr(base, y)
    # restore for next config
    with torch.no_grad():
        for name, p in tr.named_parameters():
            p.copy_(backup[name].to(p.device))

print("\n=== AUDIO quality vs fp16 baseline (mel-spectrogram correlation) ===")
for tag, c in results.items():
    print(f"  {tag:12}: {c:.4f}")
print("\nwavs saved to scratchpad: quant_fp16.wav + quant_{8bit,4bit_gs32,4bit_gs64}.wav")
print("DONE", flush=True)
