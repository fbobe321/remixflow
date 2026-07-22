"""Full-song var-50 variation that PRESERVES the original vocals.

Pipeline: separate vocals from instrumental (Demucs) -> SDEdit-vary only the
instrumental with the 'moderate' steering -> remix the original vocals back on
top (tempo/length preserved, so they stay aligned) -> save MP3.
"""
import time
import numpy as np
import soundfile as sf
import torch

from remixflow.audio.io import Clip, load
from remixflow.generation.acestep import AceStepGenerator
from remixflow.models import Steering

SRC = "/data3/remixflow/I_Will_Never_Fall.mp3"
OUT = "/data3/remixflow/I_Will_Never_Fall__var50_vocals.mp3"
VOCAL_GAIN = 1.0        # relative level of the preserved vocals in the remix
INSTR_GAIN = 1.0

clip = load(SRC)
sr = clip.sample_rate
print(f"source: {clip.duration:.1f}s @ {sr}Hz", flush=True)

# --- 1. Separate vocals / instrumental with Demucs ------------------------
from demucs.api import Separator

print("separating stems (Demucs htdemucs)…", flush=True)
t0 = time.time()
sep = Separator(model="htdemucs", device="cuda:1")
# separate_tensor wants [channels, samples] at model.samplerate (44100).
wav = torch.from_numpy(clip.samples.astype(np.float32))
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)
_, stems = sep.separate_tensor(wav, sr)
vocals = stems["vocals"].cpu().numpy()                       # [2, N]
instrumental = sum(stems[k] for k in ("drums", "bass", "other")).cpu().numpy()
print(f"  separated in {time.time()-t0:.1f}s", flush=True)

# --- 2. SDEdit-vary ONLY the instrumental ---------------------------------
instr_clip = Clip(samples=instrumental, sample_rate=sr)
steer = Steering(
    controls={"variation_amount": 0.5, "rock": 0.6, "energy": 0.5,
              "brightness": 0.5, "bass": 0.4},
    locks=["melody"],
)
g = AceStepGenerator()
print(f"varying instrumental: strength={g._strength(steer)} prompt='{g.build_prompt(steer)}'", flush=True)
t0 = time.time()
varied = g.generate(instr_clip, steer, seed=42).clip     # 48 kHz
print(f"  varied in {time.time()-t0:.1f}s -> {varied.duration:.1f}s @ {varied.sample_rate}Hz", flush=True)

# --- 3. Remix original vocals on top (resample vocals to 48k, align) ------
import librosa
vsr = varied.sample_rate
voc48 = np.stack([librosa.resample(vocals[c], orig_sr=sr, target_sr=vsr) for c in range(vocals.shape[0])])
n = min(voc48.shape[-1], varied.samples.shape[-1])
mix = INSTR_GAIN * varied.samples[:, :n] + VOCAL_GAIN * voc48[:, :n]
peak = float(np.max(np.abs(mix)))
if peak > 1e-6:
    mix = mix * (0.944 / peak)   # normalize to -0.5 dBFS

sf.write(OUT, np.clip(mix.T, -1, 1), vsr, format="MP3", subtype="MPEG_LAYER_III")
import os
print(f"SAVED {OUT} ({os.path.getsize(OUT)//1024} KB, {n/vsr:.1f}s)", flush=True)
print("DONE", flush=True)
