"""Generate the FULL song at the 'moderate' (var 50%) steering, save as MP3."""
import time
import numpy as np
import soundfile as sf

from remixflow.audio import analysis
from remixflow.audio.io import load
from remixflow.generation.acestep import AceStepGenerator
from remixflow.models import Steering

SRC = "/data3/remixflow/I_Will_Never_Fall.mp3"
OUT = "/data3/remixflow/I_Will_Never_Fall__var50.mp3"

clip = load(SRC)
print(f"source: {clip.duration:.1f}s", flush=True)
src_feats = analysis.analyze(clip)

steer = Steering(
    controls={"variation_amount": 0.5, "rock": 0.6, "energy": 0.5,
              "brightness": 0.5, "bass": 0.4},
    locks=["melody"],
)
g = AceStepGenerator()
print(f"strength={g._strength(steer)}  prompt='{g.build_prompt(steer)}'", flush=True)

t0 = time.time()
res = g.generate(clip, steer, seed=42)
dt = time.time() - t0
out = res.clip
print(f"GENERATED in {dt:.1f}s  ({out.duration:.1f}s, RTF {out.duration/dt:.1f}x)", flush=True)
print(f"note: {res.note}", flush=True)

data = out.samples.T if out.samples.ndim == 2 else out.samples[:, None]
sf.write(OUT, np.clip(data, -1, 1), out.sample_rate, format="MP3", subtype="MPEG_LAYER_III")

gen_feats = analysis.analyze(out)
print(f"tempo {src_feats.tempo_bpm}->{gen_feats.tempo_bpm}  key {src_feats.key}->{gen_feats.key}", flush=True)
import os
print(f"SAVED {OUT}  ({os.path.getsize(OUT)//1024} KB)", flush=True)
print("DONE", flush=True)
