"""Generate a spread of ACE-Step variants on a 30s excerpt for A/B listening.

Writes compressed OGG excerpts (original + 3 variants) to the scratchpad so they
can be embedded in a self-contained comparison page.
"""
import numpy as np
import soundfile as sf

from remixflow.audio.io import Clip, load
from remixflow.generation.acestep import AceStepGenerator
from remixflow.models import Steering

SRC = "/data3/remixflow/I_Will_Never_Fall.mp3"
OUT = "/tmp/claude-1000/-data3-remixflow/eebcdae0-d882-407a-89c2-f62d2eb7fb47/scratchpad"
WIN = (45.0, 75.0)  # a 30s window into the song

clip = load(SRC)
sr = clip.sample_rate
a, b = int(WIN[0] * sr), int(WIN[1] * sr)
excerpt = Clip(samples=clip.samples[:, a:b], sample_rate=sr)
print(f"excerpt: {excerpt.duration:.1f}s @ {sr}Hz", flush=True)


def write_ogg(c: Clip, name: str):
    data = c.samples.T if c.samples.ndim == 2 else c.samples[:, None]
    path = f"{OUT}/ab_{name}.ogg"
    sf.write(path, np.clip(data, -1, 1), c.sample_rate, format="OGG", subtype="VORBIS")
    import os
    print(f"  wrote {name}: {os.path.getsize(path)//1024} KB", flush=True)


write_ogg(excerpt, "original")

g = AceStepGenerator()
g._pipeline()  # warm

configs = [
    ("subtle", Steering(controls={"variation_amount": 0.2}, locks=["melody", "rhythm"])),
    ("moderate", Steering(controls={"variation_amount": 0.5, "rock": 0.6, "energy": 0.5,
                                     "brightness": 0.5, "bass": 0.4}, locks=["melody"])),
    ("bold", Steering(controls={"variation_amount": 0.8, "jazz": 0.7, "warmth": 0.6,
                                 "complexity": 0.6, "swing": 0.4, "electronic": 0.3}, locks=[])),
]
for name, steer in configs:
    print(f"generating '{name}' (strength {g._strength(steer)}, prompt: {g.build_prompt(steer)})", flush=True)
    res = g.generate(excerpt, steer, seed=42)
    write_ogg(res.clip, name)
    print(f"  note: {res.note}", flush=True)

print("DONE", flush=True)
