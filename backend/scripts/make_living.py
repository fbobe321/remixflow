"""Render a Living Songs stream from the seed and validate quality.

Prints per-window identity/strength, checks seam continuity, and saves an MP3.
Run in the acestep env with ACESTEP_DEVICE set.
"""
import sys
import time

import numpy as np
import soundfile as sf

from remixflow.audio.io import load
from remixflow.generation.acestep import AceStepGenerator
from remixflow.living import LivingConfig, LivingEngine, TensionCurve
from remixflow.models import Steering

SRC = "/data3/remixflow/I_Will_Never_Fall.mp3"
OUT = "/data3/remixflow/I_Will_Never_Fall__living.mp3"
DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 75.0
LO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
HI = float(sys.argv[3]) if len(sys.argv) > 3 else 0.22

src = load(SRC)
print(f"source {src.duration:.1f}s; rendering {DUR:.0f}s living stream", flush=True)

cfg = LivingConfig(
    duration_sec=DUR,
    window_sec=12.0,
    overlap_sec=2.5,
    identity_threshold=0.85,
    preserve_vocals=True,
    base_seed=7,
    tension=TensionCurve(lo=LO, hi=HI, period=5, jitter=0.04),
    steering=Steering(controls={}),  # keep the original style; just let it breathe
)

eng = LivingEngine(AceStepGenerator())
t0 = time.time()
res = eng.render(src, cfg, progress=lambda f, m: print(f"  [{f*100:4.0f}%] {m}", flush=True))
dt = time.time() - t0

out = res.clip
sf.write(OUT, np.clip(out.samples.T, -1, 1), out.sample_rate, format="MP3", subtype="MPEG_LAYER_III")
print(f"\nRENDERED {out.duration:.1f}s in {dt:.1f}s (RTF {out.duration/dt:.1f}x)", flush=True)
print("note:", res.note, flush=True)
print("\nper-window (idx  src_pos  strength  identity  retries):", flush=True)
for r in res.windows:
    print(f"  {r.index:2d}  {r.src_pos:6.1f}s   {r.strength:.2f}     {r.identity:.3f}    {r.retries}", flush=True)

# Seam continuity: sample step at each window boundary in the output.
y = out.samples.mean(axis=0)
osr = out.sample_rate
hop = (cfg.window_sec - cfg.overlap_sec)
print("\nseam continuity (|Δ| at window boundaries, lower=smoother):", flush=True)
for k in range(1, len(res.windows)):
    b = int(k * hop * osr)
    if 0 < b < len(y):
        print(f"  ~{k*hop:5.1f}s: {abs(y[b]-y[b-1]):.4f}", flush=True)
# Non-repetition proof: split into passes at each wrap, then compare every
# pass-2 window to the NEAREST pass-1 window at the same source position.
def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

passes, prev = [[]], -1.0
for r in res.windows:
    if r.src_pos < prev - 5:      # src position jumped back = new pass
        passes.append([])
    passes[-1].append(r)
    prev = r.src_pos

print(f"\nINFINITE PROOF — {len(passes)} pass(es) over the song "
      f"(window counts {[len(p) for p in passes]}):", flush=True)
if len(passes) < 2:
    print("  (stream didn't wrap; render longer)", flush=True)
else:
    p1 = passes[0]
    for r2 in passes[1]:
        r1 = min(p1, key=lambda r: abs(r.src_pos - r2.src_pos))
        print(f"  same source ~{r2.src_pos:5.1f}s: pass1 strength {r1.strength:.2f} → "
              f"pass2 {r2.strength:.2f}; variant similarity {cos(r1.embedding, r2.embedding):.3f} "
              f"(1.0 would mean identical replay)", flush=True)

import os
print(f"\nSAVED {OUT} ({os.path.getsize(OUT)//1024} KB)", flush=True)
print("DONE", flush=True)
