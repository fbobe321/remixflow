"""Manual ACE-Step generation check against a real song.

Usage (in the `acestep` conda env):
    python scripts/try_acestep.py /path/to/song.mp3 [duration_sec]

Loads the pipeline (cached weights), runs a real task_type="cover" generation
with a sample steering payload, saves the output, and prints timing + the
identity-similarity score vs. the source.
"""

import sys
import time

import numpy as np

from remixflow.audio import analysis
from remixflow.audio.io import Clip, load, save
from remixflow.generation.acestep import AceStepGenerator
from remixflow.models import Steering

path = sys.argv[1]
max_dur = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

print(f"loading source: {path}", flush=True)
clip = load(path)
if max_dur and clip.duration > max_dur:
    n = int(max_dur * clip.sample_rate)
    clip = Clip(samples=clip.samples[..., :n], sample_rate=clip.sample_rate)
print(f"source: {clip.samples.shape} sr={clip.sample_rate} dur={clip.duration:.1f}s", flush=True)

src_feats = analysis.analyze(clip)
print(f"source features: tempo={src_feats.tempo_bpm} key={src_feats.key}", flush=True)

g = AceStepGenerator()
t0 = time.time()
pipe = g._pipeline()
print(f"pipeline ready in {time.time()-t0:.1f}s (from cache)", flush=True)

steering = Steering(
    controls={
        "variation_amount": 0.4,
        "rock": 0.5,
        "energy": 0.4,
        "brightness": 0.5,
        "bass": 0.4,
    },
    locks=["melody"],
)
print(f"prompt: {g.build_prompt(steering)}", flush=True)
print(f"strength: {g._strength(steering)}", flush=True)

t0 = time.time()
result = g.generate(clip, steering, seed=42)
dt = time.time() - t0
out = result.clip
print(f"GENERATED in {dt:.1f}s  shape={out.samples.shape} sr={out.sample_rate} "
      f"dur={out.duration:.1f}s", flush=True)
print(f"note: {result.note}", flush=True)

out_path = "/tmp/claude-1000/-data3-remixflow/eebcdae0-d882-407a-89c2-f62d2eb7fb47/scratchpad/acestep_out.wav"
save(out, out_path)
gen_feats = analysis.analyze(out)
sim = analysis.similarity(src_feats.embedding, gen_feats.embedding)
print(f"OUTPUT saved: {out_path}", flush=True)
print(f"generated features: tempo={gen_feats.tempo_bpm} key={gen_feats.key}", flush=True)
print(f"IDENTITY SIMILARITY vs source: {sim}", flush=True)
print("RTF (real-time factor): %.2fx" % (out.duration / dt if dt else 0), flush=True)
