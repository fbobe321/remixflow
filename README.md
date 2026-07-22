# RemixFlow 🎵

**An AI-powered music evolution platform.** Keep everything you love about a
song — just make it slightly different. Import a track, move the steering
sliders, and get familiar-but-fresh variations you can branch from forever.

This repo implements the **first milestone** from [`PRD.md`](./PRD.md): the full
steering UI + a runnable backend with real audio analysis and a **pluggable
generation engine**. A real diffusion/transformer music model
(ACE-Step / Stable Audio / MusicGen) drops into the same `Generator` interface
later — everything around it (import → steer → generate → evaluate → branch →
learn) already works today.

> ⚙️ **Two generation backends.** Feature extraction (tempo/key/embedding),
> identity-similarity scoring, the evolution tree, A/B compare, morphing, and
> preference learning are fully implemented and backend-agnostic. Generation
> can run through either:
> - **`ace-step`** — the real generative model ([ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5),
>   diffusion, audio-to-audio SDEdit). Runs on GPU; see [ACE-Step setup](#ace-step-15-real-model-backend).
> - **`dsp`** — a dependency-light reference backend (time-stretch, EQ tilt,
>   drive, stereo width). Always available, instant, no model download. Proves
>   the pipeline and serves as a fallback.
>
> Pick per-request via the UI backend selector or `?backend=` on the API.

---

## Quick start

```bash
# 1. Backend (Python ≥ 3.10) — all deps from PyPI
cd backend
pip install -e ".[audio]"        # 'audio' extra adds librosa (tempo/key/pitch)

# 2. Frontend (only needed to (re)build the UI; end users don't need Node)
cd ../frontend
npm install && npm run build     # compiles into backend/remixflow/static/

# 3. Run — serves API + UI on one port (8770)
cd ../backend
remixflow serve                  # → http://127.0.0.1:8770
```

Open <http://127.0.0.1:8770>, drop in a song, and start steering.

### Dev mode (hot reload, two servers)

```bash
./dev.sh    # backend :8770 (reload) + Vite :5173 (proxies /api)
```

### Install from PyPI

```bash
pip install "remixflow[audio]"   # UI is bundled — no Node needed
remixflow serve                  # → http://127.0.0.1:8770
```

<https://pypi.org/project/remixflow/>

### Docker

Slim image (DSP backend + full UI, no GPU) — the fastest way to try it:

```bash
docker run --rm -p 8770:8770 fbobe3/remixflow      # → http://localhost:8770
# or:  docker compose up
```

Full **ACE-Step v1.5** generative backend (needs an NVIDIA GPU + the NVIDIA
Container Toolkit). It pulls the model from the self-hosted mirror
[`fbobe3/acestep-v15-xl-turbo-diffusers-mirror`](https://huggingface.co/fbobe3/acestep-v15-xl-turbo-diffusers-mirror)
on first generation:

```bash
docker build -f Dockerfile.gpu -t fbobe3/remixflow:gpu .
docker run --rm --gpus all -p 8770:8770 \
  -v hf-cache:/root/.cache/huggingface fbobe3/remixflow:gpu
# or:  docker compose --profile gpu up remixflow-gpu
```

Images: `fbobe3/remixflow:latest` (slim) · `fbobe3/remixflow:gpu` (build locally).

---

## How it maps to the PRD

| PRD section | Where it lives |
|-------------|----------------|
| §1 Song Import (MP3/WAV/FLAC/OGG) | `backend/remixflow/app.py` `POST /api/songs`, `audio/io.py` |
| §1/§2 Feature extraction + Musical DNA embedding | `audio/analysis.py` |
| §3 Evolution Controls (all sliders) | `params.py` → served to UI via `/api/controls` |
| §4 Identity Preservation (locks) | `params.py` `IDENTITY_ELEMENTS`, honored in `generation/dsp.py` |
| §5 Smart Evolution (branching tree) | `store.py` `tree()`, `components/EvolutionTree.tsx` |
| §6 Infinite Evolution Mode | `App.tsx` (∞ toggle: chained gentle drift) |
| §7 A/B Comparison | `components/ABPlayer.tsx` (position-preserving switch) |
| §8 Preference Learning | `service.py` `preference_profile()` |
| Advanced: Morph Between Songs | `POST /api/morph`, `service.py` `morph()` |
| Similarity Evaluator | `analysis.py` `similarity()` (cosine on embeddings) |
| Steering Engine | `generation/dsp.py` (sliders → transforms) |

## Architecture

```
 React UI (steering sliders, evolution tree, A/B player)
        │  /api
 FastAPI (app.py) ── service.py ── store.py (songs, variants, tree)
        │                 │
   audio/analysis.py   generation/ (Generator interface)
   (features, embedding,   ├─ dsp.py       ← reference backend (built in)
    similarity)            └─ <your model> ← ACE-Step / Stable Audio / MusicGen
```

The whole system degrades gracefully: with no audio libraries at all the API
still boots and serves the UI; with `soundfile` it handles WAV/FLAC/OGG; with
the `audio` extra (`librosa`) it adds tempo/key detection and time-stretch/
pitch-shift. MP3/WAV/FLAC/OGG all decode via the bundled libsndfile — **no
ffmpeg required**.

## ACE-Step 1.5 (real model backend)

The `ace-step` backend (`generation/acestep.py`) drives [ACE-Step v1.5](https://github.com/ace-step/ACE-Step-1.5)
via the diffusers `AceStepPipeline` on GPU. It needs a CUDA GPU (turbo variant
runs in <4 GB VRAM) and downloads weights on first use. Its stack
(torch/diffusers) is heavy, so keep it in its own environment:

```bash
conda create -y -n acestep python=3.11 && conda activate acestep
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "diffusers>=0.39" transformers accelerate soundfile librosa
pip install -e ./backend            # remixflow itself + fastapi/uvicorn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
remixflow serve                     # ace-step becomes the default backend
```

**How it generates variations (SDEdit img2img).** The turbo checkpoint ships
without ACE-Step's `audio_tokenizer`, so its `cover` task is unavailable. Instead
the backend VAE-encodes the source, noises it to a level set by `variation_amount`,
and denoises the truncated flow-matching schedule with the source as timbre
`reference_audio` — faithful identity preservation using only the VAE +
transformer + scheduler. Long songs are processed in crossfaded ~30 s chunks so
VRAM stays bounded at any length (verified: a 3.7-min track → 16 s generation,
tempo/key preserved, 0.996 identity similarity).

| RemixFlow | ACE-Step (SDEdit) |
|-----------|-------------------|
| parent audio | VAE-encoded source latents + `reference_audio` timbre |
| `variation_amount` | SDEdit noise level (identity locks lower it) |
| genre / tone / energy / vocal sliders | synthesized text `prompt` |
| `seed` | `torch.Generator` (reproducible) |

Env overrides: `ACESTEP_MODEL`, `ACESTEP_DEVICE` (`cuda:1` for the 2nd GPU),
`ACESTEP_DTYPE`, `ACESTEP_STEPS` (default 8), `ACESTEP_CHUNK_SEC` (30),
`ACESTEP_OVERLAP_SEC` (2). The backend self-probes: with torch/diffusers/weights
missing it reports `available: false` and the DSP backend takes over — the same
codebase runs with or without a GPU. Quick manual check:
`python backend/scripts/try_acestep.py <song.mp3> [seconds]`.

### Adding another backend

The generation seam is one abstract method — implement `Generator.generate` and
call `register_generator(...)` (see `generation/registry.py`). It then appears in
`/api/backends`, selectable via `POST /api/generate?backend=<name>`.

## Tests

```bash
cd backend && pytest -q     # end-to-end: import → generate → branch → rate → morph
```

## Roadmap (next milestones)

- ✅ Real generative backend (ACE-Step v1.5) behind the `Generator` interface — done.
- Learned music encoder (CLAP/MERT) replacing the MFCC embedding in `embed()`
  (the current identity score saturates; the A/B page uses mel-correlation instead).
- Evolution Timeline visualization; Mood/Seasonal preset packs (PRD Advanced).
- A learned preference model driving auto-suggested steering.
- Word-level vocal preservation (needs the original lyrics or a tokenizer-equipped checkpoint).
- Copyright-aware import (user-owned media / licensing) — see PRD Risks.

## License

MIT
