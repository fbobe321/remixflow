# RemixFlow — backend

FastAPI backend for the RemixFlow music-evolution platform. See the repo-root
`README.md` for the full picture; this file covers the Python package.

## Install

```bash
pip install -e ".[audio,dev]"   # audio extra adds librosa (tempo/pitch/key)
```

Core deps are pure PyPI wheels. The bundled libsndfile (via `soundfile`) reads
**and writes** MP3/WAV/FLAC/OGG with no system packages — **no ffmpeg needed**.
The `audio` extra pulls in `librosa` for tempo/key detection and
time-stretch/pitch-shift.

## Run

```bash
remixflow serve                 # http://127.0.0.1:8770  (API + built UI)
remixflow serve --reload        # dev auto-reload
```

Interactive API docs at `/docs`.

## Layout

| Path | Role |
|------|------|
| `params.py` | Single source of truth for every steering control & identity lock (PRD §3, §4). Served to the UI via `/api/controls`. |
| `models.py` | Pydantic schemas: `Song`, `Variant`, `Steering`, evolution `TreeNode`. |
| `audio/io.py` | Load/save with graceful degradation (soundfile → librosa). |
| `audio/analysis.py` | Feature extraction + Musical-DNA embedding + similarity. |
| `generation/base.py` | The `Generator` contract every backend implements. |
| `generation/dsp.py` | Reference DSP backend — real, audible variations today. |
| `generation/registry.py` | Plug point for model backends (ACE-Step, Stable Audio, MusicGen). |
| `service.py` | Orchestration: import → steer → generate → evaluate → branch. |
| `store.py` | File-backed store (JSON + audio dir); swap for a DB later. |
| `app.py` | FastAPI routes; serves the built React UI from `static/`. |

## Adding a real model backend

Implement `Generator` and register it:

```python
from remixflow.generation import Generator, register_generator

class AceStepBackend(Generator):
    name = "ace-step"
    description = "ACE-Step 1.5 diffusion"
    def generate(self, parent, steering, *, original=None, seed=None):
        ...  # map steering -> latent edits, run the model, return GenerationResult

register_generator(AceStepBackend())
```

It then appears in `/api/backends` and is selectable via `POST /api/generate?backend=ace-step`.

## Config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `REMIXFLOW_DATA_DIR` | `./remixflow_data` | Where songs/variants/audio persist. |
| `REMIXFLOW_MAX_UPLOAD_MB` | `50` | Upload size cap. |
| `REMIXFLOW_CORS` | `*` | Comma-separated allowed origins. |
