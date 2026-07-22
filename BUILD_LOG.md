# RemixFlow — Build Log

A running log of implementation progress, decisions, and current state, so work
can resume without losing context. Newest entries at the bottom.

---

## ▶ RESUME HERE (current state — 2026-07-21)

**DEPLOYED (public):**
- GitHub: https://github.com/fbobe321/remixflow (main)
- PyPI: https://pypi.org/project/remixflow/ (`pip install "remixflow[audio]"`, UI bundled)
- Docker Hub: `fbobe3/remixflow:latest` (slim, 932 MB, DSP+UI) + `Dockerfile.gpu` (ACE-Step)
- HF model mirror: https://huggingface.co/fbobe3/acestep-v15-xl-turbo-diffusers-mirror
  (credits original; GPU image pulls from it via `ACESTEP_MODEL`)

**HARD RULE (user): never publish music files.** Verified: no audio in git
(any commit), wheel, or Docker image; `.gitignore` covers `*.mp3 *.wav samples/`.
The earlier A/B artifact was republished with audio removed.

**Playlist ✅**: Living Mode plays a set of songs (checkbox playlist in
LivingControls), evolving each ~`perSongSec` then transitioning (crossfade),
looping. Frontend-orchestrated (player calls `/api/living` per song) — backend
unchanged. Built locally; needs a re-release (0.1.x) to reach PyPI/Docker.

**Tokens shared in chat — user should ROTATE**: GitHub PAT, PyPI token, HF token,
Docker PAT. Not stored in any repo/file, but they're in the transcript.

---


**Phase 1 (RemixFlow): complete & verified.** UI + FastAPI + two generation
backends (DSP fallback, **ACE-Step v1.5 SDEdit** on GPU), async jobs, evolution
tree, A/B, morph, preference learning, **vocal preservation** (Demucs, triggered
by locking lyrics/vocal_phrasing). Default port **8770** (avoid 8000/8188).

**Phase 2 (Living Songs): engine + Living Mode player shipped.** Continuous
Evolution Engine (`living.py`) + Living API (`POST /api/living`, chained via
`next_index`) + browser Living player (prefetch-ahead, Living Repeat, controls).
Verified infinite claim (6-min = 2 passes, non-repeating) and full ACE-Step path
through the app. Latest demo `I_Will_Never_Fall__living.mp3` (6-min medium
tension, morphs audibly).

**Tuning learned:** 0.08–0.22 too subtle ("can't tell a difference"); 0.18–0.5
too aggressive (artifacts). Sweet spot ~0.15–0.42, and the **instrumental prompt
hint** (fixes vocal-clash weirdness) is what makes higher strength clean.
Improvisation slider → tension range `lo 0.08+0.12·imp, hi 0.20+0.35·imp`.

**Listening-mode presets ✅**: 6 built-ins (Studio/Live/Jazz/Orchestra/Ambient/
Infinite Radio) + user-saved presets (persisted to `presets.json`). `presets.py`,
`GET/POST/DELETE /api/presets`, picker + Save-preset in `LivingControls.tsx`.

**Gapless playback ✅**: `LivingPlayer.tsx` rewritten to use the **Web Audio API**
— segments decoded to AudioBuffers and scheduled back-to-back (sample-accurate),
45s buffered ahead, play/pause via ctx suspend/resume. Fixes the ~1s stall from
the old single-`<audio>` src-swap.

**Segment-overlap fix**: user heard ~0.5–1s doubling at joins. Cause: next
segment started at window-index `i*hop`, but the previous OUTPUT (trimmed to
`duration_sec`, 1:1 with source time) ended at a *different* source position, so
the window-crossfade region was re-covered by both segments. Fix: engine now
tracks the exact source cursor — `render(start_pos_sec=…)` and returns
`next_pos_sec = start_pos + duration`; the player passes `next_pos` back as the
next `start_pos`, so segments abut with **no shared source content**. Added a
0.18s per-segment crossfade in the player to blend the seed change at the join.

**pkill self-match gotcha**: `pkill -9 -f "bin/remixflow serve"` run in the SAME
bash command as `exec remixflow serve` matched the wrapper's own command line and
killed the shell before exec → server never started (exit 1, no output). Run
pkill in a separate command, or just start clean when nothing's bound. (The
`exec` form DOES make TaskStop kill the server directly — no orphan, confirmed.)

**"Skipped beat" fix (time-aligned crossfade)**: the first crossfade overlapped
two DIFFERENT source moments → stole ~0.18s from the timeline at each join (heard
as a skipped beat). Fix: engine renders a `crossfade_sec` (0.25s) TAIL past the
advance — the tail is the SAME source region the next segment starts with — and
reports `advance_sec` (= duration_sec) separately from clip length. Player
advances the timeline by `advance` and crossfades the `clip - advance` tail
against the next segment's head (same source moment, time-aligned) → zero time
compression, smooth blend. Verified: clip 20.25s, advance 20.0s, next_pos 20.0s.

**GPU orphan fix**: `TaskStop` on a server task killed the bash wrapper but left
the uvicorn child holding ~11 GB on GPU 1 → CUDA OOM on the next start. Now start
with `exec remixflow serve …` (directly signal-killable) and `pkill -9 -f
"bin/remixflow serve"` before restarts. See [[remixflow-ports]] memory.

**Next (user to pick):** deeper DNA + real identity score · RemoteGenerator
(point pip player at a hosted model via `REMIXFLOW_MODEL_URL`) · per-feature
morphing (tempo/melody/chords independent).

**Envs:** `base` (app+DSP+frontend build, py3.13) · `acestep` (py3.11, torch cu124
+ diffusers + ACE-Step + demucs). Run ACE-Step from `acestep` env with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `ACESTEP_DEVICE=cuda:1`.

---

## 2026-07-21 — Milestone 1: app scaffold (UI + pluggable backend)

**Decision:** React + TypeScript (Vite) frontend, Python + FastAPI backend.
First milestone = full steering UI + runnable backend with real audio analysis
and a *pluggable* generation engine (real model plugs in later).

**Environment discovered:**
- No system Node/npm/ffmpeg. Node installed via `mamba install -n base nodejs`
  (node v26 in miniforge **base**).
- Python 3.13 in miniforge base. Backend installed editable there.
- **MP3 decodes without ffmpeg** — bundled libsndfile 1.2.2 (via `soundfile`)
  handles MP3/WAV/FLAC/OGG natively.
- Hardware: **2× RTX 4060 Ti (16 GB each)**, 125 GB RAM, ~835 GB free on `/`.

**Built (backend, `backend/remixflow/`):**
- `params.py` — single source of truth for all 19 PRD steering controls +
  8 identity locks; served to UI via `/api/controls`.
- `models.py` — `Song`, `Variant`, `Steering`, evolution `TreeNode`.
- `audio/io.py` — load/save with graceful degradation (soundfile→librosa).
- `audio/analysis.py` — tempo/key/centroid + Musical-DNA embedding + cosine
  similarity (identity-preservation score).
- `generation/base.py` — the `Generator` contract.
- `generation/dsp.py` — DSP reference backend (time-stretch/EQ/drive/width).
- `generation/registry.py` — backend registry + default selection.
- `service.py` — orchestration (import→steer→generate→evaluate→branch), morph,
  preference learning.
- `store.py` — file-backed store (JSON + audio dir).
- `app.py` — FastAPI routes; serves built UI from `static/`.
- `tests/test_pipeline.py` — end-to-end smoke tests.

**Built (frontend, `frontend/src/`):** dynamic steering panel (all sliders +
identity locks), similarity/variation meters, A/B player (position-preserving
switch), evolution tree, ∞ Evolution mode, ratings, library/import. Builds into
`backend/remixflow/static/` so `pip install` ships the UI (no Node for end users).

**Verified:** all tests pass; full HTTP flow works with a synthesized tone.

---

## 2026-07-21 — Milestone 2: real model backend (ACE-Step v1.5) + async

**Decision (user request):** use **ACE-Step v1.5** as the real generation
backend. GPUs available → feasible.

**Research (accurate API):**
- v1.5 repo `ace-step/ACE-Step-1.5`; HF model `ACE-Step/Ace-Step1.5`.
- Best path: **diffusers `AceStepPipeline`** (v0.39.0). Turbo variant
  `ACE-Step/acestep-v15-xl-turbo-diffusers` — 8 steps, <4 GB VRAM, 48 kHz stereo.
- Audio-to-audio via `task_type="cover"` with `src_audio`+`reference_audio`
  (stereo 48 kHz tensors) and `audio_cover_strength` (0..1).

**Built:**
- `generation/acestep.py` — `AceStepGenerator` mapping steering→ACE-Step:
  parent→`src_audio`/`reference_audio`, `variation_amount`→`audio_cover_strength`
  (identity locks lower it), sliders→text `prompt`, seed→`torch.Generator`.
  Lazy GPU singleton; self-probes availability.
- `jobs.py` — in-process async job queue (single GPU worker). `POST /api/generate`
  now returns a job; poll `/api/jobs/{id}`. `POST /api/generate/sync` kept for
  tests/fast backends.
- Frontend: async job polling + progress bar + backend picker (Auto/ace-step/dsp).
- Analysis sped up (resample to 22 kHz): 12.5s→7.9s on a full song.

**Isolated env:** `conda env acestep` (Python 3.11) holds torch cu124 + diffusers
+ ACE-Step, plus the remixflow backend. Run the app from this env to get the
`ace-step` backend; base env falls back to DSP.

**Install status (this session):**
- ✅ torch 2.6.0+cu124, CUDA True, 2 devices.
- ✅ diffusers 0.39.0, transformers 5.14.1; `AceStepPipeline` imports.
- ✅ remixflow installed into acestep env; `ace-step` reports `available: true`.
- ⏳ **CURRENT STEP:** downloading ACE-Step v1.5 turbo weights (19 files) and
  loading the pipeline on GPU. Next: run `I_Will_Never_Fall.mp3` through a real
  `task_type="cover"` generation and verify audio out + similarity.

**Test file:** `/data3/remixflow/I_Will_Never_Fall.mp3` (3.7 min, 44.1 kHz stereo,
detected 129 BPM / key G#).

### Update — first real generation working (SDEdit)

- ✅ Weights downloaded (HF token from user sped it ~2×); pipeline loads on GPU 0
  in **~5s from cache**, sample_rate 48000.
- ⚠️ `task_type="cover"` FAILED: this `acestep-v15-xl-turbo-diffusers` checkpoint
  ships **without** the `audio_tokenizer`/`audio_token_detokenizer` modules that
  cover-from-source needs (`prepare_src_latents` raises).
- ✅ **Fix: SDEdit img2img** (`generation/acestep.py`). Read the diffusers pipeline
  source: `prepare_latents(latents=…)` and `timesteps=[…]` are honored verbatim,
  and VAE-encoding the source needs no tokenizer. So we: VAE-encode source →
  latents, noise them to sigma≈`strength` (flow-matching interp), and denoise the
  truncated schedule with the source as `reference_audio` (timbre). Turbo has no
  CFG, so no guidance/null-embedding issues.
- ✅ **Verified (30s trim):** generated in **2.6s** (RTF **11.3×**), tempo/key
  preserved (129 BPM / G#), identity similarity **0.994**. `variation_amount` →
  SDEdit noise level; identity locks lower it.
- Full-song VAE-encode OOMed at 16 GB → **fix: chunked SDEdit** (30s windows, 2s
  equal-power crossfade, `torch.cuda.empty_cache()` between chunks). Bounds VRAM
  for any length.
- Schedule rebuild: construct `target_steps` sigmas over `[strength, 0]` then apply
  the shift (instead of slicing) → fixed step count at any strength (8, not 3).
- ✅ **Full 221s song:** generated in **16.2s** (RTF 13.6×), tempo/key preserved,
  similarity **0.996** (8 chunks).
- ✅ **Full HTTP flow verified** (server in acestep env): `POST /api/songs` →
  `POST /api/generate?backend=ace-step` (async job) → poll → done in ~30s (incl.
  first-call model load), similarity 0.994, prompt built from sliders. **Milestone
  2 complete.**

**Run with ACE-Step:** activate `acestep` env, `export
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `remixflow serve`. Manual check
script: `backend/scripts/try_acestep.py <song> [dur_sec]`. Env knobs:
`ACESTEP_STEPS` (8), `ACESTEP_CHUNK_SEC` (30), `ACESTEP_OVERLAP_SEC` (2),
`ACESTEP_DEVICE` (`cuda:1` for 2nd GPU), `ACESTEP_MODEL`.

---

## 2026-07-21 — Listening / tuning / MP3 / A/B page

**User feedback:** the strength-0.35 full variant "sounds the same"; wants MP3
(WAV too big); machine uses **port 8000 and 8188 (llama.cpp) — do NOT use them**.

**Findings & fixes:**
- Confirmed via mel-spectrogram correlation to the original: subtle(0.2)→0.949,
  moderate(0.5)→0.855, **bold(0.8)→0.562** (Δ10.5 dB). SDEdit works; 0.35 just
  sits in the near-identical zone. Low variation = faithful by design.
- MFCC identity score saturates (~97–100% even at bold); switched the A/B page's
  second meter to **"Change from original" = 1 − mel-corr** (perceptually honest).
- Fixed a real clipping bug: crossfade/resample pushed peak to 1.014 → added
  final **peak-normalize to −0.5 dBFS** in `generation/acestep.py`.
- **MP3 works with no ffmpeg** — libsndfile 1.2.2 writes `format="MP3",
  subtype="MPEG_LAYER_III"`. Converted excerpts + full variant to MP3.
- **Changed default serve port off 8000** (see below).

**Deliverables produced:**
- `samples/00_original_excerpt.mp3`, `01_subtle_var20.mp3`,
  `02_moderate_var50.mp3`, `03_bold_var80.mp3` (30s each, ~0.5 MB).
- `I_Will_Never_Fall__variant.mp3` (full song, var 35%, 4.3 MB).
- Self-contained A/B player artifact (dark RemixFlow console, embedded MP3s,
  position-preserving version switch): **https://claude.ai/code/artifact/da2987b4-7b95-4d73-a6a4-751930c0e127**
- Scripts: `backend/scripts/make_ab_excerpts.py`, `build_ab_page.py`.

**PORT CONSTRAINT (this machine):** never bind 8000 or 8188. Default serve port
changed to **8770**. `ACESTEP_DEVICE=cuda:1` also frees GPU 0.

---

## 2026-07-21 — Vocal preservation wired into the app

**User feedback:** var-50 is great (beat/melody/hook), but SDEdit turns the
**vocals into gibberish** (no lyrics passed → model invents vocal sounds).

**Fix (verified, then wired in):** stem-separate with **Demucs (htdemucs)**, run
SDEdit on the **instrumental only**, remix the **original vocals** back on top
(tempo/length preserved → aligned). Demo: `I_Will_Never_Fall__var50_vocals.mp3`.

**Now in the backend** (`generation/acestep.py`): locking `lyrics` or
`vocal_phrasing` (VOCAL_LOCKS) auto-triggers the separate→vary→remix path.
- Refactored SDEdit chunk logic into `_sdedit_audio()`; `generate()` branches on
  `_should_preserve_vocals()`.
- Lazy Demucs singleton `_separator()` — device from `ACESTEP_DEVICE`, failures
  not cached (was a bug: coupling to `self._device` before pipeline init poisoned
  the cache → feature silently off). Fixed & re-verified: note now ends
  "· original vocals preserved", triggered=True, ~10s on a 30s clip.
- `demucs==4.1.0` added to the acestep env. Mix knobs: `ACESTEP_VOCAL_GAIN`,
  `ACESTEP_INSTR_GAIN`; disable via `ACESTEP_PRESERVE_VOCALS=0`.
- UI: identity-lock section notes that locking lyrics/vocal-phrasing keeps the
  original vocal recording.

**Next:** PRD_pt2 "Living Songs" (Phase 2) — see plan under discussion.

---

## 2026-07-21 — Phase 2: Living Mode player + API + tuning

- **API** (`app.py`, `service.living_segment`): `POST /api/living` (async job) →
  segment `{audio_url, next_index, ...}`; `GET /api/living/audio/{id}`. Engine
  now resumable via `start_index` (continues window/tension/seed/position).
- **Browser player** (`components/LivingPlayer.tsx`): generate-ahead prefetch
  (~15s before end), **Living Repeat** ∞ toggle for endless play, transport, live
  status. `LivingControls.tsx`: Improvisation + Segment length + style-drift
  sliders (apply to the next stretch). App gets a **Steer | ∞ Living** mode switch.
- **Vocal-clash fix**: instrumental windows now prompt "instrumental, no vocals"
  (`build_prompt(instrumental=True)`, threaded through generate + Living engine)
  → no AI vocals bleeding under the preserved real vocals. This is what lets
  strength go higher without weirdness.
- **Tension gentler default** (0.08–0.22) then recalibrated for the app so the
  Improvisation slider's default is audibly breathing.
- Verified: gentle (0.08–0.22, too subtle), medium (0.15–0.42, morphs well),
  full ACE-Step Living segment over HTTP (30s/3 windows/identity 0.997), frontend
  builds clean, server runs on **8770** (0.0.0.0), both backends available.

---

## 2026-07-21 — Phase 2 (Living Songs): Continuous Evolution Engine

User chose "continuous engine first, validate before UI". Built `living.py`
(`LivingEngine`, `LivingConfig`, `TensionCurve`) — renders an endless, never-
exactly-repeating stream from a seed:
- Chains short SDEdit **windows** (~12s), each **re-anchored to the original**
  source segment (identity can't drift) rather than evolving from prior output.
- **Tension curve** modulates per-window `variation_amount` (raised-cosine
  lo↔hi + seeded jitter) → explore/return breathing.
- **Identity gate**: each window scored vs its source segment; regenerate at
  lower strength if below `identity_threshold` (0.85).
- **Memory**: position-keyed (only a repeat of the *same moment on a later pass*
  counts — first version compared adjacent windows and misfired, 2 wasted
  retries every window; fixed).
- **Vocals preserved** for the whole stream: separate once, evolve instrumental
  windows, overlay original vocals.
- Seamless **equal-power crossfades** (overlap 2.5s).

Validated (`scripts/make_living.py`): 75s / 8 windows in **26.5s (2.8× RT incl.
one-time 30s separation; pure gen ~7× RT)**, identity avg 0.994, retries 0,
seams ≤0.054, tension visibly breathing. Output: `I_Will_Never_Fall__living.mp3`.

**Known limits / next:** MFCC identity metric saturates (~0.99) — needs a real
melody/harmony/rhythm/timbre identity score (deeper-DNA milestone). 75s < song
length so no wrap yet — position-memory built but not exercised; a >songlen
render would demonstrate non-repeating passes. Then: buffered streaming player +
listening-mode presets (Studio/Live/Jazz/Orchestra/Ambient/Infinite Radio).

**Steering→ACE-Step mapping is now SDEdit, not cover:**
`variation_amount`→noise level (fraction of trajectory re-run); locks→lower it;
sliders→text prompt; source→`reference_audio` timbre; seed→`torch.Generator`.
