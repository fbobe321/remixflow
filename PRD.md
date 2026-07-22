# Product Requirements Document (PRD)

# Project Name

**RemixFlow** *(working title)*

## Vision

Create an AI-powered music evolution platform that allows users to continuously generate subtle variations of songs they enjoy. Instead of producing completely new music, the system preserves the musical identity that users love while introducing controlled changes, resulting in an effectively infinite playlist of familiar-but-fresh versions.

The goal is to solve the problem of **listener fatigue**—the tendency to become bored with songs after repeated listening—while preserving the emotional characteristics that made the original appealing.

---

# Problem Statement

Most music listeners experience this cycle:

1. Discover a song.
2. Listen hundreds of times.
3. Eventually become fatigued.
4. Stop listening despite still loving the song.

Current solutions include:

* Shuffle playlists
* Radio stations
* Covers
* Remixes

None preserve the exact characteristics that made the original enjoyable.

Users want:

> "Keep everything I love about this song... just make it slightly different."

---

# Goals

* Generate endless musical variations.
* Preserve recognizable identity.
* Allow user-controlled evolution.
* Create an intuitive "music steering" interface.
* Enable gradual exploration rather than complete rewrites.

---

# Non-Goals

* Piracy.
* Perfect cloning of copyrighted music.
* Commercial redistribution of copyrighted works.
* Replacing professional music production.

---

# Target Users

### Casual Listeners

"I love this song but I've heard it 500 times."

### Audiophiles

Interested in nuanced variations.

### Focus / Productivity Users

Need long listening sessions without repetition.

### DJs

Generate transition tracks.

### Musicians

Explore alternate arrangements.

### Game Developers

Procedurally evolving background music.

### Streamers

Background music that never becomes repetitive.

---

# Core User Story

> I upload a song that I enjoy.

> I move several sliders describing how I want it to evolve.

> Within seconds the AI creates a new version that still feels like the original but introduces tasteful variation.

> I save the versions I enjoy.

> The system continuously creates new branches from my favorites.

---

# Product Architecture

```
Original Song
        │
Audio Encoder
        │
Latent Representation
        │
Style Steering Layer
        │
Diffusion Music Model
        │
Generated Variant
        │
Similarity Evaluator
        │
User Rating
        │
Preference Model
```

---

# Functional Requirements

## 1. Song Import

Supported formats:

* MP3
* WAV
* FLAC
* OGG

The system extracts:

* Tempo
* Key
* Chords
* Melody
* Vocal features
* Instrumentation
* Dynamics
* Song sections

---

## 2. Music Embedding

Generate a latent representation capturing:

* Melody
* Harmony
* Rhythm
* Timbre
* Genre
* Energy
* Mood
* Instrument mix
* Vocal style

This embedding becomes the editable representation.

---

## 3. Evolution Controls

### Energy

Low ←────────→ High

---

### Tempo

Slower ←────────→ Faster

---

### Blues Influence

Less ←────────→ More

---

### Rock Influence

Less ←────────→ More

---

### Jazz

Less ←────────→ More

---

### Electronic

Less ←────────→ More

---

### Acoustic

Less ←────────→ More

---

### Instrument Density

Sparse ←────────→ Dense

---

### Emotional Tone

Sad ←────────→ Happy

---

### Vocal Style

Original

↓

Softer

↓

Aggressive

↓

Whisper

↓

Powerful

---

### Instrument Focus

* Guitar
* Piano
* Strings
* Synth
* Bass
* Drums

---

### Groove

Relaxed ←────────→ Tight

---

### Swing

None ←────────→ Heavy

---

### Chorus Strength

Subtle ←────────→ Anthemic

---

### Bass

Light ←────────→ Heavy

---

### Brightness

Dark ←────────→ Bright

---

### Warmth

Cold ←────────→ Warm

---

### Complexity

Simple ←────────→ Complex

---

### Variation Amount

Critical slider.

```
0%
Nearly identical

10%

20%

30%

40%

50%

60%

70%

80%

90%

100%
Completely reimagined
```

---

## 4. Identity Preservation

Maintain user-selected musical elements such as:

* Melody
* Chord progression
* Vocal phrasing
* Rhythm
* Hook
* Chorus
* Lyrics
* Instrumentation

Users choose what remains fixed and what can evolve.

---

## 5. Smart Evolution

Each generation can branch from:

Original

↓

Version A

↓

Version B

↓

Version C

forming an evolution tree.

Users can return to any previous version.

---

## 6. Infinite Evolution Mode

```
Song

↓

Variation 1

↓

Variation 2

↓

Variation 3

↓

Variation 4

↓

...
```

Each generation introduces only minor changes, creating an endlessly evolving listening experience.

---

## 7. A/B Comparison

Instant switching between:

Original

Version 1

Version 2

Version 3

to evaluate changes.

---

## 8. Preference Learning

After each generation:

👍 Love it

😐 Neutral

👎 Don't like it

The system learns the user's preferred balance between familiarity and novelty.

---

# Advanced Features

## Musical DNA

Represent songs as vectors such as:

* Energy
* Groove
* Harmony
* Timbre
* Melody
* Instrument mix
* Vocal tone
* Genre distribution

The editor manipulates these latent features directly.

---

## Evolution Timeline

Visualize how a song changes over time, with branching history and playback from any point.

---

## Morph Between Songs

Blend characteristics from two songs into controlled intermediate versions.

---

## Mood Adaptation

Generate variants optimized for contexts like:

* Workout
* Relaxation
* Driving
* Studying
* Sleep
* Gaming

---

## Seasonal Versions

Produce winter, summer, acoustic, orchestral, lo-fi, cinematic, or other thematic variants while retaining the song's identity.

---

# User Interface

```
----------------------------------------

Original Song

Similarity
██████████░ 92%

Variation
█████░░░░░ 45%

Energy
███████░░░

Tempo
█████░░░░░

Rock
████████░░

Blues
███░░░░░░░

Jazz
█░░░░░░░░░

Brightness
██████░░░░

Bass
████░░░░░░

Warmth
████████░░

Generate

----------------------------------------
```

---

# Technical Architecture

### Audio Understanding

* Music encoder
* Beat detection
* Chord estimation
* Vocal separation (optional)

### Latent Space

* Song embedding
* Style embedding
* Instrument embedding

### Steering Engine

Maps UI sliders to latent-space modifications.

### Music Generator

Potential backends:

* ACE-Step 1.5
* Stable Audio Open
* MusicGen derivatives
* Future diffusion or transformer-based music models

### Similarity Evaluator

Measures preservation of the original song's identity using learned audio embeddings.

---

# Success Metrics

* Average listening time increases versus original track.
* High user ratings for generated variants.
* Low perceived repetition over extended sessions.
* Strong identity preservation while maintaining novelty.
* Percentage of generated tracks users save to favorites.
* Repeat usage and session length.

---

# Risks & Considerations

### Copyright

Using copyrighted songs as direct inputs raises significant legal and licensing questions. Commercial versions would likely require licensed content, user-owned media, or operation within applicable copyright exceptions. This should be a primary product and legal design consideration.

### Quality Drift

Repeated generations may gradually lose the qualities that made the original compelling. Similarity constraints and periodic re-anchoring to the source can help limit this drift.

### User Trust

Users need confidence that "20% variation" behaves consistently. Slider effects should be predictable, reversible, and transparent.

---

# Future Vision

The long-term vision is an **AI Music Evolution Engine** that transforms static recordings into living musical experiences. Rather than replaying the same track indefinitely, songs become dynamic artifacts that subtly adapt over time, user preferences, listening context, and mood—preserving their identity while continuously introducing fresh, personalized variation.

---

# Implementation Status

*Last updated: 2026-07-21. See `BUILD_LOG.md` for the detailed running log and
`README.md` for how to run it.*

**Stack:** React + TypeScript (Vite) frontend · Python + FastAPI backend. The
frontend builds into the backend's `static/` dir, so `pip install` ships the UI
(no Node required by end users).

## Status by requirement

| PRD section | Status | Notes |
|-------------|:------:|-------|
| §1 Song Import (MP3/WAV/FLAC/OGG) | ✅ | libsndfile decodes all four; **no ffmpeg needed**. |
| §1/§2 Feature extraction + Musical DNA embedding | ✅ | tempo/key/centroid + MFCC embedding (`audio/analysis.py`). |
| §3 Evolution Controls (19 sliders/selectors) | ✅ | Catalog in `params.py`, rendered dynamically in UI. |
| §4 Identity Preservation (locks) | ✅ | 8 lockable elements; honored by both backends. |
| §5 Smart Evolution (branching tree) | ✅ | `store.tree()` + `EvolutionTree.tsx`. |
| §6 Infinite Evolution Mode | ✅ | ∞ toggle: chained gentle-drift generations. |
| §7 A/B Comparison | ✅ | Position-preserving version switch (`ABPlayer.tsx`). |
| §8 Preference Learning | ✅ | 👍/😐/👎 → learned variation/similarity profile. |
| Advanced: Morph Between Songs | ✅ | `POST /api/morph`. |
| Similarity Evaluator | ✅ | Cosine on embeddings = identity-preservation %. |
| Steering Engine | ✅ | Maps sliders → backend params. |
| **Music Generator** | ✅ | Two working backends behind one interface (below). |

## Generation backends

- **`dsp`** — ✅ done. Dependency-light reference backend (time-stretch, EQ tilt,
  drive, stereo width). Instant, always available, no model. Proves the pipeline
  and serves as fallback.
- **`ace-step`** — ✅ working & verified. Real generative model: **ACE-Step v1.5**
  (`ACE-Step/acestep-v15-xl-turbo-diffusers`) via the diffusers `AceStepPipeline`,
  on GPU. This turbo checkpoint has no `audio_tokenizer`, so instead of the
  `cover` task we use **SDEdit img2img**: VAE-encode the source, noise it to a
  level set by `variation_amount`, and denoise with the source as timbre
  reference. Long songs run in crossfaded ~30s chunks (VRAM-bounded). Verified on
  a real 3.7-min track: **16s generation (13.6× realtime), tempo/key preserved,
  0.996 identity similarity**. `variation_amount`→noise level, identity
  locks→stronger preservation, sliders→text prompt.

## Architecture notes

- Generation is **asynchronous** (`jobs.py`): `POST /api/generate` returns a job
  id; the UI polls `/api/jobs/{id}` with a progress bar. Any real model runs for
  tens of seconds, so this is required, not optional. `POST /api/generate/sync`
  remains for tests/fast backends.
- Runs in **two conda envs**: `base` (app + DSP + frontend build) and `acestep`
  (Python 3.11 + torch CUDA + diffusers + ACE-Step). The `ace-step` backend
  self-probes and reports `available:false` where its stack is absent, so the
  identical codebase runs with or without a GPU.
- Default serve port is **8770** (this host reserves 8000 and 8188). MP3/WAV/FLAC/
  OGG read+write via libsndfile — no ffmpeg. Sample outputs live in `samples/`
  and an in-browser A/B player artifact demonstrates the variation spread.
- **Tuning note:** `variation_amount` ≈ 0.3 is deliberately near-identical; audible
  divergence starts around 0.5 and is clear by 0.8 (mel-correlation to source:
  0.95 / 0.86 / 0.56 at 0.2 / 0.5 / 0.8).

## Not yet started (next milestones)

- Evolution Timeline visualization; Mood/Seasonal preset packs (Advanced Features).
- Learned preference model driving auto-suggested steering.
- Copyright-aware import (user-owned media / licensing) — see Risks.
- Learned music encoder (CLAP/MERT) to replace the MFCC embedding.

