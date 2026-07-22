# PRD_pt2.md

# Living Songs

## Phase 2 Product Requirements Document

**Version:** 1.0
**Status:** Draft
**Depends On:** PRD Phase 1 (RemixFlow)

---

# Vision

Transform a static song into a **Living Song**—music that never truly repeats.

Instead of generating discrete remixes, the AI continuously evolves the music while preserving the emotional identity of the original. Like a live jazz performance or an orchestra playing the same composition differently every night, the music breathes, adapts, and explores without ever losing its core identity.

The listener should always think:

> "This is still *my song*."

---

# Product Goal

Create an AI music engine capable of producing an effectively infinite version of a song that:

* Never exactly repeats
* Never drifts too far from the original
* Preserves emotional identity
* Preserves musical "DNA"
* Evolves naturally over time

---

# Inspiration

Examples include:

* Jazz improvisation
* Live concerts
* Classical orchestras
* Jam sessions
* Human performers who never play the same song identically twice

The AI should imitate musicians—not remix software.

---

# Core Philosophy

Traditional remixes create:

Song A

↓

Song B

↓

Song C

Living Songs create:

Song

↓

Small evolution

↓

Small evolution

↓

Small evolution

↓

Small evolution

↓

...

The listener experiences one continuously evolving performance.

---

# User Story

As a listener,

I upload one of my favorite songs.

The system studies every aspect of it.

I press **Living Mode**.

The song begins evolving naturally while I listen.

Every playback is unique.

Yet it always feels like the same song.

---

# Core Requirements

## 1. Deep Song Analysis

Before any generation begins, the system performs a comprehensive musical analysis.

### Rhythm

* BPM
* Groove
* Swing
* Syncopation
* Beat emphasis
* Rhythmic complexity

---

### Harmony

* Key
* Scale
* Chord progression
* Harmonic tension
* Resolution points
* Modulations

---

### Melody

* Primary melody
* Supporting melodies
* Vocal contour
* Phrase boundaries
* Hooks
* Motifs

---

### Instrumentation

Identify:

* Guitar
* Piano
* Bass
* Strings
* Brass
* Drums
* Synths
* Pads
* Percussion
* Vocals

Estimate:

* Instrument prominence
* Playing style
* Tone
* Dynamics

---

### Song Structure

Automatically detect:

Intro

Verse

Pre-Chorus

Chorus

Bridge

Solo

Outro

Section lengths

Transition styles

---

### Dynamics

Measure:

* Loudness
* Energy
* Crescendos
* Decrescendos
* Compression
* Density

---

### Emotional Profile

Estimate latent emotional dimensions rather than assigning a single label:

* Energy
* Tension
* Warmth
* Brightness
* Aggression
* Intimacy
* Nostalgia
* Melancholy
* Excitement
* Calmness

---

### Musical Style

Estimate weighted genre influences, for example:

* Rock: 72%
* Blues: 14%
* Country: 5%
* Pop: 9%

These become editable continuous values rather than fixed labels.

---

# Musical DNA

Create a persistent **Musical DNA** representation.

Example:

```json
{
  "tempo":118,
  "key":"A Major",
  "energy":0.74,
  "brightness":0.62,
  "warmth":0.79,
  "groove":0.81,
  "rock":0.72,
  "blues":0.18,
  "country":0.03,
  "melody_embedding":"...",
  "harmony_embedding":"...",
  "instrument_embedding":"..."
}
```

This Musical DNA becomes the anchor for all future generations.

---

# Identity Lock

One of the biggest risks is "identity drift."

Every generated segment must be compared against the original DNA.

Compute an **Identity Score** using learned audio embeddings and music-specific similarity metrics.

Possible components include:

* Melody similarity
* Harmony similarity
* Rhythm similarity
* Timbre similarity
* Structural similarity
* Emotional similarity

Overall Identity Score:

0.0 → Completely different

1.0 → Identical

Default operating range:

**0.85–0.97**

If similarity drops below the threshold, regeneration or correction is triggered before playback continues.

---

# Continuous Evolution Engine

Instead of generating an entire replacement song, generate music incrementally.

Possible generation windows:

* 2 seconds
* 4 seconds
* 8 seconds
* One musical phrase
* One measure
* One section

Each new window is conditioned on:

* Previous audio
* Original Musical DNA
* Current musical context
* Future section plan

This allows seamless evolution over long listening sessions.

---

# Controlled Morphing

Each musical feature evolves independently.

Examples:

Tempo

118 BPM

↓

118.2

↓

117.8

↓

118.5

↓

118.1

Melody

Original

↓

Tiny ornament

↓

Passing tone

↓

Alternate phrase ending

↓

Return to original

Chord progression

Original

↓

Extended chord

↓

Suspension

↓

Added ninth

↓

Resolve

The objective is tasteful variation rather than wholesale replacement.

---

# Memory System

The AI remembers what it has recently played.

Avoid:

* Repeating the same improvisation
* Identical fills
* Identical solos
* Identical transitions

Maintain rolling musical memory so the performance continues to feel fresh.

---

# Musical Tension Model

Treat variation as controlled tension.

Low tension:

Almost identical.

Medium tension:

Small embellishments.

High tension:

Short improvisations before resolving back to the familiar theme.

The system should naturally alternate between exploration and return, mirroring skilled human performers.

---

# Evolution Controls

Expose high-level controls to the listener:

Identity Preservation

Loose ←────────────→ Strict

Improvisation

Low ←────────────→ High

Exploration

Conservative ←────→ Adventurous

Return Frequency

Rare ←────────────→ Frequent

Energy Drift

Stable ←──────────→ Dynamic

Instrument Creativity

Original ←────────→ Experimental

Solo Length

Short ←───────────→ Extended

Mood Stability

Constant ←────────→ Adaptive

---

# Listening Modes

## Studio Mode

Near-identical playback.

Ideal for casual listening.

---

## Live Performance

Natural variation comparable to a live band.

---

## Jazz Mode

Heavy improvisation while preserving harmonic identity.

---

## Orchestra Mode

Subtle reinterpretations of dynamics and orchestration.

---

## Ambient Mode

Continuous atmospheric evolution.

---

## Infinite Radio Mode

Generates an endless version of the seed song without obvious loops.

---

# Real-Time Generation

## Stretch Goal

Investigate whether generation can occur continuously during playback.

Potential pipeline:

Current Playback

↓

Predict next phrase

↓

Generate candidate variations ahead of time

↓

Evaluate Identity Score

↓

Select best continuation

↓

Crossfade seamlessly

↓

Continue playback

To reduce latency, maintain a generation buffer (for example, 10–30 seconds ahead of the listener). This allows continuous playback while the AI evaluates multiple candidate continuations in parallel.

If true real-time generation is not practical on available hardware, implement a streaming approach where the next musical segment is always generated before it is needed.

---

# Technical Architecture

Input Song

↓

Audio Analysis Engine

↓

Musical DNA Extraction

↓

Section Detection

↓

Embedding Generator

↓

Living Song Engine

↓

Identity Validator

↓

Playback Buffer

↓

Continuous Output

---

# Success Metrics

* Listeners report that the song still "feels like the original."
* Long listening sessions without noticeable repetition.
* High Identity Scores maintained throughout extended playback.
* Smooth transitions with no audible artifacts.
* Reduced listener fatigue compared with static playback.
* Users voluntarily spend significantly longer listening to Living Songs than to the original recordings.

---

# Future Vision

Living Songs become a new category of music—neither remixes nor covers, but adaptive performances. Every playback is unique, yet unmistakably recognizable. The system acts like an endlessly creative musician who deeply understands the essence of the original composition and continually reinterprets it without ever losing its identity.

Long term, this technology could power adaptive game soundtracks, personalized focus music, interactive concerts, and AI performers that accompany listeners in real time.

---

# Implementation Status (Phase 2)

*Last updated: 2026-07-21. Detailed log in `BUILD_LOG.md`; Phase 1 status in `PRD.md`.*

Building on Phase 1 (RemixFlow: analysis, Musical-DNA embedding, ACE-Step v1.5
SDEdit generation, identity similarity, vocal preservation). Chosen build order:
**continuous engine first → validate audio → then streaming UI + modes.**

| Phase 2 requirement | Status | Where / notes |
|---------------------|:------:|---------------|
| Continuous Evolution Engine | ✅ built & validated | `living.py` `LivingEngine`. Chains ~12s SDEdit windows, each **re-anchored to the original** so identity can't drift; seamless equal-power crossfades. |
| Identity Lock (score + regenerate) | 🟡 works, coarse | Per-window score vs source; regenerates below threshold (0.85). Uses the MFCC embedding, which **saturates (~0.99)** — needs a real melody/harmony/rhythm/timbre score. |
| Musical Tension Model | ✅ | `TensionCurve` — raised-cosine lo↔hi + jitter modulates per-window variation (explore↔return). |
| Memory (avoid repeats) | ✅ | Position-keyed: only a repeat of the *same moment on a later pass* counts. |
| Vocal preservation through stream | ✅ | Separate once (Demucs), evolve instrumental windows, overlay original vocals. |
| Living Mode player + API | ✅ | `POST /api/living` (chained via `next_index`) + `LivingPlayer.tsx` (prefetch-ahead, **Living Repeat**, transport, controls). Verified end-to-end with ACE-Step. |
| Buffered / streaming playback | ✅ (buffered) | Player generates the next segment ~15s before the current ends (gen ~4–7× realtime) → seamless endless play. True per-frame realtime still a stretch. |
| Listening Modes | ✅ | 6 built-in presets (Studio/Live/Jazz/Orchestra/Ambient/Infinite Radio) + **user-saved presets** (persisted). `presets.py`, `GET/POST/DELETE /api/presets`, picker in `LivingControls.tsx`. |
| Playlist (multiple songs) | ✅ | Living Mode weaves a set of songs into one endless set — evolve each ~`perSongSec`, crossfade the transition, loop. Checkbox playlist + "time per song" in `LivingControls`; player cycles songs (`LivingPlayer.tsx`). |
| Gapless / seamless joins | ✅ | Web Audio scheduling + source-cursor continuity + time-aligned crossfade tail (no gap, no doubling, no beat skip). |
| Deep analysis / richer DNA | 🟡 partial | Have tempo/key/embedding; sections, genre weights, emotional dims, per-instrument prominence not yet. |
| Real-time generation (stretch) | ⬜ | Buffer-ahead approach viable given >realtime generation; not built. |
| Controlled per-feature morphing | ⬜ | Tempo/melody/chords drifting independently — not yet (single strength per window today). |

**Validated:** 75s stream, 8 windows, RTF 2.8× (pure gen ~7×), identity avg
0.994, seams ≤0.054, tension breathing. Output `I_Will_Never_Fall__living.mp3`.

**Resume here → next options (awaiting user's listen/decision):**
1. Render >songlength (~4 min) to demonstrate non-repeating passes (wrap).
2. Buffered streaming player + "Living Mode" UI with listening-mode presets.
3. Tune feel (window length, tension range, per-feature morphing).
4. Deeper DNA + real identity score (make the gate meaningful).

