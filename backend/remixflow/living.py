"""Living Songs — the Continuous Evolution Engine (PRD Phase 2).

Renders an endless, never-exactly-repeating performance of a seed song by
chaining short SDEdit windows, each **re-anchored to the original** (so identity
never drifts) but varied by a time-varying *tension* curve (so it breathes).
Windows crossfade seamlessly; each is identity-scored against the source and
regenerated if it drifts below threshold; a rolling memory keeps successive
passes from repeating.

Vocals are preserved for the whole stream: the source is stem-separated **once**,
only the instrumental is evolved window-by-window, and the original vocal track
is overlaid — so the real voice carries through an infinite instrumental jam.

The engine is backend-agnostic in spirit but currently drives the ACE-Step
SDEdit backend. Because ACE-Step turbo runs ~10x faster than realtime, windows
can be produced ahead of playback (streaming comes in a later milestone).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .audio import analysis
from .audio.io import Clip
from .models import Steering


@dataclass
class TensionCurve:
    """Controlled variation over time: mostly familiar, occasional exploration.

    Value per window oscillates between ``lo`` and ``hi`` with ``period`` windows
    per cycle, plus a little seeded jitter so passes differ. This is the
    "explore ↔ return" behaviour of a live performer (PRD §Musical Tension).
    """

    # Living = mostly familiar with gentle embellishment (PRD §Tension). Kept
    # deliberately low: high strength makes ACE-Step hallucinate/artifact.
    lo: float = 0.08
    hi: float = 0.22
    period: int = 6
    jitter: float = 0.03

    def value(self, i: int, rng: np.random.Generator) -> float:
        # Raised-cosine swell from lo->hi->lo across the period.
        phase = (i % self.period) / self.period
        swell = 0.5 - 0.5 * math.cos(2 * math.pi * phase)
        v = self.lo + (self.hi - self.lo) * swell
        v += float(rng.normal(0.0, self.jitter))
        return float(min(0.95, max(0.05, v)))


@dataclass
class LivingConfig:
    duration_sec: float = 90.0
    window_sec: float = 12.0
    overlap_sec: float = 2.5
    # Extra source rendered past `duration_sec` so the player can crossfade the
    # SAME source moment from consecutive segments (time-aligned, no beat skip).
    crossfade_sec: float = 0.25
    identity_threshold: float = 0.85     # regenerate windows that drift below
    max_retries: int = 2                 # identity-correction attempts per window
    preserve_vocals: bool = True
    base_seed: int = 0
    tension: TensionCurve = field(default_factory=TensionCurve)
    #: Non-variation steering controls (genre/tone/etc.) applied to every window.
    steering: Steering = field(default_factory=Steering)


@dataclass
class WindowReport:
    index: int
    src_pos: float
    strength: float
    identity: float
    retries: int
    embedding: list = field(default_factory=list)


@dataclass
class LivingResult:
    clip: Clip
    windows: list[WindowReport]
    note: str
    next_index: int = 0        # tension/seed phase for the next render()
    next_pos_sec: float = 0.0  # source position where the ADVANCE (non-tail) ended
    advance_sec: float = 0.0   # timeline advance (clip is this + crossfade tail long)


class LivingEngine:
    def __init__(self, generator) -> None:
        # generator: a Generator exposing .generate() (the ACE-Step backend).
        self.gen = generator

    def render(
        self,
        source: Clip,
        config: LivingConfig,
        *,
        start_index: int = 0,
        start_pos_sec: float = 0.0,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> LivingResult:
        # A performance continues via TWO cursors: ``start_index`` (tension/seed
        # phase) and ``start_pos_sec`` (exact source position). The next segment
        # must begin where the previous OUTPUT ended in source-time, or its first
        # window re-covers source the previous segment already played (audible
        # doubling at the join). ``next_pos_sec`` in the result reports that point.
        rng = np.random.default_rng(config.base_seed + start_index)
        sr = source.sample_rate

        # 1. Prepare the evolvable base (instrumental) and the preserved vocals.
        vocals = None
        base = source.samples
        if config.preserve_vocals and getattr(self.gen, "_should_preserve_vocals", None):
            if self.gen._separator() is not None:  # type: ignore[attr-defined]
                if progress:
                    progress(0.02, "Separating vocals…")
                voc_np, instr_np = self.gen._separate(source)  # type: ignore[attr-defined]
                vocals, base = voc_np, instr_np
        base_len = base.shape[-1]

        # 2. Window plan.
        win = int(config.window_sec * sr)
        ov = int(config.overlap_sec * sr)
        out_sr = 48000  # ACE-Step SDEdit output rate

        # Reference embeddings per source position (for identity scoring), cached.
        out: Optional[np.ndarray] = None
        reports: list[WindowReport] = []
        # Memory keyed by source position: only a *repeat of the same moment on a
        # later pass* counts as repetition — adjacent windows are different music.
        pos_memory: dict[int, list[list[float]]] = {}

        i = start_index
        src_pos = int(start_pos_sec * sr) % max(1, base_len)
        instrumental = vocals is not None
        produced = 0.0
        # Render an extra crossfade tail; the timeline only advances by duration_sec.
        target = config.duration_sec + config.crossfade_sec
        while produced < target:
            # Source segment (wraps around for an endless stream).
            seg, seg_vocals, pos_sec = self._segment(base, vocals, src_pos, win, sr)
            seg_clip = Clip(samples=seg, sample_rate=sr)
            seg_emb = analysis.embed(seg_clip)
            pos_key = int(pos_sec)

            # 3. Identity-gated generation with tension-driven strength.
            strength = config.tension.value(i, rng)
            varied, identity, retries = self._gen_window(
                seg_clip, seg_emb, strength, config, seed=config.base_seed + i * 101,
                recent=pos_memory.get(pos_key, []), instrumental=instrumental,
            )
            varied_emb = analysis.embed(varied)
            pos_memory.setdefault(pos_key, []).append(varied_emb)

            samples = varied.samples
            # 4. Overlay preserved vocals for this window.
            if seg_vocals is not None:
                samples = self._overlay_vocals(samples, out_sr, seg_vocals, sr)

            # 5. Seamless crossfade into the running stream.
            out = self._append(out, samples, int(config.overlap_sec * out_sr))
            produced = out.shape[-1] / out_sr
            reports.append(WindowReport(i, round(pos_sec, 1), round(strength, 3),
                                        round(identity, 3), retries, varied_emb))
            if progress:
                progress(min(0.98, produced / target),
                         f"Living… {produced:.0f}/{target:.0f}s (window {i + 1})")

            i += 1
            src_pos = (src_pos + (win - ov)) % max(1, base_len)

        # Trim to duration + crossfade tail, and normalize. Output is 1:1 with
        # source time. The next segment advances by `duration_sec` (so the tail
        # overlaps the next segment's head over the SAME source moment — a
        # time-aligned crossfade at playback, no beat skip).
        out = out[:, : int(target * out_sr)]
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1e-6:
            out = out * (0.944 / peak)
        clip = Clip(samples=out.astype(np.float32), sample_rate=out_sr)
        source_dur = base_len / sr if sr else 0.0
        advance = config.duration_sec
        next_pos = (start_pos_sec + advance) % source_dur if source_dur else 0.0
        note = (f"living {len(reports)} windows, "
                f"identity avg={np.mean([r.identity for r in reports]):.3f}, "
                f"strength {config.tension.lo}-{config.tension.hi}"
                + (", vocals preserved" if vocals is not None else ""))
        return LivingResult(clip=clip, windows=reports, note=note, next_index=i,
                            next_pos_sec=round(next_pos, 3), advance_sec=round(advance, 3))

    # --- helpers -----------------------------------------------------------

    def _segment(self, base, vocals, pos, win, sr):
        """Grab a window from the base (wrapping), plus aligned vocals."""
        n = base.shape[-1]
        idx = (np.arange(pos, pos + win) % n)
        seg = base[:, idx]
        seg_vocals = vocals[:, idx] if vocals is not None else None
        return seg, seg_vocals, pos / sr

    def _gen_window(self, seg_clip, seg_emb, strength, config, *, seed, recent,
                    instrumental=False):
        """Generate one window; retry (lower strength) if it drifts too far or
        repeats a recent window (identity gate + memory, PRD §Identity Lock/Memory)."""
        best = None
        best_identity = -1.0
        for attempt in range(config.max_retries + 1):
            controls = dict(config.steering.controls)
            controls["variation_amount"] = max(0.05, strength - 0.12 * attempt)
            steer = Steering(controls=controls, locks=[])  # vocals handled by engine
            varied = self.gen.generate(seg_clip, steer, seed=seed + attempt * 7,
                                       instrumental=instrumental).clip
            emb = analysis.embed(varied)
            identity = analysis.similarity(seg_emb, emb)
            # Memory: penalize near-duplicates of recent windows.
            rep = max((analysis.similarity(emb, r) for r in recent), default=0.0)
            ok = identity >= config.identity_threshold and rep < 0.985
            if identity > best_identity:
                best, best_identity = varied, identity
            if ok:
                return varied, identity, attempt
        return best, best_identity, config.max_retries

    @staticmethod
    def _overlay_vocals(instr, instr_sr, vocals, vocals_sr):
        if vocals_sr != instr_sr:
            import librosa
            vocals = np.stack([
                librosa.resample(vocals[c], orig_sr=vocals_sr, target_sr=instr_sr)
                for c in range(vocals.shape[0])
            ])
        if instr.ndim == 1:
            instr = instr[np.newaxis, :]
        n = min(instr.shape[-1], vocals.shape[-1])
        return instr[:, :n] + vocals[:, :n]

    @staticmethod
    def _append(out, nxt, overlap):
        """Equal-power crossfade `nxt` onto the tail of `out`."""
        if out is None:
            return nxt
        ov = min(overlap, out.shape[-1], nxt.shape[-1])
        if ov <= 0:
            return np.concatenate([out, nxt], axis=-1)
        t = np.linspace(0.0, 1.0, ov, dtype=np.float32)
        fo, fi = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
        blended = out[:, -ov:] * fo + nxt[:, :ov] * fi
        return np.concatenate([out[:, :-ov], blended, nxt[:, ov:]], axis=-1)
