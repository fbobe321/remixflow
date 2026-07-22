"""A real, dependency-light reference generator.

This backend produces *audible* variations today using classic DSP rather than
a generative model: time-stretch, pitch/key nudges, spectral tilt (brightness/
warmth), low-shelf (bass), drive/compression (energy), and stereo width. It is
deliberately the reference implementation of the :class:`Generator` contract —
proof the whole pipeline (import -> steer -> generate -> evaluate -> branch)
works end-to-end, and the template a diffusion/transformer backend replaces.

The steering->DSP mapping is intentionally legible: ``variation_amount`` scales
the magnitude of every change, and identity ``locks`` suppress the transforms
that would disturb a pinned element (e.g. locking 'melody' disables pitch/key
shifting; locking 'rhythm' disables time-stretch).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..audio.io import Clip
from ..models import Steering
from .base import GenerationResult, Generator


def _blend(clip: Clip, transformed: np.ndarray, amount: float) -> np.ndarray:
    """Crossfade original<->transformed by ``amount`` (dry/wet), length-safe."""
    orig = clip.samples
    n = min(orig.shape[-1], transformed.shape[-1])
    a = orig[..., :n]
    b = transformed[..., :n]
    return (1.0 - amount) * a + amount * b


def _spectral_tilt(mono: np.ndarray, tilt: float) -> np.ndarray:
    """Apply a linear high/low frequency tilt. tilt>0 brightens, <0 darkens."""
    if mono.size == 0 or abs(tilt) < 1e-4:
        return mono
    spec = np.fft.rfft(mono)
    freqs = np.linspace(0.0, 1.0, spec.shape[0])
    gain = 1.0 + tilt * (freqs - 0.5) * 2.0
    gain = np.clip(gain, 0.05, 4.0)
    return np.fft.irfft(spec * gain, n=mono.size).astype(np.float32)


def _low_shelf(mono: np.ndarray, gain: float, sr: int) -> np.ndarray:
    """Boost/cut low frequencies (below ~250 Hz) by ``gain`` (multiplicative)."""
    if mono.size == 0 or abs(gain) < 1e-4:
        return mono
    spec = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr) if sr else np.zeros(spec.shape[0])
    shelf = np.where(freqs < 250.0, 1.0 + gain, 1.0)
    return np.fft.irfft(spec * shelf, n=mono.size).astype(np.float32)


def _drive(x: np.ndarray, amount: float) -> np.ndarray:
    """Soft saturation + makeup — raises perceived energy/loudness."""
    if abs(amount) < 1e-4:
        return x
    k = 1.0 + 4.0 * max(0.0, amount)
    return np.tanh(x * k) / np.tanh(k) if k > 0 else x


class DSPGenerator(Generator):
    name = "dsp"
    description = "Classic DSP reference backend (time-stretch, EQ, drive)."

    def __init__(self) -> None:
        self._has_librosa = self._probe_librosa()
        self.available = True  # NumPy-only path always works

    @staticmethod
    def _probe_librosa() -> bool:
        try:
            import librosa  # noqa: F401
            return True
        except Exception:
            return False

    def generate(
        self,
        parent: Clip,
        steering: Steering,
        *,
        original: Optional[Clip] = None,
        seed: Optional[int] = None,
        instrumental: bool = False,  # accepted for interface parity; DSP is stem-agnostic
    ) -> GenerationResult:
        s = steering.normalized()
        c = s.controls
        locks = set(s.locks)
        amount = float(c.get("variation_amount", 0.2))  # global magnitude
        rng = np.random.default_rng(seed)

        sr = parent.sample_rate
        stereo = parent.samples
        note_bits: list[str] = []

        # --- Tempo / time-stretch (respects 'rhythm' lock) ------------------
        tempo = float(c.get("tempo", 0.0))
        if not {"rhythm", "hook"} & locks and abs(tempo) > 1e-3 and self._has_librosa:
            rate = 1.0 + 0.25 * tempo * amount  # +/-25% at full swing
            stereo = self._time_stretch(stereo, rate)
            note_bits.append(f"tempo x{rate:.2f}")

        # Work per-channel in the spectral domain for the tone shaping.
        chans = stereo if stereo.ndim == 2 else stereo[np.newaxis, :]
        out = np.empty_like(chans)
        brightness = float(c.get("brightness", 0.0)) - float(c.get("warmth", 0.0)) * 0.5
        bass = float(c.get("bass", 0.0)) + float(c.get("electronic", 0.0)) * 0.4
        energy = float(c.get("energy", 0.0)) + float(c.get("rock", 0.0)) * 0.5

        for i in range(chans.shape[0]):
            y = chans[i].astype(np.float32)
            if not {"instrumentation"} & locks:
                y = _spectral_tilt(y, brightness * amount * 0.8)
                y = _low_shelf(y, bass * amount, sr)
            y = _drive(y, energy * amount * 0.6)
            out[i] = y
        if brightness or bass or energy:
            note_bits.append("tone/energy shaping")

        # --- Pitch / key nudge (respects 'melody' & 'chord_progression') ----
        emotional = float(c.get("emotional_tone", 0.0))
        if (
            not {"melody", "chord_progression", "lyrics"} & locks
            and abs(emotional) > 0.4
            and amount > 0.5
            and self._has_librosa
        ):
            semitones = 1.0 if emotional > 0 else -1.0  # brighter/darker mode feel
            out = self._pitch_shift(out, sr, semitones)
            note_bits.append(f"pitch {semitones:+.0f}st")

        # --- Stereo width from complexity / instrument density --------------
        width = float(c.get("complexity", 0.0)) * 0.5 + float(c.get("instrument_density", 0.0)) * 0.5
        if out.shape[0] == 2 and abs(width) > 1e-3:
            mid = (out[0] + out[1]) * 0.5
            side = (out[0] - out[1]) * 0.5 * (1.0 + width * amount)
            out = np.stack([mid + side, mid - side])
            note_bits.append("stereo width")

        # A touch of controlled randomness so repeated generations differ,
        # scaled by variation amount (never on a near-0% "nearly identical").
        if amount > 0.05:
            jitter = rng.normal(0.0, 0.0008 * amount, size=out.shape).astype(np.float32)
            out = out + jitter

        # --- Crossfade toward the transformed signal by `amount` ------------
        mixed = _blend(parent, out, amount)
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 1.0:
            mixed = mixed / peak  # prevent clipping

        result_clip = Clip(samples=mixed.astype(np.float32), sample_rate=sr)
        note = ", ".join(note_bits) if note_bits else "subtle blend"
        return GenerationResult(clip=result_clip, generator=self.name, note=note)

    # --- librosa-backed helpers (skipped when unavailable) -----------------

    def _time_stretch(self, stereo: np.ndarray, rate: float) -> np.ndarray:
        import librosa  # type: ignore

        chans = stereo if stereo.ndim == 2 else stereo[np.newaxis, :]
        stretched = [librosa.effects.time_stretch(chans[i], rate=rate) for i in range(chans.shape[0])]
        n = min(len(s) for s in stretched)
        return np.stack([s[:n] for s in stretched])

    def _pitch_shift(self, stereo: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        import librosa  # type: ignore

        chans = stereo if stereo.ndim == 2 else stereo[np.newaxis, :]
        shifted = [librosa.effects.pitch_shift(chans[i], sr=sr, n_steps=semitones) for i in range(chans.shape[0])]
        return np.stack(shifted)
