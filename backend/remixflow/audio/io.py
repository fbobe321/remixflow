"""Audio load/save with graceful degradation.

We prefer ``soundfile`` (libsndfile: WAV/FLAC/OGG) and fall back to
``librosa``/``audioread`` (which can reach MP3 via ffmpeg when present). All
imports are lazy so the API boots and serves the UI even in an environment
with none of the audio stack installed — feature extraction and DSP simply
report themselves as unavailable rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Clip:
    """Mono/stereo float32 audio in memory. Shape: (channels, samples)."""

    samples: np.ndarray  # float32, shape (channels, n)
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.samples.shape[-1] / self.sample_rate if self.sample_rate else 0.0

    def to_mono(self) -> np.ndarray:
        return self.samples.mean(axis=0) if self.samples.ndim == 2 else self.samples


class AudioUnavailable(RuntimeError):
    """Raised when no audio backend can handle a load/save request."""


def load(path: str) -> Clip:
    """Load an audio file into a :class:`Clip`. Raises AudioUnavailable if the
    audio stack cannot read it (missing libs / unsupported codec)."""
    # Try soundfile first — fast and covers WAV/FLAC/OGG.
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile gives (n, channels); transpose to (channels, n).
        return Clip(samples=np.ascontiguousarray(data.T), sample_rate=int(sr))
    except Exception:
        pass

    try:
        import librosa  # type: ignore

        data, sr = librosa.load(path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return Clip(samples=data.astype(np.float32), sample_rate=int(sr))
    except Exception as exc:  # pragma: no cover - env dependent
        raise AudioUnavailable(
            f"Could not decode {path!r}. Install 'soundfile' (WAV/FLAC/OGG) or "
            f"'librosa'+ffmpeg (MP3). Underlying error: {exc}"
        ) from exc


def save(clip: Clip, path: str) -> None:
    """Persist a clip to disk (WAV via soundfile)."""
    try:
        import soundfile as sf  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        raise AudioUnavailable(
            "Saving audio requires 'soundfile' (pip install soundfile)."
        ) from exc

    data = clip.samples
    # soundfile expects (n, channels).
    out = data.T if data.ndim == 2 else data[:, np.newaxis]
    sf.write(path, np.clip(out, -1.0, 1.0), clip.sample_rate, subtype="PCM_16")


def available() -> bool:
    """True if at least one decode backend is importable."""
    for mod in ("soundfile", "librosa"):
        try:
            __import__(mod)
            return True
        except Exception:
            continue
    return False
