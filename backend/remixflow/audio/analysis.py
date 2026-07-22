"""Feature extraction and the 'Musical DNA' embedding (PRD §1, §2, Advanced).

Everything here degrades gracefully: if ``librosa`` is missing we still return
cheap descriptors computed with NumPy (duration, RMS energy, a coarse spectral
centroid, and a deterministic embedding), flagging ``analyzed=False`` so the UI
can show what is estimated vs. measured.
"""

from __future__ import annotations

import numpy as np

from ..models import AudioFeatures
from .io import Clip

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
EMBED_DIM = 16
#: Feature extraction runs at this rate — CQT/beat/MFCC are several× cheaper at
#: 22 kHz than 44.1 kHz with negligible impact on these descriptors.
ANALYSIS_SR = 22050


def _mono_at_analysis_sr(mono: np.ndarray, sr: int):
    """Downsample mono audio to ANALYSIS_SR (via librosa) for cheaper analysis."""
    if sr and sr > ANALYSIS_SR:
        import librosa  # type: ignore

        return librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR), ANALYSIS_SR
    return mono, sr


def _spectral_centroid_np(mono: np.ndarray, sr: int) -> float:
    """A dependency-free spectral centroid (Hz) over the whole clip."""
    if mono.size == 0 or sr == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
    total = spec.sum()
    return float((freqs * spec).sum() / total) if total > 0 else 0.0


def _fallback_embedding(mono: np.ndarray, sr: int) -> list[float]:
    """A stable, low-cost latent vector derived from band energies.

    Not a learned embedding — a stand-in with the right *shape* and stability
    so the similarity evaluator and morph feature work end-to-end until a real
    music encoder backend is plugged into :func:`embed`.
    """
    if mono.size == 0:
        return [0.0] * EMBED_DIM
    spec = np.abs(np.fft.rfft(mono))
    if spec.sum() == 0:
        return [0.0] * EMBED_DIM
    bands = np.array_split(spec, EMBED_DIM)
    vec = np.array([b.mean() for b in bands], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


def analyze(clip: Clip) -> AudioFeatures:
    """Extract descriptors from a clip, using librosa when available."""
    mono = clip.to_mono().astype(np.float32)
    feats = AudioFeatures(
        duration_sec=round(clip.duration, 3),
        sample_rate=clip.sample_rate,
        rms_energy=float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0,
    )

    try:
        import librosa  # type: ignore

        y, sr = _mono_at_analysis_sr(mono, clip.sample_rate)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.tempo_bpm = round(float(np.atleast_1d(tempo)[0]), 2)
        feats.spectral_centroid = round(
            float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 2
        )
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        feats.key = _PITCH_CLASSES[int(np.argmax(chroma.mean(axis=1)))]
        feats.embedding = embed(clip)
        feats.analyzed = True
        feats.note = "librosa"
    except Exception as exc:  # librosa missing or failed — cheap fallback.
        feats.spectral_centroid = round(_spectral_centroid_np(mono, clip.sample_rate), 2)
        feats.embedding = _fallback_embedding(mono, clip.sample_rate)
        feats.analyzed = False
        feats.note = f"estimated (librosa unavailable: {type(exc).__name__})"

    return feats


def embed(clip: Clip) -> list[float]:
    """Return the Musical DNA embedding for a clip.

    Uses MFCC statistics via librosa when available (a reasonable timbral
    fingerprint), else the NumPy band-energy fallback. This is the seam where a
    real learned music encoder (CLAP / MERT / MusicFM) plugs in later.
    """
    mono = clip.to_mono().astype(np.float32)
    try:
        import librosa  # type: ignore

        y, sr = _mono_at_analysis_sr(mono, clip.sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=EMBED_DIM)
        vec = mfcc.mean(axis=1).astype(np.float64)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
    except Exception:
        return _fallback_embedding(mono, clip.sample_rate)


def similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity mapped to [0, 1] — the identity-preservation score."""
    if not a or not b or len(a) != len(b):
        return 0.0
    va, vb = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    cos = float(np.dot(va, vb) / denom)
    return round(max(0.0, min(1.0, (cos + 1.0) / 2.0)), 4)
