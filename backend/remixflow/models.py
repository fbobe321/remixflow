"""Pydantic models for the RemixFlow API and evolution tree."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field

from .params import CONTROLS_BY_KEY, IDENTITY_ELEMENTS, default_steering


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


class AudioFeatures(BaseModel):
    """Extracted musical descriptors (PRD §1 & §2)."""

    duration_sec: float = 0.0
    sample_rate: int = 0
    tempo_bpm: Optional[float] = None
    key: Optional[str] = None
    rms_energy: Optional[float] = None
    spectral_centroid: Optional[float] = None
    # "Musical DNA" embedding — a compact latent vector (PRD Advanced Features).
    embedding: list[float] = Field(default_factory=list)
    analyzed: bool = False
    note: str = ""


class Steering(BaseModel):
    """A full steering payload: control values + identity locks.

    ``controls`` holds the slider/selector values keyed by ``Control.key``.
    ``locks`` is the subset of identity elements the user pinned as fixed.
    """

    controls: dict[str, object] = Field(default_factory=default_steering)
    locks: list[str] = Field(default_factory=list)

    def normalized(self) -> "Steering":
        """Clamp/validate values against the control catalog."""
        base = default_steering()
        for key, val in self.controls.items():
            spec = CONTROLS_BY_KEY.get(key)
            if spec is None:
                continue
            if spec.kind == "bipolar":
                base[key] = max(-1.0, min(1.0, float(val)))  # type: ignore[arg-type]
            elif spec.kind == "unipolar":
                base[key] = max(0.0, min(1.0, float(val)))  # type: ignore[arg-type]
            elif spec.kind == "enum":
                base[key] = val if val in spec.options else spec.options[0]
            elif spec.kind == "multi":
                items = val if isinstance(val, list) else []
                base[key] = [x for x in items if x in spec.options]
        locks = [x for x in self.locks if x in IDENTITY_ELEMENTS]
        return Steering(controls=base, locks=locks)


class Variant(BaseModel):
    """A node in the evolution tree — the original is the root variant."""

    id: str = Field(default_factory=lambda: _uid("var"))
    song_id: str
    parent_id: Optional[str] = None
    label: str = ""
    is_original: bool = False
    created_at: float = Field(default_factory=_now)

    steering: Steering = Field(default_factory=Steering)
    features: AudioFeatures = Field(default_factory=AudioFeatures)
    # Identity preservation score vs. the original (0..1). PRD §"Similarity".
    similarity: Optional[float] = None
    audio_path: Optional[str] = None  # server-side path; served via /api/audio
    generator: str = ""

    # Preference learning (PRD §8): -1 dislike, 0 neutral, 1 love, None unrated.
    rating: Optional[int] = None


class Song(BaseModel):
    """An imported source track and the root of its evolution tree."""

    id: str = Field(default_factory=lambda: _uid("song"))
    title: str
    original_filename: str = ""
    created_at: float = Field(default_factory=_now)
    root_variant_id: Optional[str] = None


# --- Request / response payloads -------------------------------------------


class GenerateRequest(BaseModel):
    parent_id: str
    steering: Steering
    label: str = ""


class RateRequest(BaseModel):
    rating: int  # -1, 0, 1


class MorphRequest(BaseModel):
    """Blend two variants into an intermediate (PRD Advanced: Morph)."""

    variant_a: str
    variant_b: str
    blend: float = 0.5  # 0 == all A, 1 == all B
    steering: Optional[Steering] = None


class PresetParams(BaseModel):
    """A saved Living configuration: how much it evolves + the style drift."""

    improvisation: float = 0.35
    duration_sec: float = 40.0
    controls: dict[str, object] = Field(default_factory=dict)


class Preset(BaseModel):
    id: str
    name: str
    builtin: bool = False
    description: str = ""
    params: PresetParams = Field(default_factory=PresetParams)


class PresetCreate(BaseModel):
    name: str
    params: PresetParams


class LivingRequest(BaseModel):
    """Render one segment of an endless Living performance (PRD Phase 2)."""

    song_id: str
    steering: Steering = Field(default_factory=Steering)
    duration_sec: float = 40.0
    start_index: int = 0          # continue a performance (from a prior next_index)
    start_pos: float = 0.0        # source position to resume at (from next_pos)
    seed: int = 0
    improvisation: float = 0.3    # 0 = barely evolves, 1 = adventurous


class TreeNode(BaseModel):
    variant: Variant
    children: list["TreeNode"] = Field(default_factory=list)


TreeNode.model_rebuild()
