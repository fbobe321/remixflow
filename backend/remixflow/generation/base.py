"""The generator contract every backend implements.

A ``Generator`` takes a *parent* clip plus a normalized :class:`Steering`
payload and returns a new clip that preserves the parent's identity to the
degree requested. The steering engine (mapping UI sliders -> transformations)
lives inside each backend so different models can honor the same controls in
their own latent space.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from ..audio.io import Clip
from ..models import Steering


@dataclass
class GenerationResult:
    clip: Clip
    generator: str
    note: str = ""


class Generator(abc.ABC):
    """Base class for all music variation backends."""

    name: str = "base"
    #: Human-facing description shown in the UI backend picker.
    description: str = ""
    #: False when required heavy deps/models are absent — the API can surface
    #: this instead of failing a request.
    available: bool = True

    @abc.abstractmethod
    def generate(
        self,
        parent: Clip,
        steering: Steering,
        *,
        original: Optional[Clip] = None,
        seed: Optional[int] = None,
        instrumental: bool = False,
    ) -> GenerationResult:
        """Produce a variation of ``parent``.

        ``original`` (the root clip) is provided so backends can re-anchor to
        the source and limit quality drift over long evolution chains (PRD Risks
        §"Quality Drift"). ``seed`` makes generation reproducible. ``instrumental``
        hints that ``parent`` is a vocals-removed backing track, so the backend
        should avoid generating vocal content (used by Living Mode).
        """
        raise NotImplementedError
