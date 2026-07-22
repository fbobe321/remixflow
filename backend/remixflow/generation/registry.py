"""Generator registry — the plug point for future model backends.

To add a real model backend (e.g. ACE-Step, Stable Audio Open, MusicGen),
implement :class:`Generator` and call ``register_generator(YourBackend())``.
The API exposes registered backends via ``GET /api/backends`` and accepts a
``?backend=`` selector on generate requests.
"""

from __future__ import annotations

from .acestep import AceStepGenerator
from .base import Generator
from .dsp import DSPGenerator

_REGISTRY: dict[str, Generator] = {}


def register_generator(gen: Generator) -> None:
    _REGISTRY[gen.name] = gen


#: Preference order for the default backend — real model first, DSP fallback.
_DEFAULT_ORDER = ("ace-step", "dsp")


def get_generator(name: str | None = None) -> Generator:
    if name and name in _REGISTRY:
        return _REGISTRY[name]
    # Default: best available in preference order (ACE-Step if its stack is
    # installed, else the always-on DSP backend).
    for key in _DEFAULT_ORDER:
        gen = _REGISTRY.get(key)
        if gen and gen.available:
            return gen
    for gen in _REGISTRY.values():
        if gen.available:
            return gen
    return _REGISTRY["dsp"]


def list_generators() -> list[dict[str, object]]:
    return [
        {"name": g.name, "description": g.description, "available": g.available}
        for g in _REGISTRY.values()
    ]


# Register backends on import. ACE-Step self-probes availability (torch +
# weights); the DSP reference backend is always available as a fallback.
register_generator(AceStepGenerator())
register_generator(DSPGenerator())
