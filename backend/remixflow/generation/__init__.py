"""Music generation backends. The interface is the important part — real
diffusion/transformer models (ACE-Step, Stable Audio, MusicGen) implement the
same :class:`~remixflow.generation.base.Generator` contract."""

from .base import Generator, GenerationResult
from .dsp import DSPGenerator
from .registry import get_generator, register_generator, list_generators

__all__ = [
    "Generator",
    "GenerationResult",
    "DSPGenerator",
    "get_generator",
    "register_generator",
    "list_generators",
]
