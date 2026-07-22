"""RemixFlow — an AI music evolution platform (see PRD.md).

This package hosts the FastAPI backend: song import, feature extraction, a
pluggable steering/generation engine, an evolution tree, A/B comparison,
morphing, and preference learning.
"""

__version__ = "0.1.0"

from .app import create_app

__all__ = ["create_app", "__version__"]
