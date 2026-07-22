"""Module-level ASGI app for uvicorn (``remixflow.server:app``)."""

from .app import create_app

app = create_app()
