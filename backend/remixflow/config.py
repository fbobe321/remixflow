"""Runtime configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
# Built frontend assets are copied here at package build time (see pyproject).
STATIC_DIR = PKG_ROOT / "static"


@dataclass
class Settings:
    data_dir: Path
    max_upload_mb: int
    cors_origins: list[str]

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("REMIXFLOW_DATA_DIR", "./remixflow_data")).resolve()
        origins = os.environ.get("REMIXFLOW_CORS", "*").split(",")
        return cls(
            data_dir=data_dir,
            max_upload_mb=int(os.environ.get("REMIXFLOW_MAX_UPLOAD_MB", "50")),
            cors_origins=[o.strip() for o in origins if o.strip()],
        )
