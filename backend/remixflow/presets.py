"""Listening-mode presets (PRD Phase 2 §Listening Modes) + a store for
user-saved presets.

Built-in modes are read-only configurations of the Living controls; users can
save their own (persisted to ``presets.json`` in the data dir) and delete them.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from .models import Preset, PresetParams

# --- Built-in listening modes ---------------------------------------------
# Values are valid steering controls: bipolar in [-1,1], unipolar in [0,1],
# instrument_focus a list. improvisation drives the tension range.

BUILTINS: list[Preset] = [
    Preset(id="studio", name="Studio", builtin=True,
           description="Near-identical playback for casual listening.",
           params=PresetParams(improvisation=0.12, duration_sec=45, controls={})),
    Preset(id="live", name="Live Performance", builtin=True,
           description="Natural variation, like a live band.",
           params=PresetParams(improvisation=0.45, duration_sec=35,
                               controls={"energy": 0.3, "groove": 0.3})),
    Preset(id="jazz", name="Jazz", builtin=True,
           description="Heavy improvisation while keeping the harmony.",
           params=PresetParams(improvisation=0.75, duration_sec=30,
                               controls={"jazz": 0.7, "swing": 0.4, "warmth": 0.4})),
    Preset(id="orchestra", name="Orchestra", builtin=True,
           description="Subtle reinterpretation of dynamics and orchestration.",
           params=PresetParams(improvisation=0.3, duration_sec=50,
                               controls={"acoustic": 0.6, "warmth": 0.5,
                                         "instrument_focus": ["strings", "piano"]})),
    Preset(id="ambient", name="Ambient", builtin=True,
           description="Continuous atmospheric evolution.",
           params=PresetParams(improvisation=0.5, duration_sec=55,
                               controls={"energy": -0.4, "electronic": 0.4,
                                         "brightness": -0.2})),
    Preset(id="radio", name="Infinite Radio", builtin=True,
           description="An endless version of the song with no obvious loops.",
           params=PresetParams(improvisation=0.4, duration_sec=40, controls={})),
]


class PresetStore:
    def __init__(self, data_dir: str | Path) -> None:
        self._path = Path(data_dir) / "presets.json"
        self._lock = threading.RLock()
        self._user: list[Preset] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._user = [Preset(**p) for p in raw]
            except Exception:
                self._user = []

    def _flush(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([json.loads(p.model_dump_json()) for p in self._user], indent=2))
        tmp.replace(self._path)

    def list(self) -> list[Preset]:
        """Built-ins first, then user presets."""
        return list(BUILTINS) + self._user

    def add(self, name: str, params: PresetParams) -> Preset:
        with self._lock:
            preset = Preset(id=f"user_{uuid.uuid4().hex[:8]}", name=name.strip() or "My preset",
                            builtin=False, description="", params=params)
            self._user.append(preset)
            self._flush()
            return preset

    def delete(self, preset_id: str) -> bool:
        with self._lock:
            before = len(self._user)
            # Built-ins are protected.
            self._user = [p for p in self._user if p.id != preset_id]
            if len(self._user) != before:
                self._flush()
                return True
            return False
