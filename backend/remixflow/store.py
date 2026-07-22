"""A simple file-backed store for songs, variants, and the evolution tree.

Deliberately dependency-free (JSON + a data dir of audio files) so the MVP runs
without a database. The interface is narrow enough to swap for SQLModel/Postgres
later without touching the API layer.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from .models import Song, TreeNode, Variant


class Store:
    def __init__(self, data_dir: str | os.PathLike) -> None:
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "db.json"
        self._lock = threading.RLock()
        self._songs: dict[str, Song] = {}
        self._variants: dict[str, Variant] = {}
        self._load()

    # --- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        try:
            raw = json.loads(self._db_path.read_text())
            self._songs = {k: Song(**v) for k, v in raw.get("songs", {}).items()}
            self._variants = {k: Variant(**v) for k, v in raw.get("variants", {}).items()}
        except Exception:
            # Corrupt/legacy db — start clean rather than crash the server.
            self._songs, self._variants = {}, {}

    def _flush(self) -> None:
        tmp = self._db_path.with_suffix(".tmp")
        payload = {
            "songs": {k: json.loads(v.model_dump_json()) for k, v in self._songs.items()},
            "variants": {k: json.loads(v.model_dump_json()) for k, v in self._variants.items()},
        }
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._db_path)

    def audio_path_for(self, variant_id: str) -> Path:
        return self.audio_dir / f"{variant_id}.wav"

    # --- songs -------------------------------------------------------------

    def add_song(self, song: Song, root: Variant) -> None:
        with self._lock:
            song.root_variant_id = root.id
            self._songs[song.id] = song
            self._variants[root.id] = root
            self._flush()

    def get_song(self, song_id: str) -> Optional[Song]:
        return self._songs.get(song_id)

    def list_songs(self) -> list[Song]:
        return sorted(self._songs.values(), key=lambda s: s.created_at, reverse=True)

    def delete_song(self, song_id: str) -> bool:
        with self._lock:
            if song_id not in self._songs:
                return False
            for v in [v for v in self._variants.values() if v.song_id == song_id]:
                self._variants.pop(v.id, None)
                p = self.audio_path_for(v.id)
                if p.exists():
                    p.unlink()
            self._songs.pop(song_id, None)
            self._flush()
            return True

    # --- variants ----------------------------------------------------------

    def add_variant(self, variant: Variant) -> None:
        with self._lock:
            self._variants[variant.id] = variant
            self._flush()

    def get_variant(self, variant_id: str) -> Optional[Variant]:
        return self._variants.get(variant_id)

    def update_variant(self, variant: Variant) -> None:
        with self._lock:
            self._variants[variant.id] = variant
            self._flush()

    def variants_for_song(self, song_id: str) -> list[Variant]:
        return [v for v in self._variants.values() if v.song_id == song_id]

    def tree(self, song_id: str) -> Optional[TreeNode]:
        """Build the evolution tree (PRD §5) rooted at the song's original."""
        song = self._songs.get(song_id)
        if not song or not song.root_variant_id:
            return None
        variants = self.variants_for_song(song_id)
        children: dict[str, list[Variant]] = {}
        for v in variants:
            children.setdefault(v.parent_id or "", []).append(v)

        def build(v: Variant) -> TreeNode:
            kids = sorted(children.get(v.id, []), key=lambda x: x.created_at)
            return TreeNode(variant=v, children=[build(k) for k in kids])

        root = self._variants.get(song.root_variant_id)
        return build(root) if root else None
