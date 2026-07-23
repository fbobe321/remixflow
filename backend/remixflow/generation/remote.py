"""RemoteGenerator — offload generation to a hosted RemixFlow model server.

Turns a lightweight RemixFlow instance (no torch/GPU) into a thin client: set
``REMIXFLOW_MODEL_URL`` to a RemixFlow server that has a real backend (e.g. the
GPU box or the `fbobe3/remixflow:gpu` container) and generation is forwarded to
its stateless ``/api/infer`` endpoint. This is also how a phone app can drive
generation before the on-device MLX build is ready.

Env:
  REMIXFLOW_MODEL_URL       base URL of the model server (enables this backend)
  REMIXFLOW_REMOTE_BACKEND  backend to request on the server (default "ace-step")
  REMIXFLOW_REMOTE_TOKEN    optional bearer token (sent as Authorization header)
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Optional

import numpy as np

from ..audio.io import Clip
from ..models import Steering
from .base import GenerationResult, Generator


class RemoteGenerator(Generator):
    name = "remote"
    description = "Forwards generation to a hosted RemixFlow model server (REMIXFLOW_MODEL_URL)."

    def __init__(self) -> None:
        self.url = (os.environ.get("REMIXFLOW_MODEL_URL") or "").rstrip("/")
        self.remote_backend = os.environ.get("REMIXFLOW_REMOTE_BACKEND", "ace-step")
        self.token = os.environ.get("REMIXFLOW_REMOTE_TOKEN")
        self.timeout = float(os.environ.get("REMIXFLOW_REMOTE_TIMEOUT", "600"))
        self.available = bool(self.url)

    # --- http helpers (stdlib only) ---------------------------------------

    def _headers(self, extra: dict | None = None) -> dict:
        h = dict(extra or {})
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _post_wav(self, path: str, body: bytes, params: dict) -> dict:
        url = f"{self.url}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers=self._headers({"Content-Type": "application/octet-stream"}))
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _get_json(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.url}{path}", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _get_bytes(self, path: str) -> bytes:
        req = urllib.request.Request(f"{self.url}{path}", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    # --- generation --------------------------------------------------------

    def generate(
        self,
        parent: Clip,
        steering: Steering,
        *,
        original: Optional[Clip] = None,
        seed: Optional[int] = None,
        instrumental: bool = False,
    ) -> GenerationResult:
        if not self.available:
            raise RuntimeError("RemoteGenerator unavailable (set REMIXFLOW_MODEL_URL).")

        import soundfile as sf  # local to keep import light

        # Serialize the parent clip to WAV bytes.
        data = parent.samples
        out = data.T if data.ndim == 2 else data[:, np.newaxis]
        buf = io.BytesIO()
        sf.write(buf, np.clip(out, -1.0, 1.0), parent.sample_rate, format="WAV", subtype="PCM_16")

        params = {
            "steering": json.dumps(steering.normalized().model_dump()),
            "backend": self.remote_backend,
            "instrumental": "1" if instrumental else "0",
        }
        if seed is not None:
            params["seed"] = int(seed)

        job = self._post_wav("/api/infer", buf.getvalue(), params)
        job_id = job["id"]

        # Poll the remote job to completion.
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            job = self._get_json(f"/api/jobs/{job_id}")
            status = job.get("status")
            if status == "done":
                break
            if status == "error":
                raise RuntimeError(f"Remote generation failed: {job.get('error')}")
            time.sleep(0.7)
        else:
            raise RuntimeError("Remote generation timed out.")

        result = job["result"]
        audio = self._get_bytes(result["audio_url"])
        clip_data, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        clip = Clip(samples=np.ascontiguousarray(clip_data.T), sample_rate=int(sr))
        note = result.get("note", "")
        return GenerationResult(clip=clip, generator=self.name, note=f"remote({self.remote_backend}): {note}")
