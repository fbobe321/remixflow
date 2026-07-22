"""End-to-end smoke test of the import -> steer -> generate -> branch pipeline.

Runs against the in-process ASGI app with a synthesized WAV (no external
fixtures needed), so `pytest` verifies the whole backend without a real song.
"""

from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from remixflow.app import create_app
from remixflow.config import Settings


def _make_wav(seconds: float = 2.0, sr: int = 22050, freq: float = 220.0) -> bytes:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # A chord-ish tone so analysis has something to chew on.
    sig = 0.4 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 1.5 * t)
    pcm = (sig * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def make_client(tmp_path) -> TestClient:
    settings = Settings(data_dir=tmp_path, max_upload_mb=50, cors_origins=["*"])
    return TestClient(create_app(settings))


def test_health_and_controls(tmp_path):
    client = make_client(tmp_path)
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert any(b["name"] == "dsp" for b in h["backends"])

    controls = client.get("/api/controls").json()
    keys = {c["key"] for c in controls["controls"]}
    # Every PRD slider is present.
    assert {"variation_amount", "energy", "tempo", "blues", "vocal_style"} <= keys
    assert "melody" in controls["identityElements"]


def test_full_evolution_flow(tmp_path):
    client = make_client(tmp_path)
    wav = _make_wav()

    # 1. Import
    r = client.post("/api/songs?title=Test", files={"file": ("test.wav", wav, "audio/wav")})
    assert r.status_code == 200, r.text
    song_id = r.json()["song"]["id"]
    root_id = r.json()["root"]["id"]
    assert r.json()["root"]["is_original"] is True
    assert r.json()["root"]["features"]["duration_sec"] > 1.0

    # 2. Generate a variant from the original
    gen = client.post(
        "/api/generate/sync",
        params={"seed": 7},
        json={
            "parent_id": root_id,
            "label": "",
            "steering": {
                "controls": {"variation_amount": 0.5, "tempo": 0.4, "brightness": 0.6, "bass": 0.5},
                "locks": ["melody"],
            },
        },
    )
    assert gen.status_code == 200, gen.text
    v1 = gen.json()
    assert v1["parent_id"] == root_id
    assert v1["similarity"] is not None
    assert 0.0 <= v1["similarity"] <= 1.0

    # 3. Branch again from the variant (infinite evolution)
    gen2 = client.post(
        "/api/generate/sync",
        json={"parent_id": v1["id"], "steering": {"controls": {"variation_amount": 0.3}, "locks": []}},
    )
    assert gen2.status_code == 200, gen2.text
    v2 = gen2.json()

    # 4. Tree reflects the branching structure
    tree = client.get(f"/api/songs/{song_id}/tree").json()
    assert tree["variant"]["id"] == root_id
    assert tree["children"][0]["variant"]["id"] == v1["id"]
    assert tree["children"][0]["children"][0]["variant"]["id"] == v2["id"]

    # 5. Audio is retrievable (A/B compare fetches these)
    audio = client.get(f"/api/audio/{v1['id']}")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    assert len(audio.content) > 1000

    # 6. Rating + preference learning
    client.post(f"/api/variants/{v1['id']}/rate", json={"rating": 1})
    prefs = client.get(f"/api/songs/{song_id}/preferences").json()
    assert prefs["loved"] == 1
    assert prefs["preferred_variation"] == 0.5


def test_async_generation_job(tmp_path):
    client = make_client(tmp_path)
    wav = _make_wav()
    root_id = client.post("/api/songs", files={"file": ("j.wav", wav, "audio/wav")}).json()["root"]["id"]

    # Async generate returns a job (202); poll until done.
    resp = client.post(
        "/api/generate",
        json={"parent_id": root_id, "steering": {"controls": {"variation_amount": 0.4}, "locks": []}},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] in ("queued", "running", "done")

    import time

    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.2)
    assert job["status"] == "done", job.get("error")
    assert job["result"]["parent_id"] == root_id
    assert job["result"]["similarity"] is not None


def test_morph_between_variants(tmp_path):
    client = make_client(tmp_path)
    wav = _make_wav()
    root_id = client.post("/api/songs", files={"file": ("a.wav", wav, "audio/wav")}).json()["root"]["id"]
    a = client.post("/api/generate/sync", json={"parent_id": root_id, "steering": {"controls": {"variation_amount": 0.6}}}).json()
    b = client.post("/api/generate/sync", json={"parent_id": root_id, "steering": {"controls": {"variation_amount": 0.4}}}).json()

    m = client.post("/api/morph", json={"variant_a": a["id"], "variant_b": b["id"], "blend": 0.5})
    assert m.status_code == 200, m.text
    assert m.json()["similarity"] is not None
