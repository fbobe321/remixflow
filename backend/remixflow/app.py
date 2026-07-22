"""FastAPI application: import, steering, generation, evolution tree, A/B,
morphing, and preference learning. Serves the built React UI when present."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio.io import available as audio_available
from .config import STATIC_DIR, Settings
from .generation import list_generators
from .jobs import JobManager
from .models import GenerateRequest, LivingRequest, MorphRequest, PresetCreate, RateRequest
from .params import controls_manifest
from .presets import PresetStore
from .service import RemixService, ServiceError
from .store import Store

ALLOWED_EXT = {".mp3", ".wav", ".flac", ".ogg"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir)
    service = RemixService(store)
    jobs = JobManager(max_workers=1)
    presets = PresetStore(settings.data_dir)

    app = FastAPI(title="RemixFlow", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.service = service
    app.state.settings = settings
    app.state.jobs = jobs

    def get_service() -> RemixService:
        return service

    # --- meta ------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "audio_backend": audio_available(),
            "backends": list_generators(),
        }

    @app.get("/api/controls")
    def controls() -> dict:
        """The full steering control surface for the UI to render."""
        return controls_manifest()

    @app.get("/api/backends")
    def backends() -> list:
        return list_generators()

    # --- songs -----------------------------------------------------------

    @app.post("/api/songs")
    async def import_song(
        file: UploadFile = File(...),
        title: str = Query(default=""),
        svc: RemixService = Depends(get_service),
    ) -> dict:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"Unsupported format {ext!r}. Allowed: {sorted(ALLOWED_EXT)}")

        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit.")

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            song, root = svc.import_song(tmp_path, title, file.filename or "")
        except ServiceError as exc:
            raise HTTPException(422, str(exc))
        finally:
            os.unlink(tmp_path)
        return {"song": song.model_dump(), "root": root.model_dump()}

    @app.get("/api/songs")
    def list_songs() -> list:
        return [s.model_dump() for s in store.list_songs()]

    @app.get("/api/songs/{song_id}")
    def get_song(song_id: str) -> dict:
        song = store.get_song(song_id)
        if not song:
            raise HTTPException(404, "Song not found")
        return song.model_dump()

    @app.delete("/api/songs/{song_id}")
    def delete_song(song_id: str) -> dict:
        if not store.delete_song(song_id):
            raise HTTPException(404, "Song not found")
        return {"deleted": song_id}

    @app.get("/api/songs/{song_id}/tree")
    def get_tree(song_id: str) -> dict:
        tree = store.tree(song_id)
        if tree is None:
            raise HTTPException(404, "Song not found")
        return tree.model_dump()

    @app.get("/api/songs/{song_id}/preferences")
    def preferences(song_id: str, svc: RemixService = Depends(get_service)) -> dict:
        if not store.get_song(song_id):
            raise HTTPException(404, "Song not found")
        return svc.preference_profile(song_id)

    # --- variants --------------------------------------------------------

    @app.get("/api/variants/{variant_id}")
    def get_variant(variant_id: str) -> dict:
        v = store.get_variant(variant_id)
        if not v:
            raise HTTPException(404, "Variant not found")
        return v.model_dump()

    @app.post("/api/generate", status_code=202)
    def generate(
        req: GenerateRequest,
        backend: str | None = Query(default=None),
        seed: int | None = Query(default=None),
        svc: RemixService = Depends(get_service),
    ) -> dict:
        """Enqueue a generation job. Returns a job the client polls; real model
        backends (ACE-Step) run for tens of seconds, so this is never blocking."""
        parent = store.get_variant(req.parent_id)
        if parent is None:
            raise HTTPException(422, f"Unknown parent variant {req.parent_id!r}.")

        def work(report) -> dict:
            report(0.05, "Preparing…")
            report(0.15, f"Generating with {backend or 'default'} backend…")
            variant = svc.generate(req, backend=backend, seed=seed)
            report(0.98, "Finalizing…")
            return variant.model_dump()

        job = jobs.submit("generate", work, song_id=parent.song_id)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job.to_dict()

    @app.get("/api/jobs")
    def recent_jobs() -> list:
        return [j.to_dict() for j in jobs.recent()]

    @app.post("/api/generate/sync")
    def generate_sync(
        req: GenerateRequest,
        backend: str | None = Query(default=None),
        seed: int | None = Query(default=None),
        svc: RemixService = Depends(get_service),
    ) -> dict:
        """Blocking generation — convenient for tests/scripts and fast backends."""
        try:
            variant = svc.generate(req, backend=backend, seed=seed)
        except ServiceError as exc:
            raise HTTPException(422, str(exc))
        return variant.model_dump()

    @app.post("/api/morph")
    def morph(req: MorphRequest, svc: RemixService = Depends(get_service)) -> dict:
        try:
            variant = svc.morph(req)
        except ServiceError as exc:
            raise HTTPException(422, str(exc))
        return variant.model_dump()

    @app.post("/api/variants/{variant_id}/rate")
    def rate(variant_id: str, req: RateRequest, svc: RemixService = Depends(get_service)) -> dict:
        try:
            variant = svc.rate(variant_id, req.rating)
        except ServiceError as exc:
            raise HTTPException(404, str(exc))
        return variant.model_dump()

    # --- Living Mode (PRD Phase 2) ---------------------------------------

    @app.post("/api/living", status_code=202)
    def living(
        req: LivingRequest,
        backend: str | None = Query(default=None),
        svc: RemixService = Depends(get_service),
    ) -> dict:
        """Enqueue a Living segment render. Poll the returned job; its result is a
        segment {audio_url, next_index, ...}. Call again with start_index=next_index
        for a seamless continuation (Living Repeat)."""
        if not store.get_song(req.song_id):
            raise HTTPException(404, "Song not found")

        def work(report) -> dict:
            report(0.02, "Starting Living…")
            return svc.living_segment(req, backend=backend, progress=report)

        job = jobs.submit("living", work, song_id=req.song_id)
        return job.to_dict()

    @app.get("/api/presets")
    def list_presets() -> list:
        return [p.model_dump() for p in presets.list()]

    @app.post("/api/presets", status_code=201)
    def create_preset(req: PresetCreate) -> dict:
        return presets.add(req.name, req.params).model_dump()

    @app.delete("/api/presets/{preset_id}")
    def delete_preset(preset_id: str) -> dict:
        if not presets.delete(preset_id):
            raise HTTPException(404, "Preset not found (built-in presets can't be deleted)")
        return {"deleted": preset_id}

    @app.get("/api/living/audio/{seg_id}")
    def living_audio(seg_id: str) -> FileResponse:
        # seg_id is server-minted (live_<hex>); constrain to that shape.
        if not seg_id.startswith("live_") or "/" in seg_id or ".." in seg_id:
            raise HTTPException(400, "Bad segment id")
        path = store.audio_dir / f"{seg_id}.wav"
        if not path.exists():
            raise HTTPException(404, "Segment not found")
        return FileResponse(str(path), media_type="audio/wav")

    @app.get("/api/audio/{variant_id}")
    def get_audio(variant_id: str) -> FileResponse:
        v = store.get_variant(variant_id)
        if not v or not v.audio_path or not Path(v.audio_path).exists():
            raise HTTPException(404, "Audio not found")
        return FileResponse(v.audio_path, media_type="audio/wav")

    # --- static frontend (built React app) -------------------------------

    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        @app.exception_handler(404)
        async def spa_fallback(request, exc):  # noqa: ANN001
            # Serve the SPA shell for client-side routes, but keep API 404s real.
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            return FileResponse(STATIC_DIR / "index.html")
    else:

        @app.get("/")
        def index_dev() -> dict:
            return {
                "app": "RemixFlow API",
                "version": __version__,
                "note": "Frontend not built. Run the Vite dev server, or build it "
                        "so it is served here. See README.",
                "docs": "/docs",
            }

    return app
