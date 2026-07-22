"""In-process async job manager for generation.

Any real music model (ACE-Step, Stable Audio) takes tens of seconds, so
generation must not block the HTTP request. Jobs run on a single background
worker (the GPU is a serial resource) and the UI polls ``GET /api/jobs/{id}``.

Deliberately dependency-free (threads + a dict). Swap for Celery/RQ + Redis
when scaling beyond one process.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    song_id: Optional[str] = None
    result: Optional[dict] = None  # serialized Variant on success
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# A job function receives a progress reporter: report(fraction, message).
Reporter = Callable[[float, str], None]
JobFn = Callable[[Reporter], dict]


class JobManager:
    def __init__(self, max_workers: int = 1) -> None:
        # One worker: generation is GPU-bound and serial. Raise only if the
        # backend can truly run concurrently.
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gen")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def submit(self, kind: str, fn: JobFn, *, song_id: str | None = None) -> Job:
        job = Job(id=f"job_{uuid.uuid4().hex[:12]}", kind=kind, song_id=song_id)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: JobFn) -> None:
        def report(frac: float, message: str = "") -> None:
            with self._lock:
                job.progress = max(0.0, min(1.0, frac))
                if message:
                    job.message = message
                job.updated_at = time.time()

        with self._lock:
            job.status = "running"
            job.updated_at = time.time()
        try:
            result = fn(report)
            with self._lock:
                job.result = result
                job.status = "done"
                job.progress = 1.0
                job.message = "Complete"
                job.updated_at = time.time()
        except Exception as exc:  # surface the failure to the UI, don't crash
            with self._lock:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "Failed"
                job.updated_at = time.time()
            traceback.print_exc()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
