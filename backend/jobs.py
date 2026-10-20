"""In-process job store so optimisation does not block the API thread.

One worker thread: modules use matplotlib (not thread-safe), and runs take seconds,
so jobs queue and execute one at a time.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from backend.paths import REPO_ROOT

@dataclass
class Job:
    id: str
    module_id: str
    status: str = "queued"
    message: str = "Queued."
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        elapsed = None
        if self.started_at is not None:
            elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "id": self.id,
            "module_id": self.module_id,
            "status": self.status,
            "message": self.message,
            "elapsed_seconds": elapsed,
            "result": self.result,
            "error": self.error,
        }


_LOCK = threading.Lock()
_JOBS: dict[str, Job] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optivision")


def new_job_id() -> str:
    return uuid.uuid4().hex


def create_job(job_id: str, module_id: str) -> Job:
    job = Job(id=job_id, module_id=module_id, created_at=time.time())
    with _LOCK:
        _JOBS[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def _set(job_id: str, **kwargs: Any) -> None:
    with _LOCK:
        job = _JOBS[job_id]
        for key, value in kwargs.items():
            setattr(job, key, value)


def submit(job: Job, fn: Callable[[], dict[str, Any]]) -> None:
    def _run() -> None:
        _set(job.id, status="running", message="Optimisation in progress.", started_at=time.time())
        try:
            result = fn()
            _set(job.id, status="succeeded", message="Done.", result=result, error=None,
                 finished_at=time.time())
        except Exception as exc:  # noqa: BLE001 — surface module failures to the UI
            message = f"{type(exc).__name__}: {exc}".replace(str(REPO_ROOT) + os.sep, "")
            _set(
                job.id,
                status="failed",
                message=message,
                error=traceback.format_exc(),
                finished_at=time.time(),
            )

    _EXECUTOR.submit(_run)
