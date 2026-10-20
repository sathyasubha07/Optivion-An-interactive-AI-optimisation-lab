"""OptiVision FastAPI app — registry, uploads, jobs, static outputs.

Module-agnostic: everything module-specific lives in backend/adapters/ and backend/registry.py.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend.adapters.base import RunContext  # noqa: E402
from backend.jobs import create_job, get_job, new_job_id, submit  # noqa: E402
from backend.jsonutil import download, path_to_url  # noqa: E402
from backend.paths import JOB_ROOT, RUNTIME_ROOT, job_input_dir, job_output_dir  # noqa: E402
from backend.registry import TEAM, all_modules, get_adapter, get_module, public_module  # noqa: E402

CHUNK = 1024 * 1024

app = FastAPI(title="OptiVision", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(RUNTIME_ROOT)), name="outputs")


def _safe_name(filename: str) -> str:
    name = Path(filename).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "upload"
    return stem[:80] + Path(name).suffix.lower()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "optivision"}


@app.get("/contributors")
def contributors() -> dict:
    table = [
        {
            "number": m["number"],
            "module": m["name"],
            "method": m["method"],
        }
        for m in sorted(all_modules(), key=lambda m: m["number"])
    ]
    return {**TEAM, "table": table}


@app.get("/modules")
def list_modules() -> dict:
    return {"modules": [public_module(m) for m in all_modules()]}


@app.get("/modules/{module_id}")
def module_detail(module_id: str) -> dict:
    entry = get_module(module_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown module '{module_id}'.")
    return public_module(entry)


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job (the server may have restarted).")
    return job.as_dict()


async def _save_upload(file: UploadFile, dest: Path, max_bytes: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with dest.open("wb") as handle:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum is {max_bytes // (1024 * 1024)} MB.",
                    )
                handle.write(chunk)
    except HTTPException:
        shutil.rmtree(dest.parent.parent, ignore_errors=True)
        raise
    if size == 0:
        shutil.rmtree(dest.parent.parent, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return dest


@app.post("/modules/{module_id}/run")
async def run_module(
    module_id: str,
    file: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
) -> dict:
    entry = get_module(module_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown module '{module_id}'.")
    adapter = get_adapter(entry)
    if entry["status"] != "ready" or adapter is None:
        raise HTTPException(status_code=400, detail=f"{entry['name']} is coming soon and cannot run yet.")

    has_file = file is not None and bool(file.filename)
    if has_file == bool(sample_id):
        raise HTTPException(status_code=400, detail="Provide exactly one of: an uploaded file, or a sample_id.")

    if has_file:
        accepted = [t.lower() for t in entry["accepted_file_types"]]
        suffix = Path(file.filename).suffix.lower()
        if suffix not in accepted:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix or '(none)'}'. Accepted: {', '.join(accepted)}.",
            )
    else:
        known = {s.id for s in adapter.list_samples()}
        if sample_id not in known:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown sample '{sample_id}'. Available: {', '.join(sorted(known))}.",
            )

    job_id = new_job_id()
    in_dir = job_input_dir(job_id)
    out_dir = job_output_dir(job_id)
    if has_file:
        input_path = await _save_upload(file, in_dir / _safe_name(file.filename),
                                        int(entry["max_upload_mb"]) * 1024 * 1024)
    else:
        try:
            input_path = adapter.make_sample(sample_id, in_dir)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(in_dir.parent, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Could not prepare sample: {exc}") from exc
    out_dir.mkdir(parents=True, exist_ok=True)

    job = create_job(job_id, module_id)
    ctx = RunContext(input_path=input_path, job_id=job_id, out_dir=out_dir, sample_id=sample_id)
    sample_label = next((s.label for s in adapter.list_samples() if s.id == sample_id), None)

    def _work() -> dict:
        result = adapter.run(ctx)
        result["module_id"] = module_id
        result["job_id"] = job_id
        result["input"] = {
            "filename": input_path.name,
            "url": path_to_url(input_path),
            "is_sample": sample_id is not None,
            "sample_id": sample_id,
            "sample_label": sample_label,
        }
        results_json = out_dir / "results.json"
        results_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        extra = [d for d in (download("All results (JSON)", results_json),
                             download("Input file", input_path)) if d]
        result["downloads"] = [*result.get("downloads", []), *extra]
        return result

    submit(job, _work)
    return {"job_id": job.id, "status": job.status}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
