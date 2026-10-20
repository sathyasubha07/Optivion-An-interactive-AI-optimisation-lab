from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE1_ROOT = REPO_ROOT / "Module1_Network_Intrusion_Detection"
MODULE4_ROOT = REPO_ROOT / "Module4_Satellite_SuperResolution"
RUNTIME_ROOT = REPO_ROOT / "backend" / "runtime"
JOB_ROOT = RUNTIME_ROOT / "jobs"


def job_input_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id / "input"


def job_output_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id / "output"
