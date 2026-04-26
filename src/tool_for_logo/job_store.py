from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import JobRequest, JobState, JobStatus
from .runtime import outputs_root, save_json
from .state import utc_now


def jobs_root() -> Path:
    return outputs_root() / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def request_path(job_id: str) -> Path:
    return job_dir(job_id) / "request.json"


def status_path(job_id: str) -> Path:
    return job_dir(job_id) / "status.json"


def result_path(job_id: str) -> Path:
    return job_dir(job_id) / "result.json"


def log_path(job_id: str) -> Path:
    return job_dir(job_id) / "worker.log"


def _new_job_id(job_type: str) -> str:
    suffix = job_type.replace("_", "-")
    stamp = utc_now().replace(":", "").replace("+00:00", "Z")
    return f"job-{suffix}-{stamp}-{uuid4().hex[:6]}"


def create_job(*, job_type: str, payload: dict[str, Any], case_id: str | None = None) -> tuple[str, Path]:
    job_id = _new_job_id(job_type)
    run_dir = job_dir(job_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    request = JobRequest(job_id=job_id, job_type=job_type, case_id=case_id, created_at=utc_now(), payload=payload)
    status = JobStatus(
        job_id=job_id,
        job_type=job_type,
        case_id=case_id,
        state=JobState.PENDING,
        current_stage="queued",
        message="Queued and waiting for worker.",
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    save_json(request_path(job_id), request.to_dict())
    save_json(status_path(job_id), status.to_dict())
    save_json(result_path(job_id), {})
    log_event(job_id, f"Queued {job_type} job.")
    return job_id, run_dir


def log_event(job_id: str, message: str) -> None:
    target = log_path(job_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def load_request(job_id: str) -> dict[str, Any]:
    return json.loads(request_path(job_id).read_text(encoding="utf-8"))


def load_status(job_id: str) -> dict[str, Any]:
    return json.loads(status_path(job_id).read_text(encoding="utf-8"))


def load_result(job_id: str) -> dict[str, Any]:
    return json.loads(result_path(job_id).read_text(encoding="utf-8"))


def save_status(job_id: str, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = utc_now()
    save_json(status_path(job_id), payload)


def save_result(job_id: str, payload: dict[str, Any]) -> None:
    save_json(result_path(job_id), payload)


def update_status(
    job_id: str,
    *,
    state: str | None = None,
    current_stage: str | None = None,
    message: str | None = None,
    progress_percent: float | None = None,
    items_done: int | None = None,
    items_total: int | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    status = load_status(job_id)
    if state is not None:
        status["state"] = state
    if current_stage is not None:
        status["current_stage"] = current_stage
    if message is not None:
        status["message"] = message
    if progress_percent is not None:
        status["progress_percent"] = round(float(progress_percent), 1)
    if items_done is not None:
        status["items_done"] = int(items_done)
    if items_total is not None:
        status["items_total"] = int(items_total)
    if error_message is not None:
        status["error_message"] = error_message
    save_status(job_id, status)
    return status


def list_jobs(case_id: str | None = None) -> list[dict[str, Any]]:
    root = jobs_root()
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for request_file in sorted(root.glob("*/request.json"), reverse=True):
        payload = json.loads(request_file.read_text(encoding="utf-8"))
        if case_id and payload.get("case_id") != case_id:
            continue
        job_id = str(payload["job_id"])
        rows.append({"job_id": job_id, "request": payload, "status": load_status(job_id), "result": load_result(job_id), "log_path": str(log_path(job_id))})
    return rows


def list_active_jobs(case_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for job in list_jobs(case_id):
        state = str(job["status"].get("state") or "")
        if state in {JobState.PENDING.value, JobState.RUNNING.value}:
            rows.append(job)
    return rows


def next_pending_job_id() -> str | None:
    pending = [job for job in list_jobs() if str(job["status"].get("state") or "") == JobState.PENDING.value]
    if not pending:
        return None
    pending.sort(key=lambda row: str(row["request"].get("created_at") or ""))
    return str(pending[0]["job_id"])
