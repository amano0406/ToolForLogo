from __future__ import annotations

import os
import time
from typing import Any

from .generator import create_export_bundle, generate_batch
from .job_store import (
    list_jobs,
    load_request,
    next_pending_job_id,
    save_result,
    update_status,
    log_event,
)
from .model_catalog import clear_model_cache, delete_model, download_model
from .runtime import default_backend
from .settings import save_worker_capabilities, settings_snapshot
from .state import ToolForLogoStore, utc_now


def write_worker_capabilities() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generatedAt": utc_now(),
        "workerFlavor": os.getenv("TOOL_FOR_LOGO_WORKER_FLAVOR", "cpu"),
        "torchInstalled": False,
        "torchCudaBuilt": False,
        "gpuAvailable": False,
        "deviceCount": 0,
        "deviceNames": [],
        "deviceMemoryGiB": [],
        "maxGpuMemoryGiB": 0.0,
        "message": "Worker capability report created.",
    }
    try:
        import torch

        payload["torchInstalled"] = True
        payload["torchCudaBuilt"] = bool(torch.backends.cuda.is_built())
        payload["gpuAvailable"] = bool(torch.cuda.is_available())
        payload["deviceCount"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        payload["deviceNames"] = (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else []
        )
        payload["deviceMemoryGiB"] = (
            [
                round(torch.cuda.get_device_properties(index).total_memory / 1024 / 1024 / 1024, 1)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        )
        payload["maxGpuMemoryGiB"] = max(payload["deviceMemoryGiB"], default=0.0)
        payload["message"] = "GPU is available to the worker." if payload["gpuAvailable"] else "GPU is not available to the worker."
    except Exception as exc:
        payload["message"] = f"Capability check failed: {exc}"

    save_worker_capabilities(payload)
    return payload


def _process_generate_batch(job_id: str, request: dict[str, Any], store: ToolForLogoStore) -> dict[str, Any]:
    payload = dict(request.get("payload") or {})
    case_id = str(request.get("case_id") or payload.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("generate_batch job is missing case_id")

    count = int(payload.get("count") or 10)
    backend = str(payload.get("backend") or default_backend())
    direction_hint = str(payload.get("direction_hint") or "")
    seed = payload.get("seed")
    source_candidate_id = str(payload.get("source_candidate_id") or "").strip() or None
    generation_options = payload.get("generation_options") if isinstance(payload.get("generation_options"), dict) else None

    def on_progress(done: int, total: int, stage: str, message: str) -> None:
        percent = 100.0 if total <= 0 else max(0.0, min(100.0, (done / total) * 100.0))
        update_status(
            job_id,
            state="running",
            current_stage=stage,
            message=message,
            items_done=done,
            items_total=total,
            progress_percent=percent,
        )

    log_event(job_id, f"Starting batch generation for case {case_id} using backend {backend}.")
    result = generate_batch(
        store,
        case_id=case_id,
        count=count,
        direction_hint=direction_hint,
        seed=int(seed) if seed is not None else None,
        backend=backend,
        source_candidate_id=source_candidate_id,
        generation_options=generation_options,
        progress_callback=on_progress,
    )
    update_status(
        job_id,
        state="completed",
        current_stage="completed",
        message="Batch generation completed.",
        items_done=count,
        items_total=count,
        progress_percent=100.0,
    )
    log_event(job_id, f"Completed batch generation for case {case_id}.")
    return result


def _process_model_download(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    payload = dict(request.get("payload") or {})
    preset_id = str(payload.get("preset_id") or "").strip()
    if not preset_id:
        raise ValueError("model download job is missing preset_id")
    update_status(job_id, state="running", current_stage="downloading", message=f"Downloading {preset_id}.")
    log_event(job_id, f"Downloading model preset {preset_id}.")
    result = download_model(preset_id)
    update_status(job_id, state="completed", current_stage="completed", message=f"Downloaded {preset_id}.", progress_percent=100.0)
    log_event(job_id, f"Finished downloading model preset {preset_id}.")
    return result


def _process_model_delete(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    payload = dict(request.get("payload") or {})
    preset_id = str(payload.get("preset_id") or "").strip()
    if not preset_id:
        raise ValueError("model delete job is missing preset_id")
    update_status(job_id, state="running", current_stage="deleting", message=f"Deleting {preset_id} cache.")
    log_event(job_id, f"Deleting model preset {preset_id}.")
    result = delete_model(preset_id)
    update_status(job_id, state="completed", current_stage="completed", message=f"Deleted {preset_id} cache.", progress_percent=100.0)
    return result


def _process_cache_clear(job_id: str) -> dict[str, Any]:
    update_status(job_id, state="running", current_stage="clearing_cache", message="Clearing cached models.")
    log_event(job_id, "Clearing model cache.")
    result = clear_model_cache()
    update_status(job_id, state="completed", current_stage="completed", message="Cleared cached models.", progress_percent=100.0)
    return result


def process_job(job_id: str, store: ToolForLogoStore | None = None) -> dict[str, Any]:
    logo_store = store or ToolForLogoStore.from_env()
    request = load_request(job_id)
    job_type = str(request.get("job_type") or "")
    try:
        update_status(job_id, state="running", current_stage="starting", message="Worker picked up the job.")
        if job_type == "generate_batch":
            result = _process_generate_batch(job_id, request, logo_store)
        elif job_type == "download_model":
            result = _process_model_download(job_id, request)
        elif job_type == "delete_model":
            result = _process_model_delete(job_id, request)
        elif job_type == "clear_model_cache":
            result = _process_cache_clear(job_id)
        else:
            raise ValueError(f"Unsupported job type: {job_type}")
        save_result(job_id, result)
        return result
    except Exception as error:
        update_status(
            job_id,
            state="failed",
            current_stage="failed",
            message="Worker job failed.",
            error_message=str(error),
        )
        log_event(job_id, f"Job failed: {error}")
        raise


def run_daemon(*, poll_interval: int = 5) -> int:
    write_worker_capabilities()
    store = ToolForLogoStore.from_env()
    while True:
        job_id = next_pending_job_id()
        if not job_id:
            time.sleep(max(1, poll_interval))
            continue
        process_job(job_id, store)
    return 0


def worker_status_payload() -> dict[str, Any]:
    active_jobs = [job["status"] for job in list_jobs() if job["status"].get("state") in {"pending", "running"}]
    return {
        "settings": settings_snapshot(),
        "activeJobs": active_jobs,
    }
