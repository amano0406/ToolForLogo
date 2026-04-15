from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def load_local_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def environment_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def default_backend() -> str:
    backend = environment_value("TOOL_FOR_LOGO_DEFAULT_BACKEND", "comfyui").lower()
    if backend in {"mock", "comfyui", "openai"}:
        return backend
    return "comfyui"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_api_failure(logs_dir: Path, stage: str, request_payload: dict[str, Any], error: Exception) -> Path:
    payload = {
        "stage": stage,
        "request": request_payload,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
    }
    safe_stage = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage)
    target = logs_dir / f"{safe_stage}_error.json"
    save_json(target, payload)
    return target
