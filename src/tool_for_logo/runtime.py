from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SUPPORTED_BACKENDS = {'mock', 'diffusers', 'local-svg'}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_local_env(root: Path | None = None) -> None:
    env_root = root or project_root()
    env_path = env_root / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, value = stripped.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def environment_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _resolved_path(host_env: str, runtime_env: str, windows_default: str, non_windows_default: str) -> Path:
    if os.name == 'nt':
        return Path(environment_value(host_env, windows_default))
    return Path(environment_value(runtime_env, non_windows_default))


def runtime_defaults_path() -> Path:
    return Path(environment_value('TOOL_FOR_LOGO_RUNTIME_DEFAULTS', '/app/config/runtime.defaults.json'))


def appdata_root() -> Path:
    return _resolved_path(
        'TOOL_FOR_LOGO_HOST_APPDATA_ROOT',
        'TOOL_FOR_LOGO_APPDATA_ROOT',
        r'C:\Codex\workspaces\ToolForLogo\app-data',
        '/mnt/c/Codex/workspaces/ToolForLogo/app-data',
    )


def uploads_root() -> Path:
    return _resolved_path(
        'TOOL_FOR_LOGO_HOST_UPLOADS_ROOT',
        'TOOL_FOR_LOGO_UPLOADS_ROOT',
        r'C:\Codex\workspaces\ToolForLogo\uploads',
        '/mnt/c/Codex/workspaces/ToolForLogo/uploads',
    )


def outputs_root() -> Path:
    return _resolved_path(
        'TOOL_FOR_LOGO_HOST_OUTPUTS_ROOT',
        'TOOL_FOR_LOGO_OUTPUTS_ROOT',
        r'C:\Codex\workspaces\ToolForLogo\outputs',
        '/mnt/c/Codex/workspaces/ToolForLogo/outputs',
    )


def report_root() -> Path:
    return _resolved_path(
        'TOOL_FOR_LOGO_HOST_REPORT_ROOT',
        'TOOL_FOR_LOGO_REPORT_ROOT',
        r'C:\Codex\reports\ToolForLogo',
        '/mnt/c/Codex/reports/ToolForLogo',
    )


def archive_root() -> Path:
    return _resolved_path(
        'TOOL_FOR_LOGO_HOST_ARCHIVE_ROOT',
        'TOOL_FOR_LOGO_ARCHIVE_ROOT',
        r'C:\Codex\archive\ToolForLogo',
        '/mnt/c/Codex/archive/ToolForLogo',
    )


def hf_cache_root() -> Path:
    return _resolved_path(
        'TOOL_FOR_LOGO_HOST_HF_CACHE_ROOT',
        'TOOL_FOR_LOGO_HF_CACHE_ROOT',
        r'C:\Codex\workspaces\ToolForLogo\cache\huggingface',
        '/mnt/c/Codex/workspaces/ToolForLogo/cache/huggingface',
    )


def torch_cache_root() -> Path:
    return _resolved_path(
        'TOOL_FOR_LOGO_HOST_TORCH_CACHE_ROOT',
        'TOOL_FOR_LOGO_TORCH_CACHE_ROOT',
        r'C:\Codex\workspaces\ToolForLogo\cache\torch',
        '/mnt/c/Codex/workspaces/ToolForLogo/cache/torch',
    )


def default_backend() -> str:
    backend = environment_value('TOOL_FOR_LOGO_DEFAULT_BACKEND', 'diffusers').lower()
    if backend in SUPPORTED_BACKENDS:
        return backend
    return 'diffusers'


def load_runtime_defaults() -> dict[str, Any]:
    path = runtime_defaults_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_api_failure(logs_dir: Path, stage: str, request_payload: dict[str, Any], error: Exception) -> Path:
    payload = {
        'stage': stage,
        'request': request_payload,
        'error_type': error.__class__.__name__,
        'error_message': str(error),
    }
    safe_stage = re.sub(r'[^a-zA-Z0-9_.-]+', '_', stage)
    target = logs_dir / f'{safe_stage}_error.json'
    save_json(target, payload)
    return target
