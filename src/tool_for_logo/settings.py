from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import appdata_root, load_runtime_defaults


def settings_path() -> Path:
    return appdata_root() / 'settings.json'


def token_path() -> Path:
    return appdata_root() / 'secrets' / 'huggingface.token'


def worker_capabilities_path() -> Path:
    return appdata_root() / 'worker-capabilities.json'


def _default_settings() -> dict[str, Any]:
    defaults = load_runtime_defaults()
    return {
        'schemaVersion': 2,
        'uiLanguage': str(defaults.get('uiLanguage') or 'ja'),
        'computeMode': str(defaults.get('computeMode') or 'gpu'),
        'processingQuality': str(defaults.get('processingQuality') or 'standard'),
        'defaultBatchCount': int(defaults.get('defaultBatchCount') or 10),
        'defaultDirectionHint': str(defaults.get('defaultDirectionHint') or ''),
        'defaultExplorationPreset': str(defaults.get('defaultExplorationPreset') or 'balanced-saas'),
        'huggingfaceTermsConfirmed': bool(defaults.get('huggingfaceTermsConfirmed', False)),
        'allowAutoModelDownload': bool(defaults.get('allowAutoModelDownload', True)),
        'preferredConceptPreset': str(defaults.get('preferredConceptPreset') or 'concept-standard'),
        'preferredCpuPreset': str(defaults.get('preferredCpuPreset') or 'cpu-standard'),
        'preferredGpuPreset': str(defaults.get('preferredGpuPreset') or 'gpu-standard'),
        'preferredHighGpuPreset': str(defaults.get('preferredHighGpuPreset') or 'gpu-high'),
    }


def _normalize_settings(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = _default_settings()
    normalized = {**defaults, **payload}
    normalized['computeMode'] = 'gpu'
    normalized['processingQuality'] = str(normalized.get('processingQuality') or 'standard').strip().lower()
    if normalized['processingQuality'] not in {'standard', 'high'}:
        normalized['processingQuality'] = 'standard'
    normalized['defaultBatchCount'] = max(1, min(40, int(normalized.get('defaultBatchCount') or 10)))
    normalized['defaultExplorationPreset'] = str(normalized.get('defaultExplorationPreset') or 'balanced-saas').strip() or 'balanced-saas'
    normalized['allowAutoModelDownload'] = bool(normalized.get('allowAutoModelDownload', True))
    normalized['huggingfaceTermsConfirmed'] = bool(normalized.get('huggingfaceTermsConfirmed', False))
    normalized['uiLanguage'] = str(normalized.get('uiLanguage') or 'ja').strip() or 'ja'
    for key, fallback in (
        ('preferredConceptPreset', 'concept-standard'),
        ('preferredCpuPreset', 'cpu-standard'),
        ('preferredGpuPreset', 'gpu-standard'),
        ('preferredHighGpuPreset', 'gpu-high'),
    ):
        normalized[key] = str(normalized.get(key) or fallback).strip() or fallback
    return normalized


def load_settings() -> dict[str, Any]:
    if settings_path().exists():
        payload = json.loads(settings_path().read_text(encoding='utf-8'))
    else:
        payload = _default_settings()
    return _normalize_settings(payload)


def save_settings(payload: dict[str, Any]) -> dict[str, Any]:
    final = _normalize_settings({**load_settings(), **payload})
    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return final


def load_huggingface_token() -> str | None:
    path = token_path()
    if not path.exists():
        return None
    value = path.read_text(encoding='utf-8', errors='replace').strip()
    return value or None


def save_huggingface_token(token: str | None) -> None:
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if token and token.strip():
        path.write_text(token.strip(), encoding='utf-8')
        return
    if path.exists():
        path.unlink()


def load_worker_capabilities() -> dict[str, Any]:
    path = worker_capabilities_path()
    if not path.exists():
        return {
            'generatedAt': None,
            'workerFlavor': 'gpu',
            'torchInstalled': False,
            'torchCudaBuilt': False,
            'gpuAvailable': False,
            'deviceCount': 0,
            'deviceNames': [],
            'deviceMemoryGiB': [],
            'maxGpuMemoryGiB': 0.0,
            'message': 'Worker capability report has not been generated yet.',
        }
    return json.loads(path.read_text(encoding='utf-8'))


def save_worker_capabilities(payload: dict[str, Any]) -> None:
    path = worker_capabilities_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def settings_snapshot() -> dict[str, Any]:
    settings = load_settings()
    return {**settings, 'hasHuggingFaceToken': bool(load_huggingface_token()), 'workerCapabilities': load_worker_capabilities()}
