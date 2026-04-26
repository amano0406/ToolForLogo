from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import hf_cache_root, torch_cache_root
from .settings import load_settings, load_worker_capabilities, load_huggingface_token


@dataclass(frozen=True, slots=True)
class ModelPreset:
    preset_id: str
    display_name: str
    repo_id: str
    family: str
    purpose: str
    recommended_compute: str
    recommended_quality: str
    base_repo_id: str | None = None
    adapter_repo_id: str | None = None
    adapter_trigger: str | None = None
    adapter_scale: float = 1.0
    prefer_float16: bool = False
    width: int = 0
    height: int = 0
    steps: int = 0
    guidance_scale: float = 0.0
    variant: str | None = None
    max_new_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            'preset_id': self.preset_id,
            'display_name': self.display_name,
            'repo_id': self.repo_id,
            'family': self.family,
            'purpose': self.purpose,
            'recommended_compute': self.recommended_compute,
            'recommended_quality': self.recommended_quality,
            'base_repo_id': self.base_repo_id,
            'adapter_repo_id': self.adapter_repo_id,
            'adapter_trigger': self.adapter_trigger,
            'adapter_scale': self.adapter_scale,
            'prefer_float16': self.prefer_float16,
            'width': self.width,
            'height': self.height,
            'steps': self.steps,
            'guidance_scale': self.guidance_scale,
            'variant': self.variant,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
        }


MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        preset_id='concept-standard',
        display_name='Qwen 0.5B Concept',
        repo_id='Qwen/Qwen2.5-0.5B-Instruct',
        family='concept',
        purpose='Local brand direction and logo spec generation',
        recommended_compute='cpu',
        recommended_quality='standard',
        prefer_float16=False,
        max_new_tokens=220,
        temperature=0.85,
        top_p=0.92,
    ),
    ModelPreset(
        preset_id='gpu-standard',
        display_name='SDXL Turbo Logo Fast',
        repo_id='stabilityai/sdxl-turbo',
        family='image',
        purpose='Primary GPU batch exploration for 20-30 website and company logo directions',
        recommended_compute='gpu',
        recommended_quality='standard',
        width=768,
        height=768,
        steps=6,
        guidance_scale=0.0,
        prefer_float16=True,
        variant='fp16',
    ),
    ModelPreset(
        preset_id='gpu-high',
        display_name='SDXL Base Logo Refine',
        repo_id='stabilityai/stable-diffusion-xl-base-1.0',
        family='image',
        purpose='Higher quality GPU exploration for refined shortlist variants',
        recommended_compute='gpu',
        recommended_quality='high',
        width=1024,
        height=1024,
        steps=26,
        guidance_scale=6.5,
        prefer_float16=True,
        variant='fp16',
    ),
    ModelPreset(
        preset_id='gpu-logo-redmond',
        display_name='SDXL Logo Redmond',
        repo_id='stabilityai/stable-diffusion-xl-base-1.0',
        base_repo_id='stabilityai/stable-diffusion-xl-base-1.0',
        adapter_repo_id='artificialguybr/LogoRedmond-LogoLoraForSDXL-V2',
        adapter_trigger='LogoRedAF, minimalist',
        adapter_scale=0.8,
        family='image',
        purpose='Logo-specific SDXL LoRA for stronger company and website mark exploration',
        recommended_compute='gpu',
        recommended_quality='high',
        width=896,
        height=896,
        steps=22,
        guidance_scale=5.5,
        prefer_float16=True,
        variant='fp16',
    ),
)


DEFAULT_PRESET_BY_FAMILY = {
    'concept': 'concept-standard',
    'image': 'gpu-standard',
}


def list_model_presets(*, family: str | None = None) -> list[ModelPreset]:
    if family is None:
        return list(MODEL_PRESETS)
    return [preset for preset in MODEL_PRESETS if preset.family == family]


def get_model_preset(preset_id: str) -> ModelPreset:
    for preset in MODEL_PRESETS:
        if preset.preset_id == preset_id:
            return preset
    raise KeyError(f'Unknown model preset: {preset_id}')


def safe_repo_dir_name(repo_id: str) -> str:
    return repo_id.replace('/', '--')


def model_cache_dir(repo_id: str) -> Path:
    return hf_cache_root() / 'models' / safe_repo_dir_name(repo_id)


def model_is_downloaded(repo_id: str) -> bool:
    root = model_cache_dir(repo_id)
    return root.exists() and any(root.rglob('*.json'))


def preset_repo_ids(preset: ModelPreset) -> list[str]:
    repo_ids = [preset.base_repo_id or preset.repo_id]
    if preset.adapter_repo_id:
        repo_ids.append(preset.adapter_repo_id)
    return repo_ids


def preset_is_downloaded(preset: ModelPreset) -> bool:
    return all(model_is_downloaded(repo_id) for repo_id in preset_repo_ids(preset))


def preset_size_bytes(preset: ModelPreset) -> int:
    return sum(directory_size(model_cache_dir(repo_id)) for repo_id in preset_repo_ids(preset))


def snapshot_allow_patterns(preset: ModelPreset, repo_id: str) -> list[str] | None:
    if preset.family == 'concept':
        return None
    if repo_id == preset.adapter_repo_id:
        return ['*.json', '*.txt', '*.md', '*.safetensors']
    variant = str(preset.variant or '').strip()
    patterns = [
        'model_index.json',
        '**/config.json',
        'scheduler/*',
        'tokenizer/*',
        'tokenizer_2/*',
        'feature_extractor/*',
    ]
    if variant == 'fp16':
        patterns.extend(
            [
                '**/model.fp16.safetensors',
                '**/diffusion_pytorch_model.fp16.safetensors',
            ]
        )
    else:
        patterns.extend(
            [
                '**/model.safetensors',
                '**/diffusion_pytorch_model.safetensors',
                '**/*.bin',
            ]
        )
    return patterns


def directory_size(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for file in root.rglob('*'):
        if file.is_file():
            try:
                total += file.stat().st_size
            except OSError:
                continue
    return total


def cache_snapshot() -> dict[str, Any]:
    roots = [hf_cache_root(), torch_cache_root()]
    total = sum(directory_size(root) for root in roots)
    model_entries = []
    for preset in MODEL_PRESETS:
        model_entries.append(
            {
                **preset.to_dict(),
                'downloaded': preset_is_downloaded(preset),
                'cacheDir': str(model_cache_dir(preset.base_repo_id or preset.repo_id)),
                'sizeBytes': preset_size_bytes(preset),
            }
        )
    return {
        'totalBytes': total,
        'modelEntries': model_entries,
        'huggingFaceCacheRoot': str(hf_cache_root()),
        'torchCacheRoot': str(torch_cache_root()),
    }


def _active_image_preset(snapshot: dict[str, Any], worker: dict[str, Any]) -> str:
    quality = str(snapshot.get('processingQuality') or 'standard')
    if quality == 'high':
        return str(snapshot.get('preferredHighGpuPreset') or 'gpu-high')
    return str(snapshot.get('preferredGpuPreset') or 'gpu-standard')


def active_preset_id(
    settings: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
    *,
    family: str = 'image',
) -> str:
    snapshot = settings or load_settings()
    worker = capabilities or load_worker_capabilities()
    if family == 'concept':
        return str(snapshot.get('preferredConceptPreset') or DEFAULT_PRESET_BY_FAMILY['concept'])
    return _active_image_preset(snapshot, worker)


def resolve_generation_profile(
    settings: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
    *,
    preset_id: str | None = None,
    family: str = 'image',
) -> dict[str, Any]:
    snapshot = settings or load_settings()
    worker = capabilities or load_worker_capabilities()
    selected_preset = get_model_preset(preset_id or active_preset_id(snapshot, worker, family=family))
    if selected_preset.family != family:
        selected_preset = get_model_preset(active_preset_id(snapshot, worker, family=family))
    if family == 'image' and not bool(worker.get('gpuAvailable')):
        raise RuntimeError('Image exploration requires a GPU-enabled worker. Open Settings and confirm the GPU worker is available.')
    return {
        **selected_preset.to_dict(),
        'device': 'cuda' if family == 'image' else ('cuda' if snapshot.get('computeMode') == 'gpu' and worker.get('gpuAvailable') else 'cpu'),
        'allowAutoDownload': bool(snapshot.get('allowAutoModelDownload', True)),
    }


def build_model_statuses() -> list[dict[str, Any]]:
    settings = load_settings()
    capabilities = load_worker_capabilities()
    selected_image = active_preset_id(settings, capabilities, family='image')
    selected_concept = active_preset_id(settings, capabilities, family='concept')
    statuses: list[dict[str, Any]] = []
    for preset in MODEL_PRESETS:
        is_active = preset.preset_id == (selected_concept if preset.family == 'concept' else selected_image)
        statuses.append(
            {
                **preset.to_dict(),
                'downloaded': preset_is_downloaded(preset),
                'cacheDir': str(model_cache_dir(preset.base_repo_id or preset.repo_id)),
                'sizeBytes': preset_size_bytes(preset),
                'active': is_active,
            }
        )
    return statuses


def download_model(preset_id: str) -> dict[str, Any]:
    preset = get_model_preset(preset_id)
    token = load_huggingface_token()
    from huggingface_hub import snapshot_download

    downloaded_paths: list[str] = []
    for repo_id in preset_repo_ids(preset):
        target = model_cache_dir(repo_id)
        target.mkdir(parents=True, exist_ok=True)
        allow_patterns = snapshot_allow_patterns(preset, repo_id)
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
        downloaded_paths.append(str(downloaded_path))
    return {
        'presetId': preset.preset_id,
        'repoId': preset.repo_id,
        'family': preset.family,
        'path': downloaded_paths[0],
        'downloadedPaths': downloaded_paths,
        'sizeBytes': preset_size_bytes(preset),
    }


def delete_model(preset_id: str) -> dict[str, Any]:
    import shutil

    preset = get_model_preset(preset_id)
    removed = False
    for repo_id in preset_repo_ids(preset):
        target = model_cache_dir(repo_id)
        if target.exists():
            shutil.rmtree(target)
            removed = True
    return {
        'presetId': preset.preset_id,
        'repoId': preset.repo_id,
        'family': preset.family,
        'removed': removed,
        'path': str(target),
    }


def clear_model_cache() -> dict[str, Any]:
    import shutil

    cleared = 0
    for root in (hf_cache_root(), torch_cache_root()):
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            continue
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            cleared += 1
    return {
        'clearedEntries': cleared,
        'cache': cache_snapshot(),
    }
