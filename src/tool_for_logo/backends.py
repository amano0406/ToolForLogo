from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Any

from PIL import Image

from .model_catalog import download_model, model_cache_dir
from .runtime import save_api_failure


@dataclass(frozen=True, slots=True)
class ExplorationPreset:
    preset_id: str
    label: str
    prompt: str
    audience: str

    def to_dict(self) -> dict[str, str]:
        return {
            "preset_id": self.preset_id,
            "label": self.label,
            "prompt": self.prompt,
            "audience": self.audience,
        }


EXPLORATION_PRESETS: tuple[ExplorationPreset, ...] = (
    ExplorationPreset(
        preset_id="balanced-saas",
        label="Balanced SaaS",
        prompt="credible SaaS, clean homepage brand, broad appeal",
        audience="General website and product logos",
    ),
    ExplorationPreset(
        preset_id="b2b-infra",
        label="B2B Infra",
        prompt="enterprise B2B software, trustworthy, modular",
        audience="Developer tools, infrastructure, admin products",
    ),
    ExplorationPreset(
        preset_id="friendly-tool",
        label="Friendly Tool",
        prompt="approachable utility product, warm, simple",
        audience="Small teams, creator tools, approachable SaaS",
    ),
    ExplorationPreset(
        preset_id="premium-brand",
        label="Premium Brand",
        prompt="premium software brand, restrained geometry, quiet confidence",
        audience="Studios, premium SaaS, brand-forward products",
    ),
    ExplorationPreset(
        preset_id="ai-data",
        label="AI Data",
        prompt="AI data product, abstract intelligence, system thinking",
        audience="AI tools, analytics, knowledge products",
    ),
)

EXPLORATION_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "label": "Negative Space",
        "prompt": "negative space, two shapes max",
    },
    {
        "label": "Monoline",
        "prompt": "monoline, even stroke, compact silhouette",
    },
    {
        "label": "Cut Corner",
        "prompt": "cut-corner geometric, crisp angles",
    },
    {
        "label": "Ribbon Fold",
        "prompt": "flat folded ribbon, minimal planes",
    },
    {
        "label": "Orbital",
        "prompt": "calm orbital symbol",
    },
    {
        "label": "Block Link",
        "prompt": "linked blocks, favicon readable",
    },
)

COMPOSITION_HINTS: tuple[str, ...] = (
    "single centered symbol",
    "large isolated icon",
    "favicon readable silhouette",
    "flat vector posterization",
    "plain white background",
)


def list_exploration_presets() -> list[dict[str, str]]:
    return [preset.to_dict() for preset in EXPLORATION_PRESETS]


def get_exploration_preset(preset_id: str | None) -> ExplorationPreset:
    normalized = str(preset_id or "").strip().lower()
    for preset in EXPLORATION_PRESETS:
        if preset.preset_id == normalized:
            return preset
    return EXPLORATION_PRESETS[0]


@dataclass(slots=True)
class GeneratedMark:
    image: Image.Image
    request_prompt: str
    revised_prompt: str | None
    exploration_label: str | None = None
    preset_label: str | None = None


class DiffusersLogoBackend:
    _PIPELINES: dict[tuple[str, str, str], Any] = {}

    def __init__(self, *, logs_root: Path, profile: dict[str, Any], token: str | None) -> None:
        self._logs_root = logs_root
        self._profile = profile
        self._token = token

    def _save_failure(self, stage: str, payload: dict[str, Any], error: Exception) -> None:
        save_api_failure(self._logs_root, stage, payload, error)

    @staticmethod
    def _compact_phrase(value: str, limit: int) -> str:
        parts = ["".join(char for char in token if char.isalnum() or char in {"-", "+"}) for token in value.split()]
        compact = [part for part in parts if part]
        return " ".join(compact[:limit])

    def _prompt(
        self,
        *,
        product_name: str,
        description: str,
        direction: str,
        palette: dict[str, str],
        shape_kind: str,
        variant_index: int,
        source_title: str | None,
        exploration_preset_id: str | None,
    ) -> tuple[str, str]:
        family = EXPLORATION_FAMILIES[variant_index % len(EXPLORATION_FAMILIES)]
        composition = COMPOSITION_HINTS[(variant_index // len(EXPLORATION_FAMILIES)) % len(COMPOSITION_HINTS)]
        preset = get_exploration_preset(exploration_preset_id)
        adapter_trigger = str(self._profile.get("adapter_trigger") or "").strip()
        context_hint = self._compact_phrase(description, 7)
        direction_hint = self._compact_phrase(direction, 10)
        motif_hint = self._compact_phrase(shape_kind, 2)
        sibling_hint = self._compact_phrase(source_title or "", 5)
        prompt_parts = [
            "single minimalist software brand symbol",
            "one centered icon",
            "simple geometric silhouette",
            "flat vector",
            "two shapes max",
            "thick lines",
            "white background",
            "no text",
            "no ornament",
            "no icon sheet",
            f"brand for {self._compact_phrase(product_name, 3)}",
            preset.prompt,
            family["prompt"],
            direction_hint,
            context_hint,
            motif_hint,
        ]
        if adapter_trigger:
            prompt_parts = [
                adapter_trigger,
                "single minimalist brand mark",
                "one centered emblem",
                "large isolated symbol",
                "two shapes max",
                "thick lines",
                "white background",
                "no ornament",
                f"brand for {self._compact_phrase(product_name, 3)}",
                family["prompt"],
                direction_hint,
                context_hint,
                motif_hint,
            ]
        if sibling_hint:
            prompt_parts.append(f"derived from {sibling_hint}")
        prompt = ", ".join(prompt_parts)
        return prompt, family["label"]

    @staticmethod
    def ensure_dependencies() -> None:
        try:
            import torch  # noqa: F401
            from diffusers import AutoPipelineForText2Image  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "backend=diffusers requires the worker image with torch and diffusers installed."
            ) from error

    def _resolve_model_dir(self) -> Path:
        repo_id = str(self._profile.get("base_repo_id") or self._profile["repo_id"])
        target = model_cache_dir(repo_id)
        if target.exists() and any(target.rglob("*.json")):
            return target
        if not bool(self._profile.get("allowAutoDownload", True)):
            raise RuntimeError(f"Model '{repo_id}' is not downloaded. Download it first from Settings.")
        download_model(str(self._profile["preset_id"]))
        return target

    def _resolve_adapter_dir(self) -> Path | None:
        adapter_repo_id = str(self._profile.get("adapter_repo_id") or "").strip()
        if not adapter_repo_id:
            return None
        target = model_cache_dir(adapter_repo_id)
        if target.exists() and any(target.rglob("*.json")):
            return target
        if not bool(self._profile.get("allowAutoDownload", True)):
            raise RuntimeError(f"Adapter '{adapter_repo_id}' is not downloaded. Download it first from Settings.")
        download_model(str(self._profile["preset_id"]))
        return target

    def _load_pipeline(self) -> tuple[Any, Any, str]:
        self.ensure_dependencies()
        import torch
        from diffusers import AutoPipelineForText2Image

        model_dir = self._resolve_model_dir()
        adapter_dir = self._resolve_adapter_dir()
        device = "cuda" if self._profile.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("Diffusers image exploration requires a GPU-enabled worker.")
        dtype_name = "float16" if device == "cuda" and self._profile.get("prefer_float16") else "float32"
        cache_key = (str(model_dir), str(adapter_dir or ""), device, dtype_name)
        if cache_key in self._PIPELINES:
            return self._PIPELINES[cache_key], torch, device

        torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        variant = self._profile.get("variant")
        if variant:
            kwargs["variant"] = variant
        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(str(model_dir), **kwargs)
        except TypeError:
            kwargs.pop("variant", None)
            pipeline = AutoPipelineForText2Image.from_pretrained(str(model_dir), **kwargs)
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if adapter_dir is not None and hasattr(pipeline, "load_lora_weights"):
            pipeline.load_lora_weights(str(adapter_dir))
            if hasattr(pipeline, "fuse_lora"):
                try:
                    pipeline.fuse_lora(lora_scale=float(self._profile.get("adapter_scale") or 1.0))
                except TypeError:
                    pipeline.fuse_lora()
        if device == "cuda":
            pipeline = pipeline.to("cuda")
        self._PIPELINES[cache_key] = pipeline
        return pipeline, torch, device

    def generate_mark(
        self,
        *,
        product_name: str,
        description: str,
        direction: str,
        palette: dict[str, str],
        shape_kind: str,
        variant_index: int,
        source_title: str | None,
        exploration_preset_id: str | None,
        seed: int,
    ) -> GeneratedMark:
        preset = get_exploration_preset(exploration_preset_id)
        prompt, exploration_label = self._prompt(
            product_name=product_name,
            description=description,
            direction=direction,
            palette=palette,
            shape_kind=shape_kind,
            variant_index=variant_index,
            source_title=source_title,
            exploration_preset_id=exploration_preset_id,
        )
        payload = {
            "repo_id": self._profile.get("repo_id"),
            "device": self._profile.get("device"),
            "width": self._profile.get("width"),
            "height": self._profile.get("height"),
            "steps": self._profile.get("steps"),
            "guidance_scale": self._profile.get("guidance_scale"),
            "seed": seed,
            "prompt": prompt,
        }
        try:
            pipeline, torch, device = self._load_pipeline()
            generator = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)
            banlist = self._negative_terms_from_direction(direction)
            negative_prompt = (
                "text, letters, words, wordmark, monogram, mascot, badge, seal, border, poster, mockup, "
                "website screenshot, ui, device frame, infographic, collage, contact sheet, icon sheet, logo wall, "
                "symbol catalog, repeated icons, tiled pattern, wallpaper, many logos, multiple symbols, ornament, mandala, "
                "sunburst, rays, decorative lines, blueprint, wireframe, lattice, panels, hands, person, "
                "face, photorealistic, 3d render, gradient background, clutter, shadows, thin outline, hairline stroke, delicate line art"
            )
            if banlist:
                negative_prompt = f"{negative_prompt}, {', '.join(banlist)}"
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=int(self._profile["width"]),
                height=int(self._profile["height"]),
                num_inference_steps=int(self._profile["steps"]),
                guidance_scale=float(self._profile["guidance_scale"]),
                generator=generator,
            )
            image = result.images[0].convert("RGBA")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return GeneratedMark(
                image=Image.open(buffer).convert("RGBA"),
                request_prompt=prompt,
                revised_prompt=None,
                exploration_label=exploration_label,
                preset_label=preset.label,
            )
        except Exception as error:
            self._save_failure("diffusers_logo_mark", payload, error)
            raise

    @staticmethod
    def _negative_terms_from_direction(direction: str) -> list[str]:
        lowered = direction.lower()
        terms: list[str] = []
        explicit = re.findall(r"no\s+([a-z][a-z\s-]{1,24})", lowered)
        for phrase in explicit:
            cleaned = phrase.strip(" ,.-")
            if cleaned:
                terms.append(cleaned)
        if "quill" in lowered or "pen" in lowered:
            terms.extend(["quill", "pen nib", "fountain pen", "feather"])
        if "book" in lowered or "page" in lowered:
            terms.extend(["book", "open book", "pages"])
        if "circle" in lowered or "ring" in lowered or "badge" in lowered:
            terms.extend(["circle", "ring", "roundel", "badge"])
        deduped: list[str] = []
        for term in terms:
            if term not in deduped:
                deduped.append(term)
        return deduped
