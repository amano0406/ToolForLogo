from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from .runtime import save_api_failure


@dataclass(slots=True)
class GeneratedMark:
    image: Image.Image
    request_prompt: str
    revised_prompt: str | None


def _fetch_json(url: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is None:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode("utf-8"))
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


class ComfyUILogoBackend:
    def __init__(
        self,
        *,
        logs_root: Path,
        base_url: str,
        checkpoint: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        negative_prompt: str,
        timeout_seconds: int,
        poll_interval_seconds: float,
    ) -> None:
        self._logs_root = logs_root
        self._base_url = base_url.rstrip("/")
        self._checkpoint = checkpoint
        self._width = width
        self._height = height
        self._steps = steps
        self._cfg = cfg
        self._sampler_name = sampler_name
        self._scheduler = scheduler
        self._negative_prompt = negative_prompt
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def _save_failure(self, stage: str, payload: dict[str, Any], error: Exception) -> None:
        save_api_failure(self._logs_root, stage, payload, error)

    def ensure_available(self) -> None:
        try:
            _fetch_json(f"{self._base_url}/system_stats")
        except Exception as error:
            raise RuntimeError(
                f"ComfyUI is not reachable at {self._base_url}. "
                "Start C:\\apps\\ComfyUI_windows_portable\\run_nvidia_gpu_api.bat first."
            ) from error

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
    ) -> str:
        sibling_note = (
            f"close sibling of {source_title}, keep the same overall direction"
            if source_title
            else "explore the direction without drifting into unrelated styles"
        )
        return ", ".join(
            [
                "single centered logo symbol",
                "abstract geometric brand icon",
                "clean vector icon",
                "flat branding mark",
                "corporate software product logo",
                "pure white background",
                "no text",
                "no letters",
                "no wordmark",
                "no mascot",
                "no character",
                "simple silhouette",
                "monochrome-friendly shape language",
                "high contrast",
                f"brand name {product_name}",
                f"product context {description}",
                f"direction {direction}",
                f"shape tendency {shape_kind}",
                f"palette {palette['primary']} {palette['secondary']} {palette['accent']}",
                sibling_note,
                f"variant {variant_index + 1}",
            ]
        )

    def _build_workflow(self, *, positive_prompt: str, seed: int) -> dict[str, Any]:
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": self._steps,
                    "cfg": self._cfg,
                    "sampler_name": self._sampler_name,
                    "scheduler": self._scheduler,
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": self._checkpoint,
                },
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": self._width,
                    "height": self._height,
                    "batch_size": 1,
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": positive_prompt,
                    "clip": ["4", 1],
                },
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": self._negative_prompt,
                    "clip": ["4", 1],
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "ToolForLogo",
                    "images": ["8", 0],
                },
            },
        }

    def _queue_prompt(self, workflow: dict[str, Any], prompt_id: str) -> None:
        payload = {"prompt": workflow, "prompt_id": prompt_id}
        _fetch_json(f"{self._base_url}/prompt", payload=payload)

    def _get_history(self, prompt_id: str) -> dict[str, Any]:
        return _fetch_json(f"{self._base_url}/history/{prompt_id}")

    def _download_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        query = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"{self._base_url}/view?{query}") as response:
            return response.read()

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
        seed: int,
    ) -> GeneratedMark:
        self.ensure_available()
        prompt = self._prompt(
            product_name=product_name,
            description=description,
            direction=direction,
            palette=palette,
            shape_kind=shape_kind,
            variant_index=variant_index,
            source_title=source_title,
        )
        workflow = self._build_workflow(positive_prompt=prompt, seed=seed)
        prompt_id = str(uuid4())
        try:
            self._queue_prompt(workflow, prompt_id)
            deadline = time.time() + self._timeout_seconds
            while time.time() < deadline:
                history = self._get_history(prompt_id)
                prompt_history = history.get(prompt_id)
                if prompt_history and prompt_history.get("outputs"):
                    for node_output in prompt_history["outputs"].values():
                        images = node_output.get("images", [])
                        if not images:
                            continue
                        image_info = images[0]
                        image_bytes = self._download_image(
                            filename=image_info["filename"],
                            subfolder=image_info["subfolder"],
                            folder_type=image_info["type"],
                        )
                        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
                        return GeneratedMark(
                            image=image,
                            request_prompt=prompt,
                            revised_prompt=None,
                        )
                time.sleep(self._poll_interval_seconds)
            raise RuntimeError(f"ComfyUI prompt timed out after {self._timeout_seconds} seconds.")
        except Exception as error:
            self._save_failure(
                "comfyui_logo_mark",
                {
                    "base_url": self._base_url,
                    "checkpoint": self._checkpoint,
                    "width": self._width,
                    "height": self._height,
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                },
                error,
            )
            raise


class OpenAILogoBackend:
    def __init__(
        self,
        *,
        logs_root: Path,
        image_model: str,
        quality: str,
        size: str,
        background: str,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "backend=openai requires the optional 'openai' package. "
                "Install it manually if you want the fallback cloud backend."
            ) from error

        self._client = OpenAI()
        self._logs_root = logs_root
        self._image_model = image_model
        self._quality = quality
        self._size = size
        self._background = background

    @staticmethod
    def ensure_api_key() -> None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise RuntimeError("OPENAI_API_KEY is required when backend=openai.")

    def _save_failure(self, stage: str, payload: dict[str, Any], error: Exception) -> None:
        save_api_failure(self._logs_root, stage, payload, error)

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
    ) -> str:
        sibling_note = (
            f"Develop it as a close sibling of the approved direction '{source_title}'."
            if source_title
            else "Explore the direction without drifting into unrelated visual genres."
        )
        return "\n".join(
            [
                f"Design a single standalone logo symbol for the product '{product_name}'.",
                f"Product description: {description}",
                f"Creative direction: {direction}",
                f"Symbol tendency: {shape_kind}",
                f"Preferred palette: primary {palette['primary']}, secondary {palette['secondary']}, accent {palette['accent']}.",
                sibling_note,
                "Hard requirements:",
                "- Return only the symbol, not the product name.",
                "- No text, no letters, no monogram, no wordmark.",
                "- Centered composition with transparent background.",
                "- Clean vector-like flat logo mark suitable for software branding.",
                "- Readable at small sizes.",
                "- Avoid mockups, paper, wall signage, hands, devices, scenery, mascots, photorealism, and complex textures.",
                f"Variation note: concept variant {variant_index + 1}.",
            ]
        )

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
        seed: int,
    ) -> GeneratedMark:
        del seed
        prompt = self._prompt(
            product_name=product_name,
            description=description,
            direction=direction,
            palette=palette,
            shape_kind=shape_kind,
            variant_index=variant_index,
            source_title=source_title,
        )
        payload = {
            "model": self._image_model,
            "prompt": prompt,
            "size": self._size,
            "background": self._background,
            "quality": self._quality,
            "output_format": "png",
            "n": 1,
        }
        try:
            response = self._client.images.generate(**payload)
        except Exception as error:
            self._save_failure("openai_logo_mark", payload, error)
            raise
        if not getattr(response, "data", None):
            raise RuntimeError("Image API returned no image data.")
        item = response.data[0]
        if getattr(item, "b64_json", None):
            image_bytes = base64.b64decode(item.b64_json)
        elif getattr(item, "url", None):
            with urllib.request.urlopen(item.url) as handle:
                image_bytes = handle.read()
        else:
            raise RuntimeError("Image API response did not contain b64_json or url.")
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        return GeneratedMark(
            image=image,
            request_prompt=prompt,
            revised_prompt=getattr(item, "revised_prompt", None),
        )
