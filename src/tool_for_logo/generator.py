from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .backends import ComfyUILogoBackend, OpenAILogoBackend
from .models import CandidateRecord, CandidateStatus, ExportRecord
from .runtime import environment_value
from .state import ToolForLogoStore, utc_now

PALETTES: list[dict[str, str]] = [
    {
        "name": "harbor",
        "primary": "#125B9A",
        "secondary": "#6A9AB0",
        "accent": "#FFD166",
        "light_bg": "#F8FBFF",
        "dark_bg": "#0B1726",
        "light_text": "#F6F9FD",
        "dark_text": "#12263A",
    },
    {
        "name": "copper",
        "primary": "#7C3A2D",
        "secondary": "#D97B4A",
        "accent": "#F1E3D3",
        "light_bg": "#FFF9F5",
        "dark_bg": "#25130F",
        "light_text": "#FFF6EE",
        "dark_text": "#2A1914",
    },
    {
        "name": "grove",
        "primary": "#1E6F5C",
        "secondary": "#289672",
        "accent": "#FFE194",
        "light_bg": "#F7FFF9",
        "dark_bg": "#10271F",
        "light_text": "#F4FFF8",
        "dark_text": "#17342B",
    },
    {
        "name": "studio",
        "primary": "#3B2F7A",
        "secondary": "#8C5E99",
        "accent": "#F5B971",
        "light_bg": "#FBF9FF",
        "dark_bg": "#161129",
        "light_text": "#FAF8FF",
        "dark_text": "#251C3E",
    },
    {
        "name": "mono",
        "primary": "#1F1F1F",
        "secondary": "#6E6E6E",
        "accent": "#C4C4C4",
        "light_bg": "#FCFCFC",
        "dark_bg": "#111111",
        "light_text": "#FBFBFB",
        "dark_text": "#171717",
    },
]

FONT_FAMILIES = [
    "DejaVu Sans Bold",
    "DejaVu Sans",
    "Trebuchet MS",
    "Verdana",
    "Segoe UI",
]

SHAPE_KINDS = ["orbit", "shield", "ribbon", "spark", "monogram"]
REAL_SHAPE_KINDS = ["orbit", "shield", "ribbon", "spark"]

DIRECTION_LIBRARY = [
    "minimal and technical",
    "friendly and rounded",
    "premium editorial",
    "playful geometric",
    "solid B2B platform",
    "modern Japanese craft",
    "global SaaS clarity",
    "bold utility tooling",
]

RATIONALE_TEMPLATE = (
    "Focus on {direction} with a {shape_kind} mark so the icon survives small sizes "
    "and the wordmark can be swapped later without redesigning the full lockup."
)


def _color(name: str, palette: dict[str, str]) -> str:
    return palette[name]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _sanitize_text(value: str) -> str:
    return " ".join(value.split())


def _initials_for_name(name: str) -> str:
    parts = [part for part in _sanitize_text(name).replace("-", " ").split(" ") if part]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    alnum = "".join(char for char in name if char.isalnum())
    if len(alnum) >= 2:
        return alnum[:2].upper()
    return (name[:2] or "LG").upper()


def _stable_seed(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_existing_mark(mark_png_path: str | None) -> Image.Image | None:
    if not mark_png_path:
        return None
    path = Path(mark_png_path)
    if not path.exists():
        return None
    return Image.open(path).convert("RGBA")


def _hex_to_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    stripped = value.lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"Unsupported color value: {value}")
    return (
        int(stripped[0:2], 16),
        int(stripped[2:4], 16),
        int(stripped[4:6], 16),
        alpha,
    )


def _trim_transparency(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None:
        return rgba
    return rgba.crop(bounds)


def _fit_mark_to_canvas(mark_image: Image.Image, size: int = 512) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    fitted = _trim_transparency(mark_image.convert("RGBA"))
    fitted.thumbnail((size - 64, size - 64), Image.Resampling.LANCZOS)
    x = (size - fitted.width) // 2
    y = (size - fitted.height) // 2
    canvas.alpha_composite(fitted, (x, y))
    return canvas


def _select_direction(rng: random.Random, requested_direction: str) -> str:
    return requested_direction if requested_direction != "broad exploration" else rng.choice(DIRECTION_LIBRARY)


def _select_style(
    *,
    rng: random.Random,
    requested_direction: str,
    source_candidate: CandidateRecord | None,
    backend: str,
) -> tuple[str, str, dict[str, str], str, str]:
    if source_candidate is not None:
        return (
            requested_direction if requested_direction != "broad exploration" else source_candidate.direction,
            source_candidate.palette_name,
            dict(source_candidate.palette),
            source_candidate.font_family,
            source_candidate.shape_kind,
        )

    palette = dict(rng.choice(PALETTES))
    palette_name = palette.pop("name")
    font_family = rng.choice(FONT_FAMILIES)
    shape_library = REAL_SHAPE_KINDS if backend in {"comfyui", "openai"} else SHAPE_KINDS
    shape_kind = rng.choice(shape_library)
    direction = _select_direction(rng, requested_direction)
    return direction, palette_name, palette, font_family, shape_kind


def _remove_white_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    pixels = []
    for red, green, blue, alpha in rgba.getdata():
        floor = min(red, green, blue)
        if floor >= 250:
            pixels.append((red, green, blue, 0))
            continue
        if floor >= 220:
            fade = max(0, min(alpha, int((250 - floor) * 8)))
            pixels.append((red, green, blue, fade))
            continue
        pixels.append((red, green, blue, alpha))
    rgba.putdata(pixels)
    return rgba


def _stylize_generated_mark(mark_image: Image.Image, palette: dict[str, str]) -> Image.Image:
    cleaned = _trim_transparency(_remove_white_background(mark_image))
    alpha = cleaned.getchannel("A")
    if alpha.getbbox() is None:
        return cleaned

    halo_mask = ImageChops.subtract(alpha.filter(ImageFilter.GaussianBlur(18)), alpha)
    halo_mask = halo_mask.point(lambda value: min(112, value))

    halo = Image.new("RGBA", cleaned.size, _hex_to_rgba(_color("secondary", palette)))
    halo.putalpha(halo_mask)

    fill = Image.new("RGBA", cleaned.size, _hex_to_rgba(_color("primary", palette)))
    fill.putalpha(alpha)

    styled = Image.new("RGBA", cleaned.size, (0, 0, 0, 0))
    styled.alpha_composite(halo)
    styled.alpha_composite(fill)

    bounds = alpha.getbbox()
    if bounds is not None:
        left, top, right, bottom = bounds
        width = max(1, right - left)
        height = max(1, bottom - top)
        radius = max(10, min(width, height) // 9)
        dot_x = max(radius, right - radius * 2)
        dot_y = min(cleaned.height - radius * 2, top + radius)
        accent = Image.new("RGBA", cleaned.size, (0, 0, 0, 0))
        accent_draw = ImageDraw.Draw(accent)
        accent_draw.ellipse(
            (dot_x, dot_y, dot_x + radius * 2, dot_y + radius * 2),
            fill=_color("accent", palette),
        )
        styled.alpha_composite(accent)

    return _trim_transparency(styled)


def _draw_mark(image: Image.Image, shape_kind: str, palette: dict[str, str], initials: str, seed: int) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    cx, cy = width // 2, height // 2
    margin = 48
    primary = _color("primary", palette)
    secondary = _color("secondary", palette)
    accent = _color("accent", palette)
    light_bg = _color("light_bg", palette)

    if shape_kind == "orbit":
        draw.ellipse((margin, margin, width - margin, height - margin), outline=primary, width=26)
        draw.ellipse((margin + 62, margin + 62, width - margin - 62, height - margin - 62), outline=secondary, width=16)
        draw.ellipse((width - 170, 110, width - 90, 190), fill=accent)
    elif shape_kind == "shield":
        points = [
            (cx, margin),
            (width - margin, 130),
            (width - 88, height - 170),
            (cx, height - margin),
            (88, height - 170),
            (margin, 130),
        ]
        draw.polygon(points, fill=primary)
        draw.polygon([(cx, margin + 60), (width - 124, 150), (cx, height - 112), (124, 150)], fill=accent)
    elif shape_kind == "ribbon":
        draw.rounded_rectangle((70, 180, width - 70, 330), radius=48, fill=primary)
        draw.polygon([(135, 120), (255, 120), (width - 135, 392), (width - 255, 392)], fill=secondary)
        draw.rounded_rectangle((120, 220, width - 120, 290), radius=28, fill=light_bg)
    elif shape_kind == "spark":
        points = [
            (cx, margin),
            (cx + 68, cy - 68),
            (width - margin, cy),
            (cx + 68, cy + 68),
            (cx, height - margin),
            (cx - 68, cy + 68),
            (margin, cy),
            (cx - 68, cy - 68),
        ]
        draw.polygon(points, fill=primary)
        draw.ellipse((cx - 52, cy - 52, cx + 52, cy + 52), fill=accent)
    else:
        draw.rounded_rectangle((margin, margin, width - margin, height - margin), radius=96, fill=primary)
        draw.rounded_rectangle((margin + 52, margin + 52, width - margin - 52, height - margin - 52), radius=72, outline=accent, width=14)

    font = _load_font(140)
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(
        (cx - text_width / 2, cy - text_height / 2 - 8),
        initials,
        fill=light_bg,
        font=font,
    )


def _build_mark_svg(shape_kind: str, palette: dict[str, str], initials: str) -> str:
    primary = _color("primary", palette)
    secondary = _color("secondary", palette)
    accent = _color("accent", palette)
    light_bg = _color("light_bg", palette)

    if shape_kind == "orbit":
        body = f"""
  <circle cx="256" cy="256" r="190" fill="none" stroke="{primary}" stroke-width="26" />
  <circle cx="256" cy="256" r="128" fill="none" stroke="{secondary}" stroke-width="16" />
  <circle cx="382" cy="150" r="40" fill="{accent}" />
"""
    elif shape_kind == "shield":
        body = f"""
  <path d="M256 58 L420 130 L392 350 L256 454 L120 350 L92 130 Z" fill="{primary}" />
  <path d="M256 118 L388 150 L256 400 L124 150 Z" fill="{accent}" />
"""
    elif shape_kind == "ribbon":
        body = f"""
  <rect x="70" y="180" width="372" height="150" rx="48" fill="{primary}" />
  <path d="M135 120 L255 120 L377 392 L257 392 Z" fill="{secondary}" />
  <path d="M120 220 H392 A28 28 0 0 1 420 248 V262 A28 28 0 0 1 392 290 H120 A28 28 0 0 1 92 262 V248 A28 28 0 0 1 120 220 Z" fill="{light_bg}" />
"""
    elif shape_kind == "spark":
        body = f"""
  <path d="M256 58 L324 188 L454 256 L324 324 L256 454 L188 324 L58 256 L188 188 Z" fill="{primary}" />
  <circle cx="256" cy="256" r="52" fill="{accent}" />
"""
    else:
        body = f"""
  <rect x="48" y="48" width="416" height="416" rx="96" fill="{primary}" />
  <rect x="100" y="100" width="312" height="312" rx="72" fill="none" stroke="{accent}" stroke-width="14" />
"""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
{body}
  <text x="256" y="298" text-anchor="middle" font-size="140" font-weight="700" fill="{light_bg}" font-family="DejaVu Sans, Segoe UI, sans-serif">{initials}</text>
</svg>
"""


def _draw_wordmark(image: Image.Image, name: str, palette: dict[str, str], direction: str) -> None:
    draw = ImageDraw.Draw(image)
    font_main = _load_font(92)
    font_meta = _load_font(34)
    width, height = image.size
    text_color = _color("dark_text", palette)
    accent = _color("secondary", palette)
    name = _sanitize_text(name)
    name_bbox = draw.textbbox((0, 0), name, font=font_main)
    meta_bbox = draw.textbbox((0, 0), direction, font=font_meta)
    text_width = name_bbox[2] - name_bbox[0]
    meta_width = meta_bbox[2] - meta_bbox[0]
    x = 12
    y = height / 2 - 70
    draw.text((x, y), name, font=font_main, fill=text_color)
    draw.text((x, y + 102), direction, font=font_meta, fill=accent)
    draw.line((x, y + 164, max(text_width, meta_width) + x, y + 164), fill=accent, width=6)


def _build_wordmark_svg(name: str, palette: dict[str, str], direction: str, font_family: str) -> str:
    name = _sanitize_text(name)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="220" viewBox="0 0 900 220">
  <text x="12" y="104" font-size="92" font-weight="700" fill="{_color("dark_text", palette)}" font-family="{font_family}, Segoe UI, sans-serif">{name}</text>
  <text x="12" y="188" font-size="34" fill="{_color("secondary", palette)}" font-family="{font_family}, Segoe UI, sans-serif">{direction}</text>
  <line x1="12" y1="204" x2="620" y2="204" stroke="{_color("secondary", palette)}" stroke-width="6" />
</svg>
"""


def _compose_lockup(
    mark_image: Image.Image,
    wordmark_image: Image.Image,
    palette: dict[str, str],
    layout: str,
) -> Image.Image:
    if layout == "stacked":
        canvas = Image.new("RGBA", (1100, 980), (0, 0, 0, 0))
        canvas.alpha_composite(mark_image.resize((420, 420)), (340, 96))
        canvas.alpha_composite(wordmark_image.resize((900, 220)), (100, 560))
        return canvas

    canvas = Image.new("RGBA", (1400, 700), (0, 0, 0, 0))
    canvas.alpha_composite(mark_image.resize((360, 360)), (70, 170))
    canvas.alpha_composite(wordmark_image.resize((900, 220)), (430, 240))
    return canvas


def _draw_preview(
    lockup_light: Image.Image,
    lockup_dark: Image.Image,
    palette: dict[str, str],
    title: str,
    direction: str,
    target_path: Path,
    dark: bool,
) -> None:
    if dark:
        background = _color("dark_bg", palette)
        text_color = _color("light_text", palette)
    else:
        background = _color("light_bg", palette)
        text_color = _color("dark_text", palette)

    canvas = Image.new("RGBA", (1600, 1100), background)
    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(44)
    font_meta = _load_font(26)
    draw.text((76, 54), title, font=font_title, fill=text_color)
    draw.text((76, 118), direction, font=font_meta, fill=_color("accent", palette))
    first = lockup_dark if dark else lockup_light
    second = lockup_light if dark else lockup_dark
    canvas.alpha_composite(first.resize((1120, 560)), (180, 210))
    canvas.alpha_composite(second.resize((960, 430)), (320, 650))
    canvas.save(target_path)


def render_candidate_assets(
    *,
    output_dir: Path,
    product_name: str,
    direction: str,
    palette: dict[str, str],
    font_family: str,
    shape_kind: str,
    initials: str,
    seed: int,
    mark_image: Image.Image | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mark_canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    raw_mark_path = output_dir / "mark_raw.png"
    if mark_image is None:
        _draw_mark(mark_canvas, shape_kind, palette, initials, seed)
    else:
        mark_image.convert("RGBA").save(raw_mark_path)
        mark_canvas = _fit_mark_to_canvas(_stylize_generated_mark(mark_image, palette))
    mark_path = output_dir / "mark.png"
    mark_canvas.save(mark_path)
    if mark_image is None:
        (output_dir / "mark.svg").write_text(
            _build_mark_svg(shape_kind, palette, initials),
            encoding="utf-8",
        )

    wordmark_image = Image.new("RGBA", (900, 220), (0, 0, 0, 0))
    _draw_wordmark(wordmark_image, product_name, palette, direction)
    wordmark_path = output_dir / "wordmark.png"
    wordmark_image.save(wordmark_path)
    (output_dir / "wordmark.svg").write_text(
        _build_wordmark_svg(product_name, palette, direction, font_family),
        encoding="utf-8",
    )

    horizontal = _compose_lockup(mark_canvas, wordmark_image, palette, layout="horizontal")
    horizontal_path = output_dir / "lockup_horizontal.png"
    horizontal.save(horizontal_path)

    stacked = _compose_lockup(mark_canvas, wordmark_image, palette, layout="stacked")
    stacked_path = output_dir / "lockup_stacked.png"
    stacked.save(stacked_path)

    preview_light_path = output_dir / "preview_light.png"
    preview_dark_path = output_dir / "preview_dark.png"
    _draw_preview(horizontal, stacked, palette, product_name, direction, preview_light_path, dark=False)
    _draw_preview(horizontal, stacked, palette, product_name, direction, preview_dark_path, dark=True)

    assets = {
        "mark_png": str(mark_path),
        "wordmark_png": str(wordmark_path),
        "wordmark_svg": str(output_dir / "wordmark.svg"),
        "lockup_horizontal_png": str(horizontal_path),
        "lockup_stacked_png": str(stacked_path),
        "preview_light_png": str(preview_light_path),
        "preview_dark_png": str(preview_dark_path),
    }
    mark_svg_path = output_dir / "mark.svg"
    if mark_svg_path.exists():
        assets["mark_svg"] = str(mark_svg_path)
    if raw_mark_path.exists():
        assets["mark_raw_png"] = str(raw_mark_path)
    return assets


def _generate_openai_mark(
    *,
    store: ToolForLogoStore,
    case_record,
    direction: str,
    palette: dict[str, str],
    shape_kind: str,
    index: int,
    source_candidate: CandidateRecord | None,
    seed: int,
) -> tuple[Image.Image, dict[str, object]]:
    OpenAILogoBackend.ensure_api_key()
    backend = OpenAILogoBackend(
        logs_root=store.logs_root,
        image_model=environment_value("TOOL_FOR_LOGO_IMAGE_MODEL", "gpt-image-1.5"),
        quality=environment_value("TOOL_FOR_LOGO_IMAGE_QUALITY", "medium"),
        size=environment_value("TOOL_FOR_LOGO_IMAGE_SIZE", "1024x1024"),
        background=environment_value("TOOL_FOR_LOGO_IMAGE_BACKGROUND", "transparent"),
    )
    generated = backend.generate_mark(
        product_name=case_record.product_name,
        description=case_record.description,
        direction=direction,
        palette=palette,
        shape_kind=shape_kind,
        variant_index=index,
        source_title=source_candidate.title if source_candidate is not None else None,
        seed=seed,
    )
    return generated.image, {
        "request_prompt": generated.request_prompt,
        "revised_prompt": generated.revised_prompt,
        "image_model": environment_value("TOOL_FOR_LOGO_IMAGE_MODEL", "gpt-image-1.5"),
        "image_quality": environment_value("TOOL_FOR_LOGO_IMAGE_QUALITY", "medium"),
        "image_size": environment_value("TOOL_FOR_LOGO_IMAGE_SIZE", "1024x1024"),
        "background": environment_value("TOOL_FOR_LOGO_IMAGE_BACKGROUND", "transparent"),
        "seed": seed,
    }


def _generate_comfyui_mark(
    *,
    store: ToolForLogoStore,
    case_record,
    direction: str,
    palette: dict[str, str],
    shape_kind: str,
    index: int,
    source_candidate: CandidateRecord | None,
    seed: int,
) -> tuple[Image.Image, dict[str, object]]:
    backend = ComfyUILogoBackend(
        logs_root=store.logs_root,
        base_url=environment_value("TOOL_FOR_LOGO_COMFYUI_BASE_URL", "http://host.docker.internal:8188"),
        checkpoint=environment_value(
            "TOOL_FOR_LOGO_COMFYUI_CHECKPOINT",
            "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        ),
        width=int(environment_value("TOOL_FOR_LOGO_COMFYUI_WIDTH", "1024")),
        height=int(environment_value("TOOL_FOR_LOGO_COMFYUI_HEIGHT", "1024")),
        steps=int(environment_value("TOOL_FOR_LOGO_COMFYUI_STEPS", "20")),
        cfg=float(environment_value("TOOL_FOR_LOGO_COMFYUI_CFG", "6.5")),
        sampler_name=environment_value("TOOL_FOR_LOGO_COMFYUI_SAMPLER", "dpmpp_2m_sde"),
        scheduler=environment_value("TOOL_FOR_LOGO_COMFYUI_SCHEDULER", "karras"),
        negative_prompt=environment_value(
            "TOOL_FOR_LOGO_COMFYUI_NEGATIVE_PROMPT",
            "text, letters, wordmark, watermark, signature, mockup, poster, product photo, person, hand, scenery, 3d render, glossy, realistic, photorealistic, gradient mesh, cluttered background",
        ),
        timeout_seconds=int(environment_value("TOOL_FOR_LOGO_COMFYUI_TIMEOUT_SECONDS", "300")),
        poll_interval_seconds=float(environment_value("TOOL_FOR_LOGO_COMFYUI_POLL_SECONDS", "2")),
    )
    generated = backend.generate_mark(
        product_name=case_record.product_name,
        description=case_record.description,
        direction=direction,
        palette=palette,
        shape_kind=shape_kind,
        variant_index=index,
        source_title=source_candidate.title if source_candidate is not None else None,
        seed=seed,
    )
    return generated.image, {
        "request_prompt": generated.request_prompt,
        "revised_prompt": generated.revised_prompt,
        "base_url": environment_value("TOOL_FOR_LOGO_COMFYUI_BASE_URL", "http://host.docker.internal:8188"),
        "checkpoint": environment_value(
            "TOOL_FOR_LOGO_COMFYUI_CHECKPOINT",
            "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        ),
        "width": int(environment_value("TOOL_FOR_LOGO_COMFYUI_WIDTH", "1024")),
        "height": int(environment_value("TOOL_FOR_LOGO_COMFYUI_HEIGHT", "1024")),
        "steps": int(environment_value("TOOL_FOR_LOGO_COMFYUI_STEPS", "20")),
        "cfg": float(environment_value("TOOL_FOR_LOGO_COMFYUI_CFG", "6.5")),
        "sampler": environment_value("TOOL_FOR_LOGO_COMFYUI_SAMPLER", "dpmpp_2m_sde"),
        "scheduler": environment_value("TOOL_FOR_LOGO_COMFYUI_SCHEDULER", "karras"),
        "seed": seed,
    }


def generate_batch(
    store: ToolForLogoStore,
    *,
    case_id: str,
    count: int,
    direction_hint: str,
    seed: int | None = None,
    backend: str = "comfyui",
    source_candidate_id: str | None = None,
) -> dict[str, object]:
    if backend not in {"mock", "comfyui", "openai"}:
        raise ValueError(f"Unsupported backend: {backend}")

    case_record = store.get_case(case_id)
    source_candidate = store.get_candidate(case_id, source_candidate_id) if source_candidate_id else None
    batch = store.create_batch(
        case_id=case_id,
        backend=backend,
        direction_hint=direction_hint,
        candidate_count=count,
        seed=seed,
        source_candidate_id=source_candidate_id,
    )
    requested_direction = direction_hint.strip() or "broad exploration"
    batch_seed = seed if seed is not None else _stable_seed(case_record.case_id, batch.batch_id, requested_direction, backend)
    rng = random.Random(batch_seed)

    candidates: list[CandidateRecord] = []
    for index in range(count):
        candidate_id = store._new_id("candidate")
        direction, palette_name, palette, font_family, shape_kind = _select_style(
            rng=rng,
            requested_direction=requested_direction,
            source_candidate=source_candidate,
            backend=backend,
        )
        candidate_seed = batch_seed + (index * 97)
        candidate_root = store.candidate_root(case_id, candidate_id)
        assets_dir = candidate_root / "assets"
        generation: dict[str, object] = {"backend": backend}
        mark_image: Image.Image | None = None
        if backend == "openai":
            mark_image, generation = _generate_openai_mark(
                store=store,
                case_record=case_record,
                direction=direction,
                palette=palette,
                shape_kind=shape_kind,
                index=index,
                source_candidate=source_candidate,
                seed=candidate_seed,
            )
        elif backend == "comfyui":
            mark_image, generation = _generate_comfyui_mark(
                store=store,
                case_record=case_record,
                direction=direction,
                palette=palette,
                shape_kind=shape_kind,
                index=index,
                source_candidate=source_candidate,
                seed=candidate_seed,
            )
        assets = render_candidate_assets(
            output_dir=assets_dir,
            product_name=case_record.product_name,
            direction=direction,
            palette=palette,
            font_family=font_family,
            shape_kind=shape_kind,
            initials=_initials_for_name(case_record.product_name),
            seed=candidate_seed,
            mark_image=mark_image,
        )
        rationale = RATIONALE_TEMPLATE.format(direction=direction, shape_kind=shape_kind)
        if source_candidate is not None:
            rationale = (
                f"Derived from {source_candidate.title}. "
                f"{RATIONALE_TEMPLATE.format(direction=direction, shape_kind=shape_kind)}"
            )
        record = CandidateRecord(
            candidate_id=candidate_id,
            case_id=case_id,
            batch_id=batch.batch_id,
            title=f"{case_record.product_name} concept {index + 1:02d}",
            direction=direction,
            rationale=rationale,
            palette_name=palette_name,
            palette=palette,
            font_family=font_family,
            shape_kind=shape_kind,
            seed=candidate_seed,
            initials=_initials_for_name(case_record.product_name),
            status=CandidateStatus.FRESH,
            assets=assets,
            created_at=utc_now(),
            updated_at=utc_now(),
            generation=generation,
        )
        store.save_candidate(record)
        candidates.append(record)

    batch.candidate_ids = [item.candidate_id for item in candidates]
    store.save_batch(batch)
    store.touch_case(case_id)
    payload: dict[str, object] = {
        "case": case_record.to_dict(),
        "batch": batch.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    if source_candidate is not None:
        payload["source_candidate"] = source_candidate.to_dict()
    return payload


def generate_mock_batch(
    store: ToolForLogoStore,
    *,
    case_id: str,
    count: int,
    direction_hint: str,
    seed: int | None = None,
    source_candidate_id: str | None = None,
) -> dict[str, object]:
    return generate_batch(
        store,
        case_id=case_id,
        count=count,
        direction_hint=direction_hint,
        seed=seed,
        backend="mock",
        source_candidate_id=source_candidate_id,
    )


def _build_comparison_sheet(
    *,
    export_dir: Path,
    candidate_cards: list[tuple[CandidateRecord, dict[str, str]]],
) -> str:
    card_width = 540
    card_height = 420
    columns = 2
    rows = max(1, (len(candidate_cards) + columns - 1) // columns)
    canvas = Image.new("RGBA", (columns * card_width + 80, rows * card_height + 80), "#F5F7FA")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(30)
    meta_font = _load_font(22)

    for index, (candidate, assets) in enumerate(candidate_cards):
        row = index // columns
        column = index % columns
        x = 40 + column * card_width
        y = 40 + row * card_height
        draw.rounded_rectangle((x, y, x + 500, y + 360), radius=28, fill="white", outline="#D9E2EC", width=2)
        preview = Image.open(assets["preview_light_png"]).convert("RGBA")
        preview.thumbnail((460, 240), Image.Resampling.LANCZOS)
        canvas.alpha_composite(preview, (x + (500 - preview.width) // 2, y + 24))
        draw.text((x + 24, y + 282), candidate.title, font=title_font, fill="#1F2933")
        draw.text((x + 24, y + 322), candidate.direction, font=meta_font, fill="#486581")
    target = export_dir / "comparison_sheet.png"
    canvas.save(target)
    return str(target)


def _resolve_export_candidates(
    store: ToolForLogoStore,
    case_id: str,
    candidate_ids: Iterable[str] | None,
) -> list[CandidateRecord]:
    if candidate_ids:
        return [store.get_candidate(case_id, candidate_id) for candidate_id in candidate_ids]

    candidates = store.list_candidates(case_id)
    favorites = [item for item in candidates if item.status in {CandidateStatus.FAVORITE, CandidateStatus.ADOPTED}]
    return favorites or candidates[:3]


def create_export_bundle(
    store: ToolForLogoStore,
    *,
    case_id: str,
    candidate_ids: list[str] | None = None,
    name_override: str | None = None,
) -> ExportRecord:
    case_record = store.get_case(case_id)
    selected = _resolve_export_candidates(store, case_id, candidate_ids)
    if not selected:
        raise ValueError("No candidates available to export.")

    export_id = store._new_id("export")
    export_dir = store.report_root / case_id / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_candidates: list[dict[str, object]] = []
    candidate_cards: list[tuple[CandidateRecord, dict[str, str]]] = []

    for candidate in selected:
        destination = export_dir / candidate.candidate_id
        rendered_name = name_override or case_record.product_name
        existing_mark = _load_existing_mark(candidate.assets.get("mark_png"))
        assets = render_candidate_assets(
            output_dir=destination / "assets",
            product_name=rendered_name,
            direction=candidate.direction,
            palette=candidate.palette,
            font_family=candidate.font_family,
            shape_kind=candidate.shape_kind,
            initials=_initials_for_name(rendered_name),
            seed=candidate.seed,
            mark_image=existing_mark,
        )
        manifest_candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "title": candidate.title,
                "direction": candidate.direction,
                "status": candidate.status.value,
                "assets": assets,
            }
        )
        candidate_cards.append((candidate, assets))

    comparison_sheet_path = _build_comparison_sheet(export_dir=export_dir, candidate_cards=candidate_cards)

    manifest = {
        "export_id": export_id,
        "case_id": case_id,
        "product_name": case_record.product_name,
        "name_override": name_override,
        "created_at": utc_now(),
        "comparison_sheet_png": comparison_sheet_path,
        "candidates": manifest_candidates,
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    archive_base = store.report_root / case_id / export_id
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=export_dir)
    export_record = ExportRecord(
        export_id=export_id,
        case_id=case_id,
        name_override=name_override,
        candidate_ids=[item.candidate_id for item in selected],
        export_dir=str(export_dir),
        archive_path=archive_path,
        created_at=utc_now(),
    )
    store.save_export(export_record)
    return export_record
