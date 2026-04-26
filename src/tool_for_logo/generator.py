from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .backends import DiffusersLogoBackend, get_exploration_preset
from .concept_backend import LocalConceptBackend
from .model_catalog import resolve_generation_profile
from .models import CandidateRecord, CandidateStatus, ExportRecord
from .settings import load_huggingface_token, load_settings, load_worker_capabilities
from .state import ToolForLogoStore, utc_now
from .vectorize import svg_inner_content, vectorize_mark_to_svg

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
        "primary": "#243B53",
        "secondary": "#486581",
        "accent": "#F0B429",
        "light_bg": "#F7FAFC",
        "dark_bg": "#102A43",
        "light_text": "#F8FBFF",
        "dark_text": "#1F2933",
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

PALETTES_BY_NAME = {item["name"]: item for item in PALETTES}

FONT_FAMILIES = [
    "DejaVu Sans",
    "DejaVu Sans Bold",
    "Segoe UI",
    "Trebuchet MS",
    "Verdana",
]

SHAPE_KINDS = ["link", "bridge", "grid", "path", "signal", "frame", "fold", "orbit", "monogram"]
REAL_SHAPE_KINDS = ["link", "bridge", "grid", "path", "signal", "frame", "fold", "orbit"]
WORDMARK_CASES = ["title", "upper", "lower"]
WORDMARK_WEIGHTS = ["semibold", "bold"]
TONE_PRESETS: dict[str, dict[str, object]] = {
    "candidate-auto": {
        "label": "Candidate Auto",
        "font_family": None,
        "font_key": None,
        "wordmark_case": None,
        "wordmark_weight": None,
        "tracking": 0,
        "surface_theme": "candidate",
    },
    "product-ui": {
        "label": "Product UI",
        "font_family": "DejaVu Sans",
        "font_key": "sans-bold",
        "wordmark_case": "title",
        "wordmark_weight": "bold",
        "tracking": 0,
        "surface_theme": "product",
    },
    "editorial": {
        "label": "Editorial",
        "font_family": "DejaVu Serif",
        "font_key": "serif-semibold",
        "wordmark_case": "title",
        "wordmark_weight": "semibold",
        "tracking": 4,
        "surface_theme": "editorial",
    },
    "premium": {
        "label": "Premium",
        "font_family": "DejaVu Serif",
        "font_key": "serif-semibold",
        "wordmark_case": "upper",
        "wordmark_weight": "semibold",
        "tracking": 14,
        "surface_theme": "premium",
    },
    "friendly": {
        "label": "Friendly",
        "font_family": "DejaVu Sans",
        "font_key": "condensed-bold",
        "wordmark_case": "title",
        "wordmark_weight": "bold",
        "tracking": 2,
        "surface_theme": "friendly",
    },
    "utility": {
        "label": "Utility",
        "font_family": "DejaVu Sans",
        "font_key": "condensed-bold",
        "wordmark_case": "upper",
        "wordmark_weight": "bold",
        "tracking": 8,
        "surface_theme": "product",
    },
}
REFINEMENT_TONE_PRESETS = ["product-ui", "editorial", "premium", "friendly", "utility"]

DIRECTION_LIBRARY = [
    "minimal and technical",
    "friendly and rounded",
    "premium editorial",
    "solid B2B platform",
    "global SaaS clarity",
    "bold utility tooling",
    "quiet infrastructure brand",
    "confident product studio",
]

EXPLORATION_DIRECTION_MAP: dict[str, list[str]] = {
    "balanced-saas": [
        "minimal and technical",
        "global SaaS clarity",
        "solid B2B platform",
        "confident product studio",
        "bold utility tooling",
    ],
    "b2b-infra": [
        "solid B2B platform",
        "quiet infrastructure brand",
        "minimal and technical",
        "global SaaS clarity",
        "bold utility tooling",
    ],
    "friendly-tool": [
        "friendly and rounded",
        "bold utility tooling",
        "confident product studio",
        "global SaaS clarity",
    ],
    "premium-brand": [
        "premium editorial",
        "confident product studio",
        "minimal and technical",
        "quiet infrastructure brand",
    ],
    "ai-data": [
        "minimal and technical",
        "global SaaS clarity",
        "quiet infrastructure brand",
        "solid B2B platform",
    ],
}

RATIONALE_TEMPLATE = (
    "Focus on {direction} with a {shape_kind} symbol so the mark reads cleanly in a website header and favicon."
)

IMAGE_EXPLORATION_RATIONALES = (
    "Pushes toward a simple company-logo silhouette that can be judged quickly in a gallery.",
    "Keeps the mark bold enough for a navbar while exploring a distinct visual family.",
    "Explores a new logo family while staying inside the constraints of a software brand mark.",
)


ProgressCallback = Callable[[int, int, str, str], None]


def _color(name: str, palette: dict[str, str]) -> str:
    return palette[name]


def list_tone_presets() -> list[dict[str, str]]:
    return [
        {"preset_id": preset_id, "label": str(config["label"])}
        for preset_id, config in TONE_PRESETS.items()
    ]


def _font_candidates(font_key: str | None) -> list[str]:
    mapping = {
        "sans-bold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "DejaVuSans-Bold.ttf",
        ],
        "sans": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "DejaVuSans.ttf",
        ],
        "serif-semibold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "C:/Windows/Fonts/georgiab.ttf",
            "C:/Windows/Fonts/georgia.ttf",
        ],
        "condensed-bold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ],
    }
    defaults = mapping["sans-bold"] + mapping["sans"]
    return mapping.get(str(font_key or ""), []) + defaults


def _load_font(size: int, font_key: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _font_candidates(font_key)
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _font_key_from_family(font_family: str, weight: str) -> str:
    lowered = font_family.lower()
    if "serif" in lowered:
        return "serif-semibold"
    if "condensed" in lowered:
        return "condensed-bold"
    if weight == "semibold":
        return "sans"
    return "sans-bold"


def _sanitize_text(value: str) -> str:
    return " ".join(value.split())


def _display_name(name: str, case_style: str) -> str:
    cleaned = _sanitize_text(name)
    if case_style == "upper":
        return cleaned.upper()
    if case_style == "lower":
        return cleaned.lower()
    return cleaned


def _palette_by_name(name: str) -> tuple[str, dict[str, str]]:
    entry = dict(PALETTES_BY_NAME.get(name, PALETTES_BY_NAME["harbor"]))
    palette_name = entry.pop("name")
    return palette_name, entry


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


def _connected_components(mask: list[list[int]]) -> int:
    height = len(mask)
    width = len(mask[0]) if height else 0
    seen: set[tuple[int, int]] = set()
    components = 0
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or (x, y) in seen:
                continue
            components += 1
            stack = [(x, y)]
            seen.add((x, y))
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if not mask[ny][nx] or (nx, ny) in seen:
                        continue
                    seen.add((nx, ny))
                    stack.append((nx, ny))
    return components


def _score_peak(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / tolerance)


def _evaluate_mark_quality(mark_image: Image.Image) -> dict[str, object]:
    alpha = mark_image.convert("RGBA").getchannel("A")
    thumb = alpha.resize((96, 96), Image.Resampling.LANCZOS)
    pixels = list(thumb.getdata())
    active = sum(1 for value in pixels if value >= 36)
    fill_ratio = active / float(len(pixels) or 1)
    bbox = thumb.point(lambda value: 255 if value >= 36 else 0).getbbox()
    if bbox is None:
        return {
            "total": 0.0,
            "recommended": False,
            "tier": "weak",
            "fillRatio": 0.0,
            "componentCount": 0,
            "tinyFillRatio": 0.0,
            "edgeRatio": 0.0,
            "notes": ["empty silhouette"],
        }

    left, top, right, bottom = bbox
    width = max(1, right - left)
    height = max(1, bottom - top)
    margin_ratio = min(left, top, 96 - right, 96 - bottom) / 96.0
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    center_offset = (((center_x - 48.0) ** 2 + (center_y - 48.0) ** 2) ** 0.5) / 48.0

    small = alpha.resize((40, 40), Image.Resampling.LANCZOS)
    small_mask = [
        [1 if small.getpixel((x, y)) >= 42 else 0 for x in range(40)]
        for y in range(40)
    ]
    component_count = _connected_components(small_mask)

    tiny = alpha.resize((20, 20), Image.Resampling.LANCZOS)
    tiny_pixels = list(tiny.getdata())
    tiny_fill_ratio = sum(1 for value in tiny_pixels if value >= 44) / float(len(tiny_pixels) or 1)
    edges = thumb.filter(ImageFilter.FIND_EDGES)
    edge_pixels = list(edges.getdata())
    edge_ratio = sum(1 for value in edge_pixels if value >= 24) / float(len(edge_pixels) or 1)

    fill_score = _score_peak(fill_ratio, 0.24, 0.18)
    margin_score = _score_peak(margin_ratio, 0.11, 0.10)
    center_score = max(0.0, 1.0 - center_offset / 0.45)
    component_score = 1.0 if 1 <= component_count <= 3 else 0.75 if component_count == 4 else 0.4 if component_count == 5 else 0.1
    tiny_score = _score_peak(tiny_fill_ratio, 0.22, 0.16)
    edge_score = _score_peak(edge_ratio, 0.12, 0.10)
    total = round(
        (
            fill_score * 0.22
            + margin_score * 0.16
            + center_score * 0.14
            + component_score * 0.18
            + tiny_score * 0.16
            + edge_score * 0.14
        )
        * 100.0,
        1,
    )
    notes: list[str] = []
    if fill_score >= 0.75:
        notes.append("good fill balance")
    if tiny_score >= 0.7:
        notes.append("reads at small size")
    if component_count <= 3:
        notes.append("simple silhouette")
    if margin_ratio < 0.05:
        notes.append("too close to edge")
    if component_count >= 5:
        notes.append("too fragmented")
    if not notes:
        notes.append("needs review")
    tier = "strong" if total >= 78 else "promising" if total >= 64 else "weak"
    recommended = total >= 72 and component_count <= 4 and 0.10 <= fill_ratio <= 0.45
    return {
        "total": total,
        "recommended": recommended,
        "tier": tier,
        "fillRatio": round(fill_ratio, 3),
        "componentCount": component_count,
        "tinyFillRatio": round(tiny_fill_ratio, 3),
        "edgeRatio": round(edge_ratio, 3),
        "notes": notes[:3],
    }


def _select_direction(rng: random.Random, requested_direction: str) -> str:
    return requested_direction if requested_direction != "broad exploration" else rng.choice(DIRECTION_LIBRARY)


def _select_style(
    *,
    rng: random.Random,
    requested_direction: str,
    source_candidate: CandidateRecord | None,
    backend: str,
    variant_index: int,
    exploration_preset_id: str | None,
) -> tuple[str, str, dict[str, str], str, str]:
    if source_candidate is not None:
        return (
            requested_direction if requested_direction != "broad exploration" else source_candidate.direction,
            source_candidate.palette_name,
            dict(source_candidate.palette),
            source_candidate.font_family,
            source_candidate.shape_kind,
        )

    if backend == "diffusers":
        palette = dict(PALETTES[variant_index % len(PALETTES)])
        palette_name = palette.pop("name")
        font_family = FONT_FAMILIES[variant_index % len(FONT_FAMILIES)]
        shape_kind = REAL_SHAPE_KINDS[(variant_index // len(PALETTES)) % len(REAL_SHAPE_KINDS)]
        directions = EXPLORATION_DIRECTION_MAP.get(str(exploration_preset_id or ""), DIRECTION_LIBRARY)
        direction = requested_direction if requested_direction != "broad exploration" else directions[variant_index % len(directions)]
        return direction, palette_name, palette, font_family, shape_kind

    palette = dict(rng.choice(PALETTES))
    palette_name = palette.pop("name")
    font_family = rng.choice(FONT_FAMILIES)
    shape_kind = rng.choice(SHAPE_KINDS)
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


def _retain_primary_component(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    width, height = alpha.size
    mask = [[1 if alpha.getpixel((x, y)) >= 42 else 0 for x in range(width)] for y in range(height)]
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            points: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                points.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if not mask[ny][nx] or (nx, ny) in seen:
                        continue
                    seen.add((nx, ny))
                    stack.append((nx, ny))
            components.append(points)

    if len(components) <= 6:
        return rgba

    total_active = sum(len(points) for points in components)
    sizable = [points for points in components if len(points) >= max(120, total_active // 80)]
    if len(sizable) < 6:
        return rgba

    primary = max(sizable, key=len)
    if len(primary) >= total_active * 0.7:
        return rgba

    keep = Image.new("L", (width, height), 0)
    keep_pixels = keep.load()
    for x, y in primary:
        keep_pixels[x, y] = alpha.getpixel((x, y))
    isolated = rgba.copy()
    isolated.putalpha(keep)
    return _trim_transparency(isolated)


def _stylize_generated_mark(mark_image: Image.Image, palette: dict[str, str]) -> Image.Image:
    cleaned = _trim_transparency(_retain_primary_component(_remove_white_background(mark_image)))
    alpha = cleaned.getchannel("A")
    if alpha.getbbox() is None:
        return cleaned

    fill = Image.new("RGBA", cleaned.size, _hex_to_rgba(_color("primary", palette)))
    fill.putalpha(alpha)

    shadow_mask = ImageChops.subtract(alpha.filter(ImageFilter.GaussianBlur(8)), alpha)
    shadow_mask = shadow_mask.point(lambda value: min(54, value))
    shadow = Image.new("RGBA", cleaned.size, _hex_to_rgba(_color("secondary", palette)))
    shadow.putalpha(shadow_mask)

    styled = Image.new("RGBA", cleaned.size, (0, 0, 0, 0))
    styled.alpha_composite(shadow)
    styled.alpha_composite(fill)

    return _trim_transparency(styled)


def _draw_mark(image: Image.Image, shape_kind: str, palette: dict[str, str], initials: str) -> None:
    draw = ImageDraw.Draw(image)
    primary = _color("primary", palette)
    secondary = _color("secondary", palette)
    accent = _color("accent", palette)
    light_bg = _color("light_bg", palette)

    if shape_kind == "link":
        draw.rounded_rectangle((92, 176, 286, 370), radius=92, outline=primary, width=34)
        draw.rounded_rectangle((226, 142, 420, 336), radius=92, outline=secondary, width=34)
        draw.rounded_rectangle((202, 218, 308, 324), radius=54, fill=accent)
    elif shape_kind == "bridge":
        draw.rounded_rectangle((86, 302, 426, 344), radius=20, fill=primary)
        draw.rounded_rectangle((122, 258, 164, 344), radius=14, fill=primary)
        draw.rounded_rectangle((348, 258, 390, 344), radius=14, fill=primary)
        draw.arc((118, 118, 394, 394), start=198, end=342, fill=secondary, width=30)
        draw.ellipse((226, 166, 286, 226), fill=accent)
    elif shape_kind == "grid":
        size = 108
        cells = [
            (96, 96, primary),
            (224, 96, secondary),
            (96, 224, secondary),
            (224, 224, primary),
        ]
        for x, y, fill in cells:
            draw.rounded_rectangle((x, y, x + size, y + size), radius=28, fill=fill)
        draw.polygon([(292, 96), (416, 96), (416, 220), (380, 220), (292, 132)], fill=accent)
        draw.polygon([(96, 292), (132, 292), (220, 380), (220, 416), (96, 416)], fill=accent)
    elif shape_kind == "path":
        draw.line((112, 338, 210, 240, 290, 274, 396, 160), fill=primary, width=34, joint="curve")
        draw.line((116, 170, 212, 170, 266, 224), fill=secondary, width=26, joint="curve")
        draw.polygon([(356, 160), (428, 160), (392, 104)], fill=accent)
        draw.ellipse((86, 312, 144, 370), fill=secondary)
    elif shape_kind == "signal":
        bars = [(110, 284, 44, primary), (180, 220, 52, secondary), (260, 162, 60, primary), (350, 112, 66, accent)]
        for x, y, width, fill in bars:
            draw.rounded_rectangle((x, y, x + width, 392), radius=22, fill=fill)
    elif shape_kind == "frame":
        draw.rounded_rectangle((90, 90, 422, 422), radius=88, outline=primary, width=34)
        draw.line((320, 92, 422, 92, 422, 194), fill=accent, width=34)
        draw.line((90, 322, 90, 422, 190, 422), fill=secondary, width=34)
    elif shape_kind == "fold":
        draw.polygon([(116, 156), (244, 88), (382, 156), (312, 226), (244, 188), (176, 226)], fill=primary)
        draw.polygon([(176, 226), (244, 188), (312, 226), (244, 424)], fill=secondary)
        draw.polygon([(244, 88), (312, 156), (244, 188), (176, 156)], fill=accent)
    elif shape_kind == "orbit":
        draw.ellipse((86, 118, 426, 394), outline=primary, width=28)
        draw.arc((104, 160, 408, 364), start=200, end=340, fill=secondary, width=26)
        draw.ellipse((332, 116, 404, 188), fill=accent)
    else:
        draw.rounded_rectangle((88, 88, 424, 424), radius=100, fill=primary)
        draw.rounded_rectangle((124, 124, 388, 388), radius=72, outline=secondary, width=18)
        font = _load_font(172)
        bbox = draw.textbbox((0, 0), initials, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((256 - text_width / 2, 256 - text_height / 2 - 10), initials, fill=light_bg, font=font)


def _mark_svg_body(shape_kind: str, palette: dict[str, str], initials: str) -> str:
    primary = _color("primary", palette)
    secondary = _color("secondary", palette)
    accent = _color("accent", palette)
    light_bg = _color("light_bg", palette)
    if shape_kind == "link":
        return f'''\n  <rect x="92" y="176" width="194" height="194" rx="92" fill="none" stroke="{primary}" stroke-width="34" />\n  <rect x="226" y="142" width="194" height="194" rx="92" fill="none" stroke="{secondary}" stroke-width="34" />\n  <rect x="202" y="218" width="106" height="106" rx="54" fill="{accent}" />\n'''
    if shape_kind == "bridge":
        return f'''\n  <rect x="86" y="302" width="340" height="42" rx="20" fill="{primary}" />\n  <rect x="122" y="258" width="42" height="86" rx="14" fill="{primary}" />\n  <rect x="348" y="258" width="42" height="86" rx="14" fill="{primary}" />\n  <path d="M132 292 A124 124 0 0 1 380 292" fill="none" stroke="{secondary}" stroke-width="30" stroke-linecap="round" />\n  <circle cx="256" cy="196" r="30" fill="{accent}" />\n'''
    if shape_kind == "grid":
        return f'''\n  <rect x="96" y="96" width="108" height="108" rx="28" fill="{primary}" />\n  <rect x="224" y="96" width="108" height="108" rx="28" fill="{secondary}" />\n  <rect x="96" y="224" width="108" height="108" rx="28" fill="{secondary}" />\n  <rect x="224" y="224" width="108" height="108" rx="28" fill="{primary}" />\n  <path d="M292 96 H416 V220 H380 L292 132 Z" fill="{accent}" />\n  <path d="M96 292 H132 L220 380 V416 H96 Z" fill="{accent}" />\n'''
    if shape_kind == "path":
        return f'''\n  <path d="M112 338 L210 240 L290 274 L396 160" fill="none" stroke="{primary}" stroke-width="34" stroke-linecap="round" stroke-linejoin="round" />\n  <path d="M116 170 H212 L266 224" fill="none" stroke="{secondary}" stroke-width="26" stroke-linecap="round" stroke-linejoin="round" />\n  <path d="M356 160 H428 L392 104 Z" fill="{accent}" />\n  <circle cx="115" cy="341" r="29" fill="{secondary}" />\n'''
    if shape_kind == "signal":
        return f'''\n  <rect x="110" y="284" width="44" height="108" rx="22" fill="{primary}" />\n  <rect x="180" y="220" width="52" height="172" rx="22" fill="{secondary}" />\n  <rect x="260" y="162" width="60" height="230" rx="24" fill="{primary}" />\n  <rect x="350" y="112" width="66" height="280" rx="26" fill="{accent}" />\n'''
    if shape_kind == "frame":
        return f'''\n  <rect x="90" y="90" width="332" height="332" rx="88" fill="none" stroke="{primary}" stroke-width="34" />\n  <path d="M320 92 H422 V194" fill="none" stroke="{accent}" stroke-width="34" stroke-linecap="round" stroke-linejoin="round" />\n  <path d="M90 322 V422 H190" fill="none" stroke="{secondary}" stroke-width="34" stroke-linecap="round" stroke-linejoin="round" />\n'''
    if shape_kind == "fold":
        return f'''\n  <path d="M116 156 L244 88 L382 156 L312 226 L244 188 L176 226 Z" fill="{primary}" />\n  <path d="M176 226 L244 188 L312 226 L244 424 Z" fill="{secondary}" />\n  <path d="M244 88 L312 156 L244 188 L176 156 Z" fill="{accent}" />\n'''
    if shape_kind == "orbit":
        return f'''\n  <ellipse cx="256" cy="256" rx="170" ry="138" fill="none" stroke="{primary}" stroke-width="28" />\n  <path d="M120 248 A152 102 0 0 1 388 248" fill="none" stroke="{secondary}" stroke-width="26" stroke-linecap="round" />\n  <circle cx="368" cy="152" r="36" fill="{accent}" />\n'''
    return f'''\n  <rect x="88" y="88" width="336" height="336" rx="100" fill="{primary}" />\n  <rect x="124" y="124" width="264" height="264" rx="72" fill="none" stroke="{secondary}" stroke-width="18" />\n  <text x="256" y="314" text-anchor="middle" font-size="172" font-weight="700" fill="{light_bg}" font-family="DejaVu Sans, Segoe UI, sans-serif">{initials}</text>\n'''


def _build_mark_svg(shape_kind: str, palette: dict[str, str], initials: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">\n{_mark_svg_body(shape_kind, palette, initials)}\n</svg>\n'''


def _resolve_tone_style(
    *,
    tone_preset: str,
    font_family: str,
    wordmark_case: str,
    wordmark_weight: str,
) -> dict[str, object]:
    preset_id = tone_preset if tone_preset in TONE_PRESETS else "candidate-auto"
    preset = TONE_PRESETS[preset_id]
    resolved_font_family = str(preset["font_family"] or font_family)
    resolved_case = str(preset["wordmark_case"] or wordmark_case)
    resolved_weight = str(preset["wordmark_weight"] or wordmark_weight)
    resolved_font_key = str(preset["font_key"] or _font_key_from_family(resolved_font_family, resolved_weight))
    return {
        "preset_id": preset_id,
        "label": str(preset["label"]),
        "font_family": resolved_font_family,
        "font_key": resolved_font_key,
        "wordmark_case": resolved_case,
        "wordmark_weight": resolved_weight,
        "tracking": int(preset["tracking"]),
        "surface_theme": str(preset["surface_theme"]),
    }


def _surface_colors(palette: dict[str, str], tone_style: dict[str, object], *, dark: bool) -> dict[str, str]:
    theme = str(tone_style["surface_theme"])
    if theme == "editorial":
        return {
            "background": "#F3EFE8" if not dark else "#191715",
            "card_fill": "#FFFDF8" if not dark else "#221F1B",
            "card_outline": "#DED5C9" if not dark else "#3A332B",
            "label": "#4F4338" if not dark else "#F0E7DD",
        }
    if theme == "premium":
        return {
            "background": "#F8F2E8" if not dark else "#0F1114",
            "card_fill": "#FFFDF9" if not dark else "#161A20",
            "card_outline": "#D8C2A3" if not dark else "#3A4656",
            "label": "#473526" if not dark else "#F4E8D8",
        }
    if theme == "friendly":
        return {
            "background": "#F3FAFF" if not dark else "#122331",
            "card_fill": "#FFFFFF" if not dark else "#193141",
            "card_outline": "#C8DBEA" if not dark else "#2D5168",
            "label": "#274357" if not dark else "#E5F4FF",
        }
    if theme == "product":
        return {
            "background": "#F5F7FA" if not dark else "#0F1721",
            "card_fill": "#FFFFFF" if not dark else "#162230",
            "card_outline": "#D9E2EC" if not dark else "#2A3A4A",
            "label": "#1F2933" if not dark else "#F2F7FB",
        }
    return {
        "background": _color("dark_bg", palette) if dark else _color("light_bg", palette),
        "card_fill": "#132238" if dark else "#FFFFFF",
        "card_outline": "#26384B" if dark else "#D9E2EC",
        "label": _color("light_text", palette) if dark else _color("dark_text", palette),
    }


def _tracked_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | ImageFont.FreeTypeFont, tracking: int) -> tuple[int, int]:
    if tracking <= 0:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    width = 0
    height = 0
    for index, character in enumerate(text):
        bbox = draw.textbbox((0, 0), character, font=font)
        width += bbox[2] - bbox[0]
        height = max(height, bbox[3] - bbox[1])
        if index < len(text) - 1:
            width += tracking
    return width, height


def _draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: str,
    tracking: int,
) -> tuple[int, int]:
    if tracking <= 0:
        draw.text((x, y), text, font=font, fill=fill)
        return _tracked_text_size(draw, text, font, 0)
    cursor = x
    max_height = 0
    for character in text:
        bbox = draw.textbbox((0, 0), character, font=font)
        draw.text((cursor, y), character, font=font, fill=fill)
        cursor += (bbox[2] - bbox[0]) + tracking
        max_height = max(max_height, bbox[3] - bbox[1])
    return int(cursor - x - max(0, tracking)), max_height


def _fit_wordmark_font(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font_key: str,
    base_size: int,
    tracking: int,
    max_width: int,
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    size = base_size
    while size >= 74:
        font = _load_font(size, font_key=font_key)
        width, _ = _tracked_text_size(draw, text, font, tracking)
        if width <= max_width:
            return font
        size -= 6
    return _load_font(74, font_key=font_key)


def _draw_wordmark(
    image: Image.Image,
    name: str,
    palette: dict[str, str],
    font_family: str,
    wordmark_case: str,
    wordmark_weight: str,
    *,
    tone_preset: str,
) -> None:
    draw = ImageDraw.Draw(image)
    tone_style = _resolve_tone_style(
        tone_preset=tone_preset,
        font_family=font_family,
        wordmark_case=wordmark_case,
        wordmark_weight=wordmark_weight,
    )
    font_key = str(tone_style["font_key"])
    tracking = int(tone_style["tracking"])
    text = _display_name(name, str(tone_style["wordmark_case"]))
    font = _fit_wordmark_font(
        draw,
        text=text,
        font_key=font_key,
        base_size=132 if str(tone_style["wordmark_weight"]) == "bold" else 122,
        tracking=tracking,
        max_width=image.size[0] - 24,
    )
    text_color = _color("dark_text", palette)
    text_width, text_height = _tracked_text_size(draw, text, font, tracking)
    x = 12
    y = (image.size[1] - text_height) / 2 - 8
    _draw_tracked_text(draw, x=x, y=y, text=text, font=font, fill=text_color, tracking=tracking)
    underline_width = min(text_width, 440)
    draw.rounded_rectangle((x, y + text_height + 18, x + underline_width, y + text_height + 26), radius=4, fill=_color("accent", palette))


def _build_wordmark_svg(
    name: str,
    palette: dict[str, str],
    font_family: str,
    wordmark_case: str,
    wordmark_weight: str,
    *,
    tone_preset: str,
) -> str:
    tone_style = _resolve_tone_style(
        tone_preset=tone_preset,
        font_family=font_family,
        wordmark_case=wordmark_case,
        wordmark_weight=wordmark_weight,
    )
    text = _display_name(name, str(tone_style["wordmark_case"]))
    font_weight = "700" if str(tone_style["wordmark_weight"]) == "bold" else "600"
    tracking = int(tone_style["tracking"])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="260" viewBox="0 0 1080 260">\n  <text x="12" y="142" font-size="132" font-weight="{font_weight}" fill="{_color("dark_text", palette)}" font-family="{font_family}, Segoe UI, sans-serif">{text}</text>\n  <rect x="12" y="188" width="440" height="8" rx="4" fill="{_color("accent", palette)}" />\n</svg>\n'''


def _compose_lockup(mark_image: Image.Image, wordmark_image: Image.Image, layout: str) -> Image.Image:
    if layout == "stacked":
        canvas = Image.new("RGBA", (980, 860), (0, 0, 0, 0))
        canvas.alpha_composite(mark_image.resize((360, 360)), (310, 40))
        canvas.alpha_composite(wordmark_image.resize((1000, 260)), (40, 520))
        return _trim_transparency(canvas)

    canvas = Image.new("RGBA", (1320, 420), (0, 0, 0, 0))
    canvas.alpha_composite(mark_image.resize((300, 300)), (20, 44))
    canvas.alpha_composite(wordmark_image.resize((1080, 260)), (360, 92))
    return _trim_transparency(canvas)


def _build_lockup_svg(
    *,
    product_name: str,
    palette: dict[str, str],
    font_family: str,
    wordmark_case: str,
    wordmark_weight: str,
    shape_kind: str,
    initials: str,
    layout: str,
    tone_preset: str,
) -> str:
    tone_style = _resolve_tone_style(
        tone_preset=tone_preset,
        font_family=font_family,
        wordmark_case=wordmark_case,
        wordmark_weight=wordmark_weight,
    )
    text = _display_name(product_name, str(tone_style["wordmark_case"]))
    font_weight = "700" if str(tone_style["wordmark_weight"]) == "bold" else "600"
    resolved_font_family = str(tone_style["font_family"])
    if layout == "stacked":
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1100" viewBox="0 0 1200 1100">\n  <g transform="translate(344 64) scale(0.742)">\n{_mark_svg_body(shape_kind, palette, initials)}\n  </g>\n  <text x="150" y="792" font-size="132" font-weight="{font_weight}" fill="{_color("dark_text", palette)}" font-family="{resolved_font_family}, Segoe UI, sans-serif">{text}</text>\n  <rect x="150" y="842" width="440" height="8" rx="4" fill="{_color("accent", palette)}" />\n</svg>\n'''
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="720" viewBox="0 0 1500 720">\n  <g transform="translate(120 194) scale(0.648)">\n{_mark_svg_body(shape_kind, palette, initials)}\n  </g>\n  <text x="478" y="372" font-size="132" font-weight="{font_weight}" fill="{_color("dark_text", palette)}" font-family="{resolved_font_family}, Segoe UI, sans-serif">{text}</text>\n  <rect x="478" y="422" width="440" height="8" rx="4" fill="{_color("accent", palette)}" />\n</svg>\n'''


def _build_lockup_svg_from_mark_body(
    *,
    product_name: str,
    palette: dict[str, str],
    font_family: str,
    wordmark_case: str,
    wordmark_weight: str,
    mark_body: str,
    layout: str,
    tone_preset: str,
) -> str:
    tone_style = _resolve_tone_style(
        tone_preset=tone_preset,
        font_family=font_family,
        wordmark_case=wordmark_case,
        wordmark_weight=wordmark_weight,
    )
    text = _display_name(product_name, str(tone_style["wordmark_case"]))
    font_weight = "700" if str(tone_style["wordmark_weight"]) == "bold" else "600"
    resolved_font_family = str(tone_style["font_family"])
    if layout == "stacked":
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1100" viewBox="0 0 1200 1100">\n  <g transform="translate(344 64) scale(0.742)">\n{mark_body}\n  </g>\n  <text x="150" y="792" font-size="132" font-weight="{font_weight}" fill="{_color("dark_text", palette)}" font-family="{resolved_font_family}, Segoe UI, sans-serif">{text}</text>\n  <rect x="150" y="842" width="440" height="8" rx="4" fill="{_color("accent", palette)}" />\n</svg>\n'''
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="720" viewBox="0 0 1500 720">\n  <g transform="translate(120 194) scale(0.648)">\n{mark_body}\n  </g>\n  <text x="478" y="372" font-size="132" font-weight="{font_weight}" fill="{_color("dark_text", palette)}" font-family="{resolved_font_family}, Segoe UI, sans-serif">{text}</text>\n  <rect x="478" y="422" width="440" height="8" rx="4" fill="{_color("accent", palette)}" />\n</svg>\n'''


def _draw_preview(
    lockup_horizontal: Image.Image,
    lockup_stacked: Image.Image,
    palette: dict[str, str],
    target_path: Path,
    *,
    dark: bool,
    tone_style: dict[str, object],
) -> None:
    surface = _surface_colors(palette, tone_style, dark=dark)
    canvas = Image.new("RGBA", (1500, 1100), surface["background"])
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((90, 92, 1410, 1010), radius=36, fill=surface["card_fill"], outline=surface["card_outline"], width=2)
    horizontal = lockup_horizontal.copy()
    horizontal.thumbnail((980, 340), Image.Resampling.LANCZOS)
    stacked = lockup_stacked.copy()
    stacked.thumbnail((480, 300), Image.Resampling.LANCZOS)
    canvas.alpha_composite(horizontal, (200, 170))
    canvas.alpha_composite(stacked, (510, 620))
    canvas.save(target_path)


def _draw_app_icon(mark_image: Image.Image, palette: dict[str, str], target_path: Path, *, dark: bool, tone_style: dict[str, object]) -> None:
    surface = _surface_colors(palette, tone_style, dark=dark)
    canvas = Image.new("RGBA", (1024, 1024), surface["background"])
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((112, 112, 912, 912), radius=208, fill=surface["card_fill"], outline=surface["card_outline"], width=3)
    icon = mark_image.copy()
    icon.thumbnail((520, 520), Image.Resampling.LANCZOS)
    canvas.alpha_composite(icon, ((1024 - icon.width) // 2, (1024 - icon.height) // 2))
    canvas.save(target_path)


def _build_favicon_strip(mark_image: Image.Image, palette: dict[str, str], target_path: Path, *, tone_style: dict[str, object]) -> None:
    surface = _surface_colors(palette, tone_style, dark=False)
    canvas = Image.new("RGBA", (1400, 280), surface["background"])
    draw = ImageDraw.Draw(canvas)
    sizes = [128, 64, 32, 16]
    labels_font = _load_font(24, font_key="sans")
    x = 64
    for size in sizes:
        tile_size = 180 if size >= 64 else 144
        draw.rounded_rectangle((x, 46, x + tile_size, 46 + tile_size), radius=28, fill=surface["card_fill"], outline=surface["card_outline"], width=2)
        icon = mark_image.copy()
        icon.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas.alpha_composite(icon, (x + (tile_size - icon.width) // 2, 46 + (tile_size - icon.height) // 2))
        draw.text((x, 240), f"{size}px", font=labels_font, fill=surface["label"])
        x += tile_size + 72
    canvas.save(target_path)


def _build_brand_board(
    *,
    preview_light_path: Path,
    preview_dark_path: Path,
    app_icon_light_path: Path,
    app_icon_dark_path: Path,
    favicon_strip_path: Path,
    target_path: Path,
    candidate_title: str,
    tone_label: str,
) -> None:
    canvas = Image.new("RGBA", (1920, 1320), "#EEF2F6")
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(44, font_key="sans-bold")
    meta_font = _load_font(26, font_key="sans")
    draw.text((84, 56), candidate_title, font=header_font, fill="#182430")
    draw.text((84, 116), f"Materialized tone: {tone_label}", font=meta_font, fill="#52606D")

    tiles = [
        (preview_light_path, (84, 192), (820, 560)),
        (preview_dark_path, (1016, 192), (820, 560)),
        (app_icon_light_path, (84, 824), (400, 400)),
        (app_icon_dark_path, (516, 824), (400, 400)),
        (favicon_strip_path, (948, 824), (888, 400)),
    ]
    for image_path, origin, max_size in tiles:
        x, y = origin
        width, height = max_size
        draw.rounded_rectangle((x, y, x + width, y + height), radius=28, fill="white", outline="#D9E2EC", width=2)
        asset = Image.open(image_path).convert("RGBA")
        asset.thumbnail((width - 40, height - 40), Image.Resampling.LANCZOS)
        canvas.alpha_composite(asset, (x + (width - asset.width) // 2, y + (height - asset.height) // 2))

    canvas.save(target_path)


def _build_tone_review_board(
    *,
    target_path: Path,
    candidate_title: str,
    variations: list[dict[str, object]],
) -> None:
    columns = 2
    card_width = 860
    card_height = 560
    rows = max(1, (len(variations) + columns - 1) // columns)
    canvas = Image.new("RGBA", (columns * card_width + 180, rows * card_height + 260), "#EAF0F6")
    draw = ImageDraw.Draw(canvas)
    header_font = _load_font(46, font_key="sans-bold")
    meta_font = _load_font(26, font_key="sans")
    title_font = _load_font(28, font_key="sans-bold")
    draw.text((84, 52), candidate_title, font=header_font, fill="#182430")
    draw.text((84, 112), "Tone review pack", font=meta_font, fill="#52606D")

    for index, variation in enumerate(variations):
        row = index // columns
        column = index % columns
        x = 84 + column * card_width
        y = 176 + row * card_height
        draw.rounded_rectangle((x, y, x + 792, y + 492), radius=30, fill="white", outline="#D9E2EC", width=2)
        draw.text((x + 28, y + 24), str(variation["tone_label"]), font=title_font, fill="#1F2933")
        preview = Image.open(str(variation["preview_light_png"])).convert("RGBA")
        preview.thumbnail((744, 210), Image.Resampling.LANCZOS)
        canvas.alpha_composite(preview, (x + 24 + (744 - preview.width) // 2, y + 72))
        preview_dark = Image.open(str(variation["preview_dark_png"])).convert("RGBA")
        preview_dark.thumbnail((744, 132), Image.Resampling.LANCZOS)
        canvas.alpha_composite(preview_dark, (x + 24 + (744 - preview_dark.width) // 2, y + 292))
        favicon = Image.open(str(variation["favicon_strip_png"])).convert("RGBA")
        favicon.thumbnail((744, 96), Image.Resampling.LANCZOS)
        canvas.alpha_composite(favicon, (x + 24 + (744 - favicon.width) // 2, y + 430))
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
    wordmark_case: str = "title",
    wordmark_weight: str = "bold",
    tone_preset: str = "candidate-auto",
    spec_payload: dict[str, Any] | None = None,
) -> dict[str, str]:
    del seed, direction
    output_dir.mkdir(parents=True, exist_ok=True)
    tone_style = _resolve_tone_style(
        tone_preset=tone_preset,
        font_family=font_family,
        wordmark_case=wordmark_case,
        wordmark_weight=wordmark_weight,
    )

    mark_canvas = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    raw_mark_path = output_dir / "mark_raw.png"
    if mark_image is None:
        _draw_mark(mark_canvas, shape_kind, palette, initials)
    else:
        mark_image.convert("RGBA").save(raw_mark_path)
        mark_canvas = _fit_mark_to_canvas(_stylize_generated_mark(mark_image, palette))
    mark_path = output_dir / "mark.png"
    mark_canvas.save(mark_path)
    mark_svg_path = output_dir / "mark.svg"
    if mark_image is None:
        mark_svg_path.write_text(_build_mark_svg(shape_kind, palette, initials), encoding="utf-8")
    mark_vector_svg_text, vector_report = vectorize_mark_to_svg(mark_canvas, fill_color=_color("primary", palette))
    mark_vector_svg_path = output_dir / "mark_vector.svg"
    mark_vector_svg_path.write_text(mark_vector_svg_text, encoding="utf-8")
    vector_report_path = output_dir / "mark_vector_report.json"
    vector_report_path.write_text(json.dumps(vector_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mark_vector_body = svg_inner_content(mark_vector_svg_text)

    wordmark_image = Image.new("RGBA", (1080, 260), (0, 0, 0, 0))
    _draw_wordmark(
        wordmark_image,
        product_name,
        palette,
        font_family,
        wordmark_case,
        wordmark_weight,
        tone_preset=tone_preset,
    )
    wordmark_path = output_dir / "wordmark.png"
    wordmark_image.save(wordmark_path)
    wordmark_svg_path = output_dir / "wordmark.svg"
    wordmark_svg_path.write_text(
        _build_wordmark_svg(
            product_name,
            palette,
            str(tone_style["font_family"]),
            str(tone_style["wordmark_case"]),
            str(tone_style["wordmark_weight"]),
            tone_preset=tone_preset,
        ),
        encoding="utf-8",
    )

    horizontal = _compose_lockup(mark_canvas, wordmark_image, layout="horizontal")
    horizontal_path = output_dir / "lockup_horizontal.png"
    horizontal.save(horizontal_path)
    horizontal_svg_path = output_dir / "lockup_horizontal.svg"
    horizontal_svg_path.write_text(
        _build_lockup_svg(
            product_name=product_name,
            palette=palette,
            font_family=str(tone_style["font_family"]),
            wordmark_case=str(tone_style["wordmark_case"]),
            wordmark_weight=str(tone_style["wordmark_weight"]),
            shape_kind=shape_kind,
            initials=initials,
            layout="horizontal",
            tone_preset=tone_preset,
        ),
        encoding="utf-8",
    )
    horizontal_vector_svg_path = output_dir / "lockup_horizontal_vector.svg"
    horizontal_vector_svg_path.write_text(
        _build_lockup_svg_from_mark_body(
            product_name=product_name,
            palette=palette,
            font_family=str(tone_style["font_family"]),
            wordmark_case=str(tone_style["wordmark_case"]),
            wordmark_weight=str(tone_style["wordmark_weight"]),
            mark_body=mark_vector_body,
            layout="horizontal",
            tone_preset=tone_preset,
        ),
        encoding="utf-8",
    )

    stacked = _compose_lockup(mark_canvas, wordmark_image, layout="stacked")
    stacked_path = output_dir / "lockup_stacked.png"
    stacked.save(stacked_path)
    stacked_svg_path = output_dir / "lockup_stacked.svg"
    stacked_svg_path.write_text(
        _build_lockup_svg(
            product_name=product_name,
            palette=palette,
            font_family=str(tone_style["font_family"]),
            wordmark_case=str(tone_style["wordmark_case"]),
            wordmark_weight=str(tone_style["wordmark_weight"]),
            shape_kind=shape_kind,
            initials=initials,
            layout="stacked",
            tone_preset=tone_preset,
        ),
        encoding="utf-8",
    )
    stacked_vector_svg_path = output_dir / "lockup_stacked_vector.svg"
    stacked_vector_svg_path.write_text(
        _build_lockup_svg_from_mark_body(
            product_name=product_name,
            palette=palette,
            font_family=str(tone_style["font_family"]),
            wordmark_case=str(tone_style["wordmark_case"]),
            wordmark_weight=str(tone_style["wordmark_weight"]),
            mark_body=mark_vector_body,
            layout="stacked",
            tone_preset=tone_preset,
        ),
        encoding="utf-8",
    )

    preview_light_path = output_dir / "preview_light.png"
    preview_dark_path = output_dir / "preview_dark.png"
    _draw_preview(horizontal, stacked, palette, preview_light_path, dark=False, tone_style=tone_style)
    _draw_preview(horizontal, stacked, palette, preview_dark_path, dark=True, tone_style=tone_style)
    app_icon_light_path = output_dir / "app_icon_light.png"
    app_icon_dark_path = output_dir / "app_icon_dark.png"
    _draw_app_icon(mark_canvas, palette, app_icon_light_path, dark=False, tone_style=tone_style)
    _draw_app_icon(mark_canvas, palette, app_icon_dark_path, dark=True, tone_style=tone_style)
    favicon_strip_path = output_dir / "favicon_strip.png"
    _build_favicon_strip(mark_canvas, palette, favicon_strip_path, tone_style=tone_style)
    brand_board_path = output_dir / "brand_board.png"
    _build_brand_board(
        preview_light_path=preview_light_path,
        preview_dark_path=preview_dark_path,
        app_icon_light_path=app_icon_light_path,
        app_icon_dark_path=app_icon_dark_path,
        favicon_strip_path=favicon_strip_path,
        target_path=brand_board_path,
        candidate_title=product_name,
        tone_label=str(tone_style["label"]),
    )

    if spec_payload is not None:
        (output_dir / "spec.json").write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assets = {
        "mark_png": str(mark_path),
        "wordmark_png": str(wordmark_path),
        "wordmark_svg": str(wordmark_svg_path),
        "mark_vector_svg": str(mark_vector_svg_path),
        "vector_report_json": str(vector_report_path),
        "lockup_horizontal_png": str(horizontal_path),
        "lockup_horizontal_svg": str(horizontal_svg_path),
        "lockup_horizontal_vector_svg": str(horizontal_vector_svg_path),
        "lockup_stacked_png": str(stacked_path),
        "lockup_stacked_svg": str(stacked_svg_path),
        "lockup_stacked_vector_svg": str(stacked_vector_svg_path),
        "preview_light_png": str(preview_light_path),
        "preview_dark_png": str(preview_dark_path),
        "app_icon_light_png": str(app_icon_light_path),
        "app_icon_dark_png": str(app_icon_dark_path),
        "favicon_strip_png": str(favicon_strip_path),
        "brand_board_png": str(brand_board_path),
    }
    if mark_svg_path.exists():
        assets["mark_svg"] = str(mark_svg_path)
    if raw_mark_path.exists():
        assets["mark_raw_png"] = str(raw_mark_path)
    if spec_payload is not None:
        assets["spec_json"] = str(output_dir / "spec.json")
    return assets


def _generate_diffusers_mark(
    *,
    store: ToolForLogoStore,
    case_record,
    direction: str,
    palette: dict[str, str],
    shape_kind: str,
    index: int,
    source_candidate: CandidateRecord | None,
    seed: int,
    generation_options: dict[str, object] | None,
) -> tuple[Image.Image, dict[str, object]]:
    exploration_preset_id = str(generation_options.get("exploration_preset")) if generation_options and generation_options.get("exploration_preset") else None
    exploration_preset = get_exploration_preset(exploration_preset_id)
    profile = resolve_generation_profile(
        load_settings(),
        load_worker_capabilities(),
        preset_id=str(generation_options.get("preset_id")) if generation_options and generation_options.get("preset_id") else None,
        family="image",
    )
    backend = DiffusersLogoBackend(
        logs_root=store.logs_root,
        profile=profile,
        token=load_huggingface_token(),
    )
    generated = backend.generate_mark(
        product_name=case_record.product_name,
        description=case_record.description,
        direction=direction,
        palette=palette,
        shape_kind=shape_kind,
        variant_index=index,
        source_title=source_candidate.title if source_candidate is not None else None,
        exploration_preset_id=exploration_preset.preset_id,
        seed=seed,
    )
    return generated.image, {
        "backend": "diffusers",
        "exploration_label": generated.exploration_label,
        "exploration_preset_id": exploration_preset.preset_id,
        "exploration_preset_label": generated.preset_label or exploration_preset.label,
        "request_prompt": generated.request_prompt,
        "revised_prompt": generated.revised_prompt,
        **profile,
        "seed": seed,
    }


def _generate_local_svg_spec(
    *,
    store: ToolForLogoStore,
    case_record,
    requested_direction: str,
    index: int,
    source_candidate: CandidateRecord | None,
    generation_options: dict[str, object] | None,
) -> tuple[str, str, dict[str, str], str, str, str, str, str, str, dict[str, object]]:
    profile = resolve_generation_profile(
        load_settings(),
        load_worker_capabilities(),
        preset_id=str(generation_options.get("preset_id")) if generation_options and generation_options.get("preset_id") else None,
        family="concept",
    )
    backend = LocalConceptBackend(
        logs_root=store.logs_root,
        profile=profile,
        token=load_huggingface_token(),
    )
    draft = backend.generate_spec(
        product_name=case_record.product_name,
        description=case_record.description,
        notes=case_record.notes,
        requested_direction=requested_direction if requested_direction != "broad exploration" else "",
        source_summary=(f"{source_candidate.direction} / {source_candidate.shape_kind} / {source_candidate.palette_name}" if source_candidate is not None else None),
        variant_index=index,
        choices={
            "directions": DIRECTION_LIBRARY,
            "palettes": [item["name"] for item in PALETTES],
            "fonts": FONT_FAMILIES,
            "shapes": SHAPE_KINDS,
        },
    )
    direction = requested_direction if requested_direction != "broad exploration" else draft.spec["direction"]
    palette_name, palette = _palette_by_name(draft.spec["palette_name"])
    font_family = draft.spec["font_family"]
    shape_kind = draft.spec["shape_kind"]
    wordmark_case = draft.spec["wordmark_case"]
    wordmark_weight = draft.spec["weight"]
    motif = draft.spec["motif"]
    rationale = draft.spec["rationale"] or RATIONALE_TEMPLATE.format(direction=direction, shape_kind=shape_kind)
    generation = {
        "backend": "local-svg",
        "request_prompt": draft.request_prompt,
        "raw_response": draft.raw_response,
        "spec": {
            **draft.spec,
            "direction": direction,
            "palette_name": palette_name,
            "font_family": font_family,
            "shape_kind": shape_kind,
            "wordmark_case": wordmark_case,
            "weight": wordmark_weight,
            "motif": motif,
            "rationale": rationale,
        },
        **profile,
    }
    return direction, palette_name, palette, font_family, shape_kind, wordmark_case, wordmark_weight, motif, rationale, generation


def generate_batch(
    store: ToolForLogoStore,
    *,
    case_id: str,
    count: int,
    direction_hint: str,
    seed: int | None = None,
    backend: str = "local-svg",
    source_candidate_id: str | None = None,
    generation_options: dict[str, object] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    if backend not in {"mock", "diffusers", "local-svg"}:
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
    exploration_preset_id = str(generation_options.get("exploration_preset")) if generation_options and generation_options.get("exploration_preset") else "balanced-saas"

    if progress_callback is not None:
        progress_callback(0, count, "initializing", "Preparing logo generation batch.")

    candidates: list[CandidateRecord] = []
    for index in range(count):
        candidate_id = store._new_id("candidate")
        candidate_seed = batch_seed + (index * 97)
        candidate_root = store.candidate_root(case_id, candidate_id)
        assets_dir = candidate_root / "assets"
        generation: dict[str, object] = {"backend": backend}
        mark_image: Image.Image | None = None
        wordmark_case = "title"
        wordmark_weight = "bold"
        motif = "mark"

        if progress_callback is not None:
            progress_callback(index, count, "generating", f"Generating candidate {index + 1} of {count}.")

        if backend == "local-svg":
            direction, palette_name, palette, font_family, shape_kind, wordmark_case, wordmark_weight, motif, rationale, generation = _generate_local_svg_spec(
                store=store,
                case_record=case_record,
                requested_direction=requested_direction,
                index=index,
                source_candidate=source_candidate,
                generation_options=generation_options,
            )
        else:
            direction, palette_name, palette, font_family, shape_kind = _select_style(
                rng=rng,
                requested_direction=requested_direction,
                source_candidate=source_candidate,
                backend=backend,
                variant_index=index,
                exploration_preset_id=exploration_preset_id,
            )
            rationale = RATIONALE_TEMPLATE.format(direction=direction, shape_kind=shape_kind)
            if backend == "diffusers":
                mark_image, generation = _generate_diffusers_mark(
                    store=store,
                    case_record=case_record,
                    direction=direction,
                    palette=palette,
                    shape_kind=shape_kind,
                    index=index,
                    source_candidate=source_candidate,
                    seed=candidate_seed,
                    generation_options=generation_options,
                )
                exploration_label = str(generation.get("exploration_label") or "Concept")
                preset_label = str(generation.get("exploration_preset_label") or "Balanced SaaS")
                rationale = f"{IMAGE_EXPLORATION_RATIONALES[index % len(IMAGE_EXPLORATION_RATIONALES)]} Preset: {preset_label}."
            if source_candidate is not None:
                rationale = f"Derived from {source_candidate.title}. {RATIONALE_TEMPLATE.format(direction=direction, shape_kind=shape_kind)}"

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
            wordmark_case=wordmark_case,
            wordmark_weight=wordmark_weight,
            spec_payload=generation.get("spec") if isinstance(generation.get("spec"), dict) else None,
        )
        quality = _evaluate_mark_quality(Image.open(assets["mark_png"]).convert("RGBA"))
        generation["quality"] = quality
        if backend == "local-svg":
            title_suffix = motif.title()
        elif backend == "diffusers":
            title_suffix = str(generation.get("exploration_label") or "Concept")
        else:
            title_suffix = "Concept"
        record = CandidateRecord(
            candidate_id=candidate_id,
            case_id=case_id,
            batch_id=batch.batch_id,
            title=f"{case_record.product_name} {title_suffix} {index + 1:02d}",
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
        if progress_callback is not None:
            progress_callback(index + 1, count, "rendered", f"Rendered candidate {index + 1} of {count}.")

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
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    return generate_batch(
        store,
        case_id=case_id,
        count=count,
        direction_hint=direction_hint,
        seed=seed,
        backend="mock",
        source_candidate_id=source_candidate_id,
        progress_callback=progress_callback,
    )


def _build_comparison_sheet(*, export_dir: Path, candidate_cards: list[tuple[CandidateRecord, dict[str, str]]]) -> str:
    card_width = 620
    card_height = 560
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
        draw.rounded_rectangle((x, y, x + 580, y + 500), radius=28, fill="white", outline="#D9E2EC", width=2)
        preview_path = assets.get("brand_board_png") or assets["preview_light_png"]
        preview = Image.open(preview_path).convert("RGBA")
        preview.thumbnail((540, 360), Image.Resampling.LANCZOS)
        canvas.alpha_composite(preview, (x + (580 - preview.width) // 2, y + 24))
        draw.text((x + 24, y + 404), candidate.title, font=title_font, fill="#1F2933")
        tone_label = str(assets.get("tone_label") or "Candidate Auto")
        draw.text((x + 24, y + 444), f"{candidate.direction} / {tone_label}", font=meta_font, fill="#486581")
    target = export_dir / "comparison_sheet.png"
    canvas.save(target)
    return str(target)


def _resolve_export_candidates(store: ToolForLogoStore, case_id: str, candidate_ids: Iterable[str] | None) -> list[CandidateRecord]:
    if candidate_ids:
        return [store.get_candidate(case_id, candidate_id) for candidate_id in candidate_ids]

    candidates = store.list_candidates(case_id)
    favorites = [item for item in candidates if item.status in {CandidateStatus.FAVORITE, CandidateStatus.ADOPTED}]
    if favorites:
        return sorted(favorites, key=lambda item: float((item.generation.get("quality") or {}).get("total") or 0.0), reverse=True)
    return sorted(candidates, key=lambda item: float((item.generation.get("quality") or {}).get("total") or 0.0), reverse=True)[:3]


def _render_export_variation(
    *,
    destination: Path,
    candidate: CandidateRecord,
    rendered_name: str,
    spec: dict[str, Any] | None,
    existing_mark: Image.Image | None,
    tone_preset: str,
) -> dict[str, object]:
    assets = render_candidate_assets(
        output_dir=destination,
        product_name=rendered_name,
        direction=candidate.direction,
        palette=candidate.palette,
        font_family=candidate.font_family,
        shape_kind=candidate.shape_kind,
        initials=_initials_for_name(rendered_name),
        seed=candidate.seed,
        mark_image=existing_mark,
        wordmark_case=str((spec or {}).get("wordmark_case") or "title"),
        wordmark_weight=str((spec or {}).get("weight") or "bold"),
        tone_preset=tone_preset,
        spec_payload=spec,
    )
    tone_style = _resolve_tone_style(
        tone_preset=tone_preset,
        font_family=candidate.font_family,
        wordmark_case=str((spec or {}).get("wordmark_case") or "title"),
        wordmark_weight=str((spec or {}).get("weight") or "bold"),
    )
    assets["tone_label"] = str(tone_style["label"])
    return {
        "tone_preset": tone_preset,
        "tone_label": str(tone_style["label"]),
        "assets": assets,
        "preview_light_png": assets["preview_light_png"],
        "preview_dark_png": assets["preview_dark_png"],
        "favicon_strip_png": assets["favicon_strip_png"],
    }


def create_export_bundle(
    store: ToolForLogoStore,
    *,
    case_id: str,
    candidate_ids: list[str] | None = None,
    name_override: str | None = None,
    tone_preset: str = "candidate-auto",
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
    tone_style = _resolve_tone_style(
        tone_preset=tone_preset,
        font_family=selected[0].font_family,
        wordmark_case="title",
        wordmark_weight="bold",
    )

    for candidate in selected:
        destination = export_dir / candidate.candidate_id
        rendered_name = name_override or case_record.product_name
        spec = candidate.generation.get("spec") if isinstance(candidate.generation.get("spec"), dict) else None
        existing_mark = _load_existing_mark(candidate.assets.get("mark_png")) if candidate.generation.get("backend") == "diffusers" else None
        primary_variation = _render_export_variation(
            destination=destination / "assets",
            candidate=candidate,
            rendered_name=rendered_name,
            spec=spec,
            existing_mark=existing_mark,
            tone_preset=tone_preset,
        )
        assets = dict(primary_variation["assets"])
        tone_variants: list[dict[str, object]] = []
        for variant_tone in REFINEMENT_TONE_PRESETS:
            variant = _render_export_variation(
                destination=destination / "tone_variants" / variant_tone / "assets",
                candidate=candidate,
                rendered_name=rendered_name,
                spec=spec,
                existing_mark=existing_mark.copy() if existing_mark is not None else None,
                tone_preset=variant_tone,
            )
            tone_variants.append(variant)
        tone_review_path = destination / "tone_review.png"
        _build_tone_review_board(
            target_path=tone_review_path,
            candidate_title=candidate.title,
            variations=tone_variants,
        )
        assets["tone_review_png"] = str(tone_review_path)
        manifest_candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "title": candidate.title,
                "direction": candidate.direction,
                "status": candidate.status.value,
                "quality": candidate.generation.get("quality"),
                "tone_preset": tone_preset,
                "tone_label": str(tone_style["label"]),
                "assets": assets,
                "tone_variants": tone_variants,
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
        "tone_preset": tone_preset,
        "tone_label": str(tone_style["label"]),
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
        tone_preset=tone_preset,
        tone_label=str(tone_style["label"]),
    )
    store.save_export(export_record)
    return export_record
