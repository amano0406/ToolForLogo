from __future__ import annotations

import math
from typing import Iterable

from PIL import Image


def _threshold_mask(image: Image.Image, size: int = 128, threshold: int = 40) -> list[list[int]]:
    alpha = image.convert("RGBA").getchannel("A").resize((size, size), Image.Resampling.LANCZOS)
    return [
        [1 if alpha.getpixel((x, y)) >= threshold else 0 for x in range(size)]
        for y in range(size)
    ]


def _component_masks(mask: list[list[int]], *, max_components: int = 3, min_pixels: int = 20) -> list[list[list[int]]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
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
            if len(points) >= min_pixels:
                components.append(points)

    components.sort(key=len, reverse=True)
    result: list[list[list[int]]] = []
    for points in components[:max_components]:
        component = [[0 for _ in range(width)] for _ in range(height)]
        for x, y in points:
            component[y][x] = 1
        result.append(component)
    return result


def _marching_segments(mask: list[list[int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    height = len(mask)
    width = len(mask[0]) if height else 0
    padded = [[0 for _ in range(width + 2)] for _ in range(height + 2)]
    for y in range(height):
        for x in range(width):
            padded[y + 1][x + 1] = mask[y][x]

    lookup: dict[int, list[tuple[str, str]]] = {
        1: [("left", "bottom")],
        2: [("bottom", "right")],
        3: [("left", "right")],
        4: [("top", "right")],
        5: [("top", "left"), ("bottom", "right")],
        6: [("top", "bottom")],
        7: [("top", "left")],
        8: [("top", "left")],
        9: [("top", "bottom")],
        10: [("top", "right"), ("left", "bottom")],
        11: [("top", "right")],
        12: [("left", "right")],
        13: [("bottom", "right")],
        14: [("left", "bottom")],
    }

    def point(x: int, y: int, edge: str) -> tuple[int, int]:
        if edge == "top":
            return (2 * x + 1, 2 * y)
        if edge == "right":
            return (2 * x + 2, 2 * y + 1)
        if edge == "bottom":
            return (2 * x + 1, 2 * y + 2)
        return (2 * x, 2 * y + 1)

    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for y in range(height + 1):
        for x in range(width + 1):
            tl = padded[y][x]
            tr = padded[y][x + 1]
            br = padded[y + 1][x + 1]
            bl = padded[y + 1][x]
            case = (tl << 3) | (tr << 2) | (br << 1) | bl
            for edge_a, edge_b in lookup.get(case, []):
                segments.append((point(x, y, edge_a), point(x, y, edge_b)))
    return segments


def _normalize_edge(start: tuple[int, int], end: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (start, end) if start <= end else (end, start)


def _segments_to_loops(segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start, end in segments:
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    loops: list[list[tuple[int, int]]] = []
    for start, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = _normalize_edge(start, neighbor)
            if edge in visited:
                continue
            loop = [start]
            previous = start
            current = neighbor
            visited.add(edge)
            while True:
                loop.append(current)
                next_options = [item for item in adjacency.get(current, []) if item != previous]
                if not next_options:
                    break
                next_point = next_options[0]
                next_edge = _normalize_edge(current, next_point)
                if next_edge in visited:
                    if next_point == loop[0]:
                        loop.append(next_point)
                    break
                visited.add(next_edge)
                previous, current = current, next_point
                if current == loop[0]:
                    loop.append(current)
                    break
            if len(loop) >= 4 and loop[0] == loop[-1]:
                loops.append(loop[:-1])
    return loops


def _distance_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return math.dist(point, start)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
    projection = (sx + t * dx, sy + t * dy)
    return math.dist(point, projection)


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    max_distance = -1.0
    index = 0
    for position in range(1, len(points) - 1):
        distance = _distance_to_segment(points[position], start, end)
        if distance > max_distance:
            max_distance = distance
            index = position
    if max_distance <= epsilon:
        return [start, end]
    left = _rdp(points[: index + 1], epsilon)
    right = _rdp(points[index:], epsilon)
    return left[:-1] + right


def _snap_points(points: Iterable[tuple[float, float]], grid: float) -> list[tuple[float, float]]:
    snapped: list[tuple[float, float]] = []
    for x, y in points:
        point = (round(x / grid) * grid, round(y / grid) * grid)
        if not snapped or snapped[-1] != point:
            snapped.append(point)
    if len(snapped) >= 2 and snapped[0] == snapped[-1]:
        snapped.pop()
    return snapped


def _scale_loop(loop: list[tuple[int, int]], *, size: int, canvas_size: int) -> list[tuple[float, float]]:
    scaled: list[tuple[float, float]] = []
    scale = canvas_size / float(size)
    for x, y in loop:
        scaled.append(((x / 2.0 - 1.0) * scale, (y / 2.0 - 1.0) * scale))
    return scaled


def _closed_rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) < 4:
        return points
    return _rdp(points + [points[0]], epsilon)[:-1]


def _catmull_rom_path(points: list[tuple[float, float]]) -> str:
    if len(points) < 3:
        commands = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
        for x, y in points[1:]:
            commands.append(f"L {x:.1f} {y:.1f}")
        commands.append("Z")
        return " ".join(commands)
    commands = [f"M {points[0][0]:.1f} {points[0][1]:.1f}"]
    total = len(points)
    for index in range(total):
        p0 = points[(index - 1) % total]
        p1 = points[index % total]
        p2 = points[(index + 1) % total]
        p3 = points[(index + 2) % total]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        commands.append(
            f"C {c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}"
        )
    commands.append("Z")
    return " ".join(commands)


def svg_inner_content(svg: str) -> str:
    start = svg.find(">")
    end = svg.rfind("</svg>")
    return svg[start + 1 : end].strip()


def vectorize_mark_to_svg(
    mark_image: Image.Image,
    *,
    fill_color: str,
    canvas_size: int = 512,
    grid: float = 2.0,
) -> tuple[str, dict[str, object]]:
    size = 128
    mask = _threshold_mask(mark_image, size=size)
    components = _component_masks(mask)
    paths: list[str] = []
    anchor_total = 0
    raw_total = 0
    for component in components:
        loops = _segments_to_loops(_marching_segments(component))
        for loop in loops:
            scaled = _scale_loop(loop, size=size, canvas_size=canvas_size)
            simplified = _closed_rdp(scaled, epsilon=3.2)
            snapped = _snap_points(simplified, grid=grid)
            if len(snapped) < 3:
                continue
            raw_total += len(loop)
            anchor_total += len(snapped)
            paths.append(_catmull_rom_path(snapped))

    if not paths:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" '
            f'viewBox="0 0 {canvas_size} {canvas_size}">\n'
            f'  <rect width="{canvas_size}" height="{canvas_size}" fill="none" />\n'
            "</svg>\n"
        )
        return svg, {
            "status": "empty",
            "recommended": False,
            "contours": 0,
            "anchors": 0,
            "grid": grid,
            "definition": [
                "closed silhouette",
                "limited anchor count",
                "grid-snapped coordinates",
                "smooth cubic curves",
            ],
            "notes": ["No usable contour found."],
        }

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" '
        f'viewBox="0 0 {canvas_size} {canvas_size}">\n'
        f'  <path d="{" ".join(paths)}" fill="{fill_color}" fill-rule="evenodd" />\n'
        "</svg>\n"
    )
    recommended = len(paths) <= 3 and 8 <= anchor_total <= 72
    notes: list[str] = []
    if len(paths) <= 3:
        notes.append("limited closed contours")
    if anchor_total <= 72:
        notes.append("anchor count stays compact")
    if anchor_total < raw_total:
        notes.append("simplified from raster boundary")
    notes.append("coordinates snapped to a 2px grid")
    return svg, {
        "status": "ok",
        "recommended": recommended,
        "contours": len(paths),
        "anchors": anchor_total,
        "rawPoints": raw_total,
        "grid": grid,
        "definition": [
            "closed silhouette",
            "limited anchor count",
            "grid-snapped coordinates",
            "smooth cubic curves",
        ],
        "notes": notes,
    }
