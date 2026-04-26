from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_for_logo.generator import (
    create_export_bundle,
    generate_batch,
    generate_mock_batch,
    render_candidate_assets,
)
from tool_for_logo.models import CandidateStatus
from tool_for_logo.state import ToolForLogoStore


def test_mock_batch_generation_creates_candidate_assets(tmp_path: Path) -> None:
    store = ToolForLogoStore(
        state_root=tmp_path / "state",
        report_root=tmp_path / "reports",
        archive_root=tmp_path / "archive",
    )
    case_record = store.create_case(
        product_name="Northwind Atlas",
        description="Global logistics orchestration platform",
    )

    payload = generate_mock_batch(
        store,
        case_id=case_record.case_id,
        count=4,
        direction_hint="premium editorial",
        seed=42,
    )

    assert payload["batch"]["candidate_count"] == 4
    candidates = store.list_candidates(case_record.case_id)
    assert len(candidates) == 4
    assert candidates[0].direction == "premium editorial"
    assert Path(candidates[0].assets["preview_light_png"]).exists()
    assert Path(candidates[0].assets["mark_svg"]).exists()


def test_export_bundle_renders_name_override(tmp_path: Path) -> None:
    store = ToolForLogoStore(
        state_root=tmp_path / "state",
        report_root=tmp_path / "reports",
        archive_root=tmp_path / "archive",
    )
    case_record = store.create_case(
        product_name="Dockyard Metrics",
        description="Usage intelligence for developer tools",
    )
    payload = generate_mock_batch(
        store,
        case_id=case_record.case_id,
        count=2,
        direction_hint="bold utility tooling",
        seed=10,
    )
    candidate_id = payload["candidates"][0]["candidate_id"]
    store.update_candidate_status(case_record.case_id, candidate_id, CandidateStatus.FAVORITE)

    export_record = create_export_bundle(
        store,
        case_id=case_record.case_id,
        name_override="Dockyard Pulse",
        tone_preset="editorial",
    )

    manifest_path = Path(export_record.export_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name_override"] == "Dockyard Pulse"
    assert manifest["tone_preset"] == "editorial"
    assert manifest["tone_label"] == "Editorial"
    assert Path(manifest["comparison_sheet_png"]).exists()
    assert manifest["candidates"][0]["candidate_id"] == candidate_id
    assert Path(manifest["candidates"][0]["assets"]["brand_board_png"]).exists()
    assert Path(manifest["candidates"][0]["assets"]["app_icon_light_png"]).exists()
    assert Path(manifest["candidates"][0]["assets"]["favicon_strip_png"]).exists()
    assert Path(manifest["candidates"][0]["assets"]["tone_review_png"]).exists()
    assert len(manifest["candidates"][0]["tone_variants"]) == 5
    assert Path(export_record.archive_path).exists()
    assert export_record.tone_preset == "editorial"
    assert export_record.tone_label == "Editorial"


def test_generate_batch_can_derive_from_selected_candidate(tmp_path: Path) -> None:
    store = ToolForLogoStore(
        state_root=tmp_path / "state",
        report_root=tmp_path / "reports",
        archive_root=tmp_path / "archive",
    )
    case_record = store.create_case(
        product_name="Vector Harbor",
        description="Private logo exploration workspace",
    )
    initial = generate_mock_batch(
        store,
        case_id=case_record.case_id,
        count=2,
        direction_hint="friendly and rounded",
        seed=5,
    )
    source_candidate_id = initial["candidates"][0]["candidate_id"]
    store.update_candidate_status(case_record.case_id, source_candidate_id, CandidateStatus.FAVORITE)

    derived = generate_batch(
        store,
        case_id=case_record.case_id,
        count=3,
        direction_hint="",
        seed=7,
        backend="mock",
        source_candidate_id=source_candidate_id,
    )

    assert derived["batch"]["source_candidate_id"] == source_candidate_id
    assert len(derived["candidates"]) == 3
    assert all(item["direction"] == initial["candidates"][0]["direction"] for item in derived["candidates"])
    assert all("Derived from" in item["rationale"] for item in derived["candidates"])


def test_render_candidate_assets_stylizes_external_mark(tmp_path: Path) -> None:
    source = Image.new("RGBA", (320, 320), "white")
    draw = ImageDraw.Draw(source)
    draw.rounded_rectangle((48, 72, 272, 248), radius=72, fill="black")

    output_dir = tmp_path / "candidate-assets"
    assets = render_candidate_assets(
        output_dir=output_dir,
        product_name="ToolForLogo",
        direction="bold utility tooling",
        palette={
            "primary": "#125B9A",
            "secondary": "#6A9AB0",
            "accent": "#FFD166",
            "light_bg": "#F8FBFF",
            "dark_bg": "#0B1726",
            "light_text": "#F6F9FD",
            "dark_text": "#12263A",
        },
        font_family="DejaVu Sans",
        shape_kind="orbit",
        initials="TL",
        seed=11,
        mark_image=source,
    )

    assert Path(assets["mark_png"]).exists()
    assert Path(assets["mark_raw_png"]).exists()
    assert Path(assets["mark_vector_svg"]).exists()
    assert Path(assets["vector_report_json"]).exists()
    assert Path(assets["lockup_horizontal_vector_svg"]).exists()
    assert Path(assets["lockup_stacked_vector_svg"]).exists()
    assert "mark_svg" not in assets
