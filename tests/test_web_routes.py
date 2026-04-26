from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_for_logo.generator import generate_mock_batch
from tool_for_logo.models import CandidateStatus
from tool_for_logo.state import ToolForLogoStore
from tool_for_logo.web_app import app


def _configure_env(tmp_path: Path) -> None:
    os.environ["TOOL_FOR_LOGO_APPDATA_ROOT"] = str(tmp_path / "app-data")
    os.environ["TOOL_FOR_LOGO_UPLOADS_ROOT"] = str(tmp_path / "uploads")
    os.environ["TOOL_FOR_LOGO_OUTPUTS_ROOT"] = str(tmp_path / "outputs")
    os.environ["TOOL_FOR_LOGO_REPORT_ROOT"] = str(tmp_path / "reports")
    os.environ["TOOL_FOR_LOGO_ARCHIVE_ROOT"] = str(tmp_path / "archive")
    os.environ["TOOL_FOR_LOGO_HF_CACHE_ROOT"] = str(tmp_path / "cache" / "huggingface")
    os.environ["TOOL_FOR_LOGO_TORCH_CACHE_ROOT"] = str(tmp_path / "cache" / "torch")
    os.environ["TOOL_FOR_LOGO_RUNTIME_DEFAULTS"] = str(PROJECT_ROOT / "configs" / "runtime.defaults.json")
    os.environ["TOOL_FOR_LOGO_DEFAULT_BACKEND"] = "mock"


def test_web_routes_render_basic_pages(tmp_path: Path) -> None:
    _configure_env(tmp_path)
    store = ToolForLogoStore.from_env()
    case_record = store.create_case("ToolForLogo", "Local-first logo exploration workspace")
    generate_mock_batch(
        store,
        case_id=case_record.case_id,
        count=2,
        direction_hint="bold utility tooling",
        seed=12,
    )

    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json == {"status": "ok"}

    settings_page = client.get("/settings")
    assert settings_page.status_code == 200
    assert b"Worker and model settings" in settings_page.data
    assert b"GPU only" in settings_page.data

    cases_page = client.get("/cases")
    assert cases_page.status_code == 200
    assert b"ToolForLogo" in cases_page.data

    case_page = client.get(f"/cases/{case_record.case_id}")
    assert case_page.status_code == 200
    assert case_record.case_id.encode("utf-8") in case_page.data
    assert b"Select all" in case_page.data
    assert b"Tone preset" in case_page.data

    api_case = client.get(f"/api/cases/{case_record.case_id}")
    assert api_case.status_code == 200
    assert api_case.json["case"]["case_id"] == case_record.case_id


def test_bulk_candidate_status_update_route(tmp_path: Path) -> None:
    _configure_env(tmp_path)
    store = ToolForLogoStore.from_env()
    case_record = store.create_case("ToolForLogo", "Local-first logo exploration workspace")
    payload = generate_mock_batch(
        store,
        case_id=case_record.case_id,
        count=3,
        direction_hint="bold utility tooling",
        seed=21,
    )
    candidate_ids = [item["candidate_id"] for item in payload["candidates"][:2]]

    client = app.test_client()
    response = client.post(
        f"/cases/{case_record.case_id}/candidates/bulk-status",
        data={"candidate_ids": candidate_ids, "status": "favorite"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    updated = [store.get_candidate(case_record.case_id, candidate_id) for candidate_id in candidate_ids]
    assert all(candidate.status == CandidateStatus.FAVORITE for candidate in updated)


def test_export_route_accepts_tone_preset(tmp_path: Path) -> None:
    _configure_env(tmp_path)
    store = ToolForLogoStore.from_env()
    case_record = store.create_case("ToolForLogo", "Local-first logo exploration workspace")
    payload = generate_mock_batch(
        store,
        case_id=case_record.case_id,
        count=2,
        direction_hint="bold utility tooling",
        seed=31,
    )
    selected_candidate_id = payload["candidates"][0]["candidate_id"]

    client = app.test_client()
    response = client.post(
        f"/cases/{case_record.case_id}/exports",
        data={
            "candidate_ids": [selected_candidate_id],
            "name_override": "Tool For Logo",
            "tone_preset": "premium",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    exports = store.list_exports(case_record.case_id)
    assert len(exports) == 1
    assert exports[0].tone_preset == "premium"
    assert exports[0].tone_label == "Premium"
