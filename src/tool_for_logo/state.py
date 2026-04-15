from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import BatchRecord, CandidateRecord, CandidateStatus, ExportRecord, ProjectCase


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_windows_or_wsl_path(windows_path: str, wsl_path: str) -> str:
    return windows_path if os.name == "nt" else wsl_path


class ToolForLogoStore:
    def __init__(self, state_root: Path, report_root: Path, archive_root: Path) -> None:
        self.state_root = state_root
        self.report_root = report_root
        self.archive_root = archive_root
        self.cases_root = self.state_root / "cases"
        self.logs_root = self.state_root / "logs"
        self._ensure_roots()

    @classmethod
    def from_env(cls) -> "ToolForLogoStore":
        state_root = Path(
            os.getenv(
                "TOOL_FOR_LOGO_STATE_ROOT",
                _default_windows_or_wsl_path(
                    r"C:\Codex\workspaces\ToolForLogo\state",
                    "/mnt/c/Codex/workspaces/ToolForLogo/state",
                ),
            )
        )
        report_root = Path(
            os.getenv(
                "TOOL_FOR_LOGO_REPORT_ROOT",
                _default_windows_or_wsl_path(
                    r"C:\Codex\reports\ToolForLogo",
                    "/mnt/c/Codex/reports/ToolForLogo",
                ),
            )
        )
        archive_root = Path(
            os.getenv(
                "TOOL_FOR_LOGO_ARCHIVE_ROOT",
                _default_windows_or_wsl_path(
                    r"C:\Codex\archive\ToolForLogo",
                    "/mnt/c/Codex/archive/ToolForLogo",
                ),
            )
        )
        return cls(state_root=state_root, report_root=report_root, archive_root=archive_root)

    def _ensure_roots(self) -> None:
        for path in (self.cases_root, self.logs_root, self.report_root, self.archive_root):
            path.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _new_id(self, prefix: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{prefix}-{timestamp}-{uuid4().hex[:6]}"

    def case_root(self, case_id: str) -> Path:
        return self.cases_root / case_id

    def case_path(self, case_id: str) -> Path:
        return self.case_root(case_id) / "case.json"

    def batches_root(self, case_id: str) -> Path:
        return self.case_root(case_id) / "batches"

    def batch_path(self, case_id: str, batch_id: str) -> Path:
        return self.batches_root(case_id) / f"{batch_id}.json"

    def candidates_root(self, case_id: str) -> Path:
        return self.case_root(case_id) / "candidates"

    def candidate_root(self, case_id: str, candidate_id: str) -> Path:
        return self.candidates_root(case_id) / candidate_id

    def candidate_path(self, case_id: str, candidate_id: str) -> Path:
        return self.candidate_root(case_id, candidate_id) / "candidate.json"

    def exports_root(self, case_id: str) -> Path:
        return self.case_root(case_id) / "exports"

    def export_state_path(self, case_id: str, export_id: str) -> Path:
        return self.exports_root(case_id) / f"{export_id}.json"

    def create_case(self, product_name: str, description: str, notes: str = "") -> ProjectCase:
        record = ProjectCase(
            case_id=self._new_id("case"),
            product_name=product_name,
            description=description,
            notes=notes,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        self.save_case(record)
        return record

    def save_case(self, case_record: ProjectCase) -> None:
        self._write_json(self.case_path(case_record.case_id), case_record.to_dict())

    def get_case(self, case_id: str) -> ProjectCase:
        return ProjectCase.from_dict(self._read_json(self.case_path(case_id)))

    def list_cases(self) -> list[ProjectCase]:
        paths = sorted(self.cases_root.glob("*/case.json"))
        return [ProjectCase.from_dict(self._read_json(path)) for path in paths]

    def touch_case(self, case_id: str) -> ProjectCase:
        case_record = self.get_case(case_id)
        case_record.updated_at = utc_now()
        self.save_case(case_record)
        return case_record

    def create_batch(
        self,
        case_id: str,
        backend: str,
        direction_hint: str,
        candidate_count: int,
        seed: int | None,
        source_candidate_id: str | None = None,
    ) -> BatchRecord:
        batch = BatchRecord(
            batch_id=self._new_id("batch"),
            case_id=case_id,
            backend=backend,
            direction_hint=direction_hint,
            candidate_count=candidate_count,
            created_at=utc_now(),
            seed=seed,
            source_candidate_id=source_candidate_id,
            candidate_ids=[],
        )
        self.save_batch(batch)
        return batch

    def save_batch(self, batch: BatchRecord) -> None:
        self._write_json(self.batch_path(batch.case_id, batch.batch_id), batch.to_dict())

    def list_batches(self, case_id: str) -> list[BatchRecord]:
        paths = sorted(self.batches_root(case_id).glob("*.json"))
        return [BatchRecord.from_dict(self._read_json(path)) for path in paths]

    def save_candidate(self, candidate: CandidateRecord) -> None:
        self._write_json(self.candidate_path(candidate.case_id, candidate.candidate_id), candidate.to_dict())

    def get_candidate(self, case_id: str, candidate_id: str) -> CandidateRecord:
        return CandidateRecord.from_dict(self._read_json(self.candidate_path(case_id, candidate_id)))

    def list_candidates(self, case_id: str) -> list[CandidateRecord]:
        paths = sorted(self.candidates_root(case_id).glob("*/candidate.json"))
        return [CandidateRecord.from_dict(self._read_json(path)) for path in paths]

    def update_candidate_status(
        self,
        case_id: str,
        candidate_id: str,
        status: CandidateStatus,
    ) -> CandidateRecord:
        candidate = self.get_candidate(case_id, candidate_id)
        candidate.status = status
        candidate.updated_at = utc_now()
        self.save_candidate(candidate)
        self.touch_case(case_id)
        return candidate

    def save_export(self, export_record: ExportRecord) -> None:
        self._write_json(self.export_state_path(export_record.case_id, export_record.export_id), export_record.to_dict())

    def list_exports(self, case_id: str) -> list[ExportRecord]:
        paths = sorted(self.exports_root(case_id).glob("*.json"))
        return [ExportRecord.from_dict(self._read_json(path)) for path in paths]

    def status_payload(self) -> dict[str, Any]:
        cases = self.list_cases()
        payload_cases: list[dict[str, Any]] = []
        total_candidates = 0
        for case_record in cases:
            candidates = self.list_candidates(case_record.case_id)
            total_candidates += len(candidates)
            payload_cases.append(
                {
                    **asdict(case_record),
                    "candidate_count": len(candidates),
                    "favorite_count": sum(1 for item in candidates if item.status == CandidateStatus.FAVORITE),
                    "adopted_count": sum(1 for item in candidates if item.status == CandidateStatus.ADOPTED),
                }
            )
        return {
            "state_root": str(self.state_root),
            "report_root": str(self.report_root),
            "archive_root": str(self.archive_root),
            "case_count": len(cases),
            "candidate_count": total_candidates,
            "cases": payload_cases,
        }
