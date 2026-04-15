from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class CandidateStatus(StrEnum):
    FRESH = "fresh"
    FAVORITE = "favorite"
    EXCLUDED = "excluded"
    ADOPTED = "adopted"


@dataclass(slots=True)
class ProjectCase:
    case_id: str
    product_name: str
    description: str
    notes: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectCase":
        return cls(**payload)


@dataclass(slots=True)
class BatchRecord:
    batch_id: str
    case_id: str
    backend: str
    direction_hint: str
    candidate_count: int
    created_at: str
    seed: int | None
    source_candidate_id: str | None = None
    candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchRecord":
        return cls(**payload)


@dataclass(slots=True)
class CandidateRecord:
    candidate_id: str
    case_id: str
    batch_id: str
    title: str
    direction: str
    rationale: str
    palette_name: str
    palette: dict[str, str]
    font_family: str
    shape_kind: str
    seed: int
    initials: str
    status: CandidateStatus
    assets: dict[str, str]
    created_at: str
    updated_at: str
    generation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateRecord":
        raw = dict(payload)
        raw["status"] = CandidateStatus(raw["status"])
        return cls(**raw)


@dataclass(slots=True)
class ExportRecord:
    export_id: str
    case_id: str
    name_override: str | None
    candidate_ids: list[str]
    export_dir: str
    archive_path: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExportRecord":
        return cls(**payload)
