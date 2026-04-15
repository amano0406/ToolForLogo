from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .generator import create_export_bundle, generate_batch
from .models import CandidateStatus
from .runtime import default_backend
from .state import ToolForLogoStore


class ToolForLogoHandler(BaseHTTPRequestHandler):
    server_version = "ToolForLogo/0.1"

    @property
    def store(self) -> ToolForLogoStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _not_found(self) -> None:
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def _bad_request(self, message: str) -> None:
        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": message})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        segments = [segment for segment in parsed.path.split("/") if segment]

        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if parsed.path == "/":
            self._send_json(
                HTTPStatus.OK,
                {
                    "service": "ToolForLogo",
                    "default_backend": default_backend(),
                    "endpoints": [
                        "/health",
                        "/api/status",
                        "/api/cases",
                    ],
                },
            )
            return

        if parsed.path == "/api/status":
            self._send_json(HTTPStatus.OK, self.store.status_payload())
            return

        if parsed.path == "/api/cases":
            payload = {"cases": [case.to_dict() for case in self.store.list_cases()]}
            self._send_json(HTTPStatus.OK, payload)
            return

        if len(segments) == 3 and segments[:2] == ["api", "cases"]:
            case_id = segments[2]
            case_record = self.store.get_case(case_id)
            candidates = [candidate.to_dict() for candidate in self.store.list_candidates(case_id)]
            batches = [batch.to_dict() for batch in self.store.list_batches(case_id)]
            exports = [item.to_dict() for item in self.store.list_exports(case_id)]
            self._send_json(
                HTTPStatus.OK,
                {
                    "case": case_record.to_dict(),
                    "batches": batches,
                    "candidates": candidates,
                    "exports": exports,
                },
            )
            return

        self._not_found()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        segments = [segment for segment in parsed.path.split("/") if segment]
        payload = self._read_json()

        if parsed.path == "/api/cases":
            name = str(payload.get("name", "")).strip()
            description = str(payload.get("description", "")).strip()
            notes = str(payload.get("notes", "")).strip()
            if not name or not description:
                self._bad_request("name and description are required")
                return
            case_record = self.store.create_case(name, description, notes=notes)
            self._send_json(HTTPStatus.CREATED, {"case": case_record.to_dict()})
            return

        if len(segments) == 4 and segments[:2] == ["api", "cases"] and segments[3] == "batches":
            case_id = segments[2]
            count = int(payload.get("count", 20))
            direction_hint = str(payload.get("direction_hint", "")).strip()
            seed = payload.get("seed")
            try:
                result = generate_batch(
                    self.store,
                    case_id=case_id,
                    count=count,
                    direction_hint=direction_hint,
                    seed=int(seed) if seed is not None else None,
                    backend=str(payload.get("backend", default_backend())).strip() or default_backend(),
                    source_candidate_id=str(payload.get("source_candidate_id", "")).strip() or None,
                )
            except (RuntimeError, ValueError) as error:
                self._bad_request(str(error))
                return
            self._send_json(HTTPStatus.CREATED, result)
            return

        if (
            len(segments) == 6
            and segments[:2] == ["api", "cases"]
            and segments[3] == "candidates"
            and segments[5] == "status"
        ):
            case_id = segments[2]
            candidate_id = segments[4]
            status_value = str(payload.get("status", "")).strip()
            try:
                status = CandidateStatus(status_value)
            except ValueError:
                self._bad_request(f"invalid status: {status_value}")
                return
            candidate = self.store.update_candidate_status(case_id, candidate_id, status)
            self._send_json(HTTPStatus.OK, {"candidate": candidate.to_dict()})
            return

        if len(segments) == 4 and segments[:2] == ["api", "cases"] and segments[3] == "exports":
            case_id = segments[2]
            candidate_ids = payload.get("candidate_ids")
            if candidate_ids is not None and not isinstance(candidate_ids, list):
                self._bad_request("candidate_ids must be a list when provided")
                return
            try:
                export_record = create_export_bundle(
                    self.store,
                    case_id=case_id,
                    candidate_ids=candidate_ids,
                    name_override=str(payload.get("name_override", "")).strip() or None,
                )
            except ValueError as error:
                self._bad_request(str(error))
                return
            self._send_json(HTTPStatus.CREATED, {"export": export_record.to_dict()})
            return

        self._not_found()


class ToolForLogoHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: ToolForLogoStore) -> None:
        super().__init__(server_address, ToolForLogoHandler)
        self.store = store


def serve(host: str, port: int, store: ToolForLogoStore) -> None:
    httpd = ToolForLogoHTTPServer((host, port), store)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
