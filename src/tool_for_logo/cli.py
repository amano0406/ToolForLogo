from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .generator import create_export_bundle, generate_batch
from .models import CandidateStatus
from .runtime import default_backend, load_local_env
from .server import serve
from .state import ToolForLogoStore


def _print(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if "case" in payload and "batch" not in payload:
        case_record = payload["case"]
        print(f"Created case {case_record['case_id']} for {case_record['product_name']}")
        return

    if "batch" in payload:
        batch = payload["batch"]
        print(f"Generated {batch['candidate_count']} {batch['backend']} candidates in {batch['batch_id']}")
        return

    if "export" in payload:
        export = payload["export"]
        print(f"Created export {export['export_id']} at {export['archive_path']}")
        return

    if "candidate" in payload:
        candidate = payload["candidate"]
        print(f"Updated {candidate['candidate_id']} to {candidate['status']}")
        return

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    backend_default = default_backend()
    parser = argparse.ArgumentParser(description="ToolForLogo CLI")
    parser.add_argument("--json", action="store_true", help="print full JSON payload")

    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="run local HTTP API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8080)

    status_parser = subparsers.add_parser("status", help="show workspace status")
    status_parser.add_argument("--json", action="store_true", help="print full JSON payload")

    create_case_parser = subparsers.add_parser("create-case", help="create a new case")
    create_case_parser.add_argument("--name", required=True)
    create_case_parser.add_argument("--description", required=True)
    create_case_parser.add_argument("--notes", default="")

    generate_parser = subparsers.add_parser("generate-batch", help="generate candidate batch")
    generate_parser.add_argument("--case-id", required=True)
    generate_parser.add_argument("--count", type=int, default=20)
    generate_parser.add_argument("--direction-hint", default="")
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--backend", choices=["mock", "comfyui", "openai"], default=backend_default)
    generate_parser.add_argument("--from-candidate", dest="source_candidate_id")

    update_parser = subparsers.add_parser("set-status", help="update candidate status")
    update_parser.add_argument("--case-id", required=True)
    update_parser.add_argument("--candidate-id", required=True)
    update_parser.add_argument(
        "--status",
        required=True,
        choices=[status.value for status in CandidateStatus],
    )

    export_parser = subparsers.add_parser("export", help="export selected candidates")
    export_parser.add_argument("--case-id", required=True)
    export_parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    export_parser.add_argument("--name-override")

    return parser


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_local_env(project_root)
    parser = build_parser()
    args = parser.parse_args()
    store = ToolForLogoStore.from_env()

    if args.command == "serve":
        serve(args.host, args.port, store)
        return

    if args.command == "status":
        payload = store.status_payload()
        _print(payload, getattr(args, "json", False))
        return

    if args.command == "create-case":
        case_record = store.create_case(args.name, args.description, args.notes)
        _print({"case": case_record.to_dict()}, args.json)
        return

    if args.command == "generate-batch":
        payload = generate_batch(
            store,
            case_id=args.case_id,
            count=args.count,
            direction_hint=args.direction_hint,
            seed=args.seed,
            backend=args.backend,
            source_candidate_id=args.source_candidate_id,
        )
        _print(payload, args.json)
        return

    if args.command == "set-status":
        candidate = store.update_candidate_status(
            args.case_id,
            args.candidate_id,
            CandidateStatus(args.status),
        )
        _print({"candidate": candidate.to_dict()}, args.json)
        return

    if args.command == "export":
        export_record = create_export_bundle(
            store,
            case_id=args.case_id,
            candidate_ids=args.candidate_ids,
            name_override=args.name_override,
        )
        _print({"export": export_record.to_dict()}, args.json)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")
