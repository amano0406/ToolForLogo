from __future__ import annotations

import argparse
import json

from .generator import create_export_bundle, generate_batch
from .job_store import create_job, list_jobs, load_request, load_result, load_status
from .model_catalog import build_model_statuses, cache_snapshot, clear_model_cache, delete_model, download_model
from .models import CandidateStatus
from .runtime import default_backend, load_local_env, project_root
from .settings import load_huggingface_token, save_huggingface_token, save_settings, settings_snapshot
from .state import ToolForLogoStore
from .web_app import app as flask_app
from .worker import process_job, run_daemon, worker_status_payload, write_worker_capabilities


def _print(payload: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ToolForLogo CLI")
    parser.add_argument("--json", action="store_true", help="print full JSON payload")
    subparsers = parser.add_subparsers(dest="command", required=True)

    web_parser = subparsers.add_parser("web", help="run Flask web UI")
    web_parser.add_argument("--host", default="0.0.0.0")
    web_parser.add_argument("--port", type=int, default=8080)

    daemon_parser = subparsers.add_parser("daemon", help="run worker daemon")
    daemon_parser.add_argument("--poll-interval", type=int, default=5)

    status_parser = subparsers.add_parser("status", help="show workspace status")
    status_parser.add_argument("--json", action="store_true")

    create_case_parser = subparsers.add_parser("create-case", help="create a new case")
    create_case_parser.add_argument("--name", required=True)
    create_case_parser.add_argument("--description", required=True)
    create_case_parser.add_argument("--notes", default="")

    generate_parser = subparsers.add_parser("generate-batch", help="generate one batch immediately")
    generate_parser.add_argument("--case-id", required=True)
    generate_parser.add_argument("--count", type=int, default=10)
    generate_parser.add_argument("--direction-hint", default="")
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--backend", choices=["mock", "diffusers", "local-svg"], default=default_backend())
    generate_parser.add_argument("--from-candidate", dest="source_candidate_id")
    generate_parser.add_argument("--preset-id")
    generate_parser.add_argument("--exploration-preset")

    update_parser = subparsers.add_parser("set-status", help="update candidate status")
    update_parser.add_argument("--case-id", required=True)
    update_parser.add_argument("--candidate-id", required=True)
    update_parser.add_argument("--status", required=True, choices=[status.value for status in CandidateStatus])

    export_parser = subparsers.add_parser("export", help="export selected candidates")
    export_parser.add_argument("--case-id", required=True)
    export_parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    export_parser.add_argument("--name-override")

    settings_parser = subparsers.add_parser("settings", help="show or update settings")
    settings_subparsers = settings_parser.add_subparsers(dest="settings_command", required=True)
    settings_status = settings_subparsers.add_parser("status", help="show current settings")
    settings_status.add_argument("--json", action="store_true")
    settings_save = settings_subparsers.add_parser("save", help="save settings")
    settings_save.add_argument("--token")
    settings_save.add_argument("--compute-mode", choices=["cpu", "gpu"])
    settings_save.add_argument("--processing-quality", choices=["standard", "high"])
    settings_save.add_argument("--default-batch-count", type=int)
    settings_save.add_argument("--default-direction-hint")
    settings_save.add_argument("--default-exploration-preset")
    settings_save.add_argument("--allow-auto-model-download", action="store_true")
    settings_save.add_argument("--json", action="store_true")

    models_parser = subparsers.add_parser("models", help="manage local model presets")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)
    models_list = models_subparsers.add_parser("list", help="list preset status")
    models_list.add_argument("--json", action="store_true")
    model_download = models_subparsers.add_parser("download", help="download one preset")
    model_download.add_argument("--preset-id", required=True)
    model_download.add_argument("--json", action="store_true")
    model_delete = models_subparsers.add_parser("delete", help="delete one preset")
    model_delete.add_argument("--preset-id", required=True)
    model_delete.add_argument("--json", action="store_true")
    model_clear = models_subparsers.add_parser("clear-cache", help="clear model cache")
    model_clear.add_argument("--json", action="store_true")

    jobs_parser = subparsers.add_parser("jobs", help="create or inspect worker jobs")
    jobs_subparsers = jobs_parser.add_subparsers(dest="jobs_command", required=True)
    jobs_list = jobs_subparsers.add_parser("list", help="list jobs")
    jobs_list.add_argument("--case-id")
    jobs_list.add_argument("--json", action="store_true")
    jobs_show = jobs_subparsers.add_parser("show", help="show one job")
    jobs_show.add_argument("--job-id", required=True)
    jobs_show.add_argument("--json", action="store_true")
    jobs_run = jobs_subparsers.add_parser("run", help="run one queued job immediately")
    jobs_run.add_argument("--job-id", required=True)
    jobs_run.add_argument("--json", action="store_true")
    jobs_create = jobs_subparsers.add_parser("create-batch", help="queue one generation job")
    jobs_create.add_argument("--case-id", required=True)
    jobs_create.add_argument("--count", type=int, default=10)
    jobs_create.add_argument("--direction-hint", default="")
    jobs_create.add_argument("--backend", choices=["mock", "diffusers", "local-svg"], default=default_backend())
    jobs_create.add_argument("--source-candidate-id")
    jobs_create.add_argument("--preset-id")
    jobs_create.add_argument("--exploration-preset")
    jobs_create.add_argument("--json", action="store_true")

    worker_status = subparsers.add_parser("worker-status", help="show worker capability snapshot")
    worker_status.add_argument("--json", action="store_true")

    return parser


def main() -> None:
    load_local_env(project_root())
    parser = build_parser()
    args = parser.parse_args()
    store = ToolForLogoStore.from_env()

    if args.command == "web":
        flask_app.run(host=args.host, port=args.port, debug=False)
        return

    if args.command == "daemon":
        run_daemon(poll_interval=args.poll_interval)
        return

    if args.command == "status":
        _print(store.status_payload(), getattr(args, "json", False))
        return

    if args.command == "worker-status":
        write_worker_capabilities()
        _print(worker_status_payload(), getattr(args, "json", False))
        return

    if args.command == "create-case":
        record = store.create_case(args.name, args.description, args.notes)
        _print({"case": record.to_dict()}, args.json)
        return

    if args.command == "generate-batch":
        generation_options = {}
        if args.preset_id:
            generation_options["preset_id"] = args.preset_id
        if args.exploration_preset:
            generation_options["exploration_preset"] = args.exploration_preset
        payload = generate_batch(
            store,
            case_id=args.case_id,
            count=args.count,
            direction_hint=args.direction_hint,
            seed=args.seed,
            backend=args.backend,
            source_candidate_id=args.source_candidate_id,
            generation_options=generation_options or None,
        )
        _print(payload, args.json)
        return

    if args.command == "set-status":
        candidate = store.update_candidate_status(args.case_id, args.candidate_id, CandidateStatus(args.status))
        _print({"candidate": candidate.to_dict()}, args.json)
        return

    if args.command == "export":
        export_record = create_export_bundle(store, case_id=args.case_id, candidate_ids=args.candidate_ids, name_override=args.name_override)
        _print({"export": export_record.to_dict()}, args.json)
        return

    if args.command == "settings":
        if args.settings_command == "status":
            _print(settings_snapshot(), args.json)
            return
        if args.settings_command == "save":
            changes = {}
            if args.compute_mode is not None:
                changes["computeMode"] = args.compute_mode
            if args.processing_quality is not None:
                changes["processingQuality"] = args.processing_quality
            if args.default_batch_count is not None:
                changes["defaultBatchCount"] = args.default_batch_count
            if args.default_direction_hint is not None:
                changes["defaultDirectionHint"] = args.default_direction_hint
            if args.default_exploration_preset is not None:
                changes["defaultExplorationPreset"] = args.default_exploration_preset
            if args.allow_auto_model_download:
                changes["allowAutoModelDownload"] = True
            snapshot = save_settings(changes)
            if args.token is not None:
                save_huggingface_token(args.token)
            _print({**snapshot, "hasHuggingFaceToken": bool(load_huggingface_token())}, args.json)
            return

    if args.command == "models":
        if args.models_command == "list":
            _print({"models": build_model_statuses(), "cache": cache_snapshot()}, args.json)
            return
        if args.models_command == "download":
            _print(download_model(args.preset_id), args.json)
            return
        if args.models_command == "delete":
            _print(delete_model(args.preset_id), args.json)
            return
        if args.models_command == "clear-cache":
            _print(clear_model_cache(), args.json)
            return

    if args.command == "jobs":
        if args.jobs_command == "list":
            _print(list_jobs(args.case_id), args.json)
            return
        if args.jobs_command == "show":
            _print({"request": load_request(args.job_id), "status": load_status(args.job_id), "result": load_result(args.job_id)}, args.json)
            return
        if args.jobs_command == "run":
            _print(process_job(args.job_id, store), args.json)
            return
        if args.jobs_command == "create-batch":
            generation_options = {}
            if args.preset_id:
                generation_options["preset_id"] = args.preset_id
            if args.exploration_preset:
                generation_options["exploration_preset"] = args.exploration_preset
            job_id, _ = create_job(
                job_type="generate_batch",
                case_id=args.case_id,
                payload={
                    "case_id": args.case_id,
                    "count": args.count,
                    "direction_hint": args.direction_hint,
                    "backend": args.backend,
                    "source_candidate_id": args.source_candidate_id,
                    "generation_options": generation_options,
                },
            )
            _print({"job_id": job_id}, args.json)
            return

    raise RuntimeError(f"Unhandled command: {args.command}")
