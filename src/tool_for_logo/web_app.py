from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for

from .backends import list_exploration_presets
from .generator import create_export_bundle, list_tone_presets
from .job_store import create_job, list_active_jobs, list_jobs
from .model_catalog import build_model_statuses, cache_snapshot
from .models import CandidateStatus
from .runtime import default_backend, load_local_env, project_root
from .settings import load_huggingface_token, load_settings, load_worker_capabilities, save_huggingface_token, save_settings, settings_snapshot
from .state import ToolForLogoStore


app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "tool-for-logo-local-dev"


@app.template_filter("filesize")
def filesize_filter(value: int) -> str:
    size = int(value or 0)
    if size >= 1024 * 1024 * 1024:
        return f"{size / 1024 / 1024 / 1024:.1f} GB"
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _store() -> ToolForLogoStore:
    return ToolForLogoStore.from_env()


def _settings() -> dict[str, Any]:
    return load_settings()


def _active_jobs(case_id: str | None = None) -> list[dict[str, Any]]:
    return list_active_jobs(case_id)


def _candidate_score(candidate: Any) -> float:
    return float((candidate.generation.get("quality") or {}).get("total") or 0.0)


def _generation_options_from_form() -> dict[str, str]:
    payload: dict[str, str] = {}
    preset_id = request.form.get("preset_id", "").strip()
    exploration_preset = request.form.get("exploration_preset", "").strip()
    if preset_id:
        payload["preset_id"] = preset_id
    if exploration_preset:
        payload["exploration_preset"] = exploration_preset
    return payload


def _sorted_candidates(case_id: str) -> list[Any]:
    status_rank = {
        CandidateStatus.ADOPTED: 0,
        CandidateStatus.FAVORITE: 1,
        CandidateStatus.FRESH: 2,
        CandidateStatus.EXCLUDED: 3,
    }
    return sorted(
        _store().list_candidates(case_id),
        key=lambda item: (status_rank.get(item.status, 9), -_candidate_score(item), item.created_at),
        reverse=False,
    )


@app.context_processor
def inject_layout() -> dict[str, Any]:
    return {
        "nav_path": request.path,
        "active_jobs": _active_jobs(),
        "CandidateStatus": CandidateStatus,
    }


@app.get("/")
def index() -> Any:
    return redirect(url_for("cases_index"))


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/api/status")
def api_status() -> Any:
    store = _store()
    return jsonify({**store.status_payload(), "settings": settings_snapshot(), "activeJobs": [job["status"] for job in _active_jobs()]})


@app.route("/settings", methods=["GET", "POST"])
def settings_page() -> Any:
    if request.method == "POST":
        current = load_settings()
        save_settings(
            {
                "computeMode": request.form.get("compute_mode", "gpu"),
                "processingQuality": request.form.get("processing_quality", "standard"),
                "defaultBatchCount": int(request.form.get("default_batch_count", "10") or 10),
                "defaultDirectionHint": request.form.get("default_direction_hint", "").strip(),
                "defaultExplorationPreset": request.form.get("default_exploration_preset", current.get("defaultExplorationPreset", "balanced-saas")),
                "allowAutoModelDownload": request.form.get("allow_auto_model_download") == "on",
                "preferredConceptPreset": request.form.get("preferred_concept_preset", current.get("preferredConceptPreset", "concept-standard")),
                "preferredCpuPreset": current.get("preferredCpuPreset", "cpu-standard"),
                "preferredGpuPreset": request.form.get("preferred_gpu_preset", "gpu-standard"),
                "preferredHighGpuPreset": request.form.get("preferred_high_gpu_preset", "gpu-high"),
            }
        )
        token_value = request.form.get("hf_token")
        if token_value is not None and token_value.strip():
            save_huggingface_token(token_value.strip())
        flash("Settings saved.", "success")
        return redirect(url_for("settings_page"))

    return render_template(
        "settings.html",
        title="Settings",
        settings=_settings(),
        worker_capabilities=load_worker_capabilities(),
        model_statuses=build_model_statuses(),
        cache=cache_snapshot(),
        has_hf_token=bool(load_huggingface_token()),
        default_backend=default_backend(),
        exploration_presets=list_exploration_presets(),
    )


@app.post("/settings/models/<preset_id>/download")
def queue_model_download(preset_id: str) -> Any:
    create_job(job_type="download_model", payload={"preset_id": preset_id})
    flash(f"Queued download for {preset_id}.", "success")
    return redirect(url_for("settings_page"))


@app.post("/settings/models/<preset_id>/delete")
def queue_model_delete(preset_id: str) -> Any:
    create_job(job_type="delete_model", payload={"preset_id": preset_id})
    flash(f"Queued cache delete for {preset_id}.", "success")
    return redirect(url_for("settings_page"))


@app.post("/settings/cache/clear")
def queue_cache_clear() -> Any:
    create_job(job_type="clear_model_cache", payload={})
    flash("Queued cached model cleanup.", "success")
    return redirect(url_for("settings_page"))


@app.route("/cases/new", methods=["GET", "POST"])
def new_case() -> Any:
    store = _store()
    settings = _settings()
    if request.method == "POST":
        product_name = request.form.get("product_name", "").strip()
        description = request.form.get("description", "").strip()
        notes = request.form.get("notes", "").strip()
        direction_hint = request.form.get("direction_hint", "").strip()
        count = int(request.form.get("count", settings.get("defaultBatchCount", 10)) or 10)
        backend_choice = request.form.get("backend", default_backend()).strip() or default_backend()
        if not product_name or not description:
            flash("Product name and description are required.", "danger")
        else:
            case_record = store.create_case(product_name=product_name, description=description, notes=notes)
            create_job(
                job_type="generate_batch",
                case_id=case_record.case_id,
                payload={
                    "case_id": case_record.case_id,
                    "count": count,
                    "direction_hint": direction_hint,
                    "backend": backend_choice,
                    "generation_options": _generation_options_from_form(),
                },
            )
            flash("Case created and generation queued.", "success")
            return redirect(url_for("case_detail", case_id=case_record.case_id))

    return render_template(
        "new_case.html",
        title="New Case",
        settings=settings,
        model_statuses=build_model_statuses(),
        default_backend=default_backend(),
        exploration_presets=list_exploration_presets(),
        tone_presets=list_tone_presets(),
    )


@app.get("/cases")
def cases_index() -> Any:
    store = _store()
    cases = []
    for case_record in store.list_cases():
        candidates = store.list_candidates(case_record.case_id)
        cases.append(
            {
                "record": case_record,
                "candidate_count": len(candidates),
                "favorite_count": sum(1 for item in candidates if item.status in {CandidateStatus.FAVORITE, CandidateStatus.ADOPTED}),
                "jobs": list_jobs(case_record.case_id)[:3],
            }
        )
    return render_template("cases.html", title="Cases", cases=cases, active_jobs=_active_jobs())


@app.get("/cases/<case_id>")
def case_detail(case_id: str) -> Any:
    store = _store()
    case_record = store.get_case(case_id)
    candidates = _sorted_candidates(case_id)
    favorite_count = sum(1 for item in candidates if item.status in {CandidateStatus.FAVORITE, CandidateStatus.ADOPTED})
    top_candidates = [item for item in candidates if item.status != CandidateStatus.EXCLUDED][:6]
    status_counts = {
        "fresh": sum(1 for item in candidates if item.status == CandidateStatus.FRESH),
        "favorite": sum(1 for item in candidates if item.status == CandidateStatus.FAVORITE),
        "adopted": sum(1 for item in candidates if item.status == CandidateStatus.ADOPTED),
        "excluded": sum(1 for item in candidates if item.status == CandidateStatus.EXCLUDED),
    }
    return render_template(
        "case_detail.html",
        title=case_record.product_name,
        case_record=case_record,
        candidates=candidates,
        top_candidates=top_candidates,
        favorite_count=favorite_count,
        status_counts=status_counts,
        batches=store.list_batches(case_id),
        exports=store.list_exports(case_id),
        jobs=list_jobs(case_id),
        active_jobs=_active_jobs(case_id),
        settings=_settings(),
        model_statuses=build_model_statuses(),
        default_backend=default_backend(),
        exploration_presets=list_exploration_presets(),
        tone_presets=list_tone_presets(),
    )


@app.post("/cases/<case_id>/batches")
def queue_batch(case_id: str) -> Any:
    _store().get_case(case_id)
    count = int(request.form.get("count", _settings().get("defaultBatchCount", 10)) or 10)
    direction_hint = request.form.get("direction_hint", "").strip()
    source_candidate_id = request.form.get("source_candidate_id", "").strip() or None
    backend_choice = request.form.get("backend", default_backend()).strip() or default_backend()
    create_job(
        job_type="generate_batch",
        case_id=case_id,
        payload={
            "case_id": case_id,
            "count": count,
            "direction_hint": direction_hint,
            "backend": backend_choice,
            "source_candidate_id": source_candidate_id,
            "generation_options": _generation_options_from_form(),
        },
    )
    flash("Queued additional generation batch.", "success")
    return redirect(url_for("case_detail", case_id=case_id))


@app.post("/cases/<case_id>/candidates/<candidate_id>/status")
def update_candidate_status(case_id: str, candidate_id: str) -> Any:
    status_raw = request.form.get("status", "fresh")
    try:
        status = CandidateStatus(status_raw)
    except ValueError:
        abort(400)
    _store().update_candidate_status(case_id, candidate_id, status)
    flash(f"Updated candidate to {status.value}.", "success")
    return redirect(url_for("case_detail", case_id=case_id))


@app.post("/cases/<case_id>/candidates/bulk-status")
def bulk_update_candidate_status(case_id: str) -> Any:
    store = _store()
    store.get_case(case_id)
    candidate_ids = [item.strip() for item in request.form.getlist("candidate_ids") if item.strip()]
    status_raw = request.form.get("status", "favorite")
    if not candidate_ids:
        flash("Select at least one candidate first.", "warning")
        return redirect(url_for("case_detail", case_id=case_id))
    try:
        status = CandidateStatus(status_raw)
    except ValueError:
        abort(400)
    for candidate_id in candidate_ids:
        store.update_candidate_status(case_id, candidate_id, status)
    flash(f"Updated {len(candidate_ids)} candidates to {status.value}.", "success")
    return redirect(url_for("case_detail", case_id=case_id))


@app.post("/cases/<case_id>/exports")
def export_case(case_id: str) -> Any:
    candidate_ids = request.form.getlist("candidate_ids") or None
    name_override = request.form.get("name_override", "").strip() or None
    tone_preset = request.form.get("tone_preset", "candidate-auto").strip() or "candidate-auto"
    create_export_bundle(
        _store(),
        case_id=case_id,
        candidate_ids=candidate_ids,
        name_override=name_override,
        tone_preset=tone_preset,
    )
    flash("Created export bundle.", "success")
    return redirect(url_for("case_detail", case_id=case_id))


@app.get("/files/candidate/<case_id>/<candidate_id>/<asset_key>")
def candidate_asset(case_id: str, candidate_id: str, asset_key: str) -> Any:
    candidate = _store().get_candidate(case_id, candidate_id)
    path_value = candidate.assets.get(asset_key)
    if not path_value:
        abort(404)
    path = Path(path_value)
    if not path.exists():
        abort(404)
    return send_file(path)


@app.get("/files/export/<case_id>/<export_id>/archive")
def export_archive(case_id: str, export_id: str) -> Any:
    for record in _store().list_exports(case_id):
        if record.export_id == export_id:
            return send_file(record.archive_path, as_attachment=True)
    abort(404)


@app.get("/files/export/<case_id>/<export_id>/comparison")
def export_comparison(case_id: str, export_id: str) -> Any:
    target = _store().report_root / case_id / export_id / "comparison_sheet.png"
    if not target.exists():
        abort(404)
    return send_file(target)


@app.get("/api/cases/<case_id>")
def api_case(case_id: str) -> Any:
    store = _store()
    return jsonify(
        {
            "case": store.get_case(case_id).to_dict(),
            "batches": [item.to_dict() for item in store.list_batches(case_id)],
            "candidates": [item.to_dict() for item in store.list_candidates(case_id)],
            "exports": [item.to_dict() for item in store.list_exports(case_id)],
            "jobs": list_jobs(case_id),
        }
    )


def main() -> None:
    load_local_env(project_root())
    app.run(host="0.0.0.0", port=8080, debug=False)


if __name__ == "__main__":
    main()
