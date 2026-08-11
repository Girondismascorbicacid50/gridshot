"""Durable state for the single-tool selection and cutout editor.

The segmentation server's image embedding is intentionally not persisted: it is
process-local and is recreated lazily from the lossless editor image when the
next SAM prompt arrives. Everything the user can see or edit is persisted.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from .models import Calibration, Poly
from .readiness import ReadinessReport


SESSION_SCHEMA_VERSION = "single-session.v1"
STATE_FILE = "single-session.json"


class SessionStoreError(RuntimeError):
    """A saved session is unreadable or from an unsupported future schema."""


def _dump_model(value):
    return value.model_dump(mode="json") if value is not None else None


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _relative_file(project: Path, value: Path | str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    try:
        relative = path.resolve().relative_to(project.resolve())
    except ValueError as exc:
        raise SessionStoreError(f"session file is outside its project: {path}") from exc
    if len(relative.parts) != 1:
        raise SessionStoreError(f"session file must be at the project root: {path}")
    return relative.name


def _project_file(project: Path, name: str | None) -> Path | None:
    if name is None:
        return None
    if Path(name).name != name:
        raise SessionStoreError("saved session contains an unsafe file path")
    return project / name


def save(project: Path, session: dict) -> Path:
    """Atomically publish one complete session generation."""
    project.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    mask_name = f"single-session-masks-{generation}.npz"
    mask_path = project / mask_name

    arrays: dict[str, np.ndarray] = {
        "current": np.asarray(session["mask"], dtype=np.uint8),
        "initial": np.asarray(
            session.get("initial_mask", session["mask"]), dtype=np.uint8
        ),
    }
    history_payload = []
    for index, snapshot in enumerate(session.get("_edit_history", [])):
        key = f"history_{index}"
        arrays[key] = np.asarray(snapshot["mask"], dtype=np.uint8)
        history_payload.append({
            "revision": int(snapshot["revision"]),
            "operation": str(snapshot["operation"]),
            "mask": key,
            "points": snapshot.get("points", []),
            "labels": snapshot.get("labels", []),
            "diagnostics": snapshot.get("diagnostics", {}),
        })

    mask_tmp = project / f".{mask_name}.tmp"
    try:
        with mask_tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(mask_tmp, mask_path)
    finally:
        mask_tmp.unlink(missing_ok=True)

    payload = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "saved_ts": int(time.time()),
        "mask_archive": mask_name,
        "photo1": _relative_file(project, session.get("photo1")),
        "photo2": _relative_file(project, session.get("photo2")),
        "mat_id": session.get("mat_id"),
        "points": session.get("points", []),
        "labels": session.get("labels", []),
        "initial_points": session.get("initial_points", []),
        "calibration": session.get("calibration", {}),
        "calibration_model": _dump_model(session.get("cal_full")),
        "warnings": session.get("warnings", []),
        "capture_warnings": session.get("capture_warnings", []),
        "readiness": _dump_model(session.get("readiness")),
        "revision": int(session.get("revision", 0)),
        "edit_cursor": int(session.get("_edit_cursor", 0)),
        "next_revision": int(
            session.get("_next_revision", session.get("revision", 0))
        ),
        "edit_history": history_payload,
        "cleanup_default": session.get("cleanup_default"),
        "segmentation_confidence": session.get("segmentation_confidence"),
        "physical_override": _dump_model(session.get("physical_override")),
        "physical_override_mask_revision": session.get(
            "physical_override_mask_revision"
        ),
        "physical_override_diagnostics": session.get(
            "physical_override_diagnostics"
        ),
        "physical_override_revision": int(
            session.get("physical_override_revision", 0)
        ),
        "physical_editor_poly": _dump_model(session.get("physical_editor_poly")),
        "physical_reconstruction": session.get("physical_reconstruction"),
    }
    _atomic_json(project / STATE_FILE, payload)

    # The JSON publish above is the commit point. Old generations are now safe
    # to remove; an interrupted write always leaves at least one complete pair.
    for old in project.glob("single-session-masks-*.npz"):
        if old.name != mask_name:
            old.unlink(missing_ok=True)
    return project / STATE_FILE


def load(project: Path) -> dict:
    path = project / STATE_FILE
    if not path.is_file():
        raise KeyError(project.name)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionStoreError("saved single-tool session metadata is corrupt") from exc
    version = payload.get("schema_version")
    if version != SESSION_SCHEMA_VERSION:
        raise SessionStoreError(f"unsupported single-tool session schema: {version!r}")

    mask_name = payload.get("mask_archive")
    mask_path = _project_file(project, mask_name)
    if mask_path is None or not mask_path.is_file():
        raise SessionStoreError("saved single-tool session mask archive is missing")
    try:
        with np.load(mask_path, allow_pickle=False) as masks:
            session = {
                "photo1": _project_file(project, payload.get("photo1")),
                "photo2": _project_file(project, payload.get("photo2")),
                "mat_id": payload.get("mat_id"),
                "image_id": None,
                "mask": masks["current"].astype(np.uint8),
                "initial_mask": masks["initial"].astype(np.uint8),
                "points": payload.get("points", []),
                "labels": payload.get("labels", []),
                "initial_points": payload.get("initial_points", []),
                "calibration": payload.get("calibration", {}),
                "warnings": payload.get("warnings", []),
                "capture_warnings": payload.get("capture_warnings", []),
                "revision": int(payload.get("revision", 0)),
                "_edit_cursor": int(payload.get("edit_cursor", 0)),
                "_next_revision": int(payload.get("next_revision", 0)),
            }
            history = []
            for snapshot in payload.get("edit_history", []):
                key = snapshot["mask"]
                history.append({
                    "revision": int(snapshot["revision"]),
                    "operation": snapshot["operation"],
                    "mask": masks[key].astype(np.uint8),
                    "points": snapshot.get("points", []),
                    "labels": snapshot.get("labels", []),
                    "diagnostics": snapshot.get("diagnostics", {}),
                })
            if history:
                session["_edit_history"] = history
    except (OSError, KeyError, ValueError) as exc:
        raise SessionStoreError("saved single-tool session masks are corrupt") from exc

    if session["mask"].ndim != 2 or session["mask"].shape != session["initial_mask"].shape:
        raise SessionStoreError("saved single-tool session masks have invalid dimensions")
    history = session.get("_edit_history", [])
    cursor = session.get("_edit_cursor", 0)
    if history and not 0 <= cursor < len(history):
        raise SessionStoreError("saved single-tool session history cursor is invalid")
    if session["photo1"] is None or not session["photo1"].is_file():
        raise SessionStoreError("saved single-tool session source photo is missing")

    if payload.get("calibration_model") is not None:
        session["cal_full"] = Calibration.model_validate(
            payload["calibration_model"]
        )
    if payload.get("readiness") is not None:
        session["readiness"] = ReadinessReport.model_validate(payload["readiness"])
    for key in ("physical_override", "physical_editor_poly"):
        if payload.get(key) is not None:
            session[key] = Poly.model_validate(payload[key])
    for key in (
        "cleanup_default",
        "segmentation_confidence",
        "physical_override_mask_revision",
        "physical_override_diagnostics",
        "physical_reconstruction",
    ):
        if payload.get(key) is not None:
            session[key] = payload[key]
    session["physical_override_revision"] = int(
        payload.get("physical_override_revision", 0)
    )
    return session
