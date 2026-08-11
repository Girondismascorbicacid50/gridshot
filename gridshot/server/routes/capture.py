"""Single-tool capture, correction-session, and result endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import File, Form, HTTPException, UploadFile

from ._builder import RouteSpec, build_domain_router

_OWNER: Any | None = None

ROUTES: tuple[RouteSpec, ...] = (
    ("POST", "/api/trace", "trace"),
    ("POST", "/api/session", "session_start"),
    ("GET", "/api/session/{sid}", "session_get"),
    ("POST", "/api/session/{sid}/click", "session_click"),
    ("POST", "/api/session/{sid}/outline", "session_set_outline"),
    ("POST", "/api/session/{sid}/history/{direction}", "session_edit_history"),
    ("POST", "/api/session/{sid}/physical-outline", "session_set_physical_outline"),
    ("POST", "/api/session/{sid}/generate", "session_generate"),
    ("POST", "/api/session/{sid}/library", "session_add_to_library"),
    ("GET", "/api/result/{project}", "get_result"),
)


def configure(owner: Any) -> None:
    global _OWNER
    _OWNER = owner


def _owner() -> Any:
    if _OWNER is None:
        raise RuntimeError("capture router has not been configured")
    return _OWNER


async def trace(
    file: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    thickness: Optional[float] = Form(None),
    clearance: float = Form(1.0),
    bin_style: Literal["pocket", "corral", "grid"] = Form("pocket"),
    depth: Optional[float] = Form(None),
    full_height: Optional[float] = Form(None),
    overall_height: Optional[float] = Form(None),
    finger_hole: bool = Form(True),
    lip: bool = Form(True),
    round_tool: bool = Form(False),
    mat_id: Optional[str] = Form(None),
) -> dict:
    owner = _owner()
    if not isinstance(full_height, (int, float)):
        full_height = None
    project_id = f"{int(owner.time.time())}-{owner.uuid.uuid4().hex[:6]}"
    project = owner.PROJECTS / project_id
    project.mkdir(parents=True, exist_ok=True)
    photo_path = project / f"photo1{owner._photo_ext(file.filename)}"
    photo_path.write_bytes(await file.read())

    photo2_path: Optional[Path] = None
    if file2 is not None:
        photo2_path = project / f"photo2{owner._photo_ext(file2.filename)}"
        photo2_path.write_bytes(await file2.read())
    elif thickness is None:
        raise HTTPException(
            status_code=422,
            detail="add a second photo (from a different angle) for automatic "
            "thickness, or enter a thickness",
        )

    try:
        result = owner.trace_mod.run(
            photo_path,
            thickness_mm=thickness,
            photo2=photo2_path,
            clearance_mm=clearance,
            bin_style=bin_style,
            pocket_depth_mm=depth,
            full_height_mm=full_height,
            overall_height_mm=overall_height,
            finger_hole=finger_hole,
            lip=lip,
            round_tool=round_tool,
            mat_id=mat_id,
            out_dir=project,
            stem="bin",
        )
    except (owner.contour_mod.NoToolFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = owner._result_payload(
        result,
        project_id,
        clearance,
        lip,
        round_tool,
        finger_hole,
        pocket_depth_override_mm=depth,
        overall_height_override_mm=overall_height,
    )
    (project / "result.json").write_text(json.dumps(payload))
    return payload


def get_result(project: str) -> dict:
    owner = _owner()
    path = (owner.PROJECTS / project / "result.json").resolve()
    if not str(path).startswith(str(owner.PROJECTS.resolve())) or not path.is_file():
        raise HTTPException(status_code=404, detail="no such project")
    return json.loads(path.read_text())


def build_router(owner):
    configure(owner)
    return build_domain_router(owner, tag="capture", specs=ROUTES)
