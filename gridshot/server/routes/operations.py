"""Health, mat, device-profile, and camera-calibration endpoints."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ._builder import RouteSpec, build_domain_router

_OWNER: Any | None = None

ROUTES: tuple[RouteSpec, ...] = (
    ("GET", "/api/health/live", "health_live"),
    ("GET", "/api/health/ready", "health_ready"),
    ("GET", "/api/health/capabilities", "health_capabilities"),
    ("GET", "/api/health", "health"),
    ("GET", "/api/mats", "mats"),
    ("GET", "/api/device-profiles", "device_profiles"),
    ("DELETE", "/api/device-profiles", "device_profiles_delete_all"),
    ("DELETE", "/api/device-profiles/{device_id}", "device_profile_delete"),
    ("POST", "/api/calibration/intrinsics", "calibration_intrinsics"),
)


def configure(owner: Any) -> None:
    global _OWNER
    _OWNER = owner


def _owner() -> Any:
    if _OWNER is None:
        raise RuntimeError("operations router has not been configured")
    return _OWNER


def storage_health() -> dict[str, bool]:
    owner = _owner()

    def accessible(path: Path) -> bool:
        candidate = path.resolve()
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate.is_dir() and os.access(candidate, os.R_OK | os.W_OK)

    return {
        "projects": accessible(owner.PROJECTS),
        "config": accessible(owner.config_dir()),
    }


def health_live() -> dict:
    """Web-process liveness; no disk, model, or network checks."""
    return {"status": "alive"}


def health_ready(response: Response) -> dict:
    """Dependencies required for normal capture and editing workflows."""
    owner = _owner()
    storage = owner._storage_health()
    segserver = owner.seg_client.readiness()
    is_ready = all(storage.values()) and segserver
    response.status_code = 200 if is_ready else 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "storage": storage,
        "segserver": segserver,
    }


def health_capabilities() -> dict:
    """Detailed feature availability without changing application readiness."""
    owner = _owner()
    verified = [p.mat_id for p in owner.mat_mod.list_profiles() if p.verified]
    segserver = owner.seg_client.capabilities()
    return {
        "status": "ok" if segserver is not None else "degraded",
        "segserver_live": owner.seg_client.liveness(),
        "segserver": segserver,
        "verified_mats": verified,
    }


def health() -> dict:
    """Compatibility summary consumed by the current SPA."""
    owner = _owner()
    verified = [p.mat_id for p in owner.mat_mod.list_profiles() if p.verified]
    return {
        "status": "ok",
        "segserver": owner.seg_client.available(),
        "mats": verified,
    }


def mats() -> list[dict]:
    owner = _owner()
    return [
        {
            "mat_id": profile.mat_id,
            "paper": profile.spec.paper,
            "verified": profile.verified,
            "scale_x": profile.scale_x,
            "scale_y": profile.scale_y,
        }
        for profile in owner.mat_mod.list_profiles()
    ]


def device_profile_json(profile) -> dict:
    return {
        "device_id": profile.device_id,
        "revision": profile.revision,
        "created_at": profile.created_at,
        "device_make": profile.device_make,
        "device_model": profile.device_model,
        "lens_model": profile.lens_model,
        "image_size": list(profile.image_size),
        "orientation_deg": profile.orientation_deg,
        "focal_mm": profile.focal_mm,
        "focal_35mm": profile.focal_35mm,
        "digital_zoom_ratio": profile.digital_zoom_ratio,
        "mat_id": profile.mat_id,
        "n_views": profile.n_views,
        "reproj_rms_px": profile.reproj_rms_px,
    }


def device_profiles() -> list[dict]:
    owner = _owner()
    return [owner._device_profile_json(profile) for profile in owner.devices_mod.list_profiles()]


def device_profiles_delete_all() -> dict:
    return {"deleted": _owner().devices_mod.delete_all_profiles()}


def device_profile_delete(device_id: str) -> dict:
    owner = _owner()
    try:
        owner.devices_mod.delete_profile(device_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": device_id}


async def calibration_intrinsics(
    files: list[UploadFile] = File(...),
    mat_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
) -> dict:
    owner = _owner()
    if len(files) < owner.calibrate_mod.MIN_INTRINSICS_VIEWS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"upload at least {owner.calibrate_mod.MIN_INTRINSICS_VIEWS} "
                "calibration photos"
            ),
        )

    mat_profile = owner._pick_verified_mat(mat_id)
    calibration_id = f"calibration-{int(owner.time.time())}-{owner.uuid.uuid4().hex[:6]}"
    calibration_dir = owner.PROJECTS / calibration_id
    calibration_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    source_names: list[str] = []
    for index, upload in enumerate(files, start=1):
        path = calibration_dir / f"view-{index:03d}{owner._photo_ext(upload.filename)}"
        path.write_bytes(await upload.read())
        paths.append(path)
        source_names.append(upload.filename or path.name)

    try:
        sources = [owner.ingest_mod.load(path) for path in paths]
        signature = owner.devices_mod.calibration_signature(sources)
        K, dist, rms, n_views, warnings = owner.calibrate_mod.calibrate_intrinsics(
            [source.pixels for source in sources],
            mat_profile,
            view_names=source_names,
        )
    except (ValueError, owner.calibrate_mod.DetectionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    revision = owner.devices_mod.next_revision(signature)
    profile = owner.devices_mod.DeviceProfile(
        device_id=owner.devices_mod.profile_id(signature, revision, name),
        revision=revision,
        created_at=datetime.now(timezone.utc).isoformat(),
        device_make=signature.device_make,
        device_model=signature.device_model,
        lens_model=signature.lens_model,
        image_size=signature.image_size,
        orientation_deg=signature.orientation_deg,
        focal_mm=signature.focal_mm,
        focal_35mm=signature.focal_35mm,
        digital_zoom_ratio=signature.digital_zoom_ratio,
        mat_id=mat_profile.mat_id,
        n_views=n_views,
        source_images=source_names,
        K=K.tolist(),
        dist=dist.tolist(),
        reproj_rms_px=rms,
    )
    try:
        owner.devices_mod.save_profile(profile)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "profile": owner._device_profile_json(profile),
        "capture_signature": signature.model_dump(mode="json"),
        "views_uploaded": len(files),
        "views_used": n_views,
        "warnings": warnings,
    }


def build_router(owner):
    configure(owner)
    return build_domain_router(owner, tag="operations", specs=ROUTES)
