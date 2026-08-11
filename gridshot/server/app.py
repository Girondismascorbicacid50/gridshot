"""GridShot local web app: FastAPI over the core pipeline, serves the SPA.

Thin route layer — every endpoint is a small wrapper over `gridshot.core`.
Served on the tailnet (fronted by `tailscale serve` for HTTPS); no auth, the
tailnet ACL is the boundary.  Projects persist as directories on disk.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel, Field

from gridshot.core import batch as batch_mod
from gridshot.core import bench as bench_mod
from gridshot.core import calibrate as calibrate_mod
from gridshot.core import contour as contour_mod
from gridshot.core import devices as devices_mod
from gridshot.core import diffseg as diffseg_mod
from gridshot.core import derive as derive_mod
from gridshot.core import gridfinity as grid_mod
from gridshot.core import ingest as ingest_mod
from gridshot.core import library as library_mod
from gridshot.core import mat as mat_mod
from gridshot.core import nesting as nesting_mod
from gridshot.core import parallax as parallax_mod
from gridshot.core import quality as quality_mod
from gridshot.core import readiness as readiness_mod
from gridshot.core import session_store
from gridshot.core import trace as trace_mod
from gridshot.core.models import Poly, config_dir
from gridshot.seg import client as seg_client

PROJECTS = Path("projects")
WEB_DIST = Path("web/dist")

app = FastAPI(title="GridShot")


@app.exception_handler(library_mod.LibrarySchemaError)
async def _library_schema_error(
    _request: Request, exc: library_mod.LibrarySchemaError
) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})

# Process caches. Single-tool and batch source-of-truth state is durable on disk;
# only segmentation embeddings and library edit scratch state remain ephemeral.
_SESSIONS: dict[str, dict] = {}
_BATCH_SESSIONS: dict[str, dict] = {}  # durable zip-upload pairing state cache
_BATCH_EDIT_SESSIONS: dict[str, dict] = {}
_LIB_EDIT_SESSIONS: dict[str, dict] = {}  # SAM re-editing a stored library tool

# Batch archives are user-controlled input. These limits are intentionally large
# enough for a 100-pair phone-photo corpus while bounding memory, disk, and model
# work. The archive is streamed to disk and expanded entries are copied with a
# second byte counter rather than trusting ZIP metadata alone.
_BATCH_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_BATCH_MAX_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
_BATCH_MAX_IMAGE_BYTES = 128 * 1024 * 1024
_BATCH_MAX_IMAGES = 200
_BATCH_IO_CHUNK = 1024 * 1024
_BATCH_JOB_VERSION = 1
_BATCH_JOB_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="gridshot-batch"
)
_BATCH_JOB_FUTURES: dict[str, Future] = {}
_BATCH_JOB_LOCK = threading.RLock()


def _session_project(sid: str) -> Path:
    if not sid or Path(sid).name != sid:
        raise HTTPException(status_code=404, detail="no such session")
    project = (PROJECTS / sid).resolve()
    try:
        project.relative_to(PROJECTS.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="no such session")
    return project


def _single_session(sid: str) -> dict:
    cached = _SESSIONS.get(sid)
    if cached is not None:
        return cached
    project = _session_project(sid)
    try:
        restored = session_store.load(project)
    except KeyError:
        raise HTTPException(status_code=404, detail="session expired or not found")
    except session_store.SessionStoreError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    _SESSIONS[sid] = restored
    return restored


def _single_session_optional(sid: str) -> dict | None:
    try:
        return _single_session(sid)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise


def _persist_single_session(sid: str, sess: dict) -> None:
    """Persist production sessions; tolerate minimal in-memory unit fixtures."""
    photo1 = sess.get("photo1")
    if photo1 is None:
        return
    project = _session_project(sid)
    try:
        if Path(photo1).resolve().parent != project:
            return
    except OSError:
        return
    _ensure_edit_history(sess)
    session_store.save(project, sess)


def _ensure_session_embedding(sid: str, sess: dict) -> None:
    if sess.get("image_id"):
        return
    project = _session_project(sid)
    source = project / "session-image.png"
    if not source.is_file():
        source = project / "display.jpg"
    if not source.is_file():
        raise HTTPException(status_code=409, detail="session editor image is missing")
    if not seg_client.available():
        raise HTTPException(status_code=503, detail="segserver offline")
    try:
        image_id, _width, _height = seg_client.embed(
            np.asarray(Image.open(source).convert("RGB"))
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"segserver re-embed failed: {exc}"
        )
    sess["image_id"] = image_id


def _raw_outline_for_photo(corrected, calibration, thickness):
    """Return the visible pre-parallax outline for a stored physical footprint."""
    if corrected is None:
        return None
    if calibration is not None and thickness:
        try:
            return parallax_mod.uncorrect_polygon(corrected, calibration, thickness)
        except (parallax_mod.MissingPoseError, ValueError):
            pass
    return corrected


def _library_raw_outline(tool):
    return tool.raw_outline or _raw_outline_for_photo(
        tool.outline, tool.calibration, tool.thickness_mm
    )


def _tool_readiness(tool) -> readiness_mod.ReadinessReport:
    if tool.readiness is not None:
        return tool.readiness
    provenance = tool.provenance
    return readiness_mod.evaluate(
        calibration=tool.calibration,
        warnings=provenance.warnings if provenance else [],
        outline=_library_raw_outline(tool) or tool.outline,
        thickness_mm=tool.thickness_mm,
        thickness_source=provenance.thickness_source if provenance else "legacy",
        # Stored outlines are already physical geometry. Legacy entries without
        # capture calibration remain reviewable rather than becoming unusable.
        require_calibration=False,
    )


def _refresh_tool_readiness(tool, thickness_source: str | None = None):
    provenance = tool.provenance or readiness_mod.ArtifactProvenance(
        flow="legacy",
        mat_id=tool.calibration.mat_id if tool.calibration else None,
        device_profile_id=(
            tool.calibration.device_profile_id if tool.calibration else None
        ),
        device_profile_revision=(
            tool.calibration.device_profile_revision
            if tool.calibration
            else None
        ),
        intrinsics_source=(
            tool.calibration.intrinsics_source if tool.calibration else None
        ),
        capture_signature=(
            tool.calibration.capture_signature if tool.calibration else None
        ),
        thickness_source="legacy",
    )
    if thickness_source is not None:
        provenance = provenance.model_copy(
            update={"thickness_source": thickness_source}
        )
    refreshed = readiness_mod.evaluate(
        calibration=tool.calibration,
        warnings=provenance.warnings,
        outline=tool.outline or _library_raw_outline(tool),
        thickness_mm=tool.thickness_mm,
        thickness_source=provenance.thickness_source,
        require_calibration=False,
    )
    return tool.model_copy(update={
        "readiness": refreshed,
        "provenance": provenance,
    })


def _require_tool_ready(tool) -> readiness_mod.ReadinessReport:
    value = _tool_readiness(tool)
    if value.blocked:
        label = tool.label or tool.id
        raise HTTPException(
            status_code=422,
            detail=f"{label} is not ready: {readiness_mod.blocking_message(value)}",
        )
    return value


def _store_lib_photo(tid: str, src: Path) -> bool:
    """Store a source photo beside the library entry so the tool can later be
    SAM-re-segmented against its own image. Re-encoded to JPEG (handles HEIC/PNG
    sources via ingest). Returns whether a photo was stored."""
    if src is None or not src.is_file():
        return False
    try:
        pixels = ingest_mod.load(src).pixels  # np RGB, EXIF-rotated, HEIC-aware
        Image.fromarray(pixels).save(library_mod.library_dir() / f"{tid}-photo.jpg", quality=90)
        return True
    except Exception:
        return False


def _regen_photo_thumb(t) -> None:
    """A photo crop around the tool — the library card's real-photo thumbnail, so
    tools with near-identical outlines (screwdrivers) are told apart at a glance."""
    photo = library_mod.library_dir() / f"{t.id}-photo.jpg"
    raw = _library_raw_outline(t)
    if not (t.has_photo and raw is not None and t.calibration is not None and photo.is_file()):
        return
    try:
        px = _poly_to_px(raw, t.calibration)
        xs = [p[0] for p in px]
        ys = [p[1] for p in px]
        img = Image.open(photo).convert("RGB")
        W, H = img.size
        pad = 0.18 * max(max(xs) - min(xs), max(ys) - min(ys)) + 10
        x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x1, y1 = min(W, max(xs) + pad), min(H, max(ys) + pad)
        crop = img.crop((x0, y0, x1, y1))
        # draw the outline on the crop (dark halo + bright line → visible on any
        # background) so the card shows which tools need outline fixing
        ring = [(x - x0, y - y0) for x, y in px]
        ring.append(ring[0])
        lw = max(3, int((x1 - x0) / 110))
        draw = ImageDraw.Draw(crop)
        draw.line(ring, fill=(0, 0, 0), width=lw + 3, joint="curve")
        draw.line(ring, fill=(255, 214, 90), width=lw, joint="curve")
        crop.thumbnail((320, 320))
        crop.save(library_mod.library_dir() / f"{t.id}-photo-thumb.jpg", quality=88)
    except Exception:
        pass


def _outline_px(mask: np.ndarray) -> list[list[float]]:
    """Largest mask component's exterior as [[x, y], ...] display px (simplified)."""
    comps = contour_mod.mask_to_polygons_px(mask)
    if not comps:
        return []
    import cv2

    ext = comps[0][0].astype(np.float32).reshape(-1, 1, 2)
    simp = cv2.approxPolyDP(ext, 1.0, True).reshape(-1, 2)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in simp]


def _seed_points(mask: np.ndarray, k: int = 3) -> list[list[float]]:
    """Foreground seed clicks at the mask's distance-transform peaks — deep
    inside the tool, so the editor opens in a click-driven state matching the
    auto segmentation."""
    import cv2

    dist = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
    peaks: list[list[float]] = []
    work = dist.copy()
    for _ in range(k):
        _, mv, _, ml = cv2.minMaxLoc(work)
        if mv < 8:
            break
        peaks.append([float(ml[0]), float(ml[1])])
        cv2.circle(work, ml, int(max(mv * 1.5, 20)), 0.0, -1)
    return peaks


def _mask_change(previous: np.ndarray, current: np.ndarray) -> dict:
    """Small user-facing diagnostics for whether a refinement actually moved."""
    a = previous > 127
    b = current > 127
    union = int(np.logical_or(a, b).sum())
    intersection = int(np.logical_and(a, b).sum())
    prior_area = int(a.sum())
    current_area = int(b.sum())
    iou = intersection / union if union else 1.0
    area_change = (
        100.0 * (current_area - prior_area) / prior_area
        if prior_area
        else (100.0 if current_area else 0.0)
    )
    return {
        "iou_with_previous": round(iou, 4),
        "area_change_pct": round(area_change, 2),
    }


EDITOR_SIMPLIFY_MM = 0.20


class PixelPoly(BaseModel):
    """An editor polygon in display-image pixels, including interior holes."""

    exterior: list[list[float]]
    holes: list[list[list[float]]] = Field(default_factory=list)


class PhysicalPoly(BaseModel):
    """A directly edited physical footprint in millimetres."""

    exterior: list[list[float]]
    holes: list[list[list[float]]] = Field(default_factory=list)


def _validated_physical_outline(polygon: PhysicalPoly) -> Poly:
    """Validate and lightly clean a directly edited millimetre-space outline."""
    if len(polygon.exterior) < 3 or any(len(ring) < 3 for ring in polygon.holes):
        raise HTTPException(
            status_code=422, detail="outline and holes need at least 3 points"
        )
    values = [
        *polygon.exterior,
        *(point for ring in polygon.holes for point in ring),
    ]
    if not values or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise HTTPException(status_code=422, detail="outline coordinates must be finite")
    candidate = Poly(
        exterior=[(float(x), float(y)) for x, y in polygon.exterior],
        holes=[
            [(float(x), float(y)) for x, y in ring]
            for ring in polygon.holes
        ],
    )
    raw_shape = contour_mod.to_shapely(candidate)
    if raw_shape.is_empty or not raw_shape.is_valid or raw_shape.area <= 0:
        raise HTTPException(status_code=422, detail="physical outline is invalid")
    if raw_shape.area > trace_mod.MAX_TOOL_AREA_MM2:
        raise HTTPException(status_code=422, detail="physical outline is implausibly large")
    try:
        return contour_mod.clean(candidate, simplify_tol=0.05)
    except (ValueError, contour_mod.NoToolFoundError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid physical outline: {exc}")


def _manual_physical_diagnostics(
    baseline: Poly,
    current: Poly,
    source: dict | None = None,
) -> dict[str, float | str]:
    """Describe a physical edit while retaining its reconstruction provenance."""
    previous_shape = contour_mod.to_shapely(baseline)
    current_shape = contour_mod.to_shapely(current)
    union = previous_shape.union(current_shape).area
    iou = previous_shape.intersection(current_shape).area / union if union else 1.0
    area_change = (
        100.0 * (current_shape.area - previous_shape.area) / previous_shape.area
        if previous_shape.area
        else 0.0
    )
    inherited = dict(source or {})
    source_method = str(inherited.get("method", "single_height_parallax"))
    return {
        **inherited,
        **_physical_measurements(current),
        "method": "manual_physical_outline",
        "source_method": source_method,
        "manual_iou_with_previous": round(float(iou), 4),
        "manual_area_change_pct": round(float(area_change), 3),
        "manual_hausdorff_mm": round(
            float(previous_shape.hausdorff_distance(current_shape)), 4
        ),
    }


def _prompt_lists(sess: dict) -> tuple[list[list[float]], list[int]]:
    points = sess.get("points", [])
    if points and isinstance(points[0], dict):
        return (
            [[float(point["x"]), float(point["y"])] for point in points],
            [int(point.get("label", 1)) for point in points],
        )
    return (
        [[float(point[0]), float(point[1])] for point in points],
        [int(value) for value in sess.get("labels", [1] * len(points))],
    )


def _prompt_json(sess: dict) -> list[dict]:
    points, labels = _prompt_lists(sess)
    return [
        {"x": point[0], "y": point[1], "label": label}
        for point, label in zip(points, labels, strict=True)
    ]


def _edit_diagnostics(
    previous: np.ndarray, current: np.ndarray, initial: np.ndarray
) -> dict:
    diagnostics = _mask_change(previous, current)
    initial_change = _mask_change(initial, current)
    diagnostics.update({
        "iou_with_initial": initial_change["iou_with_previous"],
        "area_change_from_initial_pct": initial_change["area_change_pct"],
        "mask_area_px": int((current > 127).sum()),
    })
    return diagnostics


def _edit_snapshot(
    sess: dict, *, revision: int, operation: str, diagnostics: dict
) -> dict:
    points, labels = _prompt_lists(sess)
    return {
        "revision": revision,
        "operation": operation,
        "mask": sess["mask"].copy(),
        "points": points,
        "labels": labels,
        "diagnostics": dict(diagnostics),
    }


def _ensure_edit_history(sess: dict) -> None:
    if sess.get("_edit_history"):
        return
    revision = int(sess.get("revision", 0))
    initial = sess.get("initial_mask", sess["mask"])
    diagnostics = _edit_diagnostics(sess["mask"], sess["mask"], initial)
    sess["_edit_history"] = [
        _edit_snapshot(
            sess, revision=revision, operation="initial", diagnostics=diagnostics
        )
    ]
    sess["_edit_cursor"] = 0
    sess["_next_revision"] = revision


def _record_edit(
    sess: dict,
    mask: np.ndarray,
    points: list[list[float]],
    labels: list[int],
    *,
    operation: str,
) -> None:
    _ensure_edit_history(sess)
    previous = sess["mask"]
    binary = ((mask > 127) * 255).astype("uint8")
    sess["mask"] = binary
    # A changed segmentation gets a freshly measured recommendation. An
    # already accepted library/batch baseline may set cleanup_default explicitly.
    sess.pop("cleanup_default", None)
    sess["points"] = [[float(x), float(y)] for x, y in points]
    sess["labels"] = [int(value) for value in labels]
    sess["_next_revision"] += 1
    sess["revision"] = sess["_next_revision"]
    diagnostics = _edit_diagnostics(
        previous, binary, sess.get("initial_mask", binary)
    )
    history = sess["_edit_history"][: sess["_edit_cursor"] + 1]
    history.append(
        _edit_snapshot(
            sess,
            revision=sess["revision"],
            operation=operation,
            diagnostics=diagnostics,
        )
    )
    sess["_edit_history"] = history
    sess["_edit_cursor"] = len(history) - 1


def _move_edit_history(sess: dict, direction: Literal["undo", "redo"]) -> None:
    _ensure_edit_history(sess)
    delta = -1 if direction == "undo" else 1
    target = sess["_edit_cursor"] + delta
    if target < 0 or target >= len(sess["_edit_history"]):
        raise HTTPException(status_code=409, detail=f"nothing to {direction}")
    snap = sess["_edit_history"][target]
    sess["_edit_cursor"] = target
    sess["mask"] = snap["mask"].copy()
    sess["points"] = [list(point) for point in snap["points"]]
    sess["labels"] = list(snap["labels"])
    sess["revision"] = snap["revision"]


def _public_edit_history(sess: dict) -> list[dict]:
    _ensure_edit_history(sess)
    return [
        {
            "revision": snap["revision"],
            "operation": snap["operation"],
            "diagnostics": snap["diagnostics"],
        }
        for snap in sess["_edit_history"]
    ]


def _accepted_outline_revisions(
    sess: dict,
    calibration,
    thickness: float,
    current_corrected=None,
    outline_variant: str = "recommended",
) -> list[library_mod.OutlineEditRevision]:
    """Convert accepted masks and cleanup choices into physical revisions."""
    _ensure_edit_history(sess)
    revisions: list[library_mod.OutlineEditRevision] = []
    baseline = sess["_edit_history"][0]
    current = sess["_edit_history"][sess["_edit_cursor"]]
    accepted = [baseline]
    if current["revision"] != baseline["revision"]:
        accepted.append(current)
    for snap in accepted:
        candidates = _editor_candidates(snap["mask"], calibration)
        variant = (
            outline_variant
            if snap["revision"] == current["revision"]
            else "recommended"
        )
        raw, _warnings, cleanup, resolved, _source = _accepted_candidate(
            candidates, variant
        )
        if raw is None:
            continue
        corrected = (
            current_corrected
            if current_corrected is not None
            and snap["revision"] == current["revision"]
            else parallax_mod.correct_polygon(raw, calibration, thickness)
            if thickness
            else raw
        )
        operation = snap["operation"]
        source = (
            "baseline"
            if operation == "initial"
            else "manual"
            if operation == "manual"
            else "sam"
        )
        diagnostics = dict(snap["diagnostics"])
        diagnostics.update({
            "cleanup_variant": resolved,
            "cleanup_noise_mm": cleanup["noise_mm"],
            "cleanup_radius_mm": cleanup["radius_mm"],
            "cleanup_max_shift_mm": cleanup["max_shift_mm"],
            "cleanup_max_shift_cap_mm": cleanup["max_shift_cap_mm"],
        })
        revisions.append(library_mod.OutlineEditRevision(
            revision=snap["revision"],
            created_ts=int(time.time()),
            source=source,
            raw_outline=raw,
            outline=corrected,
            diagnostics=diagnostics,
        ))
    return revisions


def _editor_payload(sess: dict) -> dict:
    _ensure_edit_history(sess)
    candidates = _editor_candidates_for_session(sess)
    polygon = candidates["raw_px"]
    cleaned_polygon = candidates["cleaned_px"]
    accepted_variant = sess.get("cleanup_default") or candidates["cleanup"][
        "recommended"
    ]
    snap = sess["_edit_history"][sess["_edit_cursor"]]
    diagnostics = dict(snap["diagnostics"])
    diagnostics.update({
        "vertex_count": len(polygon["exterior"]),
        "hole_count": len(polygon["holes"]),
    })
    return {
        "outline": polygon["exterior"],
        "polygon": polygon,
        "cleaned_polygon": cleaned_polygon,
        # Compatibility alias for older web clients.
        "smooth_polygon": cleaned_polygon,
        "cleanup": candidates["cleanup"],
        "accepted_variant": accepted_variant,
        "points": _prompt_json(sess),
        "revision": snap["revision"],
        "operation": snap["operation"],
        "diagnostics": diagnostics,
        "iou_with_previous": diagnostics["iou_with_previous"],
        "area_change_pct": diagnostics["area_change_pct"],
        "can_undo": sess["_edit_cursor"] > 0,
        "can_redo": sess["_edit_cursor"] < len(sess["_edit_history"]) - 1,
        "history_index": sess["_edit_cursor"],
        "history_length": len(sess["_edit_history"]),
    }


def _refine_editor(
    sess: dict,
    points: list[list[float]],
    labels: list[int],
    box: list[float] | None = None,
) -> float:
    if not points and box is None:
        mask = sess["initial_mask"].copy()
        score = 1.0
        operation = "reset"
    else:
        try:
            mask, score = seg_client.decode(
                sess["image_id"],
                points,
                labels,
                box=box,
                mask_poly=_outline_px(sess["mask"]) if box is None else None,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"segserver decode failed: {exc}"
            )
        mask = contour_mod.select_prompted_components(
            mask, points, labels, box=box
        )
        operation = "sam-box" if box is not None else "sam-points"
    _record_edit(sess, mask, points, labels, operation=operation)
    return float(score)


def _mask_from_pixel_poly(sess: dict, polygon: PixelPoly) -> np.ndarray:
    import cv2

    if len(polygon.exterior) < 3 or any(len(ring) < 3 for ring in polygon.holes):
        raise HTTPException(
            status_code=422, detail="outline and holes need at least 3 points"
        )
    calibration = sess.get("cal_full") or sess.get("calibration")
    data = polygon.model_dump()
    if calibration is not None:
        try:
            cleaned = contour_mod.clean(
                _px_data_to_poly(data, calibration), simplify_tol=0.0
            )
        except (ValueError, contour_mod.NoToolFoundError) as exc:
            raise HTTPException(status_code=422, detail=f"invalid outline: {exc}")
        data = _poly_to_px_data(cleaned, calibration)
    mask = np.zeros_like(sess["mask"], dtype=np.uint8)
    exterior = np.rint(np.asarray(data["exterior"])).astype(np.int32)
    cv2.fillPoly(mask, [exterior], 255)
    for hole in data["holes"]:
        cv2.fillPoly(mask, [np.rint(np.asarray(hole)).astype(np.int32)], 0)
    if not np.any(mask):
        raise HTTPException(status_code=422, detail="outline is empty")
    return mask


def _pick_verified_mat(mat_id: Optional[str]):
    if mat_id:
        return mat_mod.load_profile(mat_id)
    verified = [p for p in mat_mod.list_profiles() if p.verified]
    if len(verified) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"{len(verified)} verified mats — specify one",
        )
    return verified[0]


def _poly_json(poly) -> dict | None:
    if poly is None:
        return None
    return {"exterior": [list(p) for p in poly.exterior],
            "holes": [[list(p) for p in h] for h in poly.holes]}


def _reflect_poly_y(poly: Poly) -> Poly:
    """CAD/bin y-up polygon -> editor/mat y-down polygon, or the inverse."""
    return Poly(
        exterior=[(float(x), float(-y)) for x, y in poly.exterior],
        holes=[
            [(float(x), float(-y)) for x, y in ring]
            for ring in poly.holes
        ],
    )


def _physical_measurements(poly: Poly) -> dict[str, float]:
    shape = contour_mod.to_shapely(poly)
    minx, miny, maxx, maxy = shape.bounds
    major, minor = sorted((maxx - minx, maxy - miny), reverse=True)
    return {
        "reconstructed_major_extent_mm": round(float(major), 4),
        "reconstructed_minor_extent_mm": round(float(minor), 4),
        "physical_area_mm2": round(float(shape.area), 4),
    }


def _invalidate_physical_override(sess: dict) -> None:
    """A photo-space edit invalidates any override based on an older outline."""
    for key in (
        "physical_override",
        "physical_override_mask_revision",
        "physical_override_diagnostics",
        "physical_editor_poly",
        "physical_reconstruction",
    ):
        sess.pop(key, None)


def _session_physical_override(sess: dict) -> Poly | None:
    if sess.get("physical_override_mask_revision") != sess.get("revision", 0):
        return None
    return sess.get("physical_override")


def _remember_physical_result(sess: dict, result) -> None:
    if result.tool_poly is None:
        return
    # The result preview is in CAD/bin coordinates (y up). Reflecting it gives
    # the physical editor a y-down SVG polygon; sending that polygon back as a
    # mat-frame override preserves chirality when derivation flips to CAD again.
    sess["physical_editor_poly"] = _reflect_poly_y(result.tool_poly)
    sess["physical_reconstruction"] = result.reconstruction


_PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


def _photo_ext(name: Optional[str]) -> str:
    """Extension only, from an upload filename — never the whole name.

    Phones send both shots of a two-photo capture under the same generic name
    (e.g. "image.jpg"); reusing it as the on-disk path makes photo2 overwrite
    photo1, so the two-view thickness solve compares a photo against itself
    (nadir/height Δ0 → thickness clamps to its ceiling). Fixed names avoid it.
    """
    ext = Path(name or "").suffix.lower()
    return ext if ext in _PHOTO_EXTS else ".jpg"


def _result_payload(
    result, project_id: str, clearance: float, lip: bool,
    round_tool: bool = False, finger_hole: bool = True,
    pocket_depth_override_mm: float | None = None,
    overall_height_override_mm: float | None = None,
) -> dict:
    cal = result.calibration
    source_images = sorted(
        path.name for path in (PROJECTS / project_id).glob("photo*")
        if path.is_file()
    )
    provenance = readiness_mod.ArtifactProvenance(
        flow="single",
        mat_id=cal.mat_id,
        device_profile_id=cal.device_profile_id,
        device_profile_revision=cal.device_profile_revision,
        intrinsics_source=cal.intrinsics_source,
        capture_signature=cal.capture_signature,
        thickness_source=result.thickness_source,
        source_images=source_images,
        warnings=list(result.warnings),
    )
    readiness = result.readiness or readiness_mod.evaluate(
        calibration=cal,
        warnings=result.warnings,
        outline=result.raw_poly,
        thickness_mm=result.thickness_mm,
        thickness_source=result.thickness_source,
    )
    result_lip = getattr(result, "lip", lip)
    result_style = getattr(result, "bin_style", "pocket")
    result_overall_height = getattr(
        result,
        "overall_height_mm",
        grid_mod.finished_height_mm(result.height_u, result_lip),
    )
    return {
        "project": project_id,
        "bin": {
            "grid": list(result.grid),
            "height_u": result.height_u,
            "overall_height_mm": round(result_overall_height, 1),
            "bin_style": result_style,
            "pocket_depth_mm": round(result.pocket_depth_mm, 2),
            "pocket_depth_override_mm": pocket_depth_override_mm,
            "overall_height_override_mm": overall_height_override_mm,
            "thickness_mm": round(result.thickness_mm, 2),
            "silhouette_height_mm": round(
                getattr(result, "silhouette_height_mm", result.thickness_mm), 2
            ),
            "full_height_mm": getattr(result, "full_height_mm", None),
            "clearance_mm": clearance,
            "lip": result_lip,
            "round_tool": round_tool,
            "finger_hole": finger_hole,
            "derivation_key": result.derivation_key,
            "reserved_cells": [
                list(cell) for cell in getattr(result, "reserved_cells", [])
            ],
            "available_cells": [
                list(cell) for cell in getattr(result, "available_cells", [])
            ],
        },
        "calibration": {
            "corners": cal.n_corners,
            "rms_px": round(cal.reproj_rms_px, 2),
            "tilt_deg": round(cal.tilt_deg, 1) if cal.tilt_deg is not None else None,
            "camera_height_mm": round(cal.camera_height_mm) if cal.camera_height_mm else None,
            "nadir_xy_mm": list(cal.nadir_xy_mm) if cal.nadir_xy_mm else None,
            "mat_id": cal.mat_id,
            "device_profile_id": cal.device_profile_id,
            "device_profile_revision": cal.device_profile_revision,
            "intrinsics_source": cal.intrinsics_source,
            "capture_signature": (
                cal.capture_signature.model_dump(mode="json")
                if cal.capture_signature
                else None
            ),
        },
        # Keep the lossless calibration beside the human-readable summary so a
        # physical G1 run can reproduce the exact millimetre transform.
        "calibration_model": cal.model_dump(mode="json"),
        "tool_poly": _poly_json(result.tool_poly),
        "pocket_poly": _poly_json(result.pocket_poly),
        "raw_poly": _poly_json(result.raw_poly),
        "corrected_poly": _poly_json(result.corrected_poly),  # mat-mm, for library regen
        "reconstruction": getattr(result, "reconstruction", None),
        "warnings": result.warnings,
        "readiness": readiness.model_dump(),
        "provenance": provenance.model_dump(),
        "files": {kind: f"/api/files/{project_id}/{path.name}"
                  for kind, path in result.files.items()},
    }


# ---------------------------------------------------------------------------
# interactive editor: session → click-refine → generate


async def session_start(
    file: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    mat_id: Optional[str] = Form(None),
) -> dict:
    sid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    proj = PROJECTS / sid
    proj.mkdir(parents=True, exist_ok=True)
    photo1 = proj / f"photo1{_photo_ext(file.filename)}"
    photo1.write_bytes(await file.read())
    photo2 = None
    if file2 is not None:
        photo2 = proj / f"photo2{_photo_ext(file2.filename)}"
        photo2.write_bytes(await file2.read())

    profile = _pick_verified_mat(mat_id)
    src = ingest_mod.load(photo1)
    prepared = devices_mod.prepare_image(src)
    pixels = prepared.pixels
    device = prepared.profile
    try:
        cal = calibrate_mod.calibrate_image(
            pixels,
            profile,
            K=prepared.K,
            dist=None,
            exif=src.exif,
            device_profile_id=device.device_id if device else None,
            device_profile_revision=device.revision if device else None,
            capture_signature=prepared.signature,
            intrinsics_source="profile" if device else None,
        )
    except calibrate_mod.DetectionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # display = the undistorted photo (mask + clicks live in these coords)
    disp = proj / "display.jpg"
    Image.fromarray(pixels).save(disp, quality=88)
    # Restart recovery re-embeds this exact, lossless editor coordinate space.
    Image.fromarray(pixels).save(proj / "session-image.png")

    if not seg_client.available():
        raise HTTPException(status_code=503, detail="segserver offline")
    image_id, w, h = seg_client.embed(pixels)

    # Keep the independently selected auto mask intact. Seed points describe it
    # for later refinement but must not replace a good box/diff result with an
    # unchecked point-only decode before the editor even opens.
    mask, seg_warnings = trace_mod._auto_mask(pixels, profile, cal)
    seeds = _seed_points(mask)
    points, labels = [list(p) for p in seeds], [1] * len(seeds)

    capture_warnings = [*cal.warnings, *prepared.warnings]
    if device is None:
        capture_warnings.append(
            "no device profile for this camera — run `gridshot calib intrinsics` "
            "for distortion-corrected traces"
        )
    capture_warnings += seg_warnings
    try:
        session_outline, pick_warnings = trace_mod._pick_tool(mask, cal)
    except contour_mod.NoToolFoundError as exc:
        session_outline, pick_warnings = None, [str(exc)]
    warnings = [*capture_warnings, *pick_warnings]
    readiness = readiness_mod.evaluate(
        calibration=cal,
        warnings=warnings,
        outline=session_outline,
        require_thickness=False,
    )

    pts_json = [{"x": p[0], "y": p[1], "label": lb} for p, lb in zip(points, labels)]
    cal_json = {
        "corners": cal.n_corners, "rms_px": round(cal.reproj_rms_px, 2),
        "tilt_deg": round(cal.tilt_deg, 1) if cal.tilt_deg is not None else None,
        "device_profile_id": cal.device_profile_id,
        "device_profile_revision": cal.device_profile_revision,
        "intrinsics_source": cal.intrinsics_source,
    }
    _SESSIONS[sid] = {
        "photo1": photo1, "photo2": photo2, "mat_id": profile.mat_id,
        "image_id": image_id, "mask": mask, "initial_mask": mask.copy(),
        "points": pts_json, "initial_points": list(pts_json),
        "calibration": cal_json, "warnings": warnings,
        "capture_warnings": capture_warnings, "readiness": readiness,
        "revision": 0,
        "cal_full": cal,  # full Calibration → library SAM re-editing
    }
    readiness = _refresh_session_readiness(_SESSIONS[sid])
    warnings = _SESSIONS[sid]["warnings"]
    _persist_single_session(sid, _SESSIONS[sid])
    return {
        "session": sid,
        "display": f"/api/files/{sid}/display.jpg",
        "width": w, "height": h,
        "calibration": cal_json,
        "has_photo2": photo2 is not None,
        "warnings": warnings,
        "readiness": readiness.model_dump(),
        **_editor_payload(_SESSIONS[sid]),
    }


def session_get(sid: str) -> dict:
    sess = _single_session(sid)
    import cv2

    mask = sess["mask"]
    h, w = mask.shape[:2]
    return {
        "session": sid,
        "display": f"/api/files/{sid}/display.jpg",
        "width": w, "height": h,
        "calibration": sess.get("calibration", {"corners": 0, "rms_px": 0, "tilt_deg": None}),
        "has_photo2": sess["photo2"] is not None,
        "warnings": sess.get("warnings", []),
        "readiness": (
            sess["readiness"].model_dump()
            if sess.get("readiness") is not None
            else readiness_mod.ReadinessReport().model_dump()
        ),
        **_editor_payload(sess),
    }


def _refresh_session_readiness(
    sess: dict,
    segmentation_confidence: float | None = None,
) -> readiness_mod.ReadinessReport:
    # Compatibility for in-memory sessions created before readiness existed.
    if sess.get("cal_full") is None:
        value = readiness_mod.ReadinessReport()
        sess["warnings"] = sess.get("warnings", [])
        sess["readiness"] = value
        return value
    try:
        _raw_outline, pick_warnings = trace_mod._pick_tool(
            sess["mask"], sess["cal_full"]
        )
        outline, cleanup_warnings, _cleanup, _variant, _source = (
            _accepted_editor_outline(sess, "recommended")
        )
    except (contour_mod.NoToolFoundError, HTTPException) as exc:
        outline = None
        cleanup_warnings = []
        pick_warnings = [
            str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
        ]
    warnings = [
        *sess.get("capture_warnings", []),
        *pick_warnings,
        *cleanup_warnings,
    ]
    if segmentation_confidence is None:
        sess.pop("segmentation_confidence", None)
    else:
        sess["segmentation_confidence"] = float(segmentation_confidence)
        warnings.append(f"SAM refine confidence {segmentation_confidence:.2f}")
    value = readiness_mod.evaluate(
        calibration=sess.get("cal_full"),
        warnings=warnings,
        outline=outline,
        require_thickness=False,
    )
    sess["warnings"] = warnings
    sess["readiness"] = value
    return value


def session_click(
    sid: str,
    points: str = Form(...),
    labels: str = Form(...),
    box: str = Form(None),
) -> dict:
    sess = _single_session(sid)
    pts = json.loads(points)
    lbs = json.loads(labels)
    bx = json.loads(box) if isinstance(box, str) and box else None
    if pts or bx:
        _ensure_session_embedding(sid, sess)
    score = _refine_editor(sess, pts, lbs, bx)
    _invalidate_physical_override(sess)
    readiness = _refresh_session_readiness(sess, score if pts or bx else None)
    _persist_single_session(sid, sess)
    return {
        "score": round(score, 3),
        "warnings": sess["warnings"],
        "readiness": readiness.model_dump(),
        **_editor_payload(sess),
    }


def session_set_outline(sid: str, polygon: PixelPoly) -> dict:
    sess = _single_session(sid)
    mask = _mask_from_pixel_poly(sess, polygon)
    _record_edit(sess, mask, [], [], operation="manual")
    _invalidate_physical_override(sess)
    readiness = _refresh_session_readiness(sess)
    _persist_single_session(sid, sess)
    return {
        "score": 1.0,
        "warnings": sess["warnings"],
        "readiness": readiness.model_dump(),
        **_editor_payload(sess),
    }


def session_edit_history(
    sid: str, direction: Literal["undo", "redo"]
) -> dict:
    sess = _single_session(sid)
    _move_edit_history(sess, direction)
    _invalidate_physical_override(sess)
    readiness = _refresh_session_readiness(sess)
    _persist_single_session(sid, sess)
    return {
        "score": 1.0,
        "warnings": sess["warnings"],
        "readiness": readiness.model_dump(),
        **_editor_payload(sess),
    }


def session_set_physical_outline(sid: str, polygon: PhysicalPoly) -> dict:
    """Save a WYSIWYG physical footprint; generation must not parallax it again."""
    sess = _single_session(sid)
    baseline = sess.get("physical_editor_poly")
    if baseline is None:
        raise HTTPException(
            status_code=409,
            detail="generate the reconstructed cutout before editing it",
        )
    cleaned = _validated_physical_outline(polygon)
    diagnostics = _manual_physical_diagnostics(
        baseline, cleaned, sess.get("physical_reconstruction")
    )
    sess["physical_override"] = cleaned
    sess["physical_override_mask_revision"] = sess.get("revision", 0)
    sess["physical_override_diagnostics"] = diagnostics
    sess["physical_override_revision"] = int(
        sess.get("physical_override_revision", 0)
    ) + 1
    _persist_single_session(sid, sess)
    return {
        "polygon": _poly_json(cleaned),
        "revision": sess["physical_override_revision"],
        "diagnostics": diagnostics,
    }


def session_generate(
    sid: str,
    thickness: Optional[float] = Form(None),
    clearance: float = Form(1.0),
    bin_style: Literal["pocket", "corral", "grid"] = Form("pocket"),
    depth: Optional[float] = Form(None),
    full_height: Optional[float] = Form(None),
    overall_height: Optional[float] = Form(None),
    finger_hole: bool = Form(True),
    lip: bool = Form(True),
    round_tool: bool = Form(False),
    outline_variant: str = Form("recommended"),
) -> dict:
    sess = _single_session(sid)
    if not isinstance(full_height, (int, float)):
        full_height = None
    if sess["photo2"] is None and thickness is None:
        raise HTTPException(status_code=422, detail="add a second photo or a thickness")
    proj = PROJECTS / sid
    physical_override = _session_physical_override(sess)
    if bin_style not in ("pocket", "corral", "grid"):
        bin_style = "pocket"
    try:
        accepted_outline, outline_warnings, _cleanup, resolved, _source = (
            _accepted_editor_outline(sess, outline_variant)
        )
        sess["cleanup_default"] = resolved
        result = trace_mod.run(
            sess["photo1"],
            thickness_mm=thickness,
            photo2=sess["photo2"],
            clearance_mm=clearance,
            bin_style=bin_style,
            pocket_depth_mm=depth,
            full_height_mm=full_height,
            overall_height_mm=overall_height,
            finger_hole=finger_hole,
            lip=lip,
            round_tool=round_tool,
            mat_id=sess["mat_id"],
            out_dir=proj,
            stem="bin",
            mask=sess["mask"],
            smooth_mm=0.0,
            outline_override=accepted_outline,
            outline_override_warnings=outline_warnings,
            corrected_override=physical_override,
            reconstruction_override=(
                sess.get("physical_override_diagnostics")
                if physical_override is not None
                else None
            ),
        )
    except (contour_mod.NoToolFoundError, RuntimeError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    payload = _result_payload(
        result,
        sid,
        clearance,
        lip,
        round_tool,
        finger_hole,
        pocket_depth_override_mm=depth,
        overall_height_override_mm=overall_height,
    )
    _remember_physical_result(sess, result)
    payload["outline_edits"] = _public_edit_history(sess)
    payload["physical_outline_edit"] = sess.get(
        "physical_override_diagnostics"
    )
    (proj / "result.json").write_text(json.dumps(payload))
    _persist_single_session(sid, sess)
    return payload


def session_add_to_library(
    sid: str,
    thickness: Optional[float] = Form(None),
    clearance: float = Form(1.0),
    bin_style: Literal["pocket", "corral", "grid"] = Form("pocket"),
    depth: Optional[float] = Form(None),
    full_height: Optional[float] = Form(None),
    finger_hole: bool = Form(True),
    lip: bool = Form(True),
    round_tool: bool = Form(False),
    outline_variant: str = Form("recommended"),
) -> dict:
    """Save the accepted selection without constructing or exporting a bin."""
    sess = _single_session(sid)
    if not isinstance(full_height, (int, float)):
        full_height = None
    if sess["photo2"] is None and thickness is None:
        raise HTTPException(status_code=422, detail="add a second photo or a thickness")

    profile = mat_mod.load_profile(sess["mat_id"])
    # FastAPI unwraps Form defaults for HTTP requests; direct Python callers see
    # the FieldInfo object when they omit a newly-added optional argument.
    if bin_style not in ("pocket", "corral", "grid"):
        bin_style = "pocket"
    try:
        accepted_outline, outline_warnings, _cleanup, resolved, _source = (
            _accepted_editor_outline(sess, outline_variant)
        )
        sess["cleanup_default"] = resolved
        thickness_source = "automatic" if thickness is None else "manual"
        captured = trace_mod.capture_tool_geometry(
            sess["photo1"],
            profile,
            thickness_mm=thickness,
            photo2=sess["photo2"],
            smooth_mm=0.0,
            mask=sess["mask"],
            outline_override=accepted_outline,
            outline_override_warnings=outline_warnings,
        )
        calibration = captured.calibration
        raw = captured.raw_poly
        physical_override = _session_physical_override(sess)
        corrected = physical_override or captured.corrected_poly
        solved_thickness = captured.thickness_mm
        capture_warnings = list(captured.warnings)
        if physical_override is not None:
            capture_warnings.append(
                "physical cutout override: using the manually edited physical "
                "footprint without reapplying parallax"
            )
        readiness = readiness_mod.evaluate(
            calibration=calibration,
            warnings=capture_warnings,
            outline=corrected,
            thickness_mm=solved_thickness,
            thickness_source=thickness_source,
        )
        if readiness.blocked:
            raise ValueError(
                f"not ready: {readiness_mod.blocking_message(readiness)}"
            )
        spec = derive_mod.derive_bin_spec(
            derive_mod.ToolGeometry(
                outline=corrected,
                silhouette_height_mm=solved_thickness,
                full_height_mm=full_height,
            ),
            derive_mod.BinSettings(
                clearance_mm=clearance,
                bin_style=bin_style,
                pocket_depth_mm=depth,
                lip=lip,
                finger_hole=finger_hole,
                round_tool=round_tool,
            ),
            bench_mod.load_profile() or bench_mod.default_profile(),
        )
        grid = spec.grid
    except (contour_mod.NoToolFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    outline_history = _accepted_outline_revisions(
        sess,
        calibration,
        solved_thickness,
        current_corrected=corrected,
        outline_variant=resolved,
    )
    tid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    saved = _add_entry(
        tid,
        grid,
        solved_thickness,
        sid,
        "",
        corrected.exterior,
        outline=corrected,
        raw_outline=raw,
        clearance=clearance,
        bin_style=bin_style,
        depth=depth,
        full_height=full_height,
        lip=lip,
        round_tool=round_tool,
        finger_hole=finger_hole,
        calibration=calibration,
        photo_src=PROJECTS / sid / "display.jpg",
        readiness=readiness,
        outline_history=outline_history,
        outline_revision=outline_history[-1].revision if outline_history else 0,
        provenance=readiness_mod.ArtifactProvenance(
            flow="single",
            mat_id=calibration.mat_id,
            device_profile_id=calibration.device_profile_id,
            device_profile_revision=calibration.device_profile_revision,
            intrinsics_source=calibration.intrinsics_source,
            capture_signature=calibration.capture_signature,
            thickness_source=thickness_source,
            source_images=[
                path.name for path in (PROJECTS / sid).glob("photo*")
                if path.is_file()
            ],
            warnings=capture_warnings,
        ),
    )
    _persist_single_session(sid, sess)
    return _lib_json(saved)


# ---------------------------------------------------------------------------
# photo/mat coordinate transforms used by the saved-tool editor


def _project_ring_to_px(ring, calibration) -> list[list[float]]:
    import cv2

    h_inv = np.linalg.inv(np.asarray(calibration.H_img_to_mm, dtype=np.float64))
    pts = np.asarray(ring, dtype=np.float64).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in px]


def _poly_to_px_data(poly, calibration) -> dict:
    """Project a full mat-mm polygon into display pixels without losing holes."""
    return {
        "exterior": _project_ring_to_px(poly.exterior, calibration),
        "holes": [
            _project_ring_to_px(ring, calibration) for ring in poly.holes
        ],
    }


def _poly_to_px(poly, calibration) -> list[list[float]]:
    """Backward-compatible exterior-only projection for thumbnails/crops."""
    return _poly_to_px_data(poly, calibration)["exterior"]


def _px_data_to_poly(data: dict, calibration):
    """Display-image pixel polygon → mat-mm, preserving all interior rings."""
    import cv2

    H = np.asarray(calibration.H_img_to_mm, dtype=np.float64)

    def transform(ring):
        pts = np.asarray(ring, dtype=np.float64).reshape(-1, 1, 2)
        mm = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return [[float(x), float(y)] for x, y in mm]

    return contour_mod.Poly(
        exterior=transform(data["exterior"]),
        holes=[transform(ring) for ring in data.get("holes", [])],
    )


def _px_to_poly(points_px, calibration):
    """Backward-compatible exterior-only display-pixel conversion."""
    return _px_data_to_poly(
        {"exterior": points_px, "holes": []}, calibration
    )


def _editor_candidates(mask: np.ndarray, calibration) -> dict:
    """Return raw and physically bounded cleanup candidates for one mask."""
    import cv2

    components = contour_mod.mask_to_polygons_px(mask)
    if not components:
        empty = {"exterior": [], "holes": []}
        return {
            "raw_px": empty,
            "cleaned_px": empty,
            "raw_mm": None,
            "cleaned_mm": None,
            "cleanup": {
                "available": False,
                "recommended": "raw",
                "noise_mm": 0.0,
                "radius_mm": 0.0,
                "straightened": False,
                "max_shift_cap_mm": quality_mod.CLEANUP_MAX_SHIFT_MM,
                "symdiff_mm2": 0.0,
                "mean_shift_mm": 0.0,
                "max_shift_mm": 0.0,
                "area_ratio": 1.0,
                "reason": "outline is empty",
            },
        }
    exterior, holes = components[0]
    if calibration is None:
        def simplify(ring):
            value = cv2.approxPolyDP(
                ring.astype(np.float32).reshape(-1, 1, 2), 1.0, True
            ).reshape(-1, 2)
            return [[float(x), float(y)] for x, y in value]

        polygon = {
            "exterior": simplify(exterior),
            "holes": [simplify(ring) for ring in holes],
        }
        return {
            "raw_px": polygon,
            "cleaned_px": polygon,
            "raw_mm": None,
            "cleaned_mm": None,
            "cleanup": {
                "available": False,
                "recommended": "raw",
                "noise_mm": 0.0,
                "radius_mm": 0.0,
                "straightened": False,
                "max_shift_cap_mm": quality_mod.CLEANUP_MAX_SHIFT_MM,
                "symdiff_mm2": 0.0,
                "mean_shift_mm": 0.0,
                "max_shift_mm": 0.0,
                "area_ratio": 1.0,
                "reason": "cleanup requires calibrated millimetre geometry",
            },
        }

    raw_mm = contour_mod.polygon_px_to_mm(exterior, holes, calibration)
    bounded = contour_mod.clean(raw_mm, simplify_tol=EDITOR_SIMPLIFY_MM)
    try:
        cleaned, cleanup = quality_mod.bounded_cleanup(bounded)
    except (ValueError, contour_mod.NoToolFoundError) as exc:
        cleaned = bounded
        cleanup = {
            "available": False,
            "recommended": "raw",
            "noise_mm": quality_mod.estimate_noise_mm(bounded),
            "radius_mm": 0.0,
            "straightened": False,
            "max_shift_cap_mm": quality_mod.CLEANUP_MAX_SHIFT_MM,
            "symdiff_mm2": 0.0,
            "mean_shift_mm": 0.0,
            "max_shift_mm": 0.0,
            "area_ratio": 1.0,
            "reason": f"cleanup failed: {exc}",
        }
    return {
        "raw_px": _poly_to_px_data(bounded, calibration),
        "cleaned_px": _poly_to_px_data(cleaned, calibration),
        "raw_mm": bounded,
        "cleaned_mm": cleaned,
        "cleanup": cleanup,
    }


def _editor_candidates_for_session(sess: dict) -> dict:
    """Cache cleanup work for one immutable mask revision."""
    calibration = sess.get("cal_full") or sess.get("calibration")
    key = (sess.get("revision", 0), id(sess["mask"]), id(calibration))
    cached = sess.get("_editor_candidates_cache")
    if cached is not None and cached["key"] == key:
        return cached["value"]
    value = _editor_candidates(sess["mask"], calibration)
    sess["_editor_candidates_cache"] = {"key": key, "value": value}
    return value


def _editor_polygons(mask: np.ndarray, calibration) -> tuple[dict, dict, object]:
    """Compatibility view used by older tests and editor helpers."""
    candidates = _editor_candidates(mask, calibration)
    return (
        candidates["raw_px"],
        candidates["cleaned_px"],
        candidates["raw_mm"],
    )


def _accepted_candidate(candidates: dict, outline_variant: str):
    if outline_variant == "recommended":
        resolved = candidates["cleanup"]["recommended"]
    elif outline_variant in {"raw", "cleaned"}:
        resolved = outline_variant
    else:
        raise HTTPException(
            status_code=422, detail="outline variant must be raw or cleaned"
        )
    if resolved == "cleaned" and not candidates["cleanup"]["available"]:
        raise HTTPException(
            status_code=422,
            detail=candidates["cleanup"].get(
                "reason", "a bounded cleaned outline is unavailable"
            ),
        )
    selected = (
        candidates["cleaned_mm"]
        if resolved == "cleaned"
        else candidates["raw_mm"]
    )
    cleanup = candidates["cleanup"]
    warning = (
        "accepted photo outline: bounded cleanup "
        f"{cleanup['radius_mm']:.2f}mm, max shift "
        f"{cleanup['max_shift_mm']:.2f}mm / "
        f"{cleanup['max_shift_cap_mm']:.2f}mm cap"
        if resolved == "cleaned"
        else "accepted photo outline: raw segmentation; automatic cleanup not applied"
    )
    return selected, [warning], cleanup, resolved, candidates["raw_mm"]


def _accepted_editor_outline(sess: dict, outline_variant: str = "recommended"):
    if not isinstance(outline_variant, str):
        # Direct Python calls to FastAPI route functions receive the Form default
        # object rather than a parsed string. Treat those as the API default.
        outline_variant = "recommended"
    variant = (
        sess.get("cleanup_default", "recommended")
        if outline_variant == "recommended"
        else outline_variant
    )
    return _accepted_candidate(
        _editor_candidates_for_session(sess), variant
    )


# tool library: save individually-captured tools, compose them into a drawer


def _render_thumb(points, path, size: int = 160, pad: int = 12) -> None:
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, miny = min(xs), min(ys)
    w = (max(xs) - minx) or 1.0
    h = (max(ys) - miny) or 1.0
    scale = (size - 2 * pad) / max(w, h)
    off_x = (size - w * scale) / 2
    off_y = (size - h * scale) / 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).polygon(
        [((x - minx) * scale + off_x, (y - miny) * scale + off_y) for x, y in points],
        fill=(47, 143, 149, 255),
    )
    img.save(path)


def _add_entry(tool_id, grid, thickness, project, source_tool, thumb_points,
               outline=None, clearance=1.0, bin_style="pocket", depth=None,
               lip=True, calibration=None, photo_src=None,
               round_tool=False, finger_hole=True, full_height=None,
               raw_outline=None, readiness=None, provenance=None,
               outline_history=None, outline_revision=0):
    _render_thumb(thumb_points, library_mod.library_dir() / f"{tool_id}.png")
    has_photo = _store_lib_photo(tool_id, photo_src) if (photo_src and calibration is not None) else False
    outline_poly = contour_mod.Poly(**outline) if isinstance(outline, dict) else outline
    raw_poly = contour_mod.Poly(**raw_outline) if isinstance(raw_outline, dict) else raw_outline
    if raw_poly is None:
        raw_poly = _raw_outline_for_photo(outline_poly, calibration, thickness)
    tool = library_mod.LibraryTool(
        id=tool_id, grid_x=float(grid[0]), grid_y=float(grid[1]),
        thickness_mm=float(thickness or 0.0),
        silhouette_height_mm=float(thickness or 0.0) or None,
        full_height_mm=full_height,
        raw_outline=raw_poly, outline=outline_poly,
        clearance_mm=float(clearance or 1.0), bin_style=bin_style,
        pocket_depth_mm=depth,
        round_tool=bool(round_tool), finger_hole=bool(finger_hole), lip=bool(lip),
        has_photo=has_photo, calibration=calibration,
        source_project=project, source_tool=source_tool,
        readiness=readiness, provenance=provenance, created_ts=int(time.time()),
        outline_history=outline_history or [],
        outline_revision=outline_revision,
    )
    saved = library_mod.save(
        tool if tool.readiness is not None else _refresh_tool_readiness(tool)
    )
    _regen_photo_thumb(saved)
    return saved


def library_add(project: str) -> dict:
    proj = (PROJECTS / project).resolve()
    if not str(proj).startswith(str(PROJECTS.resolve())):
        raise HTTPException(status_code=404, detail="no such project")
    added = []
    sess = _single_session_optional(project)
    cal_full = sess.get("cal_full") if sess else None
    disp = proj / "display.jpg"
    if (proj / "result.json").is_file():  # single tool
        r = json.loads((proj / "result.json").read_text())
        b = r["bin"]
        outline_data = r.get("raw_poly") or r.get("corrected_poly")
        stored_readiness = (
            readiness_mod.ReadinessReport.model_validate(r["readiness"])
            if r.get("readiness") is not None
            else readiness_mod.evaluate(
                calibration=cal_full,
                warnings=r.get("warnings", []),
                outline=contour_mod.Poly.model_validate(outline_data),
                thickness_mm=b.get("thickness_mm"),
                thickness_source="legacy",
                require_calibration=cal_full is not None,
            )
        )
        if stored_readiness.blocked:
            raise HTTPException(
                status_code=422,
                detail=f"result is not ready: {readiness_mod.blocking_message(stored_readiness)}",
            )
        tid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        added.append(_add_entry(
            tid, b["grid"], b.get("thickness_mm"), project, "", r["tool_poly"]["exterior"],
            outline=r.get("corrected_poly"), clearance=b.get("clearance_mm", 1.0),
            bin_style=b.get("bin_style", "pocket"),
            depth=b.get("pocket_depth_override_mm"),
            full_height=b.get("full_height_mm"),
            lip=b.get("lip", True), raw_outline=r.get("raw_poly"),
            round_tool=b.get("round_tool", False), finger_hole=b.get("finger_hole", True),
            calibration=cal_full, photo_src=disp,
            readiness=stored_readiness, provenance=r.get("provenance")))
    else:
        raise HTTPException(status_code=422, detail="project has no result to save")
    return {"added": [a.model_dump() for a in added]}


def _lib_json(t, printer_profile=None) -> dict:
    d = t.model_dump()
    d["readiness"] = _tool_readiness(t).model_dump()
    if t.outline is not None and t.thickness_mm > 0:
        spec = library_mod.derive_tool_spec(
            t, printer_profile=printer_profile
        )
        d.update({
            "grid_x": float(spec.grid[0]),
            "grid_y": float(spec.grid[1]),
            "derived_pocket_depth_mm": round(spec.pocket_depth_mm, 2),
            "derived_height_u": spec.height_u,
            "derived_overall_height_mm": round(spec.overall_height_mm, 2),
            "derived_key": spec.derivation_key,
            "derived_reserved_cells": [list(cell) for cell in spec.reserved_cells],
            "derived_available_cells": [list(cell) for cell in spec.available_cells],
        })
    else:
        d.update({
            "derived_pocket_depth_mm": None,
            "derived_height_u": None,
            "derived_overall_height_mm": None,
            "derived_key": None,
            "derived_reserved_cells": [],
            "derived_available_cells": [],
        })
    d.pop("outline", None)       # heavy; fetched per-tool for the editor
    d.pop("raw_outline", None)   # heavy; fetched only for photo editing
    d.pop("calibration", None)   # heavy; only used server-side for SAM re-editing
    d.pop("outline_history", None)  # heavy; revision number remains public
    d["thumb"] = f"/api/library/{t.id}/thumb"  # silhouette (always present)
    # a real-photo crop when the tool carries a photo — the browsable thumbnail.
    # Backfill it for tools saved before this existed, on first list.
    thumb_path = library_mod.library_dir() / f"{t.id}-photo-thumb.jpg"
    if t.has_photo and not thumb_path.is_file():
        _regen_photo_thumb(t)
    d["photo_thumb"] = f"/api/library/{t.id}/photo-thumb" if thumb_path.is_file() else None
    return d


def library_list() -> dict:
    printer = bench_mod.load_profile() or bench_mod.default_profile()
    return {
        "tools": [
            _lib_json(t, printer_profile=printer)
            for t in library_mod.list_tools()
        ]
    }


def library_export_archive() -> Response:
    try:
        data, _manifest = library_mod.export_archive()
    except library_mod.LibrarySchemaError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="gridshot-library.zip"'
        },
    )


def library_create_backup() -> dict:
    try:
        path, manifest = library_mod.create_backup()
    except library_mod.LibrarySchemaError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "tool_count": manifest["tool_count"],
        "created_at": manifest["created_at"],
    }


def library_outline(tool_id: str) -> dict:
    try:
        t = library_mod.load(tool_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such tool")
    return {"outline": t.outline.model_dump() if t.outline else None}


def library_thumb(tool_id: str) -> FileResponse:
    p = (library_mod.library_dir() / f"{tool_id}.png").resolve()
    if not str(p).startswith(str(library_mod.library_dir().resolve())) or not p.is_file():
        raise HTTPException(status_code=404, detail="no thumbnail")
    return FileResponse(p)


def library_photo(tool_id: str) -> FileResponse:
    p = (library_mod.library_dir() / f"{tool_id}-photo.jpg").resolve()
    if not str(p).startswith(str(library_mod.library_dir().resolve())) or not p.is_file():
        raise HTTPException(status_code=404, detail="no photo stored for this tool")
    return FileResponse(p)


def library_photo_thumb(tool_id: str) -> FileResponse:
    p = (library_mod.library_dir() / f"{tool_id}-photo-thumb.jpg").resolve()
    if not str(p).startswith(str(library_mod.library_dir().resolve())) or not p.is_file():
        raise HTTPException(status_code=404, detail="no photo thumbnail")
    return FileResponse(p)


def library_photo_outline(tool_id: str) -> dict:
    """For the vertex editor: the tool's photo + its outline projected into photo
    pixels, so vertices can be dragged on top of the real image. Falls back to
    geometry-only (mat-mm outline, no photo) for tools without a stored photo."""
    try:
        t = library_mod.load(tool_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such tool")
    photo = library_mod.library_dir() / f"{tool_id}-photo.jpg"
    if not (t.has_photo and t.calibration is not None and photo.is_file()):
        return {"has_photo": False, "outline": t.outline.exterior if t.outline else []}
    w, h = Image.open(photo).size
    raw = _library_raw_outline(t)
    return {
        "has_photo": True,
        "display": f"/api/library/{tool_id}/photo",
        "width": w, "height": h,
        "outline": _poly_to_px(raw, t.calibration) if raw else [],
    }


class OutlinePx(BaseModel):
    points: list[list[float]]  # edited outline in photo px


def library_save_outline_px(tool_id: str, req: OutlinePx) -> dict:
    """Save a vertex outline edited in photo px — map back to mat-mm and store."""
    try:
        t = library_mod.load(tool_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such tool")
    if t.calibration is None:
        raise HTTPException(status_code=422, detail="no calibration for this tool")
    if len(req.points) < 3:
        raise HTTPException(status_code=422, detail="need at least 3 points")
    raw = _px_to_poly(req.points, t.calibration)
    corrected = (
        parallax_mod.correct_polygon(raw, t.calibration, t.thickness_mm)
        if t.thickness_mm else raw
    )
    return _lib_json(_apply_outline(t, corrected, raw_outline=raw))


def _library_editor_payload(sess: dict, score: float = 1.0) -> dict:
    base = _editor_payload(sess)
    raw_mm, _warnings, _cleanup, _variant, _source = _accepted_editor_outline(
        sess, base["accepted_variant"]
    )
    if raw_mm is None:
        raise HTTPException(status_code=422, detail="empty mask — add a point on the tool")
    corrected = (
        parallax_mod.correct_polygon(raw_mm, sess["calibration"], sess["thickness"])
        if sess["thickness"] else raw_mm
    )
    spec = derive_mod.derive_bin_spec(
        derive_mod.ToolGeometry(
            outline=corrected, thickness_mm=sess["thickness"]
        ),
        derive_mod.BinSettings(
            clearance_mm=sess["clearance"],
            bin_style=sess["bin_style"],
            pocket_depth_mm=sess["depth"],
            lip=sess["lip"],
            finger_hole=sess["finger_hole"],
            round_tool=sess["round_tool"],
        ),
        bench_mod.load_profile() or bench_mod.default_profile(),
    )
    return {
        **base,
        "raw": _poly_json(raw_mm),
        "corrected": _poly_json(corrected),
        "grid_x": spec.grid[0],
        "grid_y": spec.grid[1],
        "score": round(score, 3),
    }


def library_edit_start(tool_id: str) -> dict:
    """Open a SAM re-segmentation session on a library tool's stored photo —
    the same click-to-refine experience as capture, but after the fact."""
    import cv2

    try:
        t = library_mod.load(tool_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such tool")
    photo = library_mod.library_dir() / f"{tool_id}-photo.jpg"
    if not t.has_photo or t.calibration is None or not photo.is_file():
        raise HTTPException(status_code=422, detail="no photo stored — use vertex editing instead")
    if not seg_client.available():
        raise HTTPException(status_code=503, detail="segserver offline")

    disp = np.asarray(Image.open(photo).convert("RGB"))
    h_px, w_px = disp.shape[:2]
    image_id, _, _ = seg_client.embed(disp)
    # seed SAM points from the current outline, projected back into the photo
    raw = _library_raw_outline(t)
    polygon_px = (
        _poly_to_px_data(raw, t.calibration)
        if raw else {"exterior": [], "holes": []}
    )
    outline_px = polygon_px["exterior"]
    m = np.zeros((h_px, w_px), np.uint8)
    pts = np.array(outline_px, np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 255)
    for hole in polygon_px["holes"]:
        cv2.fillPoly(m, [np.asarray(hole, np.int32)], 0)
    seeds = _seed_points(m)
    sid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    _LIB_EDIT_SESSIONS[sid] = {
        "tool_id": tool_id, "image_id": image_id, "calibration": t.calibration,
        "thickness": t.thickness_mm, "clearance": t.clearance_mm,
        "bin_style": t.bin_style, "depth": t.pocket_depth_mm, "lip": t.lip,
        "finger_hole": t.finger_hole, "round_tool": t.round_tool,
        # current outline in photo px → SAM's mask prior, so clicks refine this
        # shape rather than re-segmenting from scratch; updated after each click
        "outline_px": [[float(x), float(y)] for x, y in outline_px],
        "initial_outline_px": [[float(x), float(y)] for x, y in outline_px],
        "mask": m, "initial_mask": m.copy(),
        "points": seeds, "initial_points": list(seeds),
        "labels": [1] * len(seeds), "revision": 0,
        # This raster was reconstructed from an already accepted library outline.
        # Never recommend smoothing it again unless the user changes the mask.
        "cleanup_default": "raw",
    }
    return {
        "session": sid,
        "display": f"/api/library/{tool_id}/photo",
        "width": w_px, "height": h_px,
        **_library_editor_payload(_LIB_EDIT_SESSIONS[sid]),
    }


def library_edit_click(
    sid: str, points: str = Form("[]"), labels: str = Form("[]"), box: str = Form(None)
) -> dict:
    """Refine the mask with SAM points and/or a box; return the new photo outline
    plus the parallax-corrected mat-mm outline the caller saves back to the tool.
    A box ([x0,y0,x1,y1] px) is a strong whole-tool prompt for thin parts."""
    sess = _LIB_EDIT_SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="edit session expired — reopen")
    pts = json.loads(points)
    lbs = json.loads(labels)
    bx = json.loads(box) if isinstance(box, str) and box else None
    score = _refine_editor(sess, pts, lbs, bx)
    return _library_editor_payload(sess, score)


def library_edit_outline(sid: str, polygon: PixelPoly) -> dict:
    sess = _LIB_EDIT_SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="edit session expired — reopen")
    _record_edit(
        sess, _mask_from_pixel_poly(sess, polygon), [], [], operation="manual"
    )
    return _library_editor_payload(sess)


def library_edit_history(
    sid: str, direction: Literal["undo", "redo"]
) -> dict:
    sess = _LIB_EDIT_SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="edit session expired — reopen")
    _move_edit_history(sess, direction)
    return _library_editor_payload(sess)


def library_edit_save(
    sid: str, outline_variant: str = Form("recommended")
) -> dict:
    """Accept one photo-outline candidate and reconstruct it exactly once."""
    sess = _LIB_EDIT_SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="edit session expired — reopen")
    _ensure_edit_history(sess)
    raw, outline_warnings, cleanup, resolved, _source = _accepted_editor_outline(
        sess, outline_variant
    )
    if (
        sess["_edit_cursor"] == 0
        and resolved == sess.get("cleanup_default", "raw")
    ):
        raise HTTPException(status_code=409, detail="make a correction before saving")
    if raw is None:
        raise HTTPException(status_code=422, detail="outline is empty")
    corrected = (
        parallax_mod.correct_polygon(raw, sess["calibration"], sess["thickness"])
        if sess["thickness"] else raw
    )
    try:
        tool = library_mod.load(sess["tool_id"])
    except KeyError:
        raise HTTPException(status_code=404, detail="no such tool")
    if tool.provenance is not None:
        tool = tool.model_copy(update={
            "provenance": tool.provenance.model_copy(update={
                "warnings": list(dict.fromkeys([
                    *tool.provenance.warnings, *outline_warnings
                ]))
            })
        })
    snap = sess["_edit_history"][sess["_edit_cursor"]]
    diagnostics = dict(snap["diagnostics"])
    diagnostics.update({
        "cleanup_variant": resolved,
        "cleanup_noise_mm": cleanup["noise_mm"],
        "cleanup_radius_mm": cleanup["radius_mm"],
        "cleanup_max_shift_mm": cleanup["max_shift_mm"],
        "cleanup_max_shift_cap_mm": cleanup["max_shift_cap_mm"],
    })
    saved = _apply_outline(
        tool, corrected, raw_outline=raw, source="sam", diagnostics=diagnostics
    )
    _LIB_EDIT_SESSIONS.pop(sid, None)
    return _lib_json(saved)


def library_delete(tool_id: str) -> dict:
    _LIB_EDIT_SESSIONS.clear()  # any open edit sessions may reference this tool
    return {"deleted": library_mod.delete(tool_id)}


class LibraryUpdate(BaseModel):
    label: Optional[str] = None
    thickness_mm: Optional[float] = None
    silhouette_height_mm: Optional[float] = Field(None, gt=0)
    full_height_mm: Optional[float] = Field(None, gt=0)
    clearance_mm: Optional[float] = Field(None, ge=0)
    bin_style: Optional[Literal["pocket", "corral", "grid"]] = None
    pocket_depth_mm: Optional[float] = Field(None, gt=0)  # null clears → auto
    round_tool: Optional[bool] = None  # domed/barrel → deeper auto pocket
    finger_hole: Optional[bool] = None  # scallop to lift a recessed tool out
    outline: Optional[dict] = None  # edited Poly from the outline editor
    raw_outline: Optional[dict] = None  # matching visible silhouette on the photo
    edit_source: Optional[Literal["sam", "manual", "physical"]] = None
    edit_diagnostics: Optional[dict[str, float | int]] = None


def _apply_outline(
    t,
    poly,
    raw_outline=None,
    *,
    source: Literal["sam", "manual", "physical", "thickness"] = "manual",
    diagnostics: dict | None = None,
):
    """Save a new outline and keep everything in sync: re-derive the footprint,
    thumbnail, photo crop, and immutable accepted-edit history."""
    if raw_outline is None:
        # A physical vertex edit changes only the cutout. Preserve the accepted
        # photo silhouette so a later SAM correction still starts from the
        # image-space selection the user actually accepted.
        raw_outline = t.raw_outline or _raw_outline_for_photo(
            poly, t.calibration, t.thickness_mm
        )
    if source == "physical" and t.outline is not None and diagnostics is None:
        diagnostics = _manual_physical_diagnostics(t.outline, poly)
    _render_thumb(poly.exterior, library_mod.library_dir() / f"{t.id}.png")
    history = list(t.outline_history)
    if not history and t.raw_outline is not None and t.outline is not None:
        history.append(library_mod.OutlineEditRevision(
            revision=t.outline_revision,
            created_ts=int(time.time()),
            source="baseline",
            raw_outline=t.raw_outline,
            outline=t.outline,
        ))
    revision = max(
        [t.outline_revision, *(item.revision for item in history)], default=0
    ) + 1
    history.append(library_mod.OutlineEditRevision(
        revision=revision,
        created_ts=int(time.time()),
        source=source,
        raw_outline=raw_outline,
        outline=poly,
        diagnostics=diagnostics or {},
    ))
    updated = t.model_copy(update={
        "raw_outline": raw_outline,
        "outline": poly,
        "outline_revision": revision,
        "outline_history": history,
    })
    saved = library_mod.save(_refresh_tool_readiness(updated))
    _regen_photo_thumb(saved)
    return saved


def library_edit(tool_id: str, upd: LibraryUpdate) -> dict:
    try:
        t = library_mod.load(tool_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such tool")
    changes = upd.model_dump(exclude_unset=True)  # only fields the client sent
    outline = changes.pop("outline", None)
    raw_outline = changes.pop("raw_outline", None)
    edit_source = changes.pop("edit_source", None) or "manual"
    edit_diagnostics = changes.pop("edit_diagnostics", None)
    silhouette = changes.pop("silhouette_height_mm", None)
    if silhouette is not None:
        supplied_thickness = changes.get("thickness_mm")
        if supplied_thickness is not None and not math.isclose(
            supplied_thickness, silhouette, abs_tol=1e-6
        ):
            raise HTTPException(
                status_code=422,
                detail="thickness and silhouette height disagree",
            )
        changes["thickness_mm"] = silhouette
    thickness_changed = "thickness_mm" in changes
    thickness = changes.pop("thickness_mm", None)
    if thickness_changed and (thickness is None or thickness <= 0):
        raise HTTPException(status_code=422, detail="thickness must be > 0")
    if thickness_changed and t.calibration is None:
        raise HTTPException(
            status_code=422,
            detail="cannot change thickness without capture calibration; retrace the tool",
        )
    # Recover the visible silhouette using the OLD thickness before updating it.
    raw_for_thickness = _library_raw_outline(t) if thickness_changed else None
    if changes:  # label / clearance / depth / round / finger
        t = t.model_copy(update=changes)
    if thickness_changed:
        t = _refresh_tool_readiness(
            t.model_copy(update={
                "thickness_mm": thickness,
                "silhouette_height_mm": thickness,
            }),
            thickness_source="manual",
        )
    if outline:  # a hand-edited outline → re-derive footprint + thumbs
        raw_poly = contour_mod.Poly(**raw_outline) if raw_outline else None
        physical_poly = contour_mod.Poly(**outline)
        if edit_source == "physical":
            physical_poly = _validated_physical_outline(
                PhysicalPoly.model_validate(outline)
            )
        return _lib_json(_apply_outline(
            t, physical_poly, raw_outline=raw_poly,
            source=edit_source, diagnostics=edit_diagnostics,
        ))
    if thickness_changed and raw_for_thickness is not None and t.calibration is not None:
        corrected = (
            parallax_mod.correct_polygon(raw_for_thickness, t.calibration, thickness)
            if thickness
            else raw_for_thickness
        )
        return _lib_json(_apply_outline(
            t, corrected, raw_outline=raw_for_thickness, source="thickness"
        ))
    return _lib_json(library_mod.save(_refresh_tool_readiness(t)))


class ComposeRequest(BaseModel):
    ids: list[str]
    cols: int = Field(ge=1)
    rows: int = Field(ge=1)
    overall_height: Optional[float] = None  # uniform finished height (mm), export only


def library_compose(req: ComposeRequest) -> dict:
    tools = []
    bins: list[tuple[str, float, float]] = []
    printer = bench_mod.load_profile() or bench_mod.default_profile()
    for tid in req.ids:
        try:
            t = library_mod.load(tid)
        except KeyError:
            continue
        _require_tool_ready(t)
        spec = library_mod.derive_tool_spec(t, printer_profile=printer)
        bins.append((t.id, float(spec.grid[0]), float(spec.grid[1])))
        tools.append(_lib_json(t, printer_profile=printer))
    layout = nesting_mod.nest(bins, req.cols * grid_mod.PITCH, req.rows * grid_mod.PITCH)
    return {
        "drawer": {"cols": req.cols, "rows": req.rows},
        "tools": tools,
        "layout": {
            "placed": [p.model_dump() for p in layout.placed],
            "overflow": layout.overflow,
            "used_cols": layout.used_cols, "used_rows": layout.used_rows,
        },
    }


def library_compose_preview_glb(req: ComposeRequest) -> Response:
    """Render the composed drawer as exact bins seated in a Gridfinity grid.

    Each bin is regenerated from its current physical outline and settings,
    with the same requested-height fallback used by drawer export. The base is
    contextual preview geometry only; drawer export remains one 3MF per bin.
    """
    import trimesh

    from gridshot.core import export as export_mod

    entries: dict[str, library_mod.LibraryTool] = {}
    specs = {}
    bins: list[tuple[str, float, float]] = []
    printer = bench_mod.load_profile() or bench_mod.default_profile()
    for tid in req.ids:
        try:
            tool = library_mod.load(tid)
        except KeyError:
            continue
        _require_tool_ready(tool)
        base_spec = library_mod.derive_tool_spec(tool, printer_profile=printer)
        entries[tool.id] = tool
        bins.append((tool.id, float(base_spec.grid[0]), float(base_spec.grid[1])))
        try:
            specs[tool.id] = library_mod.derive_tool_spec(
                tool,
                printer_profile=printer,
                overall_height_mm=req.overall_height,
            )
        except ValueError:
            if req.overall_height is None:
                raise
            specs[tool.id] = base_spec

    if not entries:
        raise HTTPException(status_code=422, detail="select at least one ready library tool")

    layout = nesting_mod.nest(
        bins,
        req.cols * grid_mod.PITCH,
        req.rows * grid_mod.PITCH,
    )
    scene = trimesh.Scene()
    base_mesh = grid_mod.to_trimesh(
        grid_mod.drawer_baseplate_solid(req.cols, req.rows)
    )
    scene.add_geometry(
        base_mesh,
        node_name="drawer-grid",
        geom_name="drawer-grid",
    )

    drawer_w = req.cols * grid_mod.PITCH
    drawer_d = req.rows * grid_mod.PITCH
    for placement in layout.placed:
        tool = entries[placement.bin_id]
        spec = specs[placement.bin_id]
        solid = grid_mod.bin_solid(
            spec.grid[0],
            spec.grid[1],
            spec.height_u,
            pocket=spec.pocket_poly,
            pocket_depth=spec.pocket_depth_mm,
            finger_holes=spec.finger_holes,
            lip=spec.lip,
            style=spec.bin_style,
        )
        mesh = grid_mod.to_trimesh(solid)
        if placement.rotated:
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    math.pi / 2,
                    [0, 0, 1],
                )
            )
        mesh.apply_translation([
            (placement.col + placement.grid_x / 2) * grid_mod.PITCH - drawer_w / 2,
            (placement.row + placement.grid_y / 2) * grid_mod.PITCH - drawer_d / 2,
            0,
        ])
        node_name = f"drawer-bin-{tool.id}"
        scene.add_geometry(mesh, node_name=node_name, geom_name=node_name)

    return Response(
        export_mod.glb_bytes(scene),
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "inline; filename=drawer-preview.glb"},
    )


def _drawer_layout_svg(cols, rows, layout, entries) -> str:
    """Full drawer, to scale (42mm/cell): each placed bin at its cell + label."""
    mm = 42.0
    W, H = cols * mm, rows * mm
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" '
             f'viewBox="0 0 {W} {H}">']
    for c in range(cols + 1):
        parts.append(f'<line x1="{c*mm}" y1="0" x2="{c*mm}" y2="{H}" stroke="#ccc" stroke-width="0.3"/>')
    for r in range(rows + 1):
        parts.append(f'<line x1="0" y1="{r*mm}" x2="{W}" y2="{r*mm}" stroke="#ccc" stroke-width="0.3"/>')
    for p in layout.placed:
        e = entries.get(p.bin_id)
        label = (e.label or p.bin_id[-4:]) if e else p.bin_id[-4:]
        x, y = p.col * mm, p.row * mm
        w, h = p.grid_x * mm, p.grid_y * mm
        parts.append(f'<rect x="{x+1}" y="{y+1}" width="{w-2}" height="{h-2}" '
                     f'fill="#2f8f9522" stroke="#2f8f95" stroke-width="0.6" rx="2"/>')
        parts.append(f'<text x="{x+w/2}" y="{y+h/2}" font-size="6" fill="#2f8f95" '
                     f'text-anchor="middle" dominant-baseline="middle">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def library_export(req: ComposeRequest) -> Response:
    """Compose selected library tools into a drawer and return a printable zip:
    one 3MF per placed bin + a to-scale layout.svg + a manifest."""
    import io
    import tempfile
    import zipfile

    entries = {}
    bins: list[tuple[str, float, float]] = []
    printer = bench_mod.load_profile() or bench_mod.default_profile()
    for tid in req.ids:
        try:
            t = library_mod.load(tid)
        except KeyError:
            continue
        _require_tool_ready(t)
        spec = library_mod.derive_tool_spec(t, printer_profile=printer)
        entries[tid] = t
        bins.append((t.id, float(spec.grid[0]), float(spec.grid[1])))
    layout = nesting_mod.nest(bins, req.cols * grid_mod.PITCH, req.rows * grid_mod.PITCH)

    buf = io.BytesIO()
    manifest = {"drawer": {"cols": req.cols, "rows": req.rows}, "bins": [],
                "overflow": layout.overflow, "warnings": []}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z, tempfile.TemporaryDirectory() as tmp:
        for p in layout.placed:
            t = entries[p.bin_id]
            if t.outline is None:
                continue  # legacy entry without geometry — can't regenerate
            gen = lambda oh, t=t: trace_mod.finalize_bin(  # noqa: E731
                t.outline, t.calibration, t.thickness_mm, clearance_mm=t.clearance_mm,
                bin_style=t.bin_style, pocket_depth_mm=t.pocket_depth_mm,
                overall_height_mm=oh, lip=t.lip,
                finger_hole=t.finger_hole, out_dir=Path(tmp), stem=t.id,
                pre_corrected=True, round_tool=t.round_tool,
                readiness=_tool_readiness(t),
                thickness_source=(
                    t.provenance.thickness_source if t.provenance else "legacy"
                ),
                printer_profile=printer,
            )
            try:
                res = gen(req.overall_height)
            except ValueError:  # requested height too short for this tool's pocket
                res = gen(None)  # fall back to this bin's own (taller) auto height
                manifest["warnings"].append(
                    f"{t.label or t.id}: {t.bin_style} geometry does not fit a "
                    f"{req.overall_height:.0f}mm drawer height — this bin is taller"
                )
            fname = f"bins/{(t.label or t.id)}.3mf"
            z.write(res.files["3mf"], fname)
            manifest["bins"].append({
                "file": fname, "label": t.label, "col": p.col, "row": p.row,
                "grid_x": p.grid_x, "grid_y": p.grid_y, "rotated": p.rotated,
                "thickness_mm": t.thickness_mm, "height_u": res.height_u,
                "bin_style": res.bin_style,
                "pocket_depth_mm": round(res.pocket_depth_mm, 2),
                "overall_height_mm": round(res.overall_height_mm, 1),
                "derivation_key": res.derivation_key,
                "readiness": _tool_readiness(t).model_dump(),
                "provenance": (
                    t.provenance.model_dump() if t.provenance else None
                ),
            })
        z.writestr("layout.svg", _drawer_layout_svg(req.cols, req.rows, layout, entries))
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return Response(
        buf.getvalue(), media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=drawer.zip"},
    )


class Placement(BaseModel):
    id: str
    tx: float
    ty: float
    rot: float = 0.0


class CombineToolOverride(BaseModel):
    id: str
    # Null/omitted means inherit the current library value.
    finger_hole: Optional[bool] = None


class CombineRequest(BaseModel):
    ids: list[str]
    overall_height: Optional[float] = None
    lip: bool = True
    bin_style: Literal["pocket", "corral", "grid"] = "pocket"
    placements: Optional[list[Placement]] = None  # manual arrange; else auto-pack
    overrides: Optional[list[CombineToolOverride]] = None


def _combine_layout(req: "CombineRequest") -> dict:
    """Shared multi-tool-bin geometry, used by both preview and export: each
    tool's cleared + printer-compensated + CAD-flipped pocket and finger-access
    envelope, its effective settings, and a placement (auto-packed, or from the
    request for manual arrange), all centred in the sized gridfinity footprint."""
    from shapely.affinity import rotate as srotate
    from shapely.affinity import translate as stranslate
    from shapely.ops import unary_union

    from gridshot.core import binpack as binpack_mod

    printer = bench_mod.load_profile() or bench_mod.default_profile()
    override_map = {item.id: item for item in req.overrides or []}
    effective_lip = req.lip
    wall = grid_mod.MIN_WALL_LIP if effective_lip else grid_mod.MIN_WALL
    if req.bin_style in ("corral", "grid"):
        wall = max(
            wall,
            grid_mod.CORRAL_WALL
            + grid_mod.CORRAL_BASE_FLARE
            + grid_mod.CORRAL_EDGE_MARGIN,
        )
    tools, specs, pack_stamps, pocket_stamps = [], [], [], []
    depths, fingers, inherited_fingers = [], [], []
    for tid in req.ids:
        try:
            t = library_mod.load(tid)
        except KeyError:
            continue
        _require_tool_ready(t)
        if t.outline is None:
            continue
        override = override_map.get(t.id)
        finger = (
            override.finger_hole
            if override is not None and override.finger_hole is not None
            else t.finger_hole
        )
        effective_tool = t.model_copy(update={
            "finger_hole": finger,
            "bin_style": req.bin_style,
        })
        spec = library_mod.derive_tool_spec(
            effective_tool, printer_profile=printer, lip=effective_lip
        )
        sizing_shape = contour_mod.to_shapely(spec.sizing_poly)
        origin = sizing_shape.centroid
        pocket_shape = contour_mod.to_shapely(spec.pocket_poly)
        tools.append(t)
        specs.append(spec)
        pack_stamps.append(spec.sizing_poly)
        pocket_stamps.append(contour_mod.from_shapely(
            stranslate(pocket_shape, -origin.x, -origin.y)
        ))
        depths.append(spec.pocket_depth_mm)
        fingers.append([
            (float(x - origin.x), float(y - origin.y), float(diameter))
            for x, y, diameter in spec.finger_holes
        ])
        inherited_fingers.append(t.finger_hole)
    if len(pack_stamps) < 2:
        raise HTTPException(status_code=422, detail="select at least 2 tools with outlines")

    if req.placements:  # manual arrange → honour the given transforms
        pmap = {p.id: p for p in req.placements}
        tfs = [
            {"tx": pmap[t.id].tx, "ty": pmap[t.id].ty, "rot": pmap[t.id].rot}
            if t.id in pmap else {"tx": 0.0, "ty": 0.0, "rot": 0.0}
            for t in tools
        ]
    else:
        # Pack the complete cut envelope, not just the pocket. Finger access
        # therefore cannot silently collide with a neighbour or force a larger
        # footprint only after export.
        tfs = binpack_mod.pack(pack_stamps, wall=wall)

    placed_envelopes = [
        binpack_mod.place_stamp(
            pack_stamps[i], tfs[i]["tx"], tfs[i]["ty"], tfs[i]["rot"]
        )
        for i in range(len(pack_stamps))
    ]
    placed_pockets, placed_fingers = [], []
    for i, stamp in enumerate(pocket_stamps):
        shape = contour_mod.to_shapely(stamp)
        shape = srotate(shape, tfs[i]["rot"], origin=(0, 0))
        shape = stranslate(shape, tfs[i]["tx"], tfs[i]["ty"])
        placed_pockets.append(contour_mod.from_shapely(shape))
        angle = math.radians(tfs[i]["rot"])
        c, s = math.cos(angle), math.sin(angle)
        placed_fingers.append([
            (
                x * c - y * s + tfs[i]["tx"],
                x * s + y * c + tfs[i]["ty"],
                diameter,
            )
            for x, y, diameter in fingers[i]
        ])

    union = unary_union(
        [contour_mod.to_shapely(p) for p in placed_envelopes]
    )
    minx, miny, maxx, maxy = union.bounds
    rect = contour_mod.Poly(exterior=[(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)], holes=[])
    gx, gy = grid_mod.auto_grid(rect, wall=wall)
    dx, dy = -(minx + maxx) / 2, -(miny + maxy) / 2  # centre the group in the bin

    centered, centered_fingers, ctfs = [], [], []
    for i, p in enumerate(placed_pockets):
        centered.append(contour_mod.from_shapely(stranslate(contour_mod.to_shapely(p), dx, dy)))
        centered_fingers.append([
            (float(x + dx), float(y + dy), float(diameter))
            for x, y, diameter in placed_fingers[i]
        ])
        ctfs.append({"tx": tfs[i]["tx"] + dx, "ty": tfs[i]["ty"] + dy, "rot": tfs[i]["rot"]})

    need_u = max(spec.height_u for spec in specs)
    height_u = (
        max(
            need_u,
            grid_mod.height_u_for_style_overall(
                req.overall_height, effective_lip, req.bin_style
            ),
        )
        if req.overall_height else need_u
    )
    grid_cuts = [
        (centered[i], depths[i], centered_fingers[i])
        for i in range(len(centered))
    ]
    reserved_cells = (
        grid_mod.grid_reserved_cells(gx, gy, grid_cuts, lip=effective_lip)
        if req.bin_style == "grid" else []
    )
    available_cells = (
        grid_mod.grid_available_cells(gx, gy, grid_cuts, lip=effective_lip)
        if req.bin_style == "grid" else []
    )
    return {
        "tools": tools, "specs": specs,
        "pack_stamps": pack_stamps, "pocket_stamps": pocket_stamps,
        "centered": centered, "tfs": ctfs,
        "depths": depths, "fingers": centered_fingers,
        "local_fingers": fingers, "inherited_fingers": inherited_fingers,
        "gx": gx, "gy": gy,
        "wall": wall,
        "lip": effective_lip,
        "reserved_cells": reserved_cells,
        "available_cells": available_cells,
        "outer_w": grid_mod.PITCH * gx - (grid_mod.PITCH - grid_mod.BIN_SIZE),
        "outer_d": grid_mod.PITCH * gy - (grid_mod.PITCH - grid_mod.BIN_SIZE),
        "height_u": height_u,
    }


def library_combine_preview(req: CombineRequest) -> dict:
    """Return pocket stamps, finger access, settings, and layout transforms."""

    lay = _combine_layout(req)
    tools_json = []
    for i, t in enumerate(lay["tools"]):
        stamp = lay["pocket_stamps"][i]
        requested_override = next(
            (item.finger_hole for item in req.overrides or [] if item.id == t.id),
            None,
        )
        retention_override = t.pocket_depth_mm is not None
        tools_json.append({
            "id": t.id, "label": t.label, "bin_style": req.bin_style,
            "depth_mm": round(lay["depths"][i], 1),
            "depth_mode": "library override" if retention_override else "automatic",
            "clearance_mm": round(t.clearance_mm, 2),
            "round_tool": t.round_tool,
            "finger": bool(lay["local_fingers"][i]),
            "finger_hole": bool(lay["local_fingers"][i]),
            "finger_hole_inherited": lay["inherited_fingers"][i],
            "finger_hole_override": requested_override,
            "finger_holes": [
                [round(float(x), 2), round(float(y), 2), round(float(diameter), 2)]
                for x, y, diameter in lay["local_fingers"][i]
            ],
            "derivation_key": lay["specs"][i].derivation_key,
            "stamp": [[round(float(x), 2), round(float(y), 2)] for x, y in stamp.exterior],
            "tx": round(lay["tfs"][i]["tx"], 2), "ty": round(lay["tfs"][i]["ty"], 2),
            "rot": round(lay["tfs"][i]["rot"], 1),
        })
    return {
        "bin_style": req.bin_style,
        "gx": lay["gx"], "gy": lay["gy"],
        "outer_w": round(lay["outer_w"], 2), "outer_d": round(lay["outer_d"], 2),
        "overall_height_mm": round(
            grid_mod.style_finished_height_mm(
                lay["height_u"], lay["lip"], req.bin_style
            ), 1
        ),
        "pitch": grid_mod.PITCH, "bin_size": grid_mod.BIN_SIZE, "wall": lay["wall"],
        "lip": lay["lip"],
        "reserved_cells": [list(cell) for cell in lay["reserved_cells"]],
        "available_cells": [list(cell) for cell in lay["available_cells"]],
        "tools": tools_json,
    }


def _combine_solid(req: CombineRequest, lay: dict | None = None):
    """Build the exact solid shared by interactive GLB preview and 3MF export."""
    lay = lay or _combine_layout(req)
    pockets = [
        (lay["centered"][i], lay["depths"][i], lay["fingers"][i])
        for i in range(len(lay["centered"]))
    ]
    try:
        return grid_mod.bin_solid(
            lay["gx"], lay["gy"], lay["height_u"], pockets=pockets,
            lip=lay["lip"], style=req.bin_style,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def library_combine_preview_glb(req: CombineRequest) -> Response:
    """Return the exact multi-tool solid used by export as a browser preview."""
    from gridshot.core import export as export_mod

    mesh = grid_mod.to_trimesh(_combine_solid(req))
    return Response(
        export_mod.glb_bytes(mesh),
        media_type="model/gltf-binary",
        headers={"Content-Disposition": "inline; filename=multitool-bin.glb"},
    )


def library_combine(req: CombineRequest) -> Response:
    """Pack several tools' real outlines (or apply hand-arranged placements) into
    one pocket or stackable-corral Gridfinity bin and return its 3MF."""
    from gridshot.core import export as export_mod

    solid = _combine_solid(req)
    data = export_mod.threemf_bytes(grid_mod.to_trimesh(solid), name="multitool-bin")
    return Response(
        data, media_type="model/3mf",
        headers={"Content-Disposition": "attachment; filename=multitool-bin.3mf"},
    )


# ---------------------------------------------------------------------------
# batch zip: many tools, two shots each → auto-pair → add all to the library

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


def _batch_thickness(
    outline_a,
    calibration_a,
    outline_b,
    calibration_b,
    *,
    with_residual: bool = False,
) -> float | tuple[float, float]:
    """Solve a batch pair, rejecting degenerate views and optimiser clamps."""
    if any(
        calibration.device_profile_id is None
        or calibration.intrinsics_source not in {None, "profile"}
        for calibration in (calibration_a, calibration_b)
    ):
        raise ValueError(
            "automatic thickness requires calibrated device profiles for both "
            "views; calibrate this capture setup or enter measured thickness"
        )
    d_nadir = math.hypot(
        calibration_a.nadir_xy_mm[0] - calibration_b.nadir_xy_mm[0],
        calibration_a.nadir_xy_mm[1] - calibration_b.nadir_xy_mm[1],
    )
    d_h = abs(calibration_a.camera_height_mm - calibration_b.camera_height_mm)
    if d_nadir < 30 and d_h < 40:
        raise ValueError("camera positions are too similar to measure thickness")
    thickness, residual = parallax_mod.solve_thickness(
        outline_a, calibration_a, outline_b, calibration_b
    )
    ceiling = parallax_mod.thickness_ceiling(calibration_a, calibration_b)
    if thickness >= ceiling - 1.0:
        raise ValueError("thickness solve reached its reliability ceiling")
    solved = float(thickness)
    return (solved, float(residual)) if with_residual else solved


def _commit_batch_tool(label: str, outline, thickness: float,
                       calibration=None, photo_src=None, raw_outline=None,
                       readiness=None, provenance=None, outline_history=None,
                       outline_revision=0,
                       tool_id: str | None = None) -> library_mod.LibraryTool:
    """Persist one already-validated batch item, cleaning up a failed write."""
    tid = tool_id or f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    raw = raw_outline or outline
    history = list(outline_history or [])
    if not history:
        history.append(library_mod.OutlineEditRevision(
            revision=outline_revision,
            created_ts=int(time.time()),
            source="batch",
            raw_outline=raw,
            outline=outline,
        ))
    try:
        _render_thumb(outline.exterior, library_mod.library_dir() / f"{tid}.png")
        has_photo = (
            _store_lib_photo(tid, photo_src)
            if (photo_src and calibration is not None)
            else False
        )
        saved = library_mod.save(library_mod.LibraryTool(
            id=tid, label=Path(label).stem,
            thickness_mm=round(thickness, 1), raw_outline=raw,
            outline=outline, source_project="batch",
            has_photo=has_photo, calibration=calibration,
            readiness=readiness, provenance=provenance,
            outline_history=history, outline_revision=outline_revision,
            created_ts=int(time.time()),
        ))
        _regen_photo_thumb(saved)
        return saved
    except Exception:
        library_mod.delete(tid)
        raise


class _BatchCancelled(Exception):
    """Cooperative stop between image/model operations."""


def _batch_project(sid: str) -> Path:
    if not sid or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for char in sid):
        raise HTTPException(status_code=404, detail="batch job not found")
    return PROJECTS / f"batch-{sid}"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    pending.write_text(json.dumps(payload, indent=2))
    pending.replace(path)


def _batch_job_path(sid: str) -> Path:
    return _batch_project(sid) / "batch-job.json"


def _load_batch_job(sid: str) -> dict:
    path = _batch_job_path(sid)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="batch job not found")
    try:
        job = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"batch job is unreadable: {exc}")
    if job.get("version") != _BATCH_JOB_VERSION or job.get("session") != sid:
        raise HTTPException(status_code=500, detail="batch job has an unsupported format")
    return job


def _save_batch_job(job: dict) -> None:
    job["updated_ts"] = int(time.time())
    _atomic_json(_batch_job_path(job["session"]), job)


def _batch_job_counts(job: dict) -> dict[str, int]:
    entries = job.get("entries", [])
    return {
        "total_images": len(entries),
        "processed_images": sum(
            entry.get("status") in {"complete", "failed"} for entry in entries
        ),
        "succeeded_images": sum(entry.get("status") == "complete" for entry in entries),
        "failed_images": sum(entry.get("status") == "failed" for entry in entries),
    }


def _batch_job_public(job: dict) -> dict:
    with _BATCH_JOB_LOCK:
        future = _BATCH_JOB_FUTURES.get(job["session"])
        active = future is not None and not future.done()
    status = job.get("status", "failed")
    if status in {"queued", "processing", "matching"} and not active:
        status = "paused"
    return {
        "session": job["session"],
        "status": status,
        "phase": job.get("phase", status),
        **_batch_job_counts(job),
        "current_name": job.get("current_name"),
        "error": job.get("error"),
        "can_resume": status in {"paused", "cancelled", "failed"},
        "cancel_requested": bool(job.get("cancel_requested")),
        "entries": [
            {
                "name": entry["name"],
                "status": entry.get("status", "pending"),
                "reason": entry.get("reason"),
            }
            for entry in job.get("entries", [])
        ],
        "result": job.get("result"),
        "created_ts": job.get("created_ts"),
        "updated_ts": job.get("updated_ts"),
    }


async def _stream_batch_archive(upload: UploadFile, destination: Path) -> tuple[int, str]:
    import hashlib

    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as target:
        while True:
            chunk = await upload.read(_BATCH_IO_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > _BATCH_MAX_ARCHIVE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"ZIP exceeds {_BATCH_MAX_ARCHIVE_BYTES // (1024 * 1024)} MiB",
                )
            digest.update(chunk)
            target.write(chunk)
    if size == 0:
        raise HTTPException(status_code=422, detail="ZIP is empty")
    return size, digest.hexdigest()


def _extract_batch_entries(archive_path: Path, project: Path) -> list[dict]:
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=422, detail="not a valid ZIP file")

    selected: list[zipfile.ZipInfo] = []
    names: set[str] = set()
    expanded = 0
    with archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            base = Path(name).name
            if (
                info.is_dir()
                or name.startswith("__MACOSX/")
                or Path(base).suffix.lower() not in _IMG_EXTS
            ):
                continue
            if info.flag_bits & 0x1:
                raise HTTPException(status_code=422, detail=f"encrypted ZIP entry: {base}")
            mode = info.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise HTTPException(status_code=422, detail=f"symbolic link ZIP entry: {base}")
            duplicate_key = base.casefold()
            if duplicate_key in names:
                raise HTTPException(
                    status_code=422,
                    detail=f"duplicate image filename after folder removal: {base}",
                )
            names.add(duplicate_key)
            if info.file_size > _BATCH_MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"image {base} exceeds {_BATCH_MAX_IMAGE_BYTES // (1024 * 1024)} MiB",
                )
            expanded += info.file_size
            if expanded > _BATCH_MAX_EXPANDED_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"expanded ZIP exceeds {_BATCH_MAX_EXPANDED_BYTES // (1024 * 1024)} MiB",
                )
            selected.append(info)

        if not selected:
            raise HTTPException(status_code=422, detail="ZIP contains no supported images")
        if len(selected) > _BATCH_MAX_IMAGES:
            raise HTTPException(
                status_code=413,
                detail=f"ZIP contains {len(selected)} images; limit is {_BATCH_MAX_IMAGES}",
            )

        source_dir = project / "source"
        source_dir.mkdir(exist_ok=True)
        entries: list[dict] = []
        actual_total = 0
        for archive_index, info in enumerate(sorted(selected, key=lambda value: value.filename.casefold())):
            base = Path(info.filename.replace("\\", "/")).name
            suffix = Path(base).suffix.lower()
            relative = Path("source") / f"{archive_index:03d}{suffix}"
            actual_file = 0
            with archive.open(info) as source, (project / relative).open("wb") as target:
                while True:
                    chunk = source.read(_BATCH_IO_CHUNK)
                    if not chunk:
                        break
                    actual_file += len(chunk)
                    actual_total += len(chunk)
                    if actual_file > _BATCH_MAX_IMAGE_BYTES or actual_total > _BATCH_MAX_EXPANDED_BYTES:
                        raise HTTPException(status_code=413, detail="expanded ZIP exceeds safety limits")
                    target.write(chunk)
            entries.append({
                "archive_index": archive_index,
                "name": base,
                "source": str(relative),
                "bytes": actual_file,
                "status": "pending",
                "reason": None,
            })
    return entries


def _render_batch_overlay(pixels: np.ndarray, mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray(pixels).convert("RGB")
    mask_image = Image.fromarray(mask).convert("L")
    tint = Image.new("RGB", image.size, (35, 188, 181))
    tinted = Image.blend(image, tint, 0.36)
    image.paste(tinted, mask=mask_image)
    edge = mask_image.filter(ImageFilter.FIND_EDGES).point(lambda value: 255 if value > 24 else 0)
    image.paste(Image.new("RGB", image.size, (255, 214, 90)), mask=edge)
    bbox = mask_image.getbbox()
    if bbox is not None:
        left, top, right, bottom = bbox
        pad = max(18, int(max(right - left, bottom - top) * 0.16))
        image = image.crop((
            max(0, left - pad),
            max(0, top - pad),
            min(image.width, right + pad),
            min(image.height, bottom + pad),
        ))
    image.thumbnail((420, 420))
    image.save(path, quality=90)


def _process_batch_job_image(
    project: Path,
    entry: dict,
    image_idx: int,
    profile,
    reference,
) -> dict:
    source_path = project / entry["source"]
    captured = trace_mod.capture_artifacts(
        source_path, profile, smooth_mm=None, mask=None
    )
    matching_dir = project / "matching"
    matching_dir.mkdir(exist_ok=True)
    image_rel = Path("matching") / f"{image_idx}-image.jpg"
    mask_rel = Path("matching") / f"{image_idx}-mask.png"
    overlay_rel = Path(f"{image_idx}-overlay.jpg")
    Image.fromarray(captured.pixels).save(project / image_rel, quality=95)
    Image.fromarray(captured.mask).save(project / mask_rel)
    _render_batch_overlay(captured.pixels, captured.mask, project / overlay_rel)

    canonical_rel = None
    change_mask_rel = None
    change_mask = None
    if reference is not None:
        canonical = diffseg_mod.canonical_warp(
            captured.pixels, captured.calibration, profile.spec
        )
        change = diffseg_mod.diff_mask(canonical, reference)
        change = diffseg_mod.board_region_mask(change, profile.spec)
        change_mask = diffseg_mod.largest_component_mask(change)
        canonical_rel = Path("matching") / f"{image_idx}-canonical.jpg"
        change_mask_rel = Path("matching") / f"{image_idx}-change-mask.png"
        Image.fromarray(canonical).save(project / canonical_rel, quality=95)
        Image.fromarray(change_mask).save(project / change_mask_rel)

    _render_thumb(captured.outline.exterior, project / f"{image_idx}.png")
    ys, xs = np.nonzero(captured.mask > 127)
    bbox = (
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if xs.size else None
    )
    return {
        "archive_index": entry["archive_index"],
        "idx": image_idx,
        "name": entry["name"],
        "source": entry["source"],
        "image": str(image_rel),
        "mask": str(mask_rel),
        "overlay": str(overlay_rel),
        "canonical_image": str(canonical_rel) if canonical_rel else None,
        "change_mask": str(change_mask_rel) if change_mask_rel else None,
        "change_mask_area_px": int((change_mask > 127).sum()) if change_mask is not None else None,
        "mask_shape": list(captured.mask.shape),
        "mask_area_px": int((captured.mask > 127).sum()),
        "mask_bbox_px": bbox,
        "component_count": len(contour_mod.mask_to_polygons_px(captured.mask)),
        "raw_outline": captured.raw_poly.model_dump(),
        "outline": captured.outline.model_dump(),
        "outline_variant": (
            "cleaned"
            if any(warning.lower().startswith("auto smoothing radius") for warning in captured.warnings)
            else "raw"
        ),
        "calibration": captured.calibration.model_dump(),
        "warnings": captured.warnings,
        "readiness": captured.readiness.model_dump() if captured.readiness is not None else None,
    }


def _batch_artifact_path(sid: str) -> Path:
    return _batch_project(sid) / "batch-artifacts.json"


def _load_batch_artifacts(sid: str) -> dict:
    path = _batch_artifact_path(sid)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="batch artifacts not found")
    return json.loads(path.read_text())


def _batch_committed_images(session: dict) -> set[int]:
    ledger = session.get("commit_ledger")
    if ledger is not None:
        return {int(idx) for idx in ledger.get("committed_images", [])}
    if session.get("commit_result") is not None:
        return set(range(len(session.get("images", []))))
    return set()


def _empty_batch_commit_ledger() -> dict:
    return {"version": 2, "transactions": [], "committed_images": [], "pending": None}


def _batch_session_from_artifacts(sid: str) -> dict:
    cached = _BATCH_SESSIONS.get(sid)
    if cached is not None:
        return cached
    project = _batch_project(sid)
    payload = _load_batch_artifacts(sid)
    images = []
    for artifact in sorted(payload.get("items", []), key=lambda value: value["idx"]):
        calibration = calibrate_mod.Calibration.model_validate(artifact["calibration"])
        raw_outline = Poly.model_validate(artifact.get("raw_outline") or artifact["outline"])
        outline = Poly.model_validate(artifact["outline"])
        readiness = (
            readiness_mod.ReadinessReport.model_validate(artifact["readiness"])
            if artifact.get("readiness") else None
        )
        images.append({
            "name": artifact["name"],
            "calibration": calibration,
            "raw_outline": raw_outline,
            "outline": outline,
            "outline_variant": artifact.get("outline_variant", "raw"),
            "warnings": artifact.get("warnings", []),
            "readiness": readiness,
            "source_path": project / artifact["source"],
            "image_path": project / artifact["image"],
            "mask_path": project / artifact["mask"],
            "matcher_image_path": (
                project / artifact["canonical_image"] if artifact.get("canonical_image") else None
            ),
            "matcher_mask_path": (
                project / artifact["change_mask"] if artifact.get("change_mask") else None
            ),
            "outline_revision": int(artifact.get("outline_revision", 0)),
            "outline_history": artifact.get("outline_history", []),
            "artifact": artifact,
        })
    session = {
        "images": images,
        "artifact_payload": payload,
        "artifact_path": _batch_artifact_path(sid),
    }
    commit_path = project / "batch-commit.json"
    if commit_path.is_file():
        committed = json.loads(commit_path.read_text())
        if committed.get("version") == 2:
            ledger = committed
            pending = ledger.get("pending")
            if pending is not None:
                for tool_id in pending.get("library_ids", []):
                    library_mod.delete(tool_id)
                ledger["pending"] = None
                _atomic_json(commit_path, ledger)
            session["commit_ledger"] = ledger
        elif committed.get("status") == "pending":
            for tool_id in committed.get("library_ids", []):
                library_mod.delete(tool_id)
            commit_path.unlink(missing_ok=True)
        else:
            # Status-less records were written by the first durable E9 build and
            # are promoted into the transaction ledger for forward compatibility.
            result = committed["result"]
            images = sorted({
                int(idx)
                for item in result.get("items", [])
                for idx in item.get("images", [])
            })
            ledger = _empty_batch_commit_ledger()
            ledger["transactions"].append({
                "signature": committed["signature"],
                "library_ids": committed.get("library_ids", []),
                "images": images,
                "result": result,
            })
            ledger["committed_images"] = images
            _atomic_json(commit_path, ledger)
            session["commit_ledger"] = ledger
    _BATCH_SESSIONS[sid] = session
    return session


def _refresh_batch_result_image(sid: str, image_idx: int, image: dict, revision: int) -> None:
    """Keep the restart payload in sync with an accepted outline correction."""
    try:
        job = _load_batch_job(sid)
    except HTTPException:
        return
    result = job.get("result")
    if result is None or image_idx >= len(result.get("images", [])):
        return
    public = result["images"][image_idx]
    public.update({
        "thumb": f"/api/files/batch-{sid}/{image_idx}-overlay.jpg?v={revision}",
        "overlay": f"/api/files/batch-{sid}/{image_idx}-overlay.jpg?v={revision}",
        "warnings": image.get("warnings", []),
        "readiness": image["readiness"].model_dump() if image.get("readiness") else None,
    })
    result["draft"] = None
    (_batch_project(sid) / "batch-draft.json").unlink(missing_ok=True)
    _save_batch_job(job)


def _finalize_batch_job(sid: str, job: dict, reference) -> dict:
    artifact_payload = _load_batch_artifacts(sid)
    artifact_payload["failed"] = [
        {"name": entry["name"], "reason": entry.get("reason") or "capture failed"}
        for entry in job["entries"] if entry.get("status") == "failed"
    ]
    _atomic_json(_batch_artifact_path(sid), artifact_payload)
    _BATCH_SESSIONS.pop(sid, None)
    session = _batch_session_from_artifacts(sid)
    images = session["images"]

    pairing = batch_mod.pair_images([image["outline"] for image in images])
    legacy_candidates = [*pairing["pairs"], *pairing["flagged"]]
    legacy_iou = {
        (min(i, j), max(i, j)): iou for i, j, iou in legacy_candidates
    }
    visual_pairing = None
    matcher_warning = None
    matcher_paths = [
        (image["matcher_image_path"], image["matcher_mask_path"]) for image in images
    ]
    if len(images) >= 2 and all(image_path and mask_path for image_path, mask_path in matcher_paths):
        try:
            visual_pairing = batch_mod.visual_pair_files(matcher_paths)
        except Exception as exc:
            matcher_warning = f"visual matcher unavailable: {str(exc)[:120]}"
    elif reference is None:
        matcher_warning = "visual pairing needs an empty-mat reference"

    pairs = []
    for edge in visual_pairing["pairs"] if visual_pairing is not None else []:
        a, b = edge["a"], edge["b"]
        evidence = edge["evidence"]
        pairs.append({
            "a": a,
            "b": b,
            "iou": legacy_iou.get((min(a, b), max(a, b))),
            "score": evidence.score,
            "thickness_mm": None,
            "method": "sift-magsac",
            "gate": visual_pairing["gate"]["version"],
            "reason": (
                f"{evidence.inliers} geometric inliers; "
                f"{evidence.inlier_ratio:.0%} consistency"
            ),
            "confidence": {
                "level": "high",
                "calibrated": False,
                "score": evidence.score,
                "inliers": evidence.inliers,
                "inlier_ratio": evidence.inlier_ratio,
            },
        })
    used = {idx for pair in pairs for idx in (pair["a"], pair["b"])}
    flagged = [
        {
            "a": i,
            "b": j,
            "iou": iou,
            "reason": "shape overlap is only a hint; confirm the pair manually",
        }
        for i, j, iou in legacy_candidates if i not in used and j not in used
    ]
    matcher = {
        "method": "sift-magsac" if visual_pairing is not None else None,
        "gate": visual_pairing["gate"] if visual_pairing is not None else None,
        "warning": matcher_warning,
    }
    artifact_payload["matching"] = {
        **matcher,
        "pairs": pairs,
        "edges": [
            {"a": edge["a"], "b": edge["b"], "evidence": edge["evidence"].model_dump()}
            for edge in (visual_pairing["edges"] if visual_pairing is not None else [])
        ],
    }
    _atomic_json(_batch_artifact_path(sid), artifact_payload)
    session["artifact_payload"] = artifact_payload

    project = _batch_project(sid)
    draft_path = project / "batch-draft.json"
    draft = json.loads(draft_path.read_text()) if draft_path.is_file() else None
    return {
        "session": sid,
        "artifacts": f"/api/files/batch-{sid}/batch-artifacts.json",
        "images": [
            {
                "idx": idx,
                "name": image["name"],
                "thumb": f"/api/files/batch-{sid}/{idx}-overlay.jpg",
                "overlay": f"/api/files/batch-{sid}/{idx}-overlay.jpg",
                "photo": f"/api/batch/{sid}/image/{idx}/photo",
                "outline_thumb": f"/api/files/batch-{sid}/{idx}.png",
                "warnings": image.get("warnings", []),
                "readiness": image["readiness"].model_dump() if image.get("readiness") else None,
            }
            for idx, image in enumerate(images)
        ],
        "pairs": pairs,
        "flagged": flagged,
        "singles": [idx for idx in range(len(images)) if idx not in used],
        "matcher": matcher,
        "failed": artifact_payload["failed"],
        "draft": draft,
        "committed": False,
        "committed_images": [],
        "partial_commits": 0,
    }


def _run_batch_job(sid: str) -> None:
    try:
        with _BATCH_JOB_LOCK:
            job = _load_batch_job(sid)
            job["status"] = "processing"
            job["phase"] = "segmenting"
            job["error"] = None
            job["current_name"] = None
            _save_batch_job(job)
        profile = _pick_verified_mat(job.get("mat_id"))
        reference = mat_mod.load_reference(profile.mat_id)
        artifact_payload = _load_batch_artifacts(sid)
        artifact_by_archive = {
            item.get("archive_index"): item for item in artifact_payload.get("items", [])
        }
        for entry in job["entries"]:
            existing = artifact_by_archive.get(entry["archive_index"])
            if existing is not None:
                entry.update(status="complete", reason=None, image_idx=existing["idx"])
            elif entry.get("status") == "processing":
                entry["status"] = "pending"
        _save_batch_job(job)

        for entry in job["entries"]:
            with _BATCH_JOB_LOCK:
                job = _load_batch_job(sid)
                current = next(item for item in job["entries"] if item["archive_index"] == entry["archive_index"])
                if job.get("cancel_requested"):
                    raise _BatchCancelled()
                if current.get("status") in {"complete", "failed"}:
                    continue
                current["status"] = "processing"
                job["current_name"] = current["name"]
                _save_batch_job(job)
            image_idx = sum(
                item.get("status") == "complete" for item in job["entries"]
            )
            try:
                artifact = _process_batch_job_image(
                    _batch_project(sid), current, image_idx, profile, reference
                )
            except (
                calibrate_mod.DetectionError,
                contour_mod.NoToolFoundError,
                RuntimeError,
                ValueError,
                OSError,
            ) as exc:
                with _BATCH_JOB_LOCK:
                    job = _load_batch_job(sid)
                    current = next(item for item in job["entries"] if item["archive_index"] == entry["archive_index"])
                    current.update(status="failed", reason=str(exc)[:160])
                    _save_batch_job(job)
                continue

            with _BATCH_JOB_LOCK:
                artifact_payload = _load_batch_artifacts(sid)
                artifact_payload["items"] = [
                    item for item in artifact_payload.get("items", [])
                    if item.get("archive_index") != entry["archive_index"]
                ]
                artifact_payload["items"].append(artifact)
                artifact_payload["items"].sort(key=lambda value: value["idx"])
                _atomic_json(_batch_artifact_path(sid), artifact_payload)
                job = _load_batch_job(sid)
                current = next(item for item in job["entries"] if item["archive_index"] == entry["archive_index"])
                current.update(status="complete", reason=None, image_idx=artifact["idx"])
                _save_batch_job(job)

        with _BATCH_JOB_LOCK:
            job = _load_batch_job(sid)
            if job.get("cancel_requested"):
                raise _BatchCancelled()
            job.update(status="matching", phase="matching", current_name=None)
            _save_batch_job(job)
        result = _finalize_batch_job(sid, job, reference)
        with _BATCH_JOB_LOCK:
            job = _load_batch_job(sid)
            if job.get("cancel_requested"):
                raise _BatchCancelled()
            job.update(
                status="ready",
                phase="review",
                current_name=None,
                cancel_requested=False,
                error=None,
                result=result,
            )
            _save_batch_job(job)
    except _BatchCancelled:
        with _BATCH_JOB_LOCK:
            job = _load_batch_job(sid)
            job.update(status="cancelled", phase="cancelled", current_name=None, cancel_requested=False)
            _save_batch_job(job)
    except Exception as exc:
        with _BATCH_JOB_LOCK:
            try:
                job = _load_batch_job(sid)
                job.update(
                    status="failed",
                    phase="failed",
                    current_name=None,
                    error=str(exc)[:240],
                )
                _save_batch_job(job)
            except HTTPException:
                pass


def _schedule_batch_job(sid: str) -> dict:
    with _BATCH_JOB_LOCK:
        job = _load_batch_job(sid)
        existing = _BATCH_JOB_FUTURES.get(sid)
        if existing is not None and not existing.done():
            return _batch_job_public(job)
        if job.get("status") == "ready":
            return _batch_job_public(job)
        job.update(
            status="queued", phase="queued", cancel_requested=False, error=None
        )
        for entry in job.get("entries", []):
            if entry.get("status") == "processing":
                entry["status"] = "pending"
        _save_batch_job(job)
        future = _BATCH_JOB_EXECUTOR.submit(_run_batch_job, sid)
        _BATCH_JOB_FUTURES[sid] = future
        return _batch_job_public(job)


async def batch_upload(
    file: UploadFile = File(...), mat_id: Optional[str] = Form(None)
) -> dict:
    """Create a durable, bounded batch job and return without waiting for models."""
    if not seg_client.available():
        raise HTTPException(status_code=503, detail="segserver offline")
    profile = _pick_verified_mat(mat_id)
    sid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    project = _batch_project(sid)
    project.mkdir(parents=True, exist_ok=False)
    archive_path = project / "source.zip"
    try:
        archive_size, archive_sha = await _stream_batch_archive(file, archive_path)
        entries = _extract_batch_entries(archive_path, project)
        now = int(time.time())
        job = {
            "version": _BATCH_JOB_VERSION,
            "session": sid,
            "status": "queued",
            "phase": "queued",
            "source_zip": file.filename or "batch.zip",
            "source_sha256": archive_sha,
            "archive_bytes": archive_size,
            "mat_id": profile.mat_id,
            "created_ts": now,
            "updated_ts": now,
            "cancel_requested": False,
            "current_name": None,
            "error": None,
            "entries": entries,
            "result": None,
        }
        _atomic_json(
            _batch_artifact_path(sid),
            {
                "version": 1,
                "session": sid,
                "source_zip": job["source_zip"],
                "source_sha256": archive_sha,
                "items": [],
                "failed": [],
                "matching": {},
            },
        )
        _save_batch_job(job)
    except Exception:
        shutil.rmtree(project, ignore_errors=True)
        raise
    return _schedule_batch_job(sid)


def batch_job(sid: str) -> dict:
    return _batch_job_public(_load_batch_job(sid))


def batch_cancel(sid: str) -> dict:
    with _BATCH_JOB_LOCK:
        job = _load_batch_job(sid)
        if job.get("status") == "ready":
            raise HTTPException(
                status_code=409, detail="batch processing is already complete"
            )
        job["cancel_requested"] = True
        _save_batch_job(job)
        return _batch_job_public(job)


def batch_resume(sid: str) -> dict:
    if not seg_client.available():
        raise HTTPException(status_code=503, detail="segserver offline")
    return _schedule_batch_job(sid)


def batch_jobs() -> dict:
    jobs = []
    for path in PROJECTS.glob("batch-*/batch-job.json"):
        try:
            jobs.append(_batch_job_public(json.loads(path.read_text())))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    jobs.sort(key=lambda value: value.get("updated_ts") or 0, reverse=True)
    return {"jobs": jobs[:20]}


def _batch_edit_session(edit_sid: str) -> dict:
    sess = _BATCH_EDIT_SESSIONS.get(edit_sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="batch edit expired — reopen")
    return sess


def batch_edit_photo(sid: str, idx: int) -> FileResponse:
    batch = _batch_session_from_artifacts(sid)
    if idx < 0 or idx >= len(batch["images"]):
        raise HTTPException(status_code=404, detail="batch image not found")
    return FileResponse(batch["images"][idx]["image_path"])


def batch_edit_start(sid: str, idx: int) -> dict:
    import cv2

    batch = _batch_session_from_artifacts(sid)
    if idx in _batch_committed_images(batch):
        raise HTTPException(status_code=409, detail="this tool is already in the library")
    if idx < 0 or idx >= len(batch["images"]):
        raise HTTPException(status_code=404, detail="batch image not found")
    if not seg_client.available():
        raise HTTPException(status_code=503, detail="segserver offline")
    image = batch["images"][idx]
    pixels = np.asarray(Image.open(image["image_path"]).convert("RGB"))
    mask = np.asarray(Image.open(image["mask_path"]).convert("L"))
    image_id, width, height = seg_client.embed(pixels)
    seeds = _seed_points(mask)
    edit_sid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    _BATCH_EDIT_SESSIONS[edit_sid] = {
        "batch_sid": sid,
        "image_idx": idx,
        "image_id": image_id,
        "calibration": image["calibration"],
        "mask": mask.copy(),
        "initial_mask": mask.copy(),
        "points": seeds,
        "labels": [1] * len(seeds),
        "revision": int(image.get("outline_revision", 0)),
        "cleanup_default": image.get("outline_variant", "raw"),
    }
    return {
        "session": edit_sid,
        "display": f"/api/batch/{sid}/image/{idx}/photo",
        "width": width,
        "height": height,
        **_editor_payload(_BATCH_EDIT_SESSIONS[edit_sid]),
    }


def batch_edit_click(
    edit_sid: str,
    points: str = Form("[]"),
    labels: str = Form("[]"),
    box: str = Form(None),
) -> dict:
    sess = _batch_edit_session(edit_sid)
    score = _refine_editor(
        sess,
        json.loads(points),
        json.loads(labels),
        json.loads(box) if isinstance(box, str) and box else None,
    )
    return {"score": round(score, 3), **_editor_payload(sess)}


def batch_edit_outline(edit_sid: str, polygon: PixelPoly) -> dict:
    sess = _batch_edit_session(edit_sid)
    _record_edit(
        sess, _mask_from_pixel_poly(sess, polygon), [], [], operation="manual"
    )
    return {"score": 1.0, **_editor_payload(sess)}


def batch_edit_history(
    edit_sid: str, direction: Literal["undo", "redo"]
) -> dict:
    sess = _batch_edit_session(edit_sid)
    _move_edit_history(sess, direction)
    return {"score": 1.0, **_editor_payload(sess)}


def batch_edit_save(
    edit_sid: str, outline_variant: str = Form("recommended")
) -> dict:
    sess = _batch_edit_session(edit_sid)
    batch = _batch_session_from_artifacts(sess["batch_sid"])
    if sess["image_idx"] in _batch_committed_images(batch):
        raise HTTPException(status_code=409, detail="this tool is already in the library")
    image = batch["images"][sess["image_idx"]]
    raw_mm, outline_warnings, cleanup, resolved, source_raw_mm = (
        _accepted_editor_outline(sess, outline_variant)
    )
    if raw_mm is None:
        raise HTTPException(status_code=422, detail="outline is empty")
    _ensure_edit_history(sess)
    if (
        sess["_edit_cursor"] == 0
        and resolved == image.get("outline_variant", "raw")
    ):
        raise HTTPException(status_code=409, detail="make a correction before saving")
    previous_revision = int(image.get("outline_revision", 0))
    revision = previous_revision + 1
    transient_history = _public_edit_history(sess)
    accepted_history = list(image.get("outline_history", []))
    if not accepted_history:
        baseline_raw = image.get("raw_outline") or image["outline"]
        accepted_history.append(
            library_mod.OutlineEditRevision(
                revision=previous_revision,
                created_ts=int(time.time()),
                source="baseline",
                raw_outline=baseline_raw,
                outline=baseline_raw,
            )
        )
    current_revision = _accepted_outline_revisions(
        sess, sess["calibration"], 0, outline_variant=resolved
    )[-1].model_copy(update={"revision": revision})
    accepted_history.append(current_revision)
    readiness = readiness_mod.evaluate(
        calibration=image["calibration"],
        warnings=[*image.get("warnings", []), *outline_warnings],
        outline=raw_mm,
        require_thickness=False,
    )
    image.update({
        "raw_outline": source_raw_mm or raw_mm,
        "outline": raw_mm,
        "outline_variant": resolved,
        "warnings": list(dict.fromkeys([
            *image.get("warnings", []), *outline_warnings
        ])),
        "readiness": readiness,
        "outline_revision": revision,
        "outline_history": accepted_history,
    })
    Image.fromarray(sess["mask"]).save(image["mask_path"])
    _render_thumb(
        raw_mm.exterior,
        image["mask_path"].parents[1] / f"{sess['image_idx']}.png",
    )
    ys, xs = np.nonzero(sess["mask"] > 127)
    artifact = image["artifact"]
    artifact.update({
        "mask_area_px": int((sess["mask"] > 127).sum()),
        "mask_bbox_px": (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if xs.size else None
        ),
        "component_count": len(contour_mod.mask_to_polygons_px(sess["mask"])),
        "raw_outline": (source_raw_mm or raw_mm).model_dump(),
        "outline": raw_mm.model_dump(),
        "outline_variant": resolved,
        "cleanup": cleanup,
        "readiness": readiness.model_dump(),
        "outline_revision": revision,
        "outline_edits": transient_history,
        "outline_history": [item.model_dump() for item in accepted_history],
    })
    if batch.get("artifact_path") is not None:
        _atomic_json(batch["artifact_path"], batch["artifact_payload"])
    _render_batch_overlay(
        np.asarray(Image.open(image["image_path"]).convert("RGB")),
        sess["mask"],
        image["mask_path"].parents[1] / f"{sess['image_idx']}-overlay.jpg",
    )
    _refresh_batch_result_image(sess["batch_sid"], sess["image_idx"], image, revision)
    _BATCH_EDIT_SESSIONS.pop(edit_sid, None)
    return {
        "idx": sess["image_idx"],
        "revision": revision,
        "thumb": f"/api/files/batch-{sess['batch_sid']}/{sess['image_idx']}-overlay.jpg?v={revision}",
        "readiness": readiness.model_dump(),
    }


class BatchPairSelection(BaseModel):
    a: int
    b: int
    # Optional explicit override when automatic parallax cannot be trusted.
    thickness_mm: Optional[float] = None


class BatchSingleSelection(BaseModel):
    idx: int
    # A single view cannot solve thickness; the user must measure it.
    thickness_mm: Optional[float] = None


class BatchSelectionRequest(BaseModel):
    pairs: list[BatchPairSelection] = Field(default_factory=list)
    singles: list[BatchSingleSelection] = Field(default_factory=list)
    # Millimetre-space WYSIWYG edits keyed by pair:a:b or single:idx. Keeping
    # them in the reviewed request binds each override to the exact selection
    # that is eventually committed.
    physical_outlines: dict[str, PhysicalPoly] = Field(default_factory=dict)
    # Explicitly commit only passing tools; optionally discard the blocked remainder.
    ready_only: bool = False
    discard_blocked: bool = False


def _batch_public_item(item: dict) -> dict:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _batch_review_payload(items: list[dict]) -> dict:
    public = [_batch_public_item(item) for item in items]
    ready = sum(item["status"] == "ready" for item in items)
    return {
        "items": public,
        "ready": ready,
        "blocked": len(items) - ready,
    }


def _batch_request_for_keys(
    req: BatchSelectionRequest, keys: set[str]
) -> BatchSelectionRequest:
    return BatchSelectionRequest(
        pairs=[pair for pair in req.pairs if f"pair:{pair.a}:{pair.b}" in keys],
        singles=[single for single in req.singles if f"single:{single.idx}" in keys],
        physical_outlines={
            key: value
            for key, value in req.physical_outlines.items()
            if key in keys
        },
    )


def _validate_batch_selection(req: BatchSelectionRequest, image_count: int) -> None:
    if not req.pairs and not req.singles:
        raise HTTPException(status_code=422, detail="select at least one batch tool")

    seen: set[int] = set()
    for pair in req.pairs:
        if pair.a == pair.b:
            raise HTTPException(status_code=422, detail="a photo cannot be paired with itself")
        indices = (pair.a, pair.b)
        for idx in indices:
            if idx < 0 or idx >= image_count:
                raise HTTPException(status_code=422, detail=f"image index {idx} is invalid")
            if idx in seen:
                raise HTTPException(
                    status_code=422,
                    detail=f"image {idx} is included more than once",
                )
            seen.add(idx)

    for single in req.singles:
        idx = single.idx
        if idx < 0 or idx >= image_count:
            raise HTTPException(status_code=422, detail=f"image index {idx} is invalid")
        if idx in seen:
            raise HTTPException(
                status_code=422,
                detail=f"image {idx} is included more than once",
            )
        seen.add(idx)

    allowed = {
        *(f"pair:{pair.a}:{pair.b}" for pair in req.pairs),
        *(f"single:{single.idx}" for single in req.singles),
    }
    unknown = set(req.physical_outlines) - allowed
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"physical outline does not match this selection: {sorted(unknown)[0]}",
        )


def _positive_thickness(value: Optional[float]) -> float:
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError("enter a positive measured thickness")
    return float(value)


def _batch_readiness(
    primary: dict,
    *,
    outline,
    thickness_mm: float | None,
    thickness_source: str,
    secondary: dict | None = None,
) -> readiness_mod.ReadinessReport:
    warnings = list(dict.fromkeys([
        *primary.get("warnings", []),
        *(secondary.get("warnings", []) if secondary else []),
    ]))
    reports = [
        readiness_mod.evaluate(
            calibration=primary.get("calibration"),
            warnings=warnings,
            outline=outline,
            thickness_mm=thickness_mm,
            thickness_source=thickness_source,
        )
    ]
    if secondary is not None:
        reports.append(readiness_mod.evaluate(
            calibration=secondary.get("calibration"),
            require_outline=False,
            require_thickness=False,
        ))
    return readiness_mod.combine(*reports)


def _batch_provenance(
    primary: dict,
    secondary: dict | None,
    thickness_source: str,
) -> readiness_mod.ArtifactProvenance:
    calibration = primary.get("calibration")
    images = [primary["name"]]
    warnings = list(primary.get("warnings", []))
    if secondary is not None:
        images.append(secondary["name"])
        warnings.extend(secondary.get("warnings", []))
    return readiness_mod.ArtifactProvenance(
        flow="batch",
        mat_id=getattr(calibration, "mat_id", None),
        device_profile_id=getattr(calibration, "device_profile_id", None),
        device_profile_revision=getattr(
            calibration, "device_profile_revision", None
        ),
        intrinsics_source=getattr(calibration, "intrinsics_source", None),
        capture_signature=getattr(calibration, "capture_signature", None),
        thickness_source=thickness_source,
        source_images=images,
        warnings=list(dict.fromkeys(warnings)),
    )


def _batch_status(value: readiness_mod.ReadinessReport) -> str:
    blocked = [check.code for check in value.checks if check.status == "block"]
    if not blocked:
        return "ready"
    if any(code.startswith(("calibration.", "provenance.")) for code in blocked):
        return "failed"
    if any(code.startswith("outline.") for code in blocked):
        return "needs_outline"
    if any(code.startswith("thickness.") for code in blocked):
        return "needs_thickness"
    return "failed"


def _batch_correction_failure(
    value: readiness_mod.ReadinessReport,
    message: str,
) -> readiness_mod.ReadinessReport:
    return readiness_mod.combine(
        value,
        readiness_mod.report([readiness_mod.ReadinessCheck(
            code="outline.correction",
            status="block",
            source="outline",
            message=message,
        )]),
    )


def _batch_outline_history(
    image: dict,
    thickness: float,
    current_corrected=None,
):
    revisions = []
    values = image.get("outline_history", [])
    for index, value in enumerate(values):
        revision = library_mod.OutlineEditRevision.model_validate(value)
        corrected = (
            current_corrected
            if current_corrected is not None and index == len(values) - 1
            else parallax_mod.correct_polygon(
                revision.raw_outline, image["calibration"], thickness
            )
        )
        revisions.append(revision.model_copy(update={"outline": corrected}))
    return revisions


def _batch_physical_revision(
    image: dict,
    history: list[library_mod.OutlineEditRevision],
    baseline: Poly,
    outline: Poly,
    diagnostics: dict[str, float | str] | None,
) -> tuple[list[library_mod.OutlineEditRevision], int]:
    """Append an accepted cutout-only revision without changing the photo mask."""
    revision = int(image.get("outline_revision", 0))
    if diagnostics is None:
        return history, revision
    if not history:
        history.append(library_mod.OutlineEditRevision(
            revision=revision,
            created_ts=int(time.time()),
            source="baseline",
            raw_outline=image["outline"],
            outline=baseline,
        ))
    revision = max(
        [revision, *(item.revision for item in history)], default=revision
    ) + 1
    history.append(library_mod.OutlineEditRevision(
        revision=revision,
        created_ts=int(time.time()),
        source="physical",
        raw_outline=image["outline"],
        outline=outline,
        diagnostics=diagnostics,
    ))
    return history, revision


def _prepare_batch_items(images: list[dict], req: BatchSelectionRequest) -> list[dict]:
    """Resolve reviewed selections into the shared readiness contract."""
    _validate_batch_selection(req, len(images))
    items: list[dict] = []

    for pair in req.pairs:
        ia, ib = images[pair.a], images[pair.b]
        warnings = list(dict.fromkeys([
            *ia.get("warnings", []),
            *ib.get("warnings", []),
        ]))
        item = {
            "key": f"pair:{pair.a}:{pair.b}",
            "kind": "pair",
            "images": [pair.a, pair.b],
            "primary_image": pair.a,
            "label": Path(ia["name"]).stem,
            "status": "needs_thickness",
            "thickness_mm": None,
            "thickness_source": None,
            "reason": None,
            "warnings": warnings,
        }
        scalar_residual = None
        try:
            if pair.thickness_mm is None:
                thickness, scalar_residual = _batch_thickness(
                    ia["outline"],
                    ia["calibration"],
                    ib["outline"],
                    ib["calibration"],
                    with_residual=True,
                )
                source = "automatic"
            else:
                thickness = _positive_thickness(pair.thickness_mm)
                source = "manual"
        except (parallax_mod.MissingPoseError, RuntimeError, ValueError) as exc:
            readiness = _batch_readiness(
                ia,
                outline=ia["outline"],
                thickness_mm=None,
                thickness_source="unknown",
                secondary=ib,
            )
            provenance = _batch_provenance(ia, ib, "unknown")
            item.update(
                status=_batch_status(readiness),
                reason=str(exc),
                readiness=readiness.model_dump(),
                _readiness=readiness,
                _provenance=provenance,
            )
            items.append(item)
            continue

        reconstruction = None
        physical_diagnostics = None
        provenance = _batch_provenance(ia, ib, source)
        try:
            if source == "automatic":
                local = parallax_mod.reconstruct_footprint(
                    ia["outline"],
                    ia["calibration"],
                    ib["outline"],
                    ib["calibration"],
                    scalar_height_mm=thickness,
                    scalar_residual_mm2=scalar_residual,
                )
                corrected = local.polygon
                reconstruction = local.diagnostics()
                warnings.append(
                    "local footprint: two-view silhouette reconstruction "
                    f"{local.reconstructed_major_extent_mm:.2f} × "
                    f"{local.reconstructed_minor_extent_mm:.2f}mm "
                    f"(boundary p95 {local.boundary_p95_error_mm:.2f}mm)"
                )
            else:
                corrected = parallax_mod.correct_polygon(
                    ia["outline"], ia["calibration"], thickness
                )
        except parallax_mod.LocalReconstructionError as exc:
            corrected = parallax_mod.correct_polygon(
                ia["outline"], ia["calibration"], thickness
            )
            warnings.append(
                "local footprint fallback: using one-height parallax because "
                f"{exc}"
            )
        except (parallax_mod.MissingPoseError, RuntimeError, ValueError) as exc:
            reason = f"outline correction failed: {exc}"
            readiness = _batch_correction_failure(
                _batch_readiness(
                    ia,
                    outline=ia["outline"],
                    thickness_mm=thickness,
                    thickness_source=source,
                    secondary=ib,
                ),
                reason,
            )
            item.update(
                status=_batch_status(readiness),
                thickness_mm=round(float(thickness), 2),
                thickness_source=source,
                reason=reason,
                readiness=readiness.model_dump(),
                _readiness=readiness,
                _provenance=provenance,
            )
            items.append(item)
            continue

        baseline_corrected = corrected
        physical = req.physical_outlines.get(item["key"])
        if physical is not None:
            corrected = _validated_physical_outline(physical)
            physical_diagnostics = _manual_physical_diagnostics(
                baseline_corrected, corrected, reconstruction
            )
            reconstruction = physical_diagnostics
            warnings.append(
                "physical cutout override: using the manually edited batch "
                "footprint without reapplying parallax"
            )

        provenance = _batch_provenance(ia, ib, source).model_copy(
            update={"warnings": warnings}
        )
        primary_with_warnings = dict(ia)
        primary_with_warnings["warnings"] = warnings
        readiness = _batch_readiness(
            primary_with_warnings,
            outline=corrected,
            thickness_mm=thickness,
            thickness_source=source,
            secondary=ib,
        )
        status = _batch_status(readiness)
        outline_history = _batch_outline_history(
            ia, thickness, current_corrected=baseline_corrected
        )
        outline_history, outline_revision = _batch_physical_revision(
            ia, outline_history, baseline_corrected, corrected, physical_diagnostics
        )
        item.update(
            status=status,
            thickness_mm=round(float(thickness), 2),
            thickness_source=source,
            reason=(
                readiness_mod.blocking_message(readiness)
                if status != "ready" else None
            ),
            readiness=readiness.model_dump(),
            _readiness=readiness,
            _provenance=provenance,
            reconstruction=reconstruction,
            physical_outline=_poly_json(corrected),
            _outline=corrected,
            _raw_outline=ia["outline"],
            _calibration=ia["calibration"],
            _photo_src=ia["source_path"],
            _outline_history=outline_history,
            _outline_revision=outline_revision,
        )
        items.append(item)

    for single in req.singles:
        im = images[single.idx]
        item = {
            "key": f"single:{single.idx}",
            "kind": "single",
            "images": [single.idx],
            "primary_image": single.idx,
            "label": Path(im["name"]).stem,
            "status": "needs_thickness",
            "thickness_mm": None,
            "thickness_source": None,
            "reason": None,
            "warnings": list(im.get("warnings", [])),
        }
        try:
            thickness = _positive_thickness(single.thickness_mm)
        except ValueError as exc:
            readiness = _batch_readiness(
                im,
                outline=im["outline"],
                thickness_mm=None,
                thickness_source="unknown",
            )
            provenance = _batch_provenance(im, None, "unknown")
            item.update(
                status=_batch_status(readiness),
                reason=str(exc),
                readiness=readiness.model_dump(),
                _readiness=readiness,
                _provenance=provenance,
            )
            items.append(item)
            continue

        provenance = _batch_provenance(im, None, "manual")
        physical_diagnostics = None
        try:
            corrected = parallax_mod.correct_polygon(
                im["outline"], im["calibration"], thickness
            )
        except (parallax_mod.MissingPoseError, RuntimeError, ValueError) as exc:
            reason = f"outline correction failed: {exc}"
            readiness = _batch_correction_failure(
                _batch_readiness(
                    im,
                    outline=im["outline"],
                    thickness_mm=thickness,
                    thickness_source="manual",
                ),
                reason,
            )
            item.update(
                status=_batch_status(readiness),
                thickness_mm=round(float(thickness), 2),
                thickness_source="manual",
                reason=reason,
                readiness=readiness.model_dump(),
                _readiness=readiness,
                _provenance=provenance,
            )
            items.append(item)
            continue


        baseline_corrected = corrected
        physical = req.physical_outlines.get(item["key"])
        if physical is not None:
            corrected = _validated_physical_outline(physical)
            physical_diagnostics = _manual_physical_diagnostics(
                baseline_corrected, corrected
            )
            item["warnings"].append(
                "physical cutout override: using the manually edited batch "
                "footprint without reapplying parallax"
            )
            provenance = provenance.model_copy(
                update={"warnings": item["warnings"]}
            )

        primary_with_warnings = dict(im)
        primary_with_warnings["warnings"] = item["warnings"]
        readiness = _batch_readiness(
            primary_with_warnings,
            outline=corrected,
            thickness_mm=thickness,
            thickness_source="manual",
        )
        status = _batch_status(readiness)
        outline_history = _batch_outline_history(
            im, thickness, current_corrected=baseline_corrected
        )
        outline_history, outline_revision = _batch_physical_revision(
            im, outline_history, baseline_corrected, corrected, physical_diagnostics
        )
        item.update(
            status=status,
            thickness_mm=round(float(thickness), 2),
            thickness_source="manual",
            reason=(
                readiness_mod.blocking_message(readiness)
                if status != "ready" else None
            ),
            readiness=readiness.model_dump(),
            _readiness=readiness,
            _provenance=provenance,
            reconstruction=physical_diagnostics,
            physical_outline=_poly_json(corrected),
            _outline=corrected,
            _raw_outline=im["outline"],
            _calibration=im["calibration"],
            _photo_src=im["source_path"],
            _outline_history=outline_history,
            _outline_revision=outline_revision,
        )
        items.append(item)

    return items


def batch_review(sid: str, req: BatchSelectionRequest) -> dict:
    sess = _batch_session_from_artifacts(sid)
    review = _batch_review_payload(_prepare_batch_items(sess["images"], req))
    draft = {
        "version": 1,
        "updated_ts": int(time.time()),
        "selection": req.model_dump(mode="json"),
        "review": review,
    }
    if _batch_job_path(sid).is_file():
        _atomic_json(_batch_project(sid) / "batch-draft.json", draft)
        job = _load_batch_job(sid)
        if job.get("result") is not None:
            job["result"]["draft"] = draft
            _save_batch_job(job)
    return review


def batch_commit(sid: str, req: BatchSelectionRequest) -> dict:
    sess = _batch_session_from_artifacts(sid)
    if req.discard_blocked and not req.ready_only:
        raise HTTPException(
            status_code=422, detail="discard_blocked requires ready_only"
        )

    signature = req.model_dump_json()
    durable = _batch_job_path(sid).is_file()
    if durable:
        ledger = sess.get("commit_ledger") or _empty_batch_commit_ledger()
        for transaction in ledger["transactions"]:
            if transaction["signature"] == signature:
                return transaction["result"]
    else:
        ledger = None
        previous = sess.get("commit_result")
        if previous is not None:
            if signature == sess.get("commit_signature"):
                return previous
            raise HTTPException(
                status_code=409,
                detail="this batch was already committed with a different selection",
            )

    all_items = _prepare_batch_items(sess["images"], req)
    review = _batch_review_payload(all_items)
    ready_items = [item for item in all_items if item["status"] == "ready"]
    blocked_items = [item for item in all_items if item["status"] != "ready"]
    if req.ready_only:
        if not ready_items:
            raise HTTPException(
                status_code=409,
                detail={"message": "no ready tools to add", **review},
            )
        items = ready_items
        remaining_items = [] if req.discard_blocked else blocked_items
    else:
        if blocked_items:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": (
                        "batch has unresolved tools; add the ready subset or remove "
                        "blocked tools before saving"
                    ),
                    **review,
                },
            )
        items = all_items
        remaining_items = []

    selected_images = {
        int(idx) for item in items for idx in item.get("images", [])
    }
    already_committed = _batch_committed_images(sess)
    duplicate_images = selected_images & already_committed
    if duplicate_images:
        raise HTTPException(
            status_code=409,
            detail=f"image {min(duplicate_images)} is already in the library",
        )

    commit_path = _batch_project(sid) / "batch-commit.json"
    planned_ids = [
        f"{int(time.time())}-{uuid.uuid4().hex[:6]}" for _item in items
    ] if durable else []
    if durable:
        ledger["pending"] = {
            "signature": signature,
            "library_ids": planned_ids,
            "images": sorted(selected_images),
        }
        _atomic_json(commit_path, ledger)

    saved_ids: list[str] = []
    committed_items: list[dict] = []
    try:
        for index, item in enumerate(items):
            saved = _commit_batch_tool(
                item["label"],
                item["_outline"],
                item["thickness_mm"],
                calibration=item["_calibration"],
                photo_src=item["_photo_src"],
                raw_outline=item["_raw_outline"],
                readiness=item["_readiness"],
                provenance=item["_provenance"],
                outline_history=item["_outline_history"],
                outline_revision=item["_outline_revision"],
                tool_id=planned_ids[index] if durable else None,
            )
            saved_ids.append(saved.id)
            committed_items.append({
                **_batch_public_item(item),
                "library_id": saved.id,
            })
    except Exception as exc:
        for tool_id in planned_ids if durable else saved_ids:
            library_mod.delete(tool_id)
        if durable:
            ledger["pending"] = None
            _atomic_json(commit_path, ledger)
        raise HTTPException(status_code=500, detail=f"batch commit failed: {exc}")

    partial = bool(remaining_items)
    result = {
        "added": len(committed_items),
        "committed": not partial,
        "partial": partial,
        "discarded": len(blocked_items) if req.discard_blocked else 0,
        "remaining": len(remaining_items),
        "items": committed_items,
    }
    if durable:
        transaction = {
            "signature": signature,
            "library_ids": planned_ids,
            "images": sorted(selected_images),
            "result": result,
        }
        ledger["transactions"].append(transaction)
        ledger["committed_images"] = sorted(
            already_committed | selected_images
        )
        ledger["pending"] = None
        try:
            _atomic_json(commit_path, ledger)
        except Exception as exc:
            for tool_id in planned_ids:
                library_mod.delete(tool_id)
            ledger["transactions"].pop()
            ledger["committed_images"] = sorted(already_committed)
            ledger["pending"] = None
            _atomic_json(commit_path, ledger)
            raise HTTPException(status_code=500, detail=f"batch commit failed: {exc}")
        sess["commit_ledger"] = ledger
    else:
        sess["commit_signature"] = signature
        sess["commit_result"] = result

    if durable:
        job = _load_batch_job(sid)
        if job.get("result") is not None:
            if remaining_items:
                remaining_keys = {item["key"] for item in remaining_items}
                remaining_req = _batch_request_for_keys(req, remaining_keys)
                remaining_review = _batch_review_payload(remaining_items)
                draft = {
                    "version": 1,
                    "updated_ts": int(time.time()),
                    "selection": remaining_req.model_dump(mode="json"),
                    "review": remaining_review,
                }
                _atomic_json(_batch_project(sid) / "batch-draft.json", draft)
                job["result"]["draft"] = draft
            else:
                (_batch_project(sid) / "batch-draft.json").unlink(missing_ok=True)
                job["result"]["draft"] = None
            job["result"]["committed"] = not partial
            job["result"]["committed_images"] = ledger["committed_images"]
            job["result"]["commit_result"] = result
            job["result"]["partial_commits"] = len(ledger["transactions"])
            _save_batch_job(job)
    return result


def get_file(project: str, name: str) -> FileResponse:
    # constrain to the project dir; reject traversal
    path = (PROJECTS / project / name).resolve()
    if not str(path).startswith(str(PROJECTS.resolve())) or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


# Route topology is organized by domain; handlers remain import-compatible from
# this module while service extraction can proceed independently of HTTP wiring.
from gridshot.server.routes import batch as batch_routes
from gridshot.server.routes import capture as capture_routes
from gridshot.server.routes import export as export_routes
from gridshot.server.routes import library as library_routes
from gridshot.server.routes import operations as operation_routes

_ROUTE_OWNER = sys.modules[__name__]
operation_routes.configure(_ROUTE_OWNER)
capture_routes.configure(_ROUTE_OWNER)
_storage_health = operation_routes.storage_health
health_live = operation_routes.health_live
health_ready = operation_routes.health_ready
health_capabilities = operation_routes.health_capabilities
health = operation_routes.health
mats = operation_routes.mats
device_profiles = operation_routes.device_profiles
device_profiles_delete_all = operation_routes.device_profiles_delete_all
device_profile_delete = operation_routes.device_profile_delete
_device_profile_json = operation_routes.device_profile_json
calibration_intrinsics = operation_routes.calibration_intrinsics
trace = capture_routes.trace
get_result = capture_routes.get_result

for _router_factory in (
    operation_routes.build_router,
    capture_routes.build_router,
    library_routes.build_router,
    export_routes.build_router,
    batch_routes.build_router,
):
    app.include_router(_router_factory(_ROUTE_OWNER))


# SPA last: any non-/api path falls through to the built single-page app
class _SpaStatic(StaticFiles):
    """Serve the SPA so updates land immediately: index.html is revalidated on
    every load (no-cache) — otherwise the browser keeps an old index pointing at
    stale, content-hashed JS and the UI silently freezes on an old build. The
    hashed /assets/* are immutable, so they cache forever."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        if path == "." or path.endswith(".html"):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        elif path.startswith("assets/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp


if WEB_DIST.is_dir():
    app.mount("/", _SpaStatic(directory=WEB_DIST, html=True), name="spa")
