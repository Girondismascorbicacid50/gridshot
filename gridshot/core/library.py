"""Persistent tool library — reusable tools to compose into drawers.

The library decouples capture from layout: trace tools one (or a few) at a
time, each fully on the mat for best accuracy, save each here, then select and
nest many of them into a drawer later. So a large tool set is built from
several small, accurate captures instead of one big mat.

An entry stores what nesting and later bin generation need — footprint,
thickness, and a reference back to the source project (which still holds the
outline and the generated bin). Entries live as JSON under config/library/,
persisted by the same bind-mount as mat profiles; thumbnails sit beside them
as <id>.png, rendered by the web layer.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from . import bench as bench_mod
from . import derive as derive_mod
from .models import Calibration, Poly, PrinterProfile, config_dir
from .readiness import ArtifactProvenance, ReadinessReport


LIBRARY_SCHEMA_VERSION = "library.v2"
_LEGACY_SCHEMA_VERSIONS = {None, "library.v1"}
_SCHEMA_LOCK = threading.RLock()


class LibrarySchemaError(RuntimeError):
    """A library entry cannot be safely interpreted by this release."""


class OutlineEditRevision(BaseModel):
    """One accepted outline state, retained so edits are auditable/recoverable."""

    revision: int
    created_ts: int
    source: Literal[
        "baseline", "sam", "manual", "physical", "thickness", "batch", "single"
    ]
    raw_outline: Poly
    outline: Poly
    diagnostics: dict[str, float | int | str] = Field(default_factory=dict)


class LibraryTool(BaseModel):
    schema_version: Literal["library.v2"] = LIBRARY_SCHEMA_VERSION

    @model_validator(mode="after")
    def _resolve_height_compatibility(self):
        if self.silhouette_height_mm is None and self.thickness_mm > 0:
            self.silhouette_height_mm = self.thickness_mm
        elif self.silhouette_height_mm is not None and self.thickness_mm <= 0:
            self.thickness_mm = self.silhouette_height_mm
        elif self.silhouette_height_mm is not None and not math.isclose(
            self.silhouette_height_mm, self.thickness_mm, abs_tol=1e-6
        ):
            raise ValueError("thickness and silhouette height disagree")
        if self.full_height_mm is not None and self.full_height_mm <= 0:
            raise ValueError("full tool height must be > 0")
        return self
    id: str
    label: str = ""
    grid_x: float = 1.0
    grid_y: float = 1.0
    # Legacy cache fields retained for existing JSON. Consumers must derive them.
    derived_key: str | None = None
    thickness_mm: float = 0.0
    # Explicit name for the vertical height that drives silhouette correction.
    # `thickness_mm` remains a serialized compatibility alias.
    silhouette_height_mm: float | None = None
    # Measured maximum tool height; when present it drives automatic recess depth.
    full_height_mm: float | None = None
    # Visible silhouette mapped through the mat plane before parallax correction.
    # Photo overlays and future SAM edits use this representation.
    raw_outline: Poly | None = None
    # Physical footprint used to regenerate the bin (parallax-corrected, mat-mm).
    # Kept as `outline` for backward compatibility with existing library JSON.
    outline: Poly | None = None
    clearance_mm: float = 1.0
    bin_style: Literal["pocket", "corral", "grid"] = "pocket"
    # Shared recess-depth override; effective value always comes from derivation.
    pocket_depth_mm: float | None = None
    round_tool: bool = False  # domed/barrel → auto depth off ~2× widest-outline height
    finger_hole: bool = True  # full-depth pockets need a scallop to lift the tool out
    lip: bool = True
    # stored photo + calibration → SAM re-editing against the image in the library
    has_photo: bool = False
    calibration: Optional[Calibration] = None
    source_project: str = ""  # provenance, for re-editing
    source_tool: str = ""     # which tool within a batch ("" for single-tool)
    readiness: ReadinessReport | None = None
    provenance: ArtifactProvenance | None = None
    # Accepted outline revisions are immutable snapshots. Transient editor
    # clicks live in an in-memory session; an explicit Save appends one entry.
    outline_revision: int = 0
    outline_history: list[OutlineEditRevision] = Field(default_factory=list)
    created_ts: int = 0


def library_dir() -> Path:
    d = config_dir() / "library"
    d.mkdir(parents=True, exist_ok=True)
    return d


def backup_dir() -> Path:
    d = config_dir() / "library-backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tool_path(tool_id: str) -> Path:
    if not tool_id or Path(tool_id).name != tool_id:
        raise KeyError(tool_id)
    return library_dir() / f"{tool_id}.json"


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(value)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_bytes(path: Path, value: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(value)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _decode_entry(path: Path) -> tuple[LibraryTool, bool]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LibrarySchemaError(f"library entry {path.name} is corrupt") from exc
    version = payload.get("schema_version")
    if version not in {*_LEGACY_SCHEMA_VERSIONS, LIBRARY_SCHEMA_VERSION}:
        raise LibrarySchemaError(
            f"library entry {path.name} uses unsupported schema {version!r}"
        )
    migrated = version != LIBRARY_SCHEMA_VERSION
    if migrated:
        payload = dict(payload)
        payload["schema_version"] = LIBRARY_SCHEMA_VERSION
    try:
        return LibraryTool.model_validate(payload), migrated
    except ValueError as exc:
        raise LibrarySchemaError(f"library entry {path.name} is invalid") from exc


def _library_files() -> list[Path]:
    allowed = {".json", ".png", ".jpg", ".jpeg"}
    return sorted(
        path for path in library_dir().iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in allowed
    )


def _archive(reason: str) -> tuple[bytes, dict]:
    created_at = datetime.now(timezone.utc).isoformat()
    files = _library_files()
    records = []
    for path in files:
        data = path.read_bytes()
        records.append({
            "path": f"library/{path.name}",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "schema_version": LIBRARY_SCHEMA_VERSION,
        "created_at": created_at,
        "reason": reason,
        "tool_count": len([path for path in files if path.suffix == ".json"]),
        "files": records,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "library-manifest.json", json.dumps(manifest, indent=2, sort_keys=True)
        )
        for path in files:
            archive.write(path, f"library/{path.name}")
    return output.getvalue(), manifest


def ensure_schema() -> dict[str, int]:
    """Migrate legacy entries together, after one recoverable archive snapshot."""
    with _SCHEMA_LOCK:
        decoded = []
        for path in sorted(library_dir().glob("*.json")):
            tool, migrated = _decode_entry(path)
            decoded.append((path, tool, migrated))
        pending = [(path, tool) for path, tool, migrated in decoded if migrated]
        if not pending:
            return {"scanned": len(decoded), "migrated": 0}
        create_backup(reason="pre-library-v2-migration", ensure_current=False)
        for path, tool in pending:
            _atomic_text(path, tool.model_dump_json(indent=2))
        return {"scanned": len(decoded), "migrated": len(pending)}


def export_archive() -> tuple[bytes, dict]:
    ensure_schema()
    return _archive("export")


def create_backup(
    *, reason: str = "manual", ensure_current: bool = True
) -> tuple[Path, dict]:
    if ensure_current:
        ensure_schema()
    data, manifest = _archive(reason)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = backup_dir() / f"gridshot-library-{stamp}.zip"
    _atomic_bytes(path, data)
    return path, manifest


def derive_tool_spec(
    tool: LibraryTool,
    *,
    printer_profile: PrinterProfile | None = None,
    overall_height_mm: float | None = None,
    lip: bool | None = None,
) -> derive_mod.DerivedBinSpec:
    """Derive current printable geometry without consulting legacy caches."""
    if tool.outline is None:
        raise ValueError("library tool has no physical outline")
    printer = printer_profile or bench_mod.load_profile() or bench_mod.default_profile()
    return derive_mod.derive_bin_spec(
        derive_mod.ToolGeometry(
            outline=tool.outline,
            silhouette_height_mm=tool.silhouette_height_mm or tool.thickness_mm,
            full_height_mm=tool.full_height_mm,
        ),
        derive_mod.BinSettings(
            clearance_mm=tool.clearance_mm,
            bin_style=tool.bin_style,
            pocket_depth_mm=tool.pocket_depth_mm,
            overall_height_mm=overall_height_mm,
            lip=tool.lip if lip is None else lip,
            finger_hole=tool.finger_hole,
            round_tool=tool.round_tool,
        ),
        printer,
    )


def refresh_derived(
    tool: LibraryTool, printer_profile: PrinterProfile | None = None
) -> LibraryTool:
    """Refresh compatibility caches and their content-addressed invalidation key."""
    if tool.outline is None or (tool.silhouette_height_mm or tool.thickness_mm) <= 0:
        return tool.model_copy(update={"derived_key": None})
    spec = derive_tool_spec(tool, printer_profile=printer_profile)
    return tool.model_copy(
        update={
            "grid_x": float(spec.grid[0]),
            "grid_y": float(spec.grid[1]),
            "derived_key": spec.derivation_key,
        }
    )


def save(tool: LibraryTool) -> LibraryTool:
    ensure_schema()
    tool = refresh_derived(tool)
    path = _tool_path(tool.id)
    _atomic_text(path, tool.model_dump_json(indent=2))
    return tool


def load(tool_id: str) -> LibraryTool:
    ensure_schema()
    path = _tool_path(tool_id)
    if not path.is_file():
        raise KeyError(tool_id)
    return _decode_entry(path)[0]


def list_tools() -> list[LibraryTool]:
    """Every saved tool, newest first."""
    ensure_schema()
    return sorted(
        (_decode_entry(p)[0] for p in library_dir().glob("*.json")),
        key=lambda t: t.created_ts,
        reverse=True,
    )


def delete(tool_id: str) -> bool:
    if not tool_id or Path(tool_id).name != tool_id:
        return False
    removed = False
    for suffix in (".json", ".png", "-photo.jpg", "-photo-thumb.jpg"):
        p = library_dir() / f"{tool_id}{suffix}"
        if p.is_file():
            p.unlink()
            removed = removed or suffix == ".json"
    return removed
