"""Canonical, deterministic derivation of printable bin geometry.

Capture produces a physical, parallax-corrected tool footprint. Every later
surface -- library cards, drawer composition, previews, and export -- must
derive its printable geometry from that footprint through this module. The
content key makes the dependency on tool geometry, settings, and printer
compensation explicit instead of letting persisted grid dimensions become a
second source of truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.ops import nearest_points

from . import bench as bench_mod
from . import contour as contour_mod
from . import gridfinity as grid_mod
from .models import Poly, PrinterProfile

DERIVATION_VERSION = "bin-spec-v3"


@dataclass(frozen=True)
class ToolGeometry:
    """Physical footprint plus distinct vertical measurements.

    ``thickness_mm`` remains a compatibility alias for silhouette-driving height.
    Full tool height controls automatic recess depth when it has been measured.
    """

    outline: Poly
    thickness_mm: float | None = None
    silhouette_height_mm: float | None = None
    full_height_mm: float | None = None

    def __post_init__(self) -> None:
        silhouette = self.silhouette_height_mm
        if silhouette is None:
            silhouette = self.thickness_mm
        if silhouette is None:
            raise ValueError("silhouette height is required")
        if self.thickness_mm is not None and not math.isclose(
            self.thickness_mm, silhouette, abs_tol=1e-6
        ):
            raise ValueError("thickness and silhouette height disagree")
        object.__setattr__(self, "thickness_mm", float(silhouette))
        object.__setattr__(self, "silhouette_height_mm", float(silhouette))


@dataclass(frozen=True)
class BinSettings:
    """Source settings that can change derived printable geometry."""

    clearance_mm: float = 1.0
    bin_style: grid_mod.BinStyle = "pocket"
    pocket_depth_mm: float | None = None
    height_u: int | None = None
    overall_height_mm: float | None = None
    lip: bool = True
    finger_hole: bool = False
    round_tool: bool = False


@dataclass
class DerivedBinSpec:
    """All non-mesh geometry needed to preview or manufacture one bin."""

    tool_poly: Poly
    cleared_poly: Poly
    compensated_poly: Poly
    pocket_poly: Poly
    sizing_poly: Poly
    grid: tuple[int, int]
    height_u: int
    bin_style: grid_mod.BinStyle
    lip: bool
    pocket_depth_mm: float
    overall_height_mm: float
    silhouette_height_mm: float
    full_height_mm: float | None
    finger_holes: list[tuple[float, float, float]] = field(default_factory=list)
    reserved_cells: list[tuple[float, float]] = field(default_factory=list)
    available_cells: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    derivation_key: str = ""


def _dominant_edge_angle_deg(shape) -> float:
    """Length-weighted axial mean of boundary edge orientations."""
    coords = np.asarray(shape.simplify(0.5).exterior.coords)
    vec = np.diff(coords, axis=0)
    lengths = np.linalg.norm(vec, axis=1)
    theta2 = 2 * np.arctan2(vec[:, 1], vec[:, 0])
    return math.degrees(
        0.5
        * math.atan2(
            (lengths * np.sin(theta2)).sum(),
            (lengths * np.cos(theta2)).sum(),
        )
    )


def _align_for_bin(shape, wall: float):
    """Choose the smallest useful bin rotation and return it plus a transform."""
    origin = tuple(np.asarray(shape.centroid.coords[0], dtype=np.float64))
    rect = list(shape.minimum_rotated_rectangle.exterior.coords)
    edges = [
        (rect[i + 1][0] - rect[i][0], rect[i + 1][1] - rect[i][1])
        for i in range(2)
    ]
    vx, vy = max(edges, key=lambda edge: edge[0] ** 2 + edge[1] ** 2)
    candidates = [
        _dominant_edge_angle_deg(shape),
        math.degrees(math.atan2(vy, vx)),
    ]
    best = None
    for angle in candidates:
        rotated = shapely_rotate(shape, -angle, origin=origin)
        gx, gy = grid_mod.auto_grid(contour_mod.from_shapely(rotated), wall=wall)
        candidate = (gx * gy, -angle)
        if best is None or candidate[0] < best[0]:
            best = candidate
    base = best[1]

    total = min(
        (base + delta / 4 for delta in range(-8, 9)),
        key=lambda angle: (
            lambda bounds: (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        )(shapely_rotate(shape, angle, origin=origin).bounds),
    )

    def apply(other):
        return shapely_rotate(other, total, origin=origin)

    return apply(shape), apply


def _derivation_key(
    tool: ToolGeometry, settings: BinSettings, printer: PrinterProfile
) -> str:
    payload = {
        "version": DERIVATION_VERSION,
        "tool": tool.outline.model_dump(mode="json"),
        "silhouette_height_mm": tool.silhouette_height_mm,
        "full_height_mm": tool.full_height_mm,
        "settings": asdict(settings),
        "printer_profile": printer.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def derive_bin_spec(
    tool: ToolGeometry,
    settings: BinSettings,
    printer_profile: PrinterProfile,
) -> DerivedBinSpec:
    """Derive one complete bin specification from authoritative source values.

    Tool is the physical, parallax-corrected footprint in the mat frame.
    No stored grid size, effective pocket depth, or previous generated file is
    consulted. The same inputs therefore produce the same geometry and key.
    """
    outline = tool.outline
    silhouette_height_mm = float(tool.silhouette_height_mm)
    full_height_mm = tool.full_height_mm
    tool_shape = contour_mod.to_shapely(outline)
    if tool_shape.is_empty or not tool_shape.is_valid or tool_shape.area <= 0:
        raise ValueError("tool outline must be a valid polygon with positive area")
    if not math.isfinite(silhouette_height_mm) or silhouette_height_mm <= 0:
        raise ValueError("silhouette height must be > 0")
    if full_height_mm is not None and (
        not math.isfinite(full_height_mm) or full_height_mm <= 0
    ):
        raise ValueError("full tool height must be > 0")
    if not math.isfinite(settings.clearance_mm) or settings.clearance_mm < 0:
        raise ValueError("clearance must be >= 0")
    if settings.bin_style not in ("pocket", "corral", "grid"):
        raise ValueError(f"unknown bin style: {settings.bin_style}")
    if settings.pocket_depth_mm is not None and (
        not math.isfinite(settings.pocket_depth_mm)
        or settings.pocket_depth_mm <= 0
    ):
        raise ValueError("pocket depth must be > 0")
    if settings.height_u is not None and settings.height_u < 1:
        raise ValueError("height must be at least 1u")
    if settings.overall_height_mm is not None and settings.overall_height_mm <= 0:
        raise ValueError("overall height must be > 0")

    warnings: list[str] = []
    effective_lip = settings.lip
    if printer_profile.created_at == "default":
        warnings.append(
            f"using assumed printer shrink {100 * bench_mod.DEFAULT_SHRINK:.1f}% — "
            "run gridshot bench coupon to measure it for high-shrink materials "
            "or near-maximal parts"
        )

    cleared = contour_mod.offset(outline, settings.clearance_mm)
    compensated_shape = bench_mod.compensate(
        contour_mod.to_shapely(cleared), printer_profile
    )
    compensated = contour_mod.from_shapely(compensated_shape)

    # Mat/image coordinates use y down; CAD and print coordinates use y up.
    flip_pocket = shapely_scale(compensated_shape, yfact=-1.0, origin=(0, 0))
    flip_tool = shapely_scale(tool_shape, yfact=-1.0, origin=(0, 0))
    wall = grid_mod.MIN_WALL_LIP if effective_lip else grid_mod.MIN_WALL
    if settings.bin_style in ("corral", "grid"):
        wall = max(
            wall,
            grid_mod.CORRAL_WALL
            + grid_mod.CORRAL_BASE_FLARE
            + grid_mod.CORRAL_EDGE_MARGIN,
        )
    aligned, apply_rotation = _align_for_bin(flip_pocket, wall)
    aligned_tool = apply_rotation(flip_tool)
    minx, miny, maxx, maxy = aligned.bounds
    tx, ty = -(minx + maxx) / 2, -(miny + maxy) / 2
    pocket_shape = translate(aligned, tx, ty)
    tool_bin_shape = translate(aligned_tool, tx, ty)

    if settings.pocket_depth_mm is None:
        if full_height_mm is not None:
            depth = grid_mod.auto_recess_depth(full_height_mm)
            height_basis = (
                f"measured full tool height {full_height_mm:.1f}mm + "
                f"{grid_mod.FULL_DEPTH_MARGIN:.1f}mm margin"
            )
        else:
            depth = grid_mod.auto_pocket_depth(
                silhouette_height_mm, settings.round_tool
            )
            height_basis = (
                f"legacy conservative estimate from silhouette height "
                f"{silhouette_height_mm:.1f}mm; record full tool height to replace it"
            )
        if settings.bin_style == "pocket":
            warnings.append(
                f"pocket depth defaulted to {depth:.1f}mm from {height_basis}; "
                "the tool remains below the stacking plane"
            )
        elif settings.bin_style == "corral":
            warnings.append(
                f"corral recess defaulted to {depth:.1f}mm from {height_basis}; "
                "the thin tool shelf and full-height separator retain the tool well"
            )
        else:
            warnings.append(
                f"live-grid recess defaulted to {depth:.1f}mm from {height_basis}; "
                "complete sockets are added only where they fit"
            )
    else:
        depth = settings.pocket_depth_mm

    fingers: list[tuple[float, float, float]] = []
    sizing = pocket_shape
    if settings.finger_hole:
        base_grid = grid_mod.auto_grid(
            contour_mod.from_shapely(pocket_shape), wall=wall
        )
        minx, miny, maxx, maxy = pocket_shape.bounds
        anchors = [
            pocket_shape.representative_point(),
            Point((minx + maxx) / 2, miny),
            Point((minx + maxx) / 2, maxy),
            Point(minx, (miny + maxy) / 2),
            Point(maxx, (miny + maxy) / 2),
        ]
        placed = None
        for anchor in anchors:
            point = nearest_points(pocket_shape.exterior, anchor)[0]
            candidate_shape = pocket_shape.union(point.buffer(10.0))
            candidate_grid = grid_mod.auto_grid(
                contour_mod.from_shapely(candidate_shape), wall=wall
            )
            if placed is None:
                placed = (point, candidate_shape)
            if candidate_grid == base_grid:
                placed = (point, candidate_shape)
                break
        point, sizing = placed
        fingers.append((float(point.x), float(point.y), 20.0))

    # Centre the complete cut envelope, including the optional scallop.
    sminx, sminy, smaxx, smaxy = sizing.bounds
    dx, dy = -(sminx + smaxx) / 2, -(sminy + smaxy) / 2
    pocket_shape = translate(pocket_shape, dx, dy)
    tool_bin_shape = translate(tool_bin_shape, dx, dy)
    sizing = translate(sizing, dx, dy)
    fingers = [(x + dx, y + dy, diameter) for x, y, diameter in fingers]

    gx, gy = grid_mod.auto_grid(contour_mod.from_shapely(sizing), wall=wall)
    need_u = grid_mod.auto_height_u(depth)
    if settings.height_u is not None:
        height_u = settings.height_u
        if height_u < need_u:
            requirement = f"a {depth:.1f}mm-deep {settings.bin_style} recess"
            raise ValueError(
                f"height {height_u}u is too short for {requirement}; "
                f"use at least {need_u}u"
            )
    elif settings.overall_height_mm is not None:
        height_u = grid_mod.height_u_for_style_overall(
            settings.overall_height_mm, effective_lip, settings.bin_style
        )
        if height_u < need_u:
            minimum = grid_mod.style_finished_height_mm(
                need_u, effective_lip, settings.bin_style
            )
            requirement = f"a {depth:.1f}mm-deep {settings.bin_style} recess"
            achieved = grid_mod.style_finished_height_mm(
                height_u, effective_lip, settings.bin_style
            )
            raise ValueError(
                f"overall height {settings.overall_height_mm:.1f}mm rounds to {height_u}u "
                f"= {achieved:.1f}mm, too short for {requirement} — set it to "
                f"at least {minimum:.1f}mm"
            )
        achieved = grid_mod.style_finished_height_mm(
            height_u, effective_lip, settings.bin_style
        )
        if abs(achieved - settings.overall_height_mm) > 0.1:
            warnings.append(
                f"overall height snapped to {achieved:.1f}mm "
                f"({height_u}u{' + lip' if effective_lip else ''}) — gridfinity heights "
                "come in 7mm steps"
            )
    else:
        height_u = need_u

    reserved_cells: list[tuple[float, float]] = []
    available_cells: list[tuple[float, float]] = []
    if settings.bin_style == "grid":
        cuts = [(contour_mod.from_shapely(pocket_shape), depth, fingers)]
        reserved_cells = grid_mod.grid_reserved_cells(
            gx, gy, cuts, lip=effective_lip
        )
        available_cells = grid_mod.grid_available_cells(
            gx, gy, cuts, lip=effective_lip
        )

    return DerivedBinSpec(
        tool_poly=contour_mod.from_shapely(tool_bin_shape),
        cleared_poly=cleared,
        compensated_poly=compensated,
        pocket_poly=contour_mod.from_shapely(pocket_shape),
        sizing_poly=contour_mod.from_shapely(sizing),
        grid=(gx, gy),
        height_u=height_u,
        bin_style=settings.bin_style,
        lip=effective_lip,
        pocket_depth_mm=depth,
        overall_height_mm=grid_mod.style_finished_height_mm(
            height_u, effective_lip, settings.bin_style
        ),
        silhouette_height_mm=silhouette_height_mm,
        full_height_mm=full_height_mm,
        finger_holes=fingers,
        reserved_cells=reserved_cells,
        available_cells=available_cells,
        warnings=warnings,
        derivation_key=_derivation_key(
            tool, settings, printer_profile
        ),
    )
