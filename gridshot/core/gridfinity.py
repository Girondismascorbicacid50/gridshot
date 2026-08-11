"""Gridfinity bin solids with tool-shaped pockets, built on manifold3d.

Dimensional constants follow gridfinity-rebuilt-openscad's standard.scad
(MIT, Zack Freedman's spec relicense): 42 mm pitch, 7 mm height units,
41.5 mm bin footprint, 3.75 mm corner radius, and the 4.75 mm foot profile
(0.8 chamfer / 1.8 straight / 2.15 chamfer, 45°).  Chamfered transitions are
exact: hulls between rounded-rect plates, where a 45° chamfer is a uniform
offset that grows the corner radius by the same amount.

M1 scope: foot + body + pocket + finger holes.  Stacking lip and magnet
pockets arrive with the full generator in M3.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold, OpType
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .contour import to_shapely
from .models import Poly

PITCH = 42.0
UNIT_H = 7.0
BIN_SIZE = 41.5  # per-unit footprint: 0.25mm clearance per side within the pitch
CORNER_R = 3.75
BASE_H = 4.75
FOOT_CHAMFER_BOT = 0.8  # z 0 → 0.8, offset −(2.15+1.8... see profile below
FOOT_STRAIGHT = 1.8  # z 0.8 → 2.6
FOOT_CHAMFER_TOP = 2.15  # z 2.6 → 4.75
MIN_WALL = 2.0  # pocket to outer wall
MIN_FLOOR = 1.2  # under-pocket floor above the base

# stacking lip, per gridfinity.xyz spec (gridfinity-rebuilt STACKING_LIP_LINE):
# from the opening at the outer wall going down-inward: 1.9mm 45° chamfer,
# 1.8mm vertical, 0.7mm 45° to the inner tip 2.6mm inside the wall face
LIP_H = 4.4
LIP_CH_TOP = 1.9
LIP_STRAIGHT = 1.8
LIP_CH_BOT = 0.7
LIP_INSET = LIP_CH_TOP + LIP_CH_BOT  # 2.6
MIN_WALL_LIP = LIP_INSET + 0.8  # pocket clears the lip's cavity-floor edge; the
# ledge is backed by solid body, so 0.8mm of top-face margin suffices

EPS = 1e-3

CIRCULAR_SEGMENTS = 64

BinStyle = Literal["pocket", "corral", "grid"]

# Stackable corral/shadow-tray geometry. A thin bottom deck supports loose
# parts. Each tool rests on a thin shelf at its requested recess depth, inside
# a full-height separator that keeps loose parts out of the tool well.
CORRAL_FLOOR = 1.2
CORRAL_WALL = 2.0
CORRAL_BASE_FLARE = 0.8
CORRAL_BASE_REINFORCEMENT_H = 1.0
CORRAL_EDGE_MARGIN = 1.0

# Exact inverse socket profile from gridfinity-rebuilt's baseplate cutter.
BASEPLATE_H = 5.0
BASEPLATE_OUTER_R = 4.0
BASEPLATE_CLEARANCE_H = 0.35
BASEPLATE_CHAMFER_BOT = 0.7
BASEPLATE_STRAIGHT = 1.8
BASEPLATE_CHAMFER_TOP = 2.15
BASEPLATE_INNER_R = 1.15
BASEPLATE_INNER_SIZE = 36.3
GRID_SOCKET_GAP = 0.5
DRAWER_BASE_FLOOR = 1.2


def _rounded_rect(w: float, d: float, r: float) -> CrossSection:
    """Centred rounded rectangle as a CrossSection."""
    r = min(r, w / 2 - EPS, d / 2 - EPS)
    core = CrossSection.square((w - 2 * r, d - 2 * r), center=True)
    return core.offset(r, JoinType.Round, circular_segments=CIRCULAR_SEGMENTS)


def _plate(w: float, d: float, r: float, z: float) -> Manifold:
    return Manifold.extrude(_rounded_rect(w, d, r), EPS).translate((0, 0, z))


def _foot() -> Manifold:
    """One gridfinity foot, centred, sitting on z=0, top at BASE_H.

    Profile (per side, 45° chamfers): bottom face 35.6 mm (r 0.8), out to
    37.2 mm (r 1.6) at z 0.8, straight to z 2.6, out to 41.5 mm (r 3.75)
    at z 4.75.
    """
    top_w = BIN_SIZE
    mid_w = top_w - 2 * FOOT_CHAMFER_TOP  # 37.2
    bot_w = mid_w - 2 * FOOT_CHAMFER_BOT  # 35.6
    r_top = CORNER_R
    r_mid = r_top - FOOT_CHAMFER_TOP  # 1.6
    r_bot = r_mid - FOOT_CHAMFER_BOT  # 0.8

    z_mid0 = FOOT_CHAMFER_BOT
    z_mid1 = FOOT_CHAMFER_BOT + FOOT_STRAIGHT

    chamfer_bot = Manifold.batch_hull(
        [_plate(bot_w, bot_w, r_bot, 0.0), _plate(mid_w, mid_w, r_mid, z_mid0 - EPS)]
    )
    straight = Manifold.extrude(_rounded_rect(mid_w, mid_w, r_mid), FOOT_STRAIGHT).translate(
        (0, 0, z_mid0)
    )
    chamfer_top = Manifold.batch_hull(
        [_plate(mid_w, mid_w, r_mid, z_mid1), _plate(top_w, top_w, r_top, BASE_H - EPS)]
    )
    return chamfer_bot + straight + chamfer_top


def _baseplate_socket_cutter() -> Manifold:
    """Exact negative volume for one upstream-profile baseplate socket."""
    mid_size = BASEPLATE_INNER_SIZE + 2 * BASEPLATE_CHAMFER_BOT
    mid_r = BASEPLATE_INNER_R + BASEPLATE_CHAMFER_BOT
    z0 = BASEPLATE_CLEARANCE_H
    z1 = z0 + BASEPLATE_CHAMFER_BOT
    z2 = z1 + BASEPLATE_STRAIGHT
    bottom = Manifold.extrude(
        _rounded_rect(BASEPLATE_INNER_SIZE, BASEPLATE_INNER_SIZE, BASEPLATE_INNER_R),
        z0 + EPS,
    ).translate((0, 0, -EPS))
    lower = Manifold.batch_hull([
        _plate(BASEPLATE_INNER_SIZE, BASEPLATE_INNER_SIZE, BASEPLATE_INNER_R, z0),
        _plate(mid_size, mid_size, mid_r, z1),
    ])
    straight = Manifold.extrude(
        _rounded_rect(mid_size, mid_size, mid_r), BASEPLATE_STRAIGHT + 2 * EPS
    ).translate((0, 0, z1 - EPS))
    upper = Manifold.batch_hull([
        _plate(mid_size, mid_size, mid_r, z2),
        _plate(PITCH, PITCH, BASEPLATE_OUTER_R, BASEPLATE_H),
    ])
    return bottom + lower + straight + upper


def _baseplate_socket() -> Manifold:
    """One mechanically compatible 42 mm socket ring, z=0 through z=5."""
    outer = Manifold.extrude(
        _rounded_rect(PITCH, PITCH, BASEPLATE_OUTER_R), BASEPLATE_H
    )
    return outer - _baseplate_socket_cutter()


def drawer_baseplate_solid(cols: int, rows: int) -> Manifold:
    """A complete drawer-sized Gridfinity socket grid for 3D composition.

    The preview base is deliberately separate from printable bin export: it
    represents the drawer's existing 42 mm socket grid so composed bins can be
    inspected in context. Its top is z=0 and each standard socket rises from
    there, which lets the exact bin feet seat at the same z used by fit tests.
    """
    if cols < 1 or rows < 1:
        raise ValueError("drawer grid must be at least 1×1")
    width = cols * PITCH
    depth = rows * PITCH
    floor = Manifold.extrude(
        _rounded_rect(width, depth, BASEPLATE_OUTER_R),
        DRAWER_BASE_FLOOR,
    ).translate((0, 0, -DRAWER_BASE_FLOOR))
    socket = _baseplate_socket()
    sockets = [
        socket.translate((
            (col - (cols - 1) / 2) * PITCH,
            (row - (rows - 1) / 2) * PITCH,
            0,
        ))
        for col in range(cols)
        for row in range(rows)
    ]
    return Manifold.batch_boolean([floor, *sockets], OpType.Add)


LIP_RIM_FLAT = 0.4  # top rim flat: the spec draws a knife edge, which is both
# unprintable and degenerate to mesh — a 0.4mm flat keeps 1.5mm of 45° chamfer
# engagement and clean geometry (foot seats 0.4mm shallower, laterally solid)


def _lip_ring(outer_w: float, outer_d: float, z_top: float) -> Manifold:
    """Spec stacking lip: a 4.4mm rim whose inner cavity accepts the foot of
    the bin above (45° faces mate flush).  Sits on the solid body, so the
    spec's thin-wall support section isn't needed."""
    ring = Manifold.extrude(
        _rounded_rect(outer_w, outer_d, CORNER_R), LIP_H
    ).translate((0, 0, z_top))

    z_hi = z_top + LIP_H
    f = LIP_RIM_FLAT
    ov = 0.05  # segments interpenetrate by a real overlap — plate-to-plate
    # seams at EPS thickness tessellate into sub-float32 sliver faces

    def rr(inset: float) -> CrossSection:
        return _rounded_rect(outer_w - 2 * inset, outer_d - 2 * inset, CORNER_R - inset)

    def plate(inset: float, z: float) -> Manifold:
        return _plate(outer_w - 2 * inset, outer_d - 2 * inset, CORNER_R - inset, z)

    # opening band: vertical at the rim-flat inset, over-tall for a clean cut
    opening = Manifold.extrude(rr(f), 1.0 + f + ov).translate((0, 0, z_hi - f - ov))
    # upper 45° chamfer (truncated by the rim flat, same spec plane)
    upper = Manifold.batch_hull([plate(f, z_hi - f), plate(LIP_CH_TOP, z_hi - LIP_CH_TOP)])
    # straight section, overlapping both chamfers
    straight = Manifold.extrude(
        rr(LIP_CH_TOP), (LIP_STRAIGHT) + 2 * ov
    ).translate((0, 0, z_top + LIP_CH_BOT - ov))
    # lower 45° chamfer, extended 0.2 below the cavity floor so the boolean
    # never meets the body's top face edge-on (leaves a hidden micro-groove)
    lower = Manifold.batch_hull(
        [plate(LIP_CH_TOP, z_top + LIP_CH_BOT), plate(LIP_INSET + 0.2, z_top - 0.2)]
    )
    cavity = opening + upper + straight + lower
    return ring - cavity


def _cross_section_from_poly(poly: Poly) -> CrossSection:
    shape = to_shapely(poly)
    rings = [np.asarray(shape.exterior.coords[:-1], dtype=np.float64)]
    rings += [np.asarray(r.coords[:-1], dtype=np.float64) for r in shape.interiors]
    # EvenOdd is orientation-independent: interior rings become holes
    return CrossSection(rings, FillRule.EvenOdd)


def auto_grid(pocket: Poly, wall: float = MIN_WALL) -> tuple[int, int]:
    """Smallest full-unit footprint whose usable interior fits the pocket."""
    minx, miny, maxx, maxy = to_shapely(pocket).bounds
    need_w = (maxx - minx) + 2 * wall
    need_d = (maxy - miny) + 2 * wall
    gx = max(1, math.ceil((need_w + (PITCH - BIN_SIZE)) / PITCH))
    gy = max(1, math.ceil((need_d + (PITCH - BIN_SIZE)) / PITCH))
    return gx, gy


FULL_DEPTH_MARGIN = 1.5  # sink the tool just below the rim so bins still stack


def auto_recess_depth(full_tool_height_mm: float) -> float:
    """Recess a measured full tool just below the stacking plane."""
    if not math.isfinite(full_tool_height_mm) or full_tool_height_mm <= 0:
        raise ValueError("full tool height must be > 0")
    return max(2.0, full_tool_height_mm + FULL_DEPTH_MARGIN)


def auto_pocket_depth(thickness_mm: float, round_tool: bool = False) -> float:
    """Default pocket depth = full depth for the whole tool, recessed just below
    the rim so the bin stacks and retrieval is by finger scallop (not by pinching
    a proud tool).

    This is the conservative legacy fallback when only silhouette-driving
    height is known. New callers should pass measured full height through
    :func:`auto_recess_depth`. `round_tool` remains for call-site compatibility.
    """
    return max(2.0, 2.0 * thickness_mm + FULL_DEPTH_MARGIN)


def auto_height_u(pocket_depth: float) -> int:
    """Fewest whole gridfinity units that hold base + floor + pocket."""
    return max(1, math.ceil((BASE_H + MIN_FLOOR + pocket_depth) / UNIT_H))


def style_finished_height_mm(height_u: int, lip: bool, style: BinStyle) -> float:
    """Physical top height for any tool-retention style."""
    return finished_height_mm(height_u, lip)


def height_u_for_style_overall(overall_mm: float, lip: bool, style: BinStyle) -> int:
    """Inverse of style_finished_height_mm, snapped to whole units."""
    return height_u_for_overall(overall_mm, lip)


def finished_height_mm(height_u: int, lip: bool) -> float:
    """Physical top-of-bin height: the unit stack plus the stacking lip.

    The lip rises LIP_H above height_u·UNIT_H, so a lipped bin is 4.4mm taller
    than a lipless one of the same unit count. This is the height that must
    match for bins to sit level in a drawer.
    """
    return height_u * UNIT_H + (LIP_H if lip else 0.0)


def height_u_for_overall(overall_mm: float, lip: bool) -> int:
    """Whole unit count whose finished height is closest to overall_mm.

    Inverse of finished_height_mm: subtract the lip (when present) before
    quantising to the 7mm unit, so the *finished* height — not the bare unit
    stack — is what tracks the requested value.
    """
    body = overall_mm - (LIP_H if lip else 0.0)
    return max(1, round(body / UNIT_H))


class PocketTooDeepError(ValueError):
    pass


def _rounded_rect_polygon(w: float, d: float, r: float):
    """Shapely rounded rectangle matching :func:`_rounded_rect`."""
    r = min(r, w / 2 - EPS, d / 2 - EPS)
    return box(
        -w / 2 + r,
        -d / 2 + r,
        w / 2 - r,
        d / 2 - r,
    ).buffer(r, quad_segs=16)


def grid_candidate_cells(
    gx: int, gy: int, *, lip: bool = False
) -> list[tuple[float, float]]:
    """Complete 42 mm sockets that fit inside the corral's outer wall.

    The candidate field is centred and remains on 42 mm pitch. It is never used
    to grow the bin; small or narrow corrals simply return no candidates.
    """
    outer_w = PITCH * gx - (PITCH - BIN_SIZE)
    outer_d = PITCH * gy - (PITCH - BIN_SIZE)
    inset = MIN_WALL_LIP if lip else MIN_WALL
    inner_w = outer_w - 2 * inset
    inner_d = outer_d - 2 * inset
    if inner_w <= 0 or inner_d <= 0:
        return []
    inner_r = max(EPS, CORNER_R - inset)
    usable = _rounded_rect_polygon(inner_w, inner_d, inner_r).buffer(
        -GRID_SOCKET_GAP
    )
    if usable.is_empty:
        return []
    minx, miny, maxx, maxy = usable.bounds
    nx = max(0, math.floor((maxx - minx + EPS) / PITCH))
    ny = max(0, math.floor((maxy - miny + EPS) / PITCH))
    socket_shape = _rounded_rect_polygon(PITCH, PITCH, BASEPLATE_OUTER_R)
    candidates = []
    for ix in range(nx):
        for iy in range(ny):
            cx = (ix - (nx - 1) / 2) * PITCH
            cy = (iy - (ny - 1) / 2) * PITCH
            cell = shapely_translate(socket_shape, xoff=cx, yoff=cy)
            if usable.covers(cell):
                candidates.append((float(cx), float(cy)))
    return candidates


def _grid_retention_envelope(pockets: list[tuple]):
    envelopes = []
    for entry in pockets:
        shape = to_shapely(entry[0])
        fingers = entry[2] if len(entry) > 2 else ()
        lobes = [
            Point(float(x), float(y)).buffer(float(diameter) / 2)
            for x, y, diameter in fingers
        ]
        if lobes:
            shape = unary_union([shape, *lobes])
        envelopes.append(
            shape.buffer(
                CORRAL_WALL + CORRAL_BASE_FLARE + GRID_SOCKET_GAP,
                quad_segs=16,
            )
        )
    return unary_union(envelopes)


def grid_reserved_cells(
    gx: int, gy: int, pockets: list[tuple], *, lip: bool = False
) -> list[tuple[float, float]]:
    """Fully fitting socket positions blocked by the complete tool envelope."""
    envelope = _grid_retention_envelope(pockets)
    socket_shape = _rounded_rect_polygon(PITCH, PITCH, BASEPLATE_OUTER_R)
    return [
        (cx, cy)
        for cx, cy in grid_candidate_cells(gx, gy, lip=lip)
        if not shapely_translate(socket_shape, xoff=cx, yoff=cy).disjoint(envelope)
    ]


def grid_available_cells(
    gx: int, gy: int, pockets: list[tuple], *, lip: bool = False
) -> list[tuple[float, float]]:
    """Complete, unobstructed socket centres in the corral floor."""
    envelope = _grid_retention_envelope(pockets)
    socket_shape = _rounded_rect_polygon(PITCH, PITCH, BASEPLATE_OUTER_R)
    return [
        (cx, cy)
        for cx, cy in grid_candidate_cells(gx, gy, lip=lip)
        if shapely_translate(socket_shape, xoff=cx, yoff=cy).disjoint(envelope)
    ]


def bin_solid(
    gx: int,
    gy: int,
    height_u: int,
    pocket: Poly | None = None,
    pocket_depth: float = 0.0,
    finger_holes: list[tuple[float, float, float]] = (),
    lip: bool = False,
    pockets: list[tuple[Poly, float]] | None = None,
    style: BinStyle = "pocket",
) -> Manifold:
    """A Gridfinity pocket, corral, or live-grid tool holder.

    Grid keeps the complete corral and adds only fully fitting sockets to its
    unused floor. Recess depth is measured below the stacking plane.
    """
    if style not in ("pocket", "corral", "grid"):
        raise ValueError(f"unknown bin style: {style}")
    cuts = list(pockets) if pockets else (
        [(pocket, pocket_depth, finger_holes if style != "pocket" else ())]
        if pocket is not None and pocket_depth > 0 else []
    )
    if style in ("corral", "grid") and not cuts:
        raise ValueError(f"{style} style needs at least one tool footprint")

    total_h = height_u * UNIT_H
    outer_w = PITCH * gx - (PITCH - BIN_SIZE)
    outer_d = PITCH * gy - (PITCH - BIN_SIZE)
    if style == "pocket":
        body = Manifold.extrude(
            _rounded_rect(outer_w, outer_d, CORNER_R), total_h - BASE_H
        ).translate((0, 0, BASE_H))
    else:
        deck_top = BASE_H + CORRAL_FLOOR
        deck = Manifold.extrude(
            _rounded_rect(outer_w, outer_d, CORNER_R), CORRAL_FLOOR
        ).translate((0, 0, BASE_H))
        outer = _rounded_rect(outer_w, outer_d, CORNER_R)
        inset = MIN_WALL_LIP if lip else MIN_WALL
        inner = _rounded_rect(
            outer_w - 2 * inset, outer_d - 2 * inset, CORNER_R - inset
        )
        perimeter = Manifold.extrude(
            outer - inner, total_h - deck_top + EPS
        ).translate((0, 0, deck_top - EPS))
        body = deck + perimeter
        if style == "grid":
            if total_h < deck_top + BASEPLATE_H - EPS:
                raise ValueError(
                    "grid style needs at least 2u so socket walls remain below "
                    "the stacking plane"
                )
            socket = _baseplate_socket().translate((0, 0, deck_top - EPS))
            for cx, cy in grid_available_cells(gx, gy, cuts, lip=lip):
                body = body + socket.translate((cx, cy, 0))
    if lip:
        body = body + _lip_ring(outer_w, outer_d, total_h)

    foot = _foot()
    feet = []
    for ix in range(gx):
        for iy in range(gy):
            cx = (ix - (gx - 1) / 2) * PITCH
            cy = (iy - (gy - 1) / 2) * PITCH
            feet.append(foot.translate((cx, cy, 0)))
    solid = Manifold.batch_boolean([body, *feet], OpType.Add)

    for entry in cuts:
        pk, depth = entry[0], entry[1]
        pk_fingers = entry[2] if len(entry) > 2 else ()
        if style in ("corral", "grid"):
            if depth <= 0:
                raise ValueError(f"{style} recess depth must be > 0")
            floor_z = total_h - depth
            deck_top = BASE_H + CORRAL_FLOOR
            if floor_z < deck_top - EPS:
                raise PocketTooDeepError(
                    f"{style} recess depth {depth}mm needs "
                    f"≥{deck_top + depth:.1f}mm of finished height; "
                    f"increase height_u (now {height_u})"
                )
            inner = _cross_section_from_poly(pk)
            for fx, fy, dia in pk_fingers:
                inner = inner + CrossSection.circle(
                    dia / 2, CIRCULAR_SEGMENTS
                ).translate((fx, fy))
            outer = inner.offset(
                CORRAL_WALL, JoinType.Round, circular_segments=CIRCULAR_SEGMENTS
            )
            reinforced = inner.offset(
                CORRAL_WALL + CORRAL_BASE_FLARE,
                JoinType.Round,
                circular_segments=CIRCULAR_SEGMENTS,
            )
            separator = Manifold.extrude(
                outer - inner, total_h - deck_top + EPS
            ).translate((0, 0, deck_top - EPS))
            base = Manifold.extrude(
                reinforced - inner, CORRAL_BASE_REINFORCEMENT_H + EPS
            ).translate((0, 0, deck_top - EPS))
            shelf_bottom = max(BASE_H, floor_z - CORRAL_FLOOR)
            shelf = Manifold.extrude(
                outer, floor_z - shelf_bottom + EPS
            ).translate((0, 0, shelf_bottom))
            solid = solid + separator + base + shelf
            continue
        floor_z = total_h - depth
        if floor_z < BASE_H + MIN_FLOOR - EPS:
            raise PocketTooDeepError(
                f"pocket depth {depth}mm needs ≥{BASE_H + MIN_FLOOR + depth:.1f}mm "
                f"of bin height; increase height_u (now {height_u})"
            )
        cut = Manifold.extrude(
            _cross_section_from_poly(pk), depth + EPS
        ).translate((0, 0, floor_z))
        solid = solid - cut
        for fx, fy, dia in pk_fingers:
            cyl = Manifold.cylinder(
                depth + EPS, dia / 2, dia / 2, CIRCULAR_SEGMENTS
            ).translate((fx, fy, floor_z))
            solid = solid - cyl

    if style == "pocket" and pocket is not None and pocket_depth > 0:
        floor_z = total_h - pocket_depth
        for fx, fy, dia in finger_holes:
            cyl = Manifold.cylinder(
                pocket_depth + EPS, dia / 2, dia / 2, CIRCULAR_SEGMENTS
            ).translate((fx, fy, floor_z))
            solid = solid - cyl

    return solid


def to_trimesh(solid: Manifold):
    import trimesh

    mesh = solid.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vert_properties)[:, :3],
        faces=np.asarray(mesh.tri_verts),
        process=False,
    )
