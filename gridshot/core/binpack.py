"""Pack tool outlines into one shared bin footprint (multi-tool bins).

A bottom-left-fill heuristic places the real polygons (largest first, 0/90
rotation) at the collision-free spot that keeps the overall footprint smallest,
leaving a wall gap between pockets. Because it tests the actual shapes — not
bounding rectangles — a small tool slots into the gap beside a big one instead
of claiming a whole cell. Pure geometry; a heuristic, not optimal packing.
"""

from __future__ import annotations

import numpy as np
from shapely.affinity import rotate as srotate
from shapely.affinity import translate

from .contour import from_shapely, to_shapely
from .models import Poly


def _at_origin(shape):
    b = shape.bounds
    return translate(shape, -b[0], -b[1])


def _stamp(shape):
    """Normalise a shape so its centroid is at the origin — the canonical form a
    placement (tx, ty, rot) is applied to: translate(rotate(stamp, rot), tx, ty)."""
    c = shape.centroid
    return translate(shape, -c.x, -c.y)


def stamp_poly(poly: Poly) -> Poly:
    """The centroid-normalised form of a tool outline (centroid at origin) — the
    exact shape `place_stamp` transforms, so a client can render placements too."""
    return from_shapely(_stamp(to_shapely(poly).buffer(0)))


def place_stamp(poly: Poly, tx: float, ty: float, rot: float) -> Poly:
    """Apply a placement to a tool outline — rotate about its centroid, then move
    the centroid to (tx, ty). Shared by preview, packing, and manual arrange."""
    s = _stamp(to_shapely(poly).buffer(0))
    return from_shapely(translate(srotate(s, rot, origin=(0, 0)), tx, ty))


def pack(
    polys: list[Poly], wall: float = 2.0, step: float = 2.5,
    rotations: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
) -> list[dict]:
    """Bottom-left-fill placements, one per input tool (original order).

    Each result is {"tx", "ty", "rot"} in the centroid frame — the placed pocket
    is `place_stamp(poly, tx, ty, rot)`. Largest-first; for each tool every
    rotation is tried and the spot that keeps the overall footprint smallest
    wins. More rotations than the legacy 0/90 lets asymmetric tools interlock.
    """
    stamps = [_stamp(to_shapely(p).buffer(0)) for p in polys]
    order = sorted(range(len(stamps)), key=lambda i: -stamps[i].area)
    placed: list = []  # shapes in a bottom-left (origin) frame
    out: list[dict | None] = [None] * len(stamps)

    for idx in order:
        # each rotation, normalised so its bbox-min sits at the origin
        variants = []
        for rot in rotations:
            r = srotate(stamps[idx], rot, origin=(0, 0))
            b = r.bounds
            variants.append((rot, translate(r, -b[0], -b[1])))
        if not placed:
            rot, v = min(variants, key=lambda rv: rv[1].bounds[2] * rv[1].bounds[3])
            placed.append(v)
            c = v.centroid
            out[idx] = {"tx": float(c.x), "ty": float(c.y), "rot": float(rot)}
            continue

        cur_w, cur_h = _extent(placed)
        best = None  # (footprint_area, shape, rot)
        for rot, v in variants:
            vw, vh = v.bounds[2], v.bounds[3]
            for y in np.arange(0.0, cur_h + vh + step, step):
                for x in np.arange(0.0, cur_w + vw + step, step):
                    cand = translate(v, float(x), float(y))
                    if all(cand.distance(q) >= wall for q in placed):
                        cb = cand.bounds
                        area = max(cur_w, cb[2]) * max(cur_h, cb[3])
                        if best is None or area < best[0]:
                            best = (area, cand, rot)
                        break  # first x that fits at this y (bottom-left)
        _, shape, rot = best
        placed.append(shape)
        c = shape.centroid
        out[idx] = {"tx": float(c.x), "ty": float(c.y), "rot": float(rot)}

    return out  # type: ignore[return-value]


def _extent(placed) -> tuple[float, float]:
    bs = [p.bounds for p in placed]
    return max(b[2] for b in bs), max(b[3] for b in bs)  # placed start at 0,0


def pack_polygons(
    polys: list[Poly], wall: float = 2.0, step: float = 3.0, rotate: bool = True
) -> list[Poly]:
    """Place each outline in the bin frame (bottom-left ≈ origin), keeping ≥wall
    between them. Returns the placed outlines in the original order."""
    shapes = [to_shapely(p).buffer(0) for p in polys]
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i].area)
    placed = []
    result: list[Poly | None] = [None] * len(shapes)

    for idx in order:
        variants = [
            _at_origin(srotate(shapes[idx], rot, origin=(0, 0)) if rot else shapes[idx])
            for rot in ([0.0, 90.0] if rotate else [0.0])
        ]
        if not placed:
            v = min(variants, key=lambda s: (s.bounds[2]) * (s.bounds[3]))
            placed.append(v)
            result[idx] = from_shapely(v)
            continue

        cur_w, cur_h = _extent(placed)
        best = None  # (footprint_area, shape)
        for v in variants:
            vw, vh = v.bounds[2], v.bounds[3]
            for y in np.arange(0.0, cur_h + vh + step, step):
                for x in np.arange(0.0, cur_w + vw + step, step):
                    cand = translate(v, float(x), float(y))
                    if all(cand.distance(q) >= wall for q in placed):
                        cb = cand.bounds
                        area = max(cur_w, cb[2]) * max(cur_h, cb[3])
                        if best is None or area < best[0]:
                            best = (area, cand)
                        break  # first x that fits at this y (bottom-left)
        placed.append(best[1])
        result[idx] = from_shapely(best[1])

    return [r for r in result]  # type: ignore[return-value]
