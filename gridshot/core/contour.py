"""Mask → polygon (with interior holes) → mat-mm → cleaned, clearance-offset pocket.

Pipeline position: segmentation produces a full-resolution binary mask; this
module turns it into millimetre-space geometry.  Interior rings are preserved
so hollow tools (a closed wrench ring, scissors handles) keep their islands —
those become raised pillars inside the pocket.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from . import calibrate as calibrate_mod
from .models import Calibration, Poly

MIN_COMPONENT_AREA_MM2 = 100.0  # ignore specks smaller than ~1 cm²
MIN_HOLE_AREA_MM2 = 25.0  # fill mask noise; keep real tool holes
SIMPLIFY_TOL_MM = 0.05


class NoToolFoundError(RuntimeError):
    pass


PIXEL_CENTER_COMP_PX = 0.5  # findContours traces pixel centres, half a pixel
# inside the true region boundary — grow outlines back out by that much


def mask_to_polygons_px(mask: np.ndarray) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Binary mask → [(exterior Nx2 px, [hole Nx2 px, ...]), ...], largest first.

    Outlines are compensated by +0.5px for OpenCV's pixel-centre convention
    (holes shrink by the same amount — the fit-safe direction for islands).
    """
    binary = (mask > 127).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # [next, prev, first_child, parent]

    out: list[tuple[np.ndarray, list[np.ndarray], float]] = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:  # holes handled via their parent
            continue
        exterior = cnt.reshape(-1, 2).astype(np.float64)
        if len(exterior) < 3:
            continue
        holes = []
        child = hierarchy[i][2]
        while child != -1:
            ring = contours[child].reshape(-1, 2).astype(np.float64)
            if len(ring) >= 3:
                holes.append(ring)
            child = hierarchy[child][0]

        shape = Polygon(exterior, holes).buffer(
            PIXEL_CENTER_COMP_PX, join_style="round", quad_segs=2
        )
        if isinstance(shape, MultiPolygon):
            shape = max(shape.geoms, key=lambda g: g.area)
        if shape.is_empty:
            continue
        comp_ext = np.asarray(shape.exterior.coords[:-1], dtype=np.float64)
        comp_holes = [
            np.asarray(r.coords[:-1], dtype=np.float64) for r in shape.interiors
        ]
        out.append((comp_ext, comp_holes, shape.area))

    out.sort(key=lambda t: -t[2])
    return [(e, h) for e, h, _ in out]


def select_prompted_components(
    mask: np.ndarray,
    points: list[tuple[float, float]] | None = None,
    labels: list[int] | None = None,
    box: tuple[float, float, float, float] | None = None,
    bridge_width_px: int = 5,
) -> np.ndarray:
    """Keep prompt-selected mask components and connect them minimally.

    SAM can return a thin shaft and a handle as separate foreground components.
    The application ultimately stores one physical-tool polygon, so merely
    retaining both components is insufficient: `mask_to_polygons_px` would
    sort them and downstream single-polygon callers would take the largest.

    Positive clicks are authoritative. A box selects non-trivial components
    whose centroids fall inside it. Selected components are joined at their
    nearest boundary points with a narrow bridge so the displayed and saved
    contour contains every explicitly selected part.
    """
    import cv2
    from scipy.spatial import cKDTree

    binary = ((mask > 127) * 255).astype(np.uint8)
    n, component_map, stats, centroids = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return binary

    foreground = list(range(1, n))
    largest = max(foreground, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    # Point refinement is additive around the primary object: retain the
    # largest component plus every component explicitly selected by a positive
    # click. A box is authoritative and selects all nontrivial parts inside it.
    keep: set[int] = {largest} if box is None else set()
    for (x, y), label in zip(points or [], labels or []):
        if int(label) != 1:
            continue
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < binary.shape[0] and 0 <= ix < binary.shape[1]:
            component = int(component_map[iy, ix])
            if component > 0:
                keep.add(component)

    if box is not None:
        x0, y0, x1, y1 = box
        min_area = max(16, int(stats[largest, cv2.CC_STAT_AREA] * 0.002))
        for component in foreground:
            cx, cy = centroids[component]
            if (
                x0 <= cx <= x1
                and y0 <= cy <= y1
                and int(stats[component, cv2.CC_STAT_AREA]) >= min_area
            ):
                keep.add(component)

    if not keep:
        keep.add(largest)

    selected = np.isin(component_map, list(keep)).astype(np.uint8) * 255
    if len(keep) == 1:
        return selected

    # Start with the largest selected component, then attach each remaining one
    # at the nearest contour points. cKDTree avoids an O(N*M) full-resolution
    # boundary comparison on phone photos.
    ordered = sorted(
        keep, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True
    )
    connected = (component_map == ordered[0]).astype(np.uint8) * 255
    for component in ordered[1:]:
        part = (component_map == component).astype(np.uint8) * 255
        base_contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        part_contours, _ = cv2.findContours(
            part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        base_points = np.vstack([c.reshape(-1, 2) for c in base_contours])
        part_points = np.vstack([c.reshape(-1, 2) for c in part_contours])
        distances, nearest = cKDTree(base_points).query(part_points, k=1)
        j = int(np.argmin(distances))
        a = tuple(int(v) for v in base_points[int(nearest[j])])
        b = tuple(int(v) for v in part_points[j])
        cv2.line(
            connected, a, b, 255, max(1, int(bridge_width_px)), cv2.LINE_8
        )
        connected = cv2.bitwise_or(connected, part)
    return connected


def polygon_px_to_mm(
    exterior_px: np.ndarray, holes_px: list[np.ndarray], calibration: Calibration
) -> Poly:
    ext = calibrate_mod.image_to_mm(exterior_px, calibration)
    holes = [calibrate_mod.image_to_mm(h, calibration) for h in holes_px]
    return Poly(
        exterior=[tuple(p) for p in ext],
        holes=[[tuple(p) for p in h] for h in holes],
    )


def to_shapely(poly: Poly) -> Polygon:
    return Polygon(poly.exterior, poly.holes)


def from_shapely(polygon: Polygon) -> Poly:
    polygon = orient(polygon, sign=1.0)  # exterior CCW, holes CW
    return Poly(
        exterior=[(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]],
        holes=[
            [(float(x), float(y)) for x, y in ring.coords[:-1]]
            for ring in polygon.interiors
        ],
    )


def clean(poly: Poly, simplify_tol: float = SIMPLIFY_TOL_MM) -> Poly:
    """Validity repair, small-hole removal, sub-0.05mm simplification."""
    shape = to_shapely(poly).buffer(0)  # repairs self-intersections
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda g: g.area)
    if shape.is_empty:
        raise NoToolFoundError("polygon vanished during cleanup")
    shape = Polygon(
        shape.exterior,
        [r for r in shape.interiors if Polygon(r).area >= MIN_HOLE_AREA_MM2],
    )
    return from_shapely(shape.simplify(simplify_tol, preserve_topology=True))


def _smooth_ring(pts: np.ndarray, sigma_mm: float, spacing_mm: float = 0.2) -> np.ndarray:
    """Circular Gaussian low-pass of a closed ring, arc-length parametrized."""
    pts = np.asarray(pts, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    length = float(seg.sum())
    if length < 4 * sigma_mm:
        return pts
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(length / spacing_mm), 32)
    su = np.linspace(0.0, length, n, endpoint=False)
    resampled = np.stack(
        [np.interp(su, s, closed[:, 0]), np.interp(su, s, closed[:, 1])], axis=1
    )
    half = max(int(round(3 * sigma_mm / spacing_mm)), 1)
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets * spacing_mm / sigma_mm) ** 2)
    kernel /= kernel.sum()
    idx = (np.arange(n)[:, None] + offsets[None, :]) % n
    return (resampled[idx] * kernel[None, :, None]).sum(axis=1)


def smooth(poly: Poly, radius_mm: float = 0.6) -> Poly:
    """Suppress boundary noise below `radius_mm` (segmentation jitter).

    Two complementary passes, thin features first:
    - light open+close at half the radius removes thin spikes/cracks while
      they are still thin — the arc-length filter would smear them into wide
      low bumps it can no longer distinguish from geometry.  Roughly neutral
      elsewhere (opening and closing bias in opposite directions).
    - Gaussian low-pass along the boundary: zero-mean jitter averages to the
      midline, so dimensions stay unbiased (morphological smoothing alone
      latches onto an envelope; corner-cutters shrink tips).
    Sharp corners round by roughly the radius; the fit clearance absorbs it.
    """
    if radius_mm <= 0:
        return poly
    r2 = radius_mm / 2  # opening then closing: -r2, +2·r2, -r2
    shape = (
        to_shapely(poly)
        .buffer(0)
        .buffer(-r2, join_style="round", quad_segs=8)
        .buffer(2 * r2, join_style="round", quad_segs=8)
        .buffer(-r2, join_style="round", quad_segs=8)
    )
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda g: g.area)
    if shape.is_empty:
        raise NoToolFoundError("polygon vanished during smoothing")
    shape = Polygon(
        _smooth_ring(np.asarray(shape.exterior.coords[:-1]), radius_mm),
        [
            _smooth_ring(np.asarray(ring.coords[:-1]), radius_mm)
            for ring in shape.interiors
        ],
    ).buffer(0)  # repair any self-intersection introduced at tight necks
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda g: g.area)
    if shape.is_empty:
        raise NoToolFoundError("polygon vanished during smoothing")
    return from_shapely(shape.simplify(SIMPLIFY_TOL_MM, preserve_topology=True))


STRAIGHTEN_TOL_MM = 0.35  # inlier band: a point within this of the run's line
STRAIGHTEN_MIN_LEN_MM = 12.0  # shortest run worth snapping to a line
STRAIGHTEN_INLIER = 0.80  # fraction of a run's points that must be inliers
_RESAMPLE_MM = 0.2


def _resample_closed(pts: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 16)
    su = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack(
        [np.interp(su, s, closed[:, 0]), np.interp(su, s, closed[:, 1])], axis=1
    )


def _dominant_dirs(pts: np.ndarray) -> np.ndarray:
    """The two major axes of a boundary: the length-weighted dominant edge
    direction and its perpendicular.  Straight tool edges cluster on these, so
    locking run fits to them makes a fit immune to spike-induced tilt."""
    v = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    lengths = np.linalg.norm(v, axis=1)
    th2 = 2 * np.arctan2(v[:, 1], v[:, 0])
    ang = 0.5 * np.arctan2((lengths * np.sin(th2)).sum(), (lengths * np.cos(th2)).sum())
    dom = np.array([np.cos(ang), np.sin(ang)])
    return np.array([dom, np.array([-dom[1], dom[0]])])


def _snap_dir(u: np.ndarray, major: np.ndarray | None, tol_deg: float = 8.0) -> np.ndarray:
    """Snap a fitted direction to a major axis when within tol — kills the
    tilt a spike near a seed would otherwise induce."""
    if major is None:
        return u
    best, best_ang = u, tol_deg
    for m in major:
        cosang = abs(float(m @ u))
        ang = math.degrees(math.acos(min(cosang, 1.0)))
        if ang < best_ang:
            best, best_ang = m, ang
    return best


def _robust_line(
    pts: np.ndarray, tol: float, iters: int = 3, major: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Total-least-squares line refit on its own inliers (centroid, unit dir,
    inlier fraction).  Iterating on inliers is the judgment loop: outlier bumps
    (SAM bleed, tick marks) drop out.  With `major` axes supplied, the fitted
    direction snaps to an axis so spikes shift the offset but never rotate it."""
    c = pts.mean(axis=0)
    d = pts - c
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    u = _snap_dir(vt[0], major)
    for _ in range(iters):
        nrm = np.array([-u[1], u[0]])
        inl = np.abs((pts - c) @ nrm) <= tol
        if inl.sum() < max(4, int(0.3 * len(pts))):
            break
        c = pts[inl].mean(axis=0)
        if major is None:
            d = pts[inl] - c
            _, _, vt = np.linalg.svd(d, full_matrices=False)
            u = vt[0]
    nrm = np.array([-u[1], u[0]])
    ratio = float((np.abs((pts - c) @ nrm) <= tol).mean())
    return c, u, ratio


STRAIGHTEN_BUMP_SKIP_MM = 6.0  # a run may jump a bump this wide and keep going
STRAIGHTEN_MAX_BOW_MM = 0.12  # a run bowing more than this along its length is a
# real curve (handle arc, tapered jaw), not a straight edge with jitter — don't
# flatten it. A true straight edge's residuals scatter both sides of the line
# (~0 bow); a gentle curve bows consistently to one side.


def _run_bow(seg: np.ndarray, c: np.ndarray, u: np.ndarray, tol: float) -> float:
    """Quadratic bow (mm) of a run's inlier residuals about its fit line.

    Near 0 for a straight edge (two-sided noise); grows with curvature. The
    signed normal residuals are fit as a parabola vs arc-length; the quadratic
    term is how far the edge bows over the run — the straight-vs-curve tell.
    """
    nrm = np.array([-u[1], u[0]])
    s = (seg - c) @ nrm
    inl = np.abs(s) <= tol
    if inl.sum() < 6:
        return 0.0
    t = (seg - c) @ u
    span = float(t[inl].max() - t[inl].min())
    if span < 1e-6:
        return 0.0
    tn = 2.0 * (t[inl] - t[inl].min()) / span - 1.0  # arc-length → [-1, 1]
    a = np.polyfit(tn, s[inl], 2)[0]
    return abs(float(a))


def _straight_runs(
    P: np.ndarray, tol: float, min_pts: int, step: int, major: np.ndarray | None = None
) -> list:
    """Region-grow maximal straight runs: a run is a straight EDGE even if it
    carries outlier bumps.  When growth stalls at a bump, the run looks past
    it (up to STRAIGHTEN_BUMP_SKIP_MM) and continues if the edge resumes on
    the same line — so a beam edge with periodic SAM-bleed bumps becomes one
    run instead of fragmenting.  Returns [(start, end_excl, c, u), ...]."""
    n = len(P)
    skip_max = int(STRAIGHTEN_BUMP_SKIP_MM / _RESAMPLE_MM)
    probe = max(min_pts // 2, 10)
    runs: list = []
    i = 0
    while i < n:
        j = i + min_pts
        if j > n:
            break
        c, u, ratio = _robust_line(P[i:j], tol, major=major)
        if ratio < 0.9:  # not straight enough to seed a run here
            i += step
            continue
        while j < n:
            jn = min(j + step, n)
            c2, u2, r2 = _robust_line(P[i:jn], tol, major=major)
            if r2 >= STRAIGHTEN_INLIER:
                j, c, u = jn, c2, u2
                continue
            # growth stalled — is this a bump the edge recovers from?
            nrm = np.array([-u[1], u[0]])
            jumped = False
            for skip in range(step, skip_max + 1, step):
                js = j + skip
                if js + probe > n:
                    break
                ahead = P[js : js + probe]
                if float(np.abs((ahead - c) @ nrm).max()) < 1.5 * tol:
                    c3, u3, r3 = _robust_line(P[i : js + probe], tol, major=major)
                    if r3 >= STRAIGHTEN_INLIER:
                        j, c, u = js + probe, c3, u3
                        jumped = True
                        break
            if not jumped:
                break
        # keep the run only if it's a genuine straight edge, not a gentle curve
        if _run_bow(P[i:j], c, u, tol) <= STRAIGHTEN_MAX_BOW_MM:
            runs.append((i, j, c, u))
        i = j
    return runs


STRAIGHTEN_GAP_BAND_MM = 15.0  # collinear flanking runs already prove it's a
# spike/notch (a real feature breaks collinearity); allow tall bleed spikes
STRAIGHTEN_MAX_GAP_MM = 30.0  # never bridge gaps longer than this
STRAIGHTEN_COLLINEAR_MM = 2.5  # merge parallel runs whose offsets differ by up
# to this — the bleed makes a straight edge wander ~1-2mm; distinct parallel
# edges (e.g. the two sides of a beam) are much further apart


def _merge_collinear(runs: list, P: np.ndarray, tol: float, major: np.ndarray | None = None) -> list:
    """Merge adjacent runs lying on the same line, absorbing the bump between.

    The gap between two collinear runs is a SAM-bleed bump; snapping the
    merged run to its robust line (which excludes the bump as outliers) then
    erases it.  The decision is geometric — the two runs must be parallel and
    collinear, and every gap point must stay within a bounded band of the
    shared line — so a real notch, tab, or curve (which departs far) never
    merges, but a 1-3mm bleed bump is folded in regardless of how tall it is.
    """
    if len(runs) < 2:
        return runs
    max_gap_pts = int(STRAIGHTEN_MAX_GAP_MM / _RESAMPLE_MM)
    merged = [runs[0]]
    for s, e, c, u in runs[1:]:
        ps, pe, pc, pu = merged[-1]
        nrm = np.array([-pu[1], pu[0]])
        parallel = abs(pu[0] * u[1] - pu[1] * u[0]) < 0.06  # < ~3.5°
        collinear = abs((c - pc) @ nrm) < STRAIGHTEN_COLLINEAR_MM
        gap = P[pe:s]
        gap_near = len(gap) == 0 or float(np.abs((gap - pc) @ nrm).max()) < STRAIGHTEN_GAP_BAND_MM
        if parallel and collinear and gap_near and 0 <= s - pe <= max_gap_pts:
            c2, u2, _ = _robust_line(P[ps:e], tol, major=major)  # robust: bump excluded
            merged[-1] = (ps, e, c2, u2)
            continue
        merged.append((s, e, c, u))
    return merged


STRAIGHTEN_EXTEND_MIN_MM = 25.0  # a run this long is a confident edge line
STRAIGHTEN_EXTEND_BAND_MM = 3.5  # scan across excursions (bleed spikes) up to this


def _extend_runs(runs: list, P: np.ndarray) -> list:
    """Extend confident long runs along their FIXED line to sweep up bleed
    bumps that never formed a run of their own.

    A bleed spike is a brief excursion that RETURNS to the edge line; a
    tangent curve (a rounded end meeting a straight side) departs and stays
    away.  So the extension only commits through points that come back within
    `tol` of the line inside a short window — it walks through spikes but
    stops at the onset of a real curve, which the band-only test would eat."""
    n = len(P)
    band = STRAIGHTEN_EXTEND_BAND_MM
    out = []
    for s, e, c, u in runs:
        length = np.linalg.norm(np.diff(P[s:e], axis=0), axis=1).sum()
        if length < STRAIGHTEN_EXTEND_MIN_MM:
            out.append((s, e, c, u))
            continue
        nrm = np.array([-u[1], u[0]])
        ee = e
        while ee < n and abs((P[ee] - c) @ nrm) <= band:
            ee += 1
        ss = s
        while ss > 0 and abs((P[ss - 1] - c) @ nrm) <= band:
            ss -= 1
        out.append((ss, ee, c, u))
    return out


def _straighten_ring(pts: np.ndarray, fit_tol: float, min_len: float) -> np.ndarray:
    """Replace straight boundary edges with their robust lines; keep curves.

    Region-grow straight runs (robust to outlier bumps via axis-locked fits),
    merge collinear runs, extend confident long runs to sweep up bleed bumps
    on the same edge, then project every covered point onto its line (removing
    bumps) while passing curves/short stretches through unchanged.
    """
    P = _resample_closed(np.asarray(pts, dtype=np.float64), _RESAMPLE_MM)
    n = len(P)
    if n < 40:
        return P

    step = max(int(1.0 / _RESAMPLE_MM), 3)  # ~1mm growth increments
    min_pts = max(int(min_len / _RESAMPLE_MM), 20)
    major = _dominant_dirs(P)
    runs = _merge_collinear(
        _straight_runs(P, fit_tol, min_pts, step, major), P, fit_tol, major
    )
    if not runs:
        return P
    runs = _extend_runs(runs, P)

    # assign each point to a covering run's line; longest run wins overlaps so
    # a confident edge dominates a short fragment
    line_of: list[tuple | None] = [None] * n
    for s, e, c, u in sorted(runs, key=lambda r: r[1] - r[0]):
        for k in range(s, e):
            line_of[k] = (c, u)

    out = np.empty((n, 2))
    for k in range(n):
        if line_of[k] is None:
            out[k] = P[k]
        else:
            c, u = line_of[k]
            out[k] = c + u * float((P[k] - c) @ u)  # project onto the line
    return out


def straighten(
    poly: Poly,
    fit_tol: float = STRAIGHTEN_TOL_MM,
    min_len: float = STRAIGHTEN_MIN_LEN_MM,
) -> Poly:
    """Boundary beautification for manufactured objects: see _straighten_ring."""
    if fit_tol <= 0:
        return poly
    shape = Polygon(
        _straighten_ring(np.array(poly.exterior), fit_tol, min_len),
        [
            _straighten_ring(np.array(ring), fit_tol, min_len)
            for ring in poly.holes
            if len(ring) >= 3
        ],
    ).buffer(0)
    if isinstance(shape, MultiPolygon):
        shape = max(shape.geoms, key=lambda g: g.area)
    if shape.is_empty:
        raise NoToolFoundError("polygon vanished during straightening")
    return from_shapely(shape.simplify(SIMPLIFY_TOL_MM, preserve_topology=True))


def offset(poly: Poly, clearance_mm: float) -> Poly:
    """Grow the outline by the fit clearance (round joins); holes shrink."""
    grown = to_shapely(poly).buffer(
        clearance_mm, join_style="round", quad_segs=8
    )
    if isinstance(grown, MultiPolygon):
        grown = max(grown.geoms, key=lambda g: g.area)
    # a hole narrower than 2×clearance closes itself — that's correct behaviour
    return from_shapely(grown)


def largest_tool_mm(
    mask: np.ndarray, calibration: Calibration
) -> tuple[Poly, list[Poly]]:
    """Largest mask component in mat-mm (cleaned), plus any other components.

    Returns (tool polygon, other sizable components) — callers warn on the
    latter so a stray object in frame doesn't silently become the "tool".
    """
    components = mask_to_polygons_px(mask)
    polys: list[Poly] = []
    for exterior, holes in components:
        p = polygon_px_to_mm(exterior, holes, calibration)
        if to_shapely(p).area >= MIN_COMPONENT_AREA_MM2:
            polys.append(p)
    if not polys:
        raise NoToolFoundError("no tool-sized region found in the segmentation mask")
    polys.sort(key=lambda p: -to_shapely(p).area)
    return clean(polys[0]), polys[1:]
