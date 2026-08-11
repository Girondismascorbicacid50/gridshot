"""Tool-thickness parallax correction — the error no shipping tool fixes.

A tool of thickness t photographed from height H projects its top-edge
silhouette *outward* onto the paper plane: the camera "sees over" the sides.
Mapping the image contour through the plane homography therefore lands each
point too far from the camera's nadir by a factor H/(H−t).

The exact inverse: the observed contour point P0 on the z=0 plane came from a
physical edge at height t.  The ray from the camera centre C through that
edge hits z=0 at P0, so the edge's (x, y) is the point on segment C→P0 where
z = −t (mat frame: +z into the table, camera at z = −H):

    P = nadir + (1 − t/H) · (P0 − nadir)

which depends only on camera height and nadir — both recovered by solvePnP —
and holds exactly for any camera tilt.  `thickness_mm` is the height of the
tool's *widest* outline: full thickness for prismatic tools, roughly half for
barrel-shaped ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Calibration, Poly


class MissingPoseError(RuntimeError):
    """Raised when the calibration carries no camera pose."""


class LocalReconstructionError(RuntimeError):
    """Raised when two silhouettes cannot support a trustworthy footprint."""


@dataclass(frozen=True)
class FootprintReconstruction:
    """Physical footprint selected from two calibrated silhouettes.

    ``scalar_height_mm`` remains the tool's effective widest-outline height and
    is still used for automatic pocket depth. The per-boundary heights are used
    only to recover the top-opening footprint.
    """

    polygon: Poly
    method: str
    scalar_height_mm: float
    scalar_residual_mm2: float
    boundary_mean_error_mm: float
    boundary_p95_error_mm: float
    height_p05_mm: float
    height_median_mm: float
    height_p95_mm: float
    scalar_major_extent_mm: float
    reconstructed_major_extent_mm: float
    reconstructed_minor_extent_mm: float
    area_change_from_scalar_pct: float

    def diagnostics(self) -> dict[str, float | str]:
        return {
            "method": self.method,
            "scalar_height_mm": round(self.scalar_height_mm, 4),
            "scalar_residual_mm2": round(self.scalar_residual_mm2, 4),
            "boundary_mean_error_mm": round(self.boundary_mean_error_mm, 4),
            "boundary_p95_error_mm": round(self.boundary_p95_error_mm, 4),
            "height_p05_mm": round(self.height_p05_mm, 4),
            "height_median_mm": round(self.height_median_mm, 4),
            "height_p95_mm": round(self.height_p95_mm, 4),
            "scalar_major_extent_mm": round(self.scalar_major_extent_mm, 4),
            "reconstructed_major_extent_mm": round(
                self.reconstructed_major_extent_mm, 4
            ),
            "reconstructed_minor_extent_mm": round(
                self.reconstructed_minor_extent_mm, 4
            ),
            "area_change_from_scalar_pct": round(
                self.area_change_from_scalar_pct, 4
            ),
        }


def shrink_factor(calibration: Calibration, thickness_mm: float) -> float:
    if calibration.camera_height_mm is None or calibration.nadir_xy_mm is None:
        raise MissingPoseError(
            "calibration has no camera pose — parallax correction unavailable"
        )
    H = calibration.camera_height_mm
    if thickness_mm < 0:
        raise ValueError("thickness must be >= 0")
    if thickness_mm >= 0.5 * H:
        raise ValueError(
            f"tool thickness {thickness_mm}mm vs camera height {H:.0f}mm — "
            "shoot from higher up"
        )
    return 1.0 - thickness_mm / H


def correct_points(
    points_mm: np.ndarray, calibration: Calibration, thickness_mm: float
) -> np.ndarray:
    """Shrink mat-plane points toward the camera nadir by the exact factor."""
    k = shrink_factor(calibration, thickness_mm)
    nadir = np.asarray(calibration.nadir_xy_mm, dtype=np.float64)
    pts = np.asarray(points_mm, dtype=np.float64)
    return nadir + k * (pts - nadir)


def correct_polygon(
    poly: Poly, calibration: Calibration, thickness_mm: float
) -> Poly:
    return Poly(
        exterior=[tuple(p) for p in correct_points(np.array(poly.exterior), calibration, thickness_mm)],
        holes=[
            [tuple(p) for p in correct_points(np.array(ring), calibration, thickness_mm)]
            for ring in poly.holes
        ],
    )


def uncorrect_points(
    points_mm: np.ndarray, calibration: Calibration, thickness_mm: float
) -> np.ndarray:
    """Expand a corrected footprint back to its visible plane-mapped silhouette.

    This is the exact inverse of :func:`correct_points`. It is primarily used
    to migrate older library entries that stored only the corrected footprint
    even though photo editors need the raw silhouette seen in the source image.
    """
    k = shrink_factor(calibration, thickness_mm)
    nadir = np.asarray(calibration.nadir_xy_mm, dtype=np.float64)
    pts = np.asarray(points_mm, dtype=np.float64)
    return nadir + (pts - nadir) / k


def uncorrect_polygon(
    poly: Poly, calibration: Calibration, thickness_mm: float
) -> Poly:
    """Inverse of :func:`correct_polygon` for photo-overlay reconstruction."""
    return Poly(
        exterior=[
            tuple(p)
            for p in uncorrect_points(
                np.array(poly.exterior), calibration, thickness_mm
            )
        ],
        holes=[
            [
                tuple(p)
                for p in uncorrect_points(np.array(ring), calibration, thickness_mm)
            ]
            for ring in poly.holes
        ],
    )


def thickness_ceiling(
    cal_a: Calibration, cal_b: Calibration, t_max: float = 60.0
) -> float:
    """Upper bound the two-view solve is allowed to reach.

    Capped both by an absolute ceiling (tools thicker than this want a
    higher shot) and by 0.45× the nearer camera height (the correction
    diverges as thickness approaches the camera height).
    """
    return min(t_max, 0.45 * min(cal_a.camera_height_mm, cal_b.camera_height_mm))


def solve_thickness(
    poly_a: Poly, cal_a: Calibration,
    poly_b: Poly, cal_b: Calibration,
    t_max: float = 60.0,
) -> tuple[float, float]:
    """Recover tool thickness from two calibrated views of the same tool.

    Both photos see the same physical outline, projected onto the paper from
    different camera positions — so their plane-mapped silhouettes disagree,
    and the disagreement vanishes only at the true silhouette height.  We
    scan t for the minimum symmetric-difference area between the two
    corrected outlines (coarse grid + parabolic refinement).

    Returns (thickness_mm, residual_mismatch_mm2).  Requires the tool not to
    move between shots and the views to differ (height and/or nadir).
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    def as_shapely(poly: Poly) -> ShapelyPolygon:
        return ShapelyPolygon(poly.exterior, poly.holes).buffer(0)

    t_hi = thickness_ceiling(cal_a, cal_b, t_max)

    def mismatch(t: float) -> float:
        a = as_shapely(correct_polygon(poly_a, cal_a, t))
        b = as_shapely(correct_polygon(poly_b, cal_b, t))
        return a.symmetric_difference(b).area

    ts = np.arange(0.0, t_hi + 0.25, 0.5)
    costs = np.array([mismatch(t) for t in ts])
    i = int(np.argmin(costs))

    # parabolic refinement around the grid minimum
    if 0 < i < len(ts) - 1:
        c0, c1, c2 = costs[i - 1], costs[i], costs[i + 1]
        denom = c0 - 2 * c1 + c2
        if denom > 1e-12:
            t_best = float(ts[i] + 0.5 * 0.5 * (c0 - c2) / denom)
        else:
            t_best = float(ts[i])
    else:
        t_best = float(ts[i])
    t_best = float(np.clip(t_best, 0.0, t_hi))
    return t_best, mismatch(t_best)


def _sample_closed_ring(points: list[tuple[float, float]], spacing_mm: float) -> np.ndarray:
    """Arc-length sample a closed ring without duplicating its first point."""
    from shapely.geometry import LineString

    ring = np.asarray(points, dtype=np.float64)
    if len(ring) < 3:
        raise LocalReconstructionError("outline has fewer than three points")
    line = LineString(np.vstack([ring, ring[:1]]))
    count = max(int(np.ceil(line.length / spacing_mm)), 16)
    distances = np.linspace(0.0, line.length, count, endpoint=False)
    return np.asarray(
        [[p.x, p.y] for p in (line.interpolate(d) for d in distances)],
        dtype=np.float64,
    )


def _major_minor_extent(poly: Poly) -> tuple[float, float]:
    from shapely.geometry import Polygon as ShapelyPolygon

    minx, miny, maxx, maxy = ShapelyPolygon(poly.exterior, poly.holes).bounds
    extents = sorted((maxx - minx, maxy - miny), reverse=True)
    return float(extents[0]), float(extents[1])


def reconstruct_footprint(
    poly_a: Poly,
    cal_a: Calibration,
    poly_b: Poly,
    cal_b: Calibration,
    *,
    scalar_height_mm: float | None = None,
    scalar_residual_mm2: float | None = None,
    boundary_spacing_mm: float = 0.5,
    support_spacing_mm: float = 0.2,
    height_step_mm: float = 0.25,
    smoothness_weight: float = 0.10,
    scalar_prior_weight: float = 0.02,
    max_boundary_p95_mm: float = 2.0,
) -> FootprintReconstruction:
    """Recover a locally height-corrected footprint from two silhouettes.

    A single thickness cannot describe a screwdriver: the handle boundary is
    high above the mat while the blade boundary is close to it. For every
    point on the accepted first-view contour, this solver tests candidate
    heights by projecting the resulting physical point into the second view.
    Dynamic programming selects the boundary-height sequence that best follows
    the second silhouette while remaining smooth and near the robust scalar
    solution. The first view stays authoritative, which preserves explicit web
    edits; the second view supplies geometric evidence rather than a new mask.

    The returned scalar height is deliberately retained for pocket-depth
    sizing. Only ``polygon`` should replace the old scalar-corrected footprint.
    """
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon as ShapelyPolygon

    if scalar_height_mm is None or scalar_residual_mm2 is None:
        solved_height, solved_residual = solve_thickness(
            poly_a, cal_a, poly_b, cal_b
        )
        if scalar_height_mm is None:
            scalar_height_mm = solved_height
        if scalar_residual_mm2 is None:
            scalar_residual_mm2 = solved_residual
    scalar_height_mm = float(scalar_height_mm)
    scalar_residual_mm2 = float(scalar_residual_mm2)

    primary = _sample_closed_ring(poly_a.exterior, boundary_spacing_mm)
    support = _sample_closed_ring(poly_b.exterior, support_spacing_mm)
    if len(primary) < 16 or len(support) < 16:
        raise LocalReconstructionError("two-view outlines are too small to reconstruct")

    height_limit = min(
        thickness_ceiling(cal_a, cal_b),
        # A long low shaft can dominate the scalar solve even when its compact
        # handle is much taller. Keep enough independent search range for that
        # exact mixed-height case.
        max(30.0, 2.0 * scalar_height_mm),
    )
    heights = np.arange(0.0, height_limit + 0.5 * height_step_mm, height_step_mm)
    if len(heights) < 3:
        raise LocalReconstructionError("two-view height search has no usable range")

    nadir_a = np.asarray(cal_a.nadir_xy_mm, dtype=np.float64)
    nadir_b = np.asarray(cal_b.nadir_xy_mm, dtype=np.float64)
    height_a = float(cal_a.camera_height_mm)
    height_b = float(cal_b.camera_height_mm)
    h = heights[None, :, None]
    # Candidate physical points from the accepted first-view silhouette.
    physical = nadir_a + (1.0 - h / height_a) * (
        primary[:, None, :] - nadir_a
    )
    # Where each candidate would land when mapped through the second view's
    # plane homography. Correct heights lie on the second silhouette boundary.
    predicted_b = nadir_b + (physical - nadir_b) / (1.0 - h / height_b)
    distances, _ = cKDTree(support).query(predicted_b.reshape(-1, 2), k=1)
    boundary_cost = distances.reshape(len(primary), len(heights)) ** 2
    prior_cost = scalar_prior_weight * (heights - scalar_height_mm) ** 2

    # First-order total variation keeps a continuous tool surface while still
    # allowing the shaft/handle transition. This is small enough to run during
    # an interactive Generate action (typically hundreds of points/states).
    transition = smoothness_weight * np.abs(heights[:, None] - heights[None, :])
    back = np.zeros((len(primary), len(heights)), dtype=np.int16)
    cost = boundary_cost[0] + prior_cost
    for i in range(1, len(primary)):
        candidates = cost[:, None] + transition
        previous = np.argmin(candidates, axis=0)
        back[i] = previous
        cost = boundary_cost[i] + prior_cost + candidates[previous, np.arange(len(heights))]

    # Prefer a smooth seam at the arbitrary first/last contour split. The DP
    # is open-chain, so inspect each possible end and include its seam penalty.
    best_total = np.inf
    best_path: np.ndarray | None = None
    for end in range(len(heights)):
        path = np.empty(len(primary), dtype=np.int16)
        path[-1] = end
        for i in range(len(primary) - 1, 0, -1):
            path[i - 1] = back[i, path[i]]
        total = cost[end] + transition[path[-1], path[0]]
        if total < best_total:
            best_total = float(total)
            best_path = path
    assert best_path is not None

    selected_heights = heights[best_path]
    selected = physical[np.arange(len(primary)), best_path]
    selected_errors = np.sqrt(
        boundary_cost[np.arange(len(primary)), best_path]
    )
    mean_error = float(np.mean(selected_errors))
    p95_error = float(np.percentile(selected_errors, 95))
    if not np.isfinite(p95_error) or p95_error > max_boundary_p95_mm:
        raise LocalReconstructionError(
            f"second-view boundary support is weak (p95 {p95_error:.2f}mm)"
        )

    # Exterior uses local heights; holes remain on the proven scalar model
    # until multi-height interior-ring evidence is validated separately.
    local = Poly(
        exterior=[(float(x), float(y)) for x, y in selected],
        holes=correct_polygon(poly_a, cal_a, scalar_height_mm).holes,
    )
    from . import contour as contour_mod

    local = contour_mod.clean(local, simplify_tol=0.15)
    scalar = correct_polygon(poly_a, cal_a, scalar_height_mm)
    local_shape = ShapelyPolygon(local.exterior, local.holes).buffer(0)
    scalar_shape = ShapelyPolygon(scalar.exterior, scalar.holes).buffer(0)
    if local_shape.is_empty or not local_shape.is_valid:
        raise LocalReconstructionError("local two-view footprint is invalid")
    area_ratio = local_shape.area / scalar_shape.area if scalar_shape.area else 0.0
    if not 0.65 <= area_ratio <= 1.35:
        raise LocalReconstructionError(
            f"local footprint area changed implausibly ({area_ratio:.2f}x scalar)"
        )

    scalar_major, _ = _major_minor_extent(scalar)
    local_major, local_minor = _major_minor_extent(local)
    return FootprintReconstruction(
        polygon=local,
        method="two_view_local_silhouette",
        scalar_height_mm=scalar_height_mm,
        scalar_residual_mm2=scalar_residual_mm2,
        boundary_mean_error_mm=mean_error,
        boundary_p95_error_mm=p95_error,
        height_p05_mm=float(np.percentile(selected_heights, 5)),
        height_median_mm=float(np.median(selected_heights)),
        height_p95_mm=float(np.percentile(selected_heights, 95)),
        scalar_major_extent_mm=scalar_major,
        reconstructed_major_extent_mm=local_major,
        reconstructed_minor_extent_mm=local_minor,
        area_change_from_scalar_pct=100.0 * (area_ratio - 1.0),
    )
