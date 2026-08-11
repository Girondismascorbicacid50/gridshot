"""Cleanup quality metrics and boundary-noise estimation.

The guardrail against over-tuning the smoothing/straightening: instead of
eyeballing overlays, measure two things.

- `fidelity(cleaned, raw)` — how far the cleaned outline moved from the raw
  segmentation.  Cleanup should remove sub-millimetre noise, so the average
  boundary shift stays small and the max shift is bounded; a large shift means
  a real curve got flattened (over-cleaning).

- `estimate_noise_mm(poly)` — the amplitude of high-frequency boundary noise
  in the raw outline, so the smoothing scale can be *derived from the image*
  rather than hard-coded.  This is the "analysis drives the smoothing" knob.
"""

from __future__ import annotations

import numpy as np

from . import contour as contour_mod
from .contour import _resample_closed, to_shapely
from .models import Poly


CLEANUP_MAX_SHIFT_MM = 0.30
CLEANUP_MAX_MEAN_SHIFT_MM = 0.15
CLEANUP_MIN_AREA_RATIO = 0.99
CLEANUP_MAX_AREA_RATIO = 1.01


def fidelity(cleaned: Poly, raw: Poly) -> dict:
    """How much cleanup changed the outline, relative to the raw segmentation.

    Returns:
      symdiff_mm2   — area added+removed by cleanup
      mean_shift_mm — symdiff / raw perimeter ≈ average boundary displacement
      max_shift_mm  — Hausdorff distance ≈ the single largest displacement
      area_ratio    — cleaned area / raw area (≈1 means dimensions preserved)
    """
    c = to_shapely(cleaned).buffer(0)
    r = to_shapely(raw).buffer(0)
    symdiff = c.symmetric_difference(r).area
    perim = r.exterior.length
    return {
        "symdiff_mm2": float(symdiff),
        "mean_shift_mm": float(symdiff / perim) if perim else 0.0,
        "max_shift_mm": float(c.hausdorff_distance(r)),
        "area_ratio": float(c.area / r.area) if r.area else 1.0,
    }


def estimate_noise_mm(poly: Poly, smooth_scale_mm: float = 1.5) -> float:
    """Amplitude of high-frequency boundary noise in the exterior ring.

    Resample uniformly, low-pass at `smooth_scale_mm`, and take a robust high
    percentile of the residual (distance from each raw point to the smoothed
    curve).  Percentile (not max) so a few big bleed spikes don't dominate the
    everyday jitter estimate.  Returned value is what the smoothing radius /
    straighten tolerance should track.
    """
    P = _resample_closed(np.asarray(poly.exterior, dtype=np.float64), 0.2)
    n = len(P)
    if n < 32:
        return 0.0
    half = max(int(smooth_scale_mm / 0.2), 1)
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets * 0.2 / smooth_scale_mm) ** 2)
    kernel /= kernel.sum()
    idx = (np.arange(n)[:, None] + offsets[None, :]) % n
    smoothed = (P[idx] * kernel[None, :, None]).sum(axis=1)
    residual = np.linalg.norm(P - smoothed, axis=1)
    return float(np.percentile(residual, 75))


def bounded_cleanup(
    raw: Poly,
    *,
    max_shift_mm: float = CLEANUP_MAX_SHIFT_MM,
) -> tuple[Poly, dict]:
    """Build the strongest noise-adaptive cleanup that stays physically bounded.

    The flat-tool G1 budget is 0.30 mm, so automatic cleanup is never allowed to
    move any boundary point farther than that. The radius starts at three times
    the measured high-frequency noise and backs off until both the Hausdorff cap
    and conservative mean/area guards pass. Straightening is attempted first at
    each radius, but is discarded when it would reshape a real curve or tip.

    When no non-trivial candidate is safe, the raw outline is returned and
    ``available`` is false. Callers can therefore recommend cleanup without ever
    making it an irreversible or silent geometry change.
    """
    noise_mm = estimate_noise_mm(raw)
    target_radius_mm = float(np.clip(3.0 * noise_mm, 0.10, 0.60))
    factors = (1.0, 0.75, 0.50, 0.35, 0.25, 0.15)

    def acceptable(metrics: dict) -> bool:
        return (
            metrics["max_shift_mm"] <= max_shift_mm + 1e-9
            and metrics["mean_shift_mm"] <= CLEANUP_MAX_MEAN_SHIFT_MM + 1e-9
            and CLEANUP_MIN_AREA_RATIO
            <= metrics["area_ratio"]
            <= CLEANUP_MAX_AREA_RATIO
        )

    attempts = 0
    for factor in factors:
        radius_mm = target_radius_mm * factor
        smoothed = contour_mod.smooth(raw, radius_mm)
        fit_tol_mm = min(
            contour_mod.STRAIGHTEN_TOL_MM,
            max(0.10, 2.0 * noise_mm),
        )
        for straightened in (True, False):
            attempts += 1
            candidate = (
                contour_mod.straighten(smoothed, fit_tol=fit_tol_mm)
                if straightened
                else smoothed
            )
            metrics = fidelity(candidate, raw)
            if not acceptable(metrics):
                continue
            return candidate, {
                "available": True,
                "recommended": "cleaned",
                "noise_mm": noise_mm,
                "radius_mm": radius_mm,
                "straightened": straightened,
                "max_shift_cap_mm": max_shift_mm,
                "attempts": attempts,
                **metrics,
            }

    metrics = fidelity(raw, raw)
    return raw, {
        "available": False,
        "recommended": "raw",
        "noise_mm": noise_mm,
        "radius_mm": 0.0,
        "straightened": False,
        "max_shift_cap_mm": max_shift_mm,
        "attempts": attempts,
        "reason": (
            f"no automatic cleanup stayed within the {max_shift_mm:.2f}mm "
            "boundary-displacement cap"
        ),
        **metrics,
    }


def max_edge_deviation(poly: Poly, p0, p1, tol_mm: float = 2.0) -> float:
    """Max perpendicular deviation of the exterior points lying near the
    segment p0→p1 from the line through it — a straightness probe for tests:
    a genuinely straight cleaned edge should return a tiny value."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    u = b - a
    length = np.linalg.norm(u)
    u = u / length
    nrm = np.array([-u[1], u[0]])
    pts = np.asarray(poly.exterior, dtype=np.float64)
    t = (pts - a) @ u
    on_seg = (t >= 0) & (t <= length)
    near = np.abs((pts - a) @ nrm) <= tol_mm
    sel = on_seg & near
    if not sel.any():
        return 0.0
    return float(np.abs((pts[sel] - a) @ nrm).max())
