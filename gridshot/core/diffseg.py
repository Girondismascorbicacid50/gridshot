"""Empty-mat differencing: locate the tool, so SAM can outline it.

Saliency models fail on the ChArUco background (they outline markers and
tape, not tools).  Instead: both the empty-mat reference photo and the tool
photo are rectified into the same canonical mat frame via their own
calibrations, and their difference marks where something new sits on the
mat.  The diff only needs to be roughly right — its interior points become
SAM prompts, and SAM produces the precise boundary on the original photo.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import Calibration, MatSpec

CANONICAL_SCALE = 4.0  # px per mm in the canonical mat frame
CANONICAL_MARGIN_MM = 40.0  # tools may overhang the board — cover the surround
DIFF_THRESHOLD = 40  # 0-255 per-channel max abs difference
MIN_COMPONENT_MM2 = 300.0
MAX_PROMPTS_PER_COMPONENT = 3


def _canonical_matrix(calibration: Calibration, scale: float) -> np.ndarray:
    """canonical px = scale · (mat mm + margin); the plane extends beyond the
    board, so the homography extrapolates over the margin exactly."""
    H = np.asarray(calibration.H_img_to_mm)
    m = CANONICAL_MARGIN_MM
    A = np.array([[scale, 0, scale * m], [0, scale, scale * m], [0, 0, 1.0]])
    return A @ H


def canonical_warp(
    pixels: np.ndarray, calibration: Calibration, spec: MatSpec,
    scale: float = CANONICAL_SCALE,
) -> np.ndarray:
    """Rectify a photo into the mat frame: board + margin at `scale` px/mm."""
    M = _canonical_matrix(calibration, scale)
    size = (
        round((spec.board_w_mm + 2 * CANONICAL_MARGIN_MM) * scale),
        round((spec.board_h_mm + 2 * CANONICAL_MARGIN_MM) * scale),
    )
    return cv2.warpPerspective(pixels, M, size, flags=cv2.INTER_LINEAR)


def canonical_to_image_px(
    points_canonical: np.ndarray, calibration: Calibration,
    scale: float = CANONICAL_SCALE,
) -> np.ndarray:
    """Map canonical-frame px back to original photo px."""
    inv = np.linalg.inv(_canonical_matrix(calibration, scale))
    pts = np.asarray(points_canonical, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)


def diff_mask(
    canonical_tool: np.ndarray, canonical_ref: np.ndarray,
    threshold: int = DIFF_THRESHOLD,
) -> np.ndarray:
    """Binary mask (0/255, canonical frame) of what changed vs the empty mat."""
    a = cv2.GaussianBlur(canonical_tool, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(canonical_ref, (5, 5), 0).astype(np.int16)
    d = np.abs(a - b).max(axis=2).astype(np.uint8)
    _, binary = cv2.threshold(d, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return binary


def board_region_mask(
    binary: np.ndarray,
    spec: MatSpec,
    scale: float = CANONICAL_SCALE,
) -> np.ndarray:
    """Discard canonical warp margins that are not guaranteed visible.

    Different camera positions expose different parts of the 40 mm canonical
    margin; comparing those invalid regions creates large false components.
    The printed board itself is required to be visible and is the stable region
    from which SAM can recover a tool even when it overhangs the edge.
    """
    x0 = round(CANONICAL_MARGIN_MM * scale)
    y0 = round(CANONICAL_MARGIN_MM * scale)
    x1 = min(binary.shape[1], x0 + round(spec.board_w_mm * scale))
    y1 = min(binary.shape[0], y0 + round(spec.board_h_mm * scale))
    clipped = np.zeros_like(binary, dtype=np.uint8)
    clipped[y0:y1, x0:x1] = binary[y0:y1, x0:x1]
    return clipped


def largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest changed region in a binary diff mask.

    Batch matching must not see illumination changes or tape elsewhere on the
    mat. This is the same component-selection policy used to seed automatic
    segmentation, exposed here so matcher artifacts preserve that independent
    foreground cue rather than a possibly bad SAM answer.
    """
    n, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    if n < 2:
        return np.zeros_like(binary, dtype=np.uint8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels_img == i) * 255).astype(np.uint8)


def component_box(
    binary: np.ndarray, pad_px: float = 30.0
) -> tuple[float, float, float, float] | None:
    """Bounding box (canonical px) of the largest changed component, padded."""
    n, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    return (x - pad_px, y - pad_px, x + w + pad_px, y + h + pad_px)


def extreme_points(
    binary: np.ndarray, k: int = 6
) -> list[tuple[float, float]]:
    """Farthest-point sample of the largest component — reaches extremities
    (jaw tips, handle ends) that distance-transform peaks never visit."""
    n, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    if n < 2:
        return []
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.where(labels_img == i)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) > 20000:  # subsample for speed; extremes survive sampling
        pts = pts[:: len(pts) // 20000]
    sel = [int(np.argmax(pts[:, 1]))]
    for _ in range(k - 1):
        d = np.min(
            np.linalg.norm(pts[:, None, :] - pts[sel][None, :, :], axis=2), axis=1
        )
        sel.append(int(np.argmax(d)))
    # nudge extremes toward the component interior so prompts sit on the tool
    comp = (labels_img == i).astype(np.uint8)
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    out = []
    for idx in sel:
        x, y = int(pts[idx][0]), int(pts[idx][1])
        r = 12
        window = dist[max(y - r, 0) : y + r, max(x - r, 0) : x + r]
        if window.size:
            dy, dx = np.unravel_index(np.argmax(window), window.shape)
            out.append((float(max(x - r, 0) + dx), float(max(y - r, 0) + dy)))
    return out


def prompt_points(
    binary: np.ndarray, scale: float = CANONICAL_SCALE
) -> list[list[tuple[float, float]]]:
    """Interior prompt points per changed component, largest area first.

    Points sit at distance-transform peaks — deep inside the object, where a
    SAM prompt is unambiguous even if the diff boundary is sloppy.
    """
    min_area_px = MIN_COMPONENT_MM2 * scale * scale
    n, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    components: list[tuple[float, list[tuple[float, float]]]] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        comp = (labels_img == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        pts: list[tuple[float, float]] = []
        work = dist.copy()
        for _ in range(MAX_PROMPTS_PER_COMPONENT):
            _, maxval, _, maxloc = cv2.minMaxLoc(work)
            if maxval < 3.0:  # too shallow — component exhausted
                break
            pts.append((float(maxloc[0]), float(maxloc[1])))
            # suppress the neighbourhood so the next point lands elsewhere
            cv2.circle(work, maxloc, int(max(maxval * 2, 20)), 0.0, -1)
        if pts:
            components.append((float(area), pts))
    components.sort(key=lambda t: -t[0])
    return [pts for _, pts in components]
