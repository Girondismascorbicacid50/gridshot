"""Lightweight image-correspondence baselines for batch matcher evaluation.

This is benchmark evidence, not the production auto-accept policy. The shipped
matcher remains fail-closed until a candidate clears the corpus precision gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PreparedFeatures:
    points: np.ndarray
    descriptors: np.ndarray | None
    mask_area: float


@dataclass
class MatchEvidence:
    tentative: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    coverage_a: float = 0.0
    coverage_b: float = 0.0
    median_sampson_px: float | None = None
    certainty_mean: float | None = None
    certainty_median: float | None = None
    score: float = 0.0
    reason: str = ""

    def model_dump(self) -> dict:
        return asdict(self)


def load_match_image(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not read matcher image: {image_path}")
    if mask is None:
        raise ValueError(f"could not read matcher mask: {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"image/mask shape mismatch: {image.shape[:2]} vs {mask.shape[:2]}"
        )
    return image, mask


def prepare_sift(
    image: np.ndarray,
    mask: np.ndarray,
    max_side: int = 1200,
    context_fraction: float = 0.02,
    nfeatures: int = 4096,
) -> PreparedFeatures:
    """Extract SIFT features inside a tight, slightly dilated foreground crop.

    A narrow context ring keeps boundary evidence when segmentation is a little
    short. Most of the ChArUco mat is excluded so marker matches cannot become
    the tool-identity signal.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR color image")
    if mask.shape != image.shape[:2]:
        raise ValueError("image and mask dimensions differ")

    foreground = mask > 127
    ys, xs = np.nonzero(foreground)
    if not xs.size:
        return PreparedFeatures(np.empty((0, 2), np.float32), None, 0.0)

    h, w = mask.shape
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    pad = max(24, int(round(0.12 * span)))
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    crop = image[y0:y1, x0:x1]
    crop_mask = (foreground[y0:y1, x0:x1].astype(np.uint8) * 255)

    scale = min(1.0, float(max_side) / max(crop.shape[:2]))
    if scale < 1.0:
        size = (
            max(1, int(round(crop.shape[1] * scale))),
            max(1, int(round(crop.shape[0] * scale))),
        )
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        crop_mask = cv2.resize(crop_mask, size, interpolation=cv2.INTER_NEAREST)

    radius = max(2, int(round(context_fraction * max(crop_mask.shape))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    allowed = cv2.dilate(crop_mask, kernel)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    detector = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=0.02,
        edgeThreshold=12,
    )
    keypoints, descriptors = detector.detectAndCompute(gray, allowed)
    points = (
        np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
        if keypoints
        else np.empty((0, 2), dtype=np.float32)
    )
    return PreparedFeatures(
        points=points,
        descriptors=descriptors,
        mask_area=float(np.count_nonzero(allowed)),
    )


def _ratio_matches(
    query: np.ndarray | None,
    train: np.ndarray | None,
    ratio: float,
) -> dict[int, tuple[int, float]]:
    if query is None or train is None or len(query) < 2 or len(train) < 2:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    accepted: dict[int, tuple[int, float]] = {}
    for candidates in matcher.knnMatch(query, train, k=2):
        if len(candidates) < 2:
            continue
        first, second = candidates
        if first.distance < ratio * second.distance:
            accepted[first.queryIdx] = (first.trainIdx, float(first.distance))
    return accepted


def _coverage(points: np.ndarray, mask_area: float) -> float:
    if len(points) < 3 or mask_area <= 0:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(np.clip(cv2.contourArea(hull) / mask_area, 0.0, 1.0))


def _sampson_error(F: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ah = np.column_stack([a, np.ones(len(a))])
    bh = np.column_stack([b, np.ones(len(b))])
    Fa = (F @ ah.T).T
    Ftb = (F.T @ bh.T).T
    numerator = np.sum(bh * Fa, axis=1) ** 2
    denominator = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def correspondence_magsac(
    points_a: np.ndarray,
    points_b: np.ndarray,
    certainty: np.ndarray | None,
    mask_area_a: float,
    mask_area_b: float,
    threshold_px: float = 2.0,
) -> MatchEvidence:
    """Score externally proposed correspondences with fitted epipolar geometry."""
    points_a = np.asarray(points_a, dtype=np.float32).reshape(-1, 2)
    points_b = np.asarray(points_b, dtype=np.float32).reshape(-1, 2)
    if len(points_a) != len(points_b):
        raise ValueError("correspondence arrays differ in length")
    if certainty is None:
        weights = np.ones(len(points_a), dtype=np.float32)
    else:
        weights = np.asarray(certainty, dtype=np.float32).reshape(-1)
        if len(weights) != len(points_a):
            raise ValueError("certainty array differs in length")
    finite = (
        np.isfinite(points_a).all(axis=1)
        & np.isfinite(points_b).all(axis=1)
        & np.isfinite(weights)
    )
    points_a, points_b, weights = points_a[finite], points_b[finite], weights[finite]
    tentative = len(points_a)
    if tentative < 8:
        return MatchEvidence(
            tentative=tentative,
            reason="fewer than 8 foreground correspondences",
        )

    F, inlier_mask = cv2.findFundamentalMat(
        points_a,
        points_b,
        cv2.USAC_MAGSAC,
        threshold_px,
        0.999,
        10_000,
    )
    if F is None or np.asarray(F).shape != (3, 3) or inlier_mask is None:
        return MatchEvidence(tentative=tentative, reason="MAGSAC fit failed")
    keep = np.asarray(inlier_mask).reshape(-1).astype(bool)
    inliers = int(keep.sum())
    if inliers == 0:
        return MatchEvidence(tentative=tentative, reason="MAGSAC found no inliers")

    pa, pb, inlier_weights = points_a[keep], points_b[keep], weights[keep]
    inlier_ratio = inliers / tentative
    coverage_a = _coverage(pa, mask_area_a)
    coverage_b = _coverage(pb, mask_area_b)
    errors = _sampson_error(np.asarray(F, dtype=np.float64), pa, pb)
    median_error = float(np.median(errors))
    certainty_mean = float(np.mean(inlier_weights))
    certainty_median = float(np.median(inlier_weights))
    coverage = float(np.sqrt(max(coverage_a, 1e-4) * max(coverage_b, 1e-4)))
    score = float(
        inliers
        * inlier_ratio
        * coverage
        * certainty_mean
        * np.exp(-median_error / max(threshold_px, 1e-6))
    )
    return MatchEvidence(
        tentative=tentative,
        inliers=inliers,
        inlier_ratio=round(inlier_ratio, 6),
        coverage_a=round(coverage_a, 6),
        coverage_b=round(coverage_b, 6),
        median_sampson_px=round(median_error, 6),
        certainty_mean=round(certainty_mean, 6),
        certainty_median=round(certainty_median, 6),
        score=round(score, 6),
    )


def sift_magsac(
    a: PreparedFeatures,
    b: PreparedFeatures,
    ratio: float = 0.8,
    threshold_px: float = 1.5,
) -> MatchEvidence:
    """Mutual SIFT matching followed by a USAC_MAGSAC fundamental-matrix fit."""
    forward = _ratio_matches(a.descriptors, b.descriptors, ratio)
    reverse = _ratio_matches(b.descriptors, a.descriptors, ratio)
    mutual = [
        (qa, tb)
        for qa, (tb, _distance) in forward.items()
        if tb in reverse and reverse[tb][0] == qa
    ]
    tentative = len(mutual)
    if tentative < 8:
        return MatchEvidence(
            tentative=tentative,
            reason="fewer than 8 mutual ratio-test matches",
        )

    points_a = np.asarray([a.points[qa] for qa, _ in mutual], dtype=np.float32)
    points_b = np.asarray([b.points[tb] for _, tb in mutual], dtype=np.float32)
    F, inlier_mask = cv2.findFundamentalMat(
        points_a,
        points_b,
        cv2.USAC_MAGSAC,
        threshold_px,
        0.999,
        10_000,
    )
    if F is None or np.asarray(F).shape != (3, 3) or inlier_mask is None:
        return MatchEvidence(tentative=tentative, reason="MAGSAC fit failed")

    keep = np.asarray(inlier_mask).reshape(-1).astype(bool)
    inliers = int(keep.sum())
    if inliers == 0:
        return MatchEvidence(tentative=tentative, reason="MAGSAC found no inliers")

    pa = points_a[keep]
    pb = points_b[keep]
    inlier_ratio = inliers / tentative
    coverage_a = _coverage(pa, a.mask_area)
    coverage_b = _coverage(pb, b.mask_area)
    errors = _sampson_error(np.asarray(F, dtype=np.float64), pa, pb)
    median_error = float(np.median(errors))
    coverage = float(np.sqrt(max(coverage_a, 1e-4) * max(coverage_b, 1e-4)))
    score = float(
        inliers
        * inlier_ratio
        * coverage
        * np.exp(-median_error / max(threshold_px, 1e-6))
    )
    return MatchEvidence(
        tentative=tentative,
        inliers=inliers,
        inlier_ratio=round(inlier_ratio, 6),
        coverage_a=round(coverage_a, 6),
        coverage_b=round(coverage_b, 6),
        median_sampson_px=round(median_error, 6),
        score=round(score, 6),
    )
