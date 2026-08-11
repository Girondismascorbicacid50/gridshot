"""ChArUco detection → plane homography + camera pose.

Everything accuracy-critical flows through here.  The contract:

- Object points are the board's chessboard corners in millimetres, multiplied
  per-axis by the mat profile's caliper-measured print scale — so "mat mm"
  means real physical millimetres on the sheet, not nominal ones.
- The homography maps image pixels → mat-plane mm (z = 0).  It is fit on all
  detected corners (dozens), which lets it absorb mild lens distortion when no
  distortion model is available yet.
- solvePnP gives the camera pose used later for parallax correction; it needs
  intrinsics, which fall back to an EXIF-derived estimate until a device
  profile exists (M2).  Pose quality gates are reported as warnings, never
  silently ignored.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .models import Calibration, CaptureSignature, MatProfile, MatSpec
from . import mat as mat_mod

MIN_CORNERS = 20
MAX_RMS_PX = 1.5  # pass target; marginal per-capture misses require review
BLOCK_RMS_PX = 3.0  # unusable mapping; twice the target is a hard stop

MAX_TILT_DEG = 15.0


class UnverifiedMatError(RuntimeError):
    """Raised when calibrating against a mat whose print scale was never measured."""


class DetectionError(RuntimeError):
    """Raised when the board cannot be found in the image."""


def detect_corners(
    image: np.ndarray, spec: MatSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Detect ChArUco chessboard corners.

    Returns (obj_pts Nx3 float32 in nominal board mm, img_pts Nx2 float32 px).

    Object points use the *mat frame*: origin at the board corner that appears
    top-left when the printed sheet is viewed (and photographed) from the
    front, x right, y down — matching image convention, so traced outlines are
    never mirrored.  OpenCV 5's ChArUco round trip (generateImage →
    matchImagePoints) already hands back object points in this orientation;
    the unit test `test_object_y_matches_image_y` pins that empirically so a
    future OpenCV convention change fails loudly instead of mirroring parts.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    board = mat_mod.build_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) == 0:
        raise DetectionError("no ChArUco corners detected")
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    return obj_pts.reshape(-1, 3), img_pts.reshape(-1, 2)


def estimate_K(
    width: int, height: int, exif: dict | None = None
) -> tuple[np.ndarray, bool]:
    """Camera matrix from EXIF 35mm-equivalent focal length, or a generic guess.

    Returns (K, from_exif).  36 mm is the full-frame sensor width; the
    equivalent focal length scales against the long image side.
    """
    exif = exif or {}
    long_side = max(width, height)
    if exif.get("focal_35mm"):
        f_px = long_side * float(exif["focal_35mm"]) / 36.0
        from_exif = True
    else:
        f_px = 1.2 * long_side  # ~40mm-equivalent guess
        from_exif = False
    K = np.array(
        [[f_px, 0, width / 2.0], [0, f_px, height / 2.0], [0, 0, 1]], dtype=np.float64
    )
    return K, from_exif


def _pose_stats(
    obj_xy: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> tuple[float, tuple[float, float], float]:
    """(camera height above mat plane, nadir xy on the mat, tilt in degrees).

    Mat frame is x right, y down, z=0 on the paper — a right-handed frame with
    +z pointing *into* the table.  A camera above the table therefore sits at
    negative z and a nadir shot looks along +z, so tilt is the angle between
    the optical axis (in mat frame) and +z.
    """
    R, _ = cv2.Rodrigues(rvec)
    cam_center = (-R.T @ tvec).ravel()  # camera position in mat frame
    height = float(abs(cam_center[2]))
    nadir = (float(cam_center[0]), float(cam_center[1]))
    view_dir = R.T @ np.array([0.0, 0.0, 1.0])
    cos_tilt = float(np.clip(view_dir[2], -1.0, 1.0))
    tilt = math.degrees(math.acos(cos_tilt))
    return height, nadir, tilt


def calibrate_image(
    image: np.ndarray,
    profile: MatProfile,
    K: np.ndarray | None = None,
    dist: np.ndarray | None = None,
    exif: dict | None = None,
    allow_unverified: bool = False,
    device_profile_id: str | None = None,
    device_profile_revision: int | None = None,
    capture_signature: CaptureSignature | None = None,
    intrinsics_source: str | None = None,
) -> Calibration:
    """Recover plane homography and camera pose from one photo of the mat."""
    if not profile.verified and not allow_unverified:
        raise UnverifiedMatError(
            f"mat '{profile.mat_id}' has no measured print scale; run "
            f"`gridshot mat verify {profile.mat_id} --measured-x ... --measured-y ...` first"
        )

    obj_pts, img_pts = detect_corners(image, profile.spec)
    n = len(obj_pts)
    warnings: list[str] = []
    if n < MIN_CORNERS:
        warnings.append(f"only {n} corners detected (< {MIN_CORNERS}) — result unreliable")

    # apply caliper-measured print scale: nominal board mm → physical mm
    scaled = obj_pts.astype(np.float64).copy()
    scaled[:, 0] *= profile.scale_x
    scaled[:, 1] *= profile.scale_y

    height, width = image.shape[:2]
    if K is None:
        K, from_exif = estimate_K(width, height, exif)
        intrinsics_source = "exif" if from_exif else "generic"
        if from_exif:
            warnings.append(
                "pose uses EXIF-estimated intrinsics; calibrate this camera "
                "for reliable thick-tool correction"
            )
        else:
            warnings.append(
                "no intrinsics or EXIF focal length — pose uses a generic guess"
            )
    elif intrinsics_source is None:
        intrinsics_source = "provided"
    dist_arr = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)

    # undistort image points when a real distortion model exists (device profile)
    pts_for_h = img_pts.astype(np.float64)
    if dist is not None:
        pts_for_h = cv2.undistortImagePoints(
            img_pts.reshape(-1, 1, 2).astype(np.float64), K, dist_arr
        ).reshape(-1, 2)

    H, inliers = cv2.findHomography(pts_for_h, scaled[:, :2], cv2.RANSAC, 2.0)
    if H is None:
        raise DetectionError("homography estimation failed")
    inliers = inliers.ravel().astype(bool)
    n_in = int(inliers.sum())
    if n_in < n:
        warnings.append(f"{n - n_in} corner(s) rejected as outliers")

    # refine on inliers with plain least squares
    H, _ = cv2.findHomography(pts_for_h[inliers], scaled[inliers, :2], 0)

    # reprojection RMS in *pixels*: mm → px through the inverse homography
    H_inv = np.linalg.inv(H)
    proj = cv2.perspectiveTransform(
        scaled[inliers, :2].reshape(-1, 1, 2), H_inv
    ).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - pts_for_h[inliers]) ** 2, axis=1))))
    if rms > MAX_RMS_PX:
        warnings.append(f"reprojection RMS {rms:.2f}px (> {MAX_RMS_PX}px)")

    rvec = tvec = None
    cam_height = nadir = tilt = None
    ok, rv, tv = cv2.solvePnP(
        scaled[inliers], img_pts[inliers].astype(np.float64), K, dist_arr,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if ok:
        cam_height, nadir, tilt = _pose_stats(scaled[inliers, :2], rv, tv)
        rvec, tvec = rv.ravel().tolist(), tv.ravel().tolist()
        if tilt > MAX_TILT_DEG:
            warnings.append(f"camera tilt {tilt:.1f}° (> {MAX_TILT_DEG}°) — hold the phone flatter")
    else:
        warnings.append("solvePnP failed — no camera pose (parallax correction unavailable)")

    return Calibration(
        mat_id=profile.mat_id,
        device_profile_id=device_profile_id,
        device_profile_revision=device_profile_revision,
        intrinsics_source=intrinsics_source,
        capture_signature=capture_signature,
        K=K.tolist(),
        dist=dist_arr.tolist(),
        H_img_to_mm=H.tolist(),
        rvec=rvec,
        tvec=tvec,
        camera_height_mm=cam_height,
        nadir_xy_mm=nadir,
        n_corners=n,
        reproj_rms_px=rms,
        tilt_deg=tilt,
        warnings=warnings,
    )


def _fit_intrinsics(
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    obj32 = [
        np.ascontiguousarray(obj, dtype=np.float32).reshape(-1, 1, 3)
        for obj in obj_list
    ]
    img32 = [
        np.ascontiguousarray(img, dtype=np.float32).reshape(-1, 1, 2)
        for img in img_list
    ]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj32, img32, image_size, None, None
    )
    view_rms: list[float] = []
    for obj, img, rvec, tvec in zip(
        obj32, img32, rvecs, tvecs, strict=True
    ):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        residual = projected.reshape(-1, 2) - img.reshape(-1, 2)
        view_rms.append(
            float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        )
    return K, dist.ravel()[:5], float(rms), view_rms


def calibrate_intrinsics_from_points(
    obj_list: list[np.ndarray],
    img_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Camera intrinsics from per-view (object mm, image px) correspondences.

    Returns (K 3x3, dist 5-vector, reprojection RMS px). Views with too few
    points should be filtered by the caller.
    """
    K, dist, rms, _ = _fit_intrinsics(obj_list, img_list, image_size)
    return K, dist, rms


MIN_INTRINSICS_VIEWS = 8
GOOD_INTRINSICS_VIEWS = 12
MAX_INTRINSICS_RMS_PX = 1.5
MAX_INTRINSICS_REJECT_FRACTION = 0.25


def calibrate_intrinsics(
    images: list[np.ndarray],
    profile: MatProfile,
    view_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int, list[str]]:
    """Fit intrinsics and reject a bounded set of high-residual views.

    At least 75% of usable views and never fewer than eight are retained. The
    profile is not returned unless the refit meets the 1.5 px accuracy gate.
    """
    if view_names is not None and len(view_names) != len(images):
        raise ValueError("view_names must contain one label per image")

    warnings: list[str] = []
    obj_list: list[np.ndarray] = []
    img_list: list[np.ndarray] = []
    usable_names: list[str] = []
    size: tuple[int, int] | None = None

    for idx, image in enumerate(images):
        label = view_names[idx] if view_names is not None else f"view {idx + 1}"
        h, w = image.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            warnings.append(
                f"{label}: resolution {w}x{h} != "
                f"{size[0]}x{size[1]} — skipped"
            )
            continue
        try:
            obj, img = detect_corners(image, profile.spec)
        except DetectionError:
            warnings.append(f"{label}: board not detected — skipped")
            continue
        if len(obj) < 10:
            warnings.append(
                f"{label}: only {len(obj)} corners — skipped"
            )
            continue
        scaled = obj.astype(np.float64).copy()
        scaled[:, 0] *= profile.scale_x
        scaled[:, 1] *= profile.scale_y
        obj_list.append(scaled)
        img_list.append(img.astype(np.float64))
        usable_names.append(label)

    if size is None or len(obj_list) < MIN_INTRINSICS_VIEWS:
        raise DetectionError(
            f"only {len(obj_list)} usable views (need ≥{MIN_INTRINSICS_VIEWS}); "
            "shoot the mat from more angles"
        )

    initial_count = len(obj_list)
    minimum_retained = max(
        MIN_INTRINSICS_VIEWS,
        math.ceil(initial_count * (1.0 - MAX_INTRINSICS_REJECT_FRACTION)),
    )
    active = list(range(initial_count))
    rejected: list[str] = []

    while True:
        K, dist, rms, view_rms = _fit_intrinsics(
            [obj_list[index] for index in active],
            [img_list[index] for index in active],
            size,
        )
        if rms <= MAX_INTRINSICS_RMS_PX or len(active) <= minimum_retained:
            break
        worst_local = int(np.argmax(view_rms))
        rejected.append(usable_names[active[worst_local]])
        active.pop(worst_local)

    if rejected:
        warnings.append(
            f"rejected {len(rejected)} high-residual view(s): "
            + ", ".join(rejected)
        )
    if rms > MAX_INTRINSICS_RMS_PX:
        raise DetectionError(
            f"intrinsics reprojection RMS {rms:.2f}px remains above "
            f"{MAX_INTRINSICS_RMS_PX:.2f}px after bounded outlier rejection; "
            "remove blurry or extreme views and recalibrate"
        )
    if len(active) < GOOD_INTRINSICS_VIEWS:
        warnings.append(
            f"{len(active)} views is workable but {GOOD_INTRINSICS_VIEWS}+ "
            "varied retained views gives a more stable distortion model"
        )

    return K, dist, rms, len(active), warnings


def image_to_mm(points_px: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Map Nx2 image pixels to mat-plane millimetres (z = 0)."""
    H = np.asarray(calibration.H_img_to_mm)
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    dist = np.asarray(calibration.dist)
    if dist.any():
        K = np.asarray(calibration.K)
        pts = cv2.undistortImagePoints(pts, K, dist)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
