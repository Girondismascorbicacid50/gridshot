"""Per-device (phone lens) intrinsics profile store.

A profile is keyed to the lens and resolution that produced it; calibration
auto-selects a matching profile so distortion correction applies whenever the
photo came from a calibrated device.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .models import CaptureSignature, DeviceProfile, config_dir

ProfileQuality = Literal["exact", "partial", "ambiguous", "none"]


@dataclass(frozen=True)
class ProfileSelection:
    signature: CaptureSignature
    profile: DeviceProfile | None = None
    rotation_deg: int = 0
    quality: ProfileQuality = "none"
    reason: str = ""


@dataclass
class PreparedImage:
    pixels: np.ndarray
    K: np.ndarray | None
    profile: DeviceProfile | None
    signature: CaptureSignature
    selection: ProfileSelection
    warnings: list[str] = field(default_factory=list)

    def __iter__(self):
        """Keep the historical three-value unpacking contract temporarily."""
        yield self.pixels
        yield self.K
        yield self.profile


def devices_dir() -> Path:
    d = config_dir() / "devices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "device"


def _profile_path(device_id: str) -> Path:
    if (
        not device_id
        or device_id in {".", ".."}
        or "/" in device_id
        or "\\" in device_id
    ):
        raise ValueError("invalid device profile id")
    return devices_dir() / f"{device_id}.json"


def save_profile(profile: DeviceProfile) -> Path:
    """Persist an immutable profile revision.

    A revision may be re-saved only when its complete payload is identical.
    Calibration changes always receive a new revision and device ID.
    """
    path = _profile_path(profile.device_id)
    if path.exists():
        current = DeviceProfile.model_validate_json(path.read_text())
        if current != profile:
            raise FileExistsError(
                f"device profile '{profile.device_id}' already exists; "
                "create a new revision instead of overwriting calibration data"
            )
        return path
    path.write_text(profile.model_dump_json(indent=2))
    return path


def load_profile(device_id: str) -> DeviceProfile:
    path = _profile_path(device_id)
    if not path.exists():
        raise FileNotFoundError(f"no device profile '{device_id}' in {devices_dir()}")
    return DeviceProfile.model_validate_json(path.read_text())


def list_profiles() -> list[DeviceProfile]:
    return [
        DeviceProfile.model_validate_json(p.read_text())
        for p in sorted(devices_dir().glob("*.json"))
    ]


def delete_profile(device_id: str) -> Path:
    path = _profile_path(device_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"no device profile '{device_id}' in {devices_dir()}"
        )
    path.unlink()
    return path


def delete_all_profiles() -> int:
    paths = list(devices_dir().glob("*.json"))
    for path in paths:
        path.unlink()
    return len(paths)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(str(value).strip().lower().split())
    return cleaned or None


def signature_for(src) -> CaptureSignature:
    exif = src.exif or {}
    orientation = int(
        exif.get("orientation_deg", 90 if src.height > src.width else 0)
    )
    if orientation not in {0, 90, 180, 270}:
        orientation = 0
    return CaptureSignature(
        device_make=exif.get("device_make"),
        device_model=exif.get("device_model"),
        lens_model=exif.get("lens_model"),
        image_size=(src.width, src.height),
        orientation_deg=orientation,
        mirrored=bool(exif.get("orientation_mirrored", False)),
        focal_mm=exif.get("focal_mm"),
        focal_35mm=exif.get("focal_35mm"),
        digital_zoom_ratio=exif.get("digital_zoom_ratio"),
    )


def profile_signature(profile: DeviceProfile) -> CaptureSignature:
    return CaptureSignature(
        device_make=profile.device_make,
        device_model=profile.device_model,
        lens_model=profile.lens_model,
        image_size=profile.image_size,
        orientation_deg=profile.orientation_deg,
        focal_mm=profile.focal_mm,
        focal_35mm=profile.focal_35mm,
        digital_zoom_ratio=profile.digital_zoom_ratio,
    )


def _relative_close(
    left: float | None,
    right: float | None,
    *,
    absolute: float,
    relative: float = 0.02,
) -> bool:
    if left is None or right is None:
        return True
    return abs(left - right) <= max(absolute, relative * max(abs(left), abs(right)))


def _rotation_to(
    profile: DeviceProfile, signature: CaptureSignature
) -> int | None:
    rotation = (signature.orientation_deg - profile.orientation_deg) % 360
    pw, ph = profile.image_size
    expected = (pw, ph) if rotation in {0, 180} else (ph, pw)
    return rotation if expected == signature.image_size else None


def _candidate_score(
    profile: DeviceProfile, signature: CaptureSignature
) -> tuple[int, int] | None:
    rotation = _rotation_to(profile, signature)
    if rotation is None:
        return None

    text_fields = (
        ("device_make", 8),
        ("device_model", 24),
        ("lens_model", 32),
    )
    score = 0
    known = 0
    for field_name, weight in text_fields:
        wanted = _clean(getattr(signature, field_name))
        actual = _clean(getattr(profile, field_name))
        if wanted is not None and actual is not None:
            if wanted != actual:
                return None
            score += weight
            known += 1

    numeric_fields = (
        ("focal_mm", 0.05, 8),
        ("focal_35mm", 0.5, 6),
        ("digital_zoom_ratio", 0.01, 12),
    )
    for field_name, tolerance, weight in numeric_fields:
        wanted = getattr(signature, field_name)
        actual = getattr(profile, field_name)
        if wanted is not None and actual is not None:
            if not _relative_close(wanted, actual, absolute=tolerance):
                return None
            score += weight
            known += 1
    return score, known


def _identity_key(profile: DeviceProfile) -> str:
    payload = profile_signature(profile).model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def select_profile(
    signature: CaptureSignature,
    profiles: list[DeviceProfile] | None = None,
) -> ProfileSelection:
    """Select one compatible profile deterministically, or explicitly abstain."""
    if signature.mirrored:
        return ProfileSelection(
            signature=signature,
            quality="none",
            reason="mirrored captures are not compatible with camera calibration",
        )

    candidates: list[tuple[int, int, DeviceProfile, int]] = []
    for profile in profiles if profiles is not None else list_profiles():
        scored = _candidate_score(profile, signature)
        if scored is None:
            continue
        rotation = _rotation_to(profile, signature)
        assert rotation is not None
        candidates.append((*scored, profile, rotation))

    if not candidates:
        return ProfileSelection(
            signature=signature,
            quality="none",
            reason="no calibrated profile matches this capture signature",
        )

    highest_score = max((score, known) for score, known, _, _ in candidates)
    best = [
        (profile, rotation)
        for score, known, profile, rotation in candidates
        if (score, known) == highest_score
    ]
    identities = {_identity_key(profile) for profile, _ in best}
    if len(identities) != 1:
        return ProfileSelection(
            signature=signature,
            quality="ambiguous",
            reason=(
                "multiple calibrated profiles match equally; preserve camera "
                "metadata or choose a single capture setup"
            ),
        )

    highest_revision = max(profile.revision for profile, _ in best)
    newest = [
        (profile, rotation)
        for profile, rotation in best
        if profile.revision == highest_revision
    ]
    if len(newest) != 1:
        return ProfileSelection(
            signature=signature,
            quality="ambiguous",
            reason="multiple profile files claim the same latest revision",
        )

    profile, rotation = newest[0]
    exact = all(
        _clean(getattr(signature, field_name)) is not None
        and _clean(getattr(profile, field_name)) is not None
        for field_name in ("device_model", "lens_model")
    )
    quality: ProfileQuality = "exact" if exact else "partial"
    reason = (
        "capture signature matched the calibrated device and lens"
        if exact
        else "profile matched resolution but capture metadata was incomplete"
    )
    return ProfileSelection(
        signature=signature,
        profile=profile,
        rotation_deg=rotation,
        quality=quality,
        reason=reason,
    )


def transform_camera_matrix(
    K: np.ndarray, image_size: tuple[int, int], rotation_deg: int
) -> np.ndarray:
    """Rotate intrinsics with the image, preserving the pixel-center convention."""
    K = np.asarray(K, dtype=np.float64)
    width, height = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    rotation = rotation_deg % 360
    if rotation == 0:
        return K.copy()
    if rotation == 90:
        return np.array(
            [[fy, 0.0, height - 1.0 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]]
        )
    if rotation == 180:
        return np.array(
            [
                [fx, 0.0, width - 1.0 - cx],
                [0.0, fy, height - 1.0 - cy],
                [0.0, 0.0, 1.0],
            ]
        )
    if rotation == 270:
        return np.array(
            [[fy, 0.0, cy], [0.0, fx, width - 1.0 - cx], [0.0, 0.0, 1.0]]
        )
    raise ValueError("camera matrix rotation must be 0, 90, 180, or 270 degrees")


def _rotate_clockwise(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    rotation = rotation_deg % 360
    if rotation not in {0, 90, 180, 270}:
        raise ValueError("image rotation must be 0, 90, 180, or 270 degrees")
    if rotation == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k=-(rotation // 90)))


def prepare_image(
    src, profiles: list[DeviceProfile] | None = None
) -> PreparedImage:
    """Select a profile, undistort in its native orientation, and rotate back."""
    import cv2

    signature = signature_for(src)
    selection = select_profile(signature, profiles)
    profile = selection.profile
    warnings: list[str] = []
    if profile is None:
        warnings.append(selection.reason)
        return PreparedImage(
            pixels=src.pixels,
            K=None,
            profile=None,
            signature=signature,
            selection=selection,
            warnings=warnings,
        )

    if selection.quality == "partial":
        warnings.append(selection.reason)

    profile_to_target = selection.rotation_deg
    target_to_profile = (-profile_to_target) % 360
    native_pixels = _rotate_clockwise(src.pixels, target_to_profile)
    native_K = np.asarray(profile.K, dtype=np.float64)
    dist = np.asarray(profile.dist, dtype=np.float64)
    undistorted = (
        cv2.undistort(native_pixels, native_K, dist)
        if dist.size and np.any(dist)
        else native_pixels
    )
    pixels = _rotate_clockwise(undistorted, profile_to_target)
    K = transform_camera_matrix(
        native_K, profile.image_size, profile_to_target
    )
    return PreparedImage(
        pixels=pixels,
        K=K,
        profile=profile,
        signature=signature,
        selection=selection,
        warnings=warnings,
    )


def find_profile(
    width: int, height: int, lens_model: str | None = None
) -> DeviceProfile | None:
    """Compatibility wrapper around deterministic capture-signature selection."""
    signature = CaptureSignature(
        image_size=(width, height),
        orientation_deg=90 if height > width else 0,
        lens_model=lens_model,
    )
    return select_profile(signature).profile


def same_capture_stream(
    left: CaptureSignature, right: CaptureSignature
) -> bool:
    if (
        left.image_size != right.image_size
        or left.orientation_deg != right.orientation_deg
        or left.mirrored != right.mirrored
    ):
        return False
    for field_name in ("device_make", "device_model", "lens_model"):
        if _clean(getattr(left, field_name)) != _clean(
            getattr(right, field_name)
        ):
            return False
    for field_name, tolerance in (
        ("focal_mm", 0.05),
        ("focal_35mm", 0.5),
        ("digital_zoom_ratio", 0.01),
    ):
        left_value = getattr(left, field_name)
        right_value = getattr(right, field_name)
        if (left_value is None) != (right_value is None):
            return False
        if not _relative_close(
            left_value, right_value, absolute=tolerance, relative=0.01
        ):
            return False
    return True


def calibration_signature(sources: list) -> CaptureSignature:
    if not sources:
        raise ValueError("at least one calibration image is required")
    first = signature_for(sources[0])
    if first.mirrored:
        raise ValueError("mirrored images cannot be used for calibration")
    for index, source in enumerate(sources[1:], start=2):
        candidate = signature_for(source)
        if not same_capture_stream(first, candidate):
            raise ValueError(
                f"calibration image {index} has a different capture signature; "
                "use one device, lens, resolution, orientation, and zoom"
            )
    return first


def next_revision(
    signature: CaptureSignature,
    profiles: list[DeviceProfile] | None = None,
) -> int:
    revisions = [
        profile.revision
        for profile in profiles if same_capture_stream(
            signature, profile_signature(profile)
        )
    ] if profiles is not None else [
        profile.revision
        for profile in list_profiles()
        if same_capture_stream(signature, profile_signature(profile))
    ]
    return max(revisions, default=0) + 1


def profile_id(
    signature: CaptureSignature,
    revision: int,
    name: str | None = None,
) -> str:
    label = name or signature.device_model or signature.lens_model or "camera"
    payload = json.dumps(
        signature.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    width, height = signature.image_size
    return (
        f"{slugify(label)}-{width}x{height}-o{signature.orientation_deg}"
        f"-r{revision}-{digest}"
    )
