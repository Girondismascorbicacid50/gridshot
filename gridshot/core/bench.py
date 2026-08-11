"""Versioned printer compensation and the repeated-baseline cavity coupon."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, Manifold, OpType

from . import gridfinity as grid_mod
from .models import PrinterProfile, PrinterSignature, config_dir

A_X, A_Y = 125.0, 25.0
B_X, B_Y = 25.0, 8.0
POCKET_DEPTH = 5.0
HEIGHT_U = 2
COUPON_GX = 4
A_CX, B_CX = -17.0, 64.0
COUPON_RECOMMENDED_REPEATS = 3

MAX_SANE_SCALE = 0.03
MIN_SANE_SCALE = -0.015
MAX_SANE_OFFSET = 1.2
MAX_REPEAT_RANGE_MM = 0.25
BLOCK_REPEAT_RANGE_MM = 0.50
MAX_SCALE_STDERR = 0.0025
MAX_OFFSET_STDERR_MM = 0.10
DEFAULT_SHRINK = 0.006


class CouponMeasurementError(ValueError):
    """Repeated measurements are too inconsistent to activate a profile."""


def default_signature() -> PrinterSignature:
    return PrinterSignature()


def _profile_id(signature: PrinterSignature) -> str:
    payload = json.dumps(
        signature.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return f"printer-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def default_profile() -> PrinterProfile:
    signature = default_signature()
    return PrinterProfile(
        profile_id="default",
        signature=signature,
        created_at="default",
        scale_x=DEFAULT_SHRINK,
        scale_y=DEFAULT_SHRINK,
        offset_mm=0.0,
        quality="default",
    )


def coupon_solid() -> Manifold:
    """4×1 coupon; print three copies for independent long-baseline repeats."""
    solid = grid_mod.bin_solid(COUPON_GX, 1, HEIGHT_U)
    z_floor = HEIGHT_U * grid_mod.UNIT_H - POCKET_DEPTH

    def pocket(w: float, d: float, cx: float) -> Manifold:
        return Manifold.extrude(
            CrossSection.square((w, d), center=True), POCKET_DEPTH + 1
        ).translate((cx, 0, z_floor))

    return solid - pocket(A_X, A_Y, A_CX) - pocket(B_X, B_Y, B_CX)


def profile_path() -> Path:
    """Legacy singleton path retained as a read-only migration source."""
    return config_dir() / "printer.json"


def profiles_dir() -> Path:
    return config_dir() / "printers"


def active_profile_path() -> Path:
    return profiles_dir() / "active.json"


def _revision_path(profile_id: str, revision: int) -> Path:
    return profiles_dir() / profile_id / f"v{revision}.json"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pending.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _publish_immutable_json(path: Path, payload: dict) -> bool:
    """Atomically create a complete revision without ever replacing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pending.write_text(json.dumps(payload, indent=2, sort_keys=True))
        try:
            os.link(pending, path)
        except FileExistsError:
            return False
        return True
    finally:
        pending.unlink(missing_ok=True)


def _revision_numbers(profile_id: str) -> list[int]:
    root = profiles_dir() / profile_id
    if not root.is_dir():
        return []
    revisions = []
    for path in root.glob("v*.json"):
        try:
            revisions.append(int(path.stem[1:]))
        except ValueError:
            continue
    return sorted(revisions)


def save_profile(profile: PrinterProfile, *, activate: bool = True) -> PrinterProfile:
    """Persist one immutable revision and optionally make it the active profile."""
    path = _revision_path(profile.profile_id, profile.revision)
    created = _publish_immutable_json(path, profile.model_dump(mode="json"))
    if not created:
        existing = PrinterProfile.model_validate_json(path.read_text())
        if existing != profile:
            raise ValueError(
                f"printer profile {profile.profile_id} revision {profile.revision} "
                "already exists with different content"
            )
    if activate:
        _atomic_json(
            active_profile_path(),
            {
                "schema_version": "printer-active.v1",
                "profile_id": profile.profile_id,
                "revision": profile.revision,
            },
        )
    return profile


def _migrate_legacy_profile() -> None:
    legacy_path = profile_path()
    if active_profile_path().exists() or not legacy_path.exists():
        return
    legacy = PrinterProfile.model_validate_json(legacy_path.read_text())
    signature = legacy.signature
    profile_id = _profile_id(signature)
    revisions = _revision_numbers(profile_id)
    migrated = legacy.model_copy(
        update={
            "profile_id": profile_id,
            "revision": revisions[-1] + 1 if revisions else 1,
            "quality": "legacy",
        }
    )
    save_profile(migrated, activate=True)


def load_profile(
    profile_id: str | None = None,
    revision: int | None = None,
    signature: PrinterSignature | None = None,
) -> PrinterProfile | None:
    """Load an exact revision, newest signature match, or the active profile."""
    _migrate_legacy_profile()
    if signature is not None:
        profile_id = _profile_id(signature)
    if profile_id is None:
        active = active_profile_path()
        if not active.exists():
            return None
        pointer = json.loads(active.read_text())
        profile_id = str(pointer["profile_id"])
        revision = int(pointer["revision"])
    if revision is None:
        revisions = _revision_numbers(profile_id)
        if not revisions:
            return None
        revision = revisions[-1]
    path = _revision_path(profile_id, revision)
    if not path.exists():
        return None
    profile = PrinterProfile.model_validate_json(path.read_text())
    if signature is not None and profile.signature != signature:
        return None
    return profile


def list_profiles(signature: PrinterSignature | None = None) -> list[PrinterProfile]:
    _migrate_legacy_profile()
    root = profiles_dir()
    if not root.is_dir():
        return []
    profiles = []
    for path in sorted(root.glob("printer-*/v*.json")):
        profile = PrinterProfile.model_validate_json(path.read_text())
        if signature is None or profile.signature == signature:
            profiles.append(profile)
    return sorted(
        profiles,
        key=lambda profile: (
            profile.signature.printer,
            profile.signature.material,
            profile.signature.nozzle_mm,
            profile.signature.process,
            profile.revision,
        ),
    )


def activate_profile(profile_id: str, revision: int | None = None) -> PrinterProfile:
    profile = load_profile(profile_id=profile_id, revision=revision)
    if profile is None:
        raise KeyError(profile_id)
    if profile.quality == "review":
        raise ValueError("review-quality printer profile cannot be activated")
    save_profile(profile, activate=True)
    return profile


def _fit_axis(
    long_nominal: float,
    short_nominal: float,
    long_values: list[float],
    short_values: list[float],
) -> tuple[float, float, float, float]:
    nominal = np.asarray(
        [long_nominal] * len(long_values) + [short_nominal] * len(short_values),
        dtype=np.float64,
    )
    measured = np.asarray(long_values + short_values, dtype=np.float64)
    design = np.column_stack((nominal, np.ones_like(nominal)))
    beta, intercept = np.linalg.lstsq(design, measured, rcond=None)[0]
    residual = measured - design @ np.asarray([beta, intercept])
    dof = max(1, len(measured) - 2)
    sigma2 = float(np.dot(residual, residual) / dof)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    scale_stderr = math.sqrt(max(0.0, float(covariance[0, 0])))
    offset_stderr = 0.5 * math.sqrt(max(0.0, float(covariance[1, 1])))
    return 1.0 - float(beta), -float(intercept) / 2.0, scale_stderr, offset_stderr


def _measurement_warnings(
    scale_x: float,
    scale_y: float,
    off_x: float,
    off_y: float,
) -> list[str]:
    warnings: list[str] = []
    if abs(off_x - off_y) > 0.15:
        warnings.append(
            f"per-side offset differs between axes (x {off_x:.2f} vs "
            f"y {off_y:.2f}mm) — check measurements"
        )
    offset = (off_x + off_y) / 2
    for name, value, low, high in (
        ("scale x", scale_x, MIN_SANE_SCALE, MAX_SANE_SCALE),
        ("scale y", scale_y, MIN_SANE_SCALE, MAX_SANE_SCALE),
        ("offset", offset, -MAX_SANE_OFFSET, MAX_SANE_OFFSET),
    ):
        if not low <= value <= high:
            warnings.append(
                f"{name} = {value:.3f} is outside the plausible range — check "
                "inside-jaw measurements at mid-depth"
            )
    return warnings


def solve_repeated(
    observations: list[dict[str, float]],
    *,
    signature: PrinterSignature | None = None,
    require_uncertainty: bool = True,
) -> tuple[PrinterProfile, list[str]]:
    """Fit all repeated long/short baselines and persist an immutable revision."""
    if not observations:
        raise CouponMeasurementError("at least one coupon observation is required")
    required = ("a_x", "a_y", "b_x", "b_y")
    normalized = []
    for index, observation in enumerate(observations, start=1):
        values = {name: float(observation[name]) for name in required}
        if not all(math.isfinite(value) and value > 0 for value in values.values()):
            raise CouponMeasurementError(
                f"coupon observation {index} must contain positive finite dimensions"
            )
        normalized.append(values)

    a_x = [row["a_x"] for row in normalized]
    a_y = [row["a_y"] for row in normalized]
    b_x = [row["b_x"] for row in normalized]
    b_y = [row["b_y"] for row in normalized]
    scale_x, off_x, scale_x_se, off_x_se = _fit_axis(A_X, B_X, a_x, b_x)
    scale_y, off_y, scale_y_se, off_y_se = _fit_axis(A_Y, B_Y, a_y, b_y)
    offset = (off_x + off_y) / 2

    repeat_ranges = {
        name: max(values) - min(values)
        for name, values in (("a_x", a_x), ("a_y", a_y), ("b_x", b_x), ("b_y", b_y))
    }
    max_range = max(repeat_ranges.values())
    if len(normalized) >= COUPON_RECOMMENDED_REPEATS and max_range > BLOCK_REPEAT_RANGE_MM:
        raise CouponMeasurementError(
            f"coupon repeats span {max_range:.3f}mm; limit is "
            f"{BLOCK_REPEAT_RANGE_MM:.3f}mm — reprint or remeasure"
        )

    warnings = _measurement_warnings(scale_x, scale_y, off_x, off_y)
    quality = "pass"
    if len(normalized) < COUPON_RECOMMENDED_REPEATS:
        quality = "legacy" if not require_uncertainty else "review"
        if require_uncertainty:
            warnings.append(
                f"record at least {COUPON_RECOMMENDED_REPEATS} independent coupon "
                "copies before activating compensation"
            )
    elif max_range > MAX_REPEAT_RANGE_MM:
        quality = "review"
        warnings.append(
            f"coupon repeat range {max_range:.3f}mm exceeds "
            f"{MAX_REPEAT_RANGE_MM:.3f}mm; profile was retained but not activated"
        )
    if max(scale_x_se, scale_y_se) > MAX_SCALE_STDERR:
        quality = "review"
        warnings.append("printer scale uncertainty is too high; profile was not activated")
    if max(off_x_se, off_y_se) > MAX_OFFSET_STDERR_MM:
        quality = "review"
        warnings.append("printer offset uncertainty is too high; profile was not activated")
    if warnings and quality == "pass":
        quality = "review"

    signature = signature or default_signature()
    profile_id = _profile_id(signature)
    revisions = _revision_numbers(profile_id)
    uncertainty = {
        "observation_count": len(normalized),
        "max_repeat_range_mm": max_range,
        "scale_x_stderr": scale_x_se,
        "scale_y_stderr": scale_y_se,
        "offset_x_stderr_mm": off_x_se,
        "offset_y_stderr_mm": off_y_se,
        **{f"{name}_range_mm": value for name, value in repeat_ranges.items()},
    }
    profile = PrinterProfile(
        profile_id=profile_id,
        revision=revisions[-1] + 1 if revisions else 1,
        signature=signature,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        scale_x=scale_x,
        scale_y=scale_y,
        offset_mm=offset,
        measurements={"observations": normalized},
        uncertainty=uncertainty,
        quality=quality,
    )
    save_profile(profile, activate=quality != "review")
    return profile, warnings


def solve(
    a_x: float, a_y: float, b_x: float, b_y: float
) -> tuple[PrinterProfile, list[str]]:
    """Compatibility fit for one coupon; new workflows should record repeats."""
    return solve_repeated(
        [{"a_x": a_x, "a_y": a_y, "b_x": b_x, "b_y": b_y}],
        require_uncertainty=False,
    )


def compensate(shape, profile: PrinterProfile):
    """Inflate a pocket so the simulated printed cavity lands on nominal."""
    from shapely.affinity import scale as shapely_scale

    out = shapely_scale(
        shape,
        xfact=1.0 / (1.0 - profile.scale_x),
        yfact=1.0 / (1.0 - profile.scale_y),
        origin="centroid",
    )
    if profile.offset_mm:
        out = out.buffer(profile.offset_mm, join_style="round", quad_segs=8)
    return out
