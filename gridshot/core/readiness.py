"""Shared print-readiness classification for every GridShot workflow.

Readiness is deliberately deterministic and model-agnostic. Inference warnings and
geometry measurements become a small pass/review/block contract that the CLI, API,
web UI, batch path, and saved library can all enforce consistently.
"""

from __future__ import annotations

import math
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field
from shapely.geometry import Polygon as ShapelyPolygon

from . import calibrate as calibrate_mod
from .models import Calibration, CaptureSignature, Poly

MAX_UNCALIBRATED_THICKNESS_MM = 2.0

ReadinessStatus = Literal["pass", "review", "block"]
ReadinessSource = Literal[
    "calibration",
    "segmentation",
    "cleanup",
    "outline",
    "thickness",
    "printer",
    "generation",
    "provenance",
]


class ReadinessCheck(BaseModel):
    code: str
    status: ReadinessStatus
    source: ReadinessSource
    message: str
    confidence: float | None = None


class ReadinessReport(BaseModel):
    status: ReadinessStatus = "pass"
    checks: list[ReadinessCheck] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.status == "block"

    @property
    def review_required(self) -> bool:
        return self.status == "review"


class ArtifactProvenance(BaseModel):
    flow: Literal["single", "batch", "legacy"] = "legacy"
    mat_id: Optional[str] = None
    device_profile_id: Optional[str] = None
    device_profile_revision: Optional[int] = None
    intrinsics_source: Optional[str] = None
    capture_signature: Optional[CaptureSignature] = None
    thickness_source: Literal["manual", "automatic", "legacy", "unknown"] = "unknown"
    source_images: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _status(checks: list[ReadinessCheck]) -> ReadinessStatus:
    if any(check.status == "block" for check in checks):
        return "block"
    if any(check.status == "review" for check in checks):
        return "review"
    return "pass"


def report(
    checks: list[ReadinessCheck],
    metrics: dict[str, float] | None = None,
) -> ReadinessReport:
    return ReadinessReport(
        status=_status(checks),
        checks=checks,
        metrics=metrics or {},
    )


def combine(*reports: ReadinessReport) -> ReadinessReport:
    """Combine reports while removing identical checks and retaining evidence."""
    checks: list[ReadinessCheck] = []
    metrics: dict[str, float] = {}
    seen: set[tuple[str, str, str]] = set()
    for value in reports:
        metrics.update(value.metrics)
        for check in value.checks:
            key = (check.code, check.status, check.message)
            if key not in seen:
                seen.add(key)
                checks.append(check)
    return report(checks, metrics)


def _capture_warning_checks(warnings: list[str]) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    informational = (
        "auto thickness:",
        "local footprint:",
        "physical cutout override:",
        "pocket depth defaulted",
    )
    already_structured = (
        "only ",
        "reprojection rms",
        "camera tilt",
        "solvepnp failed",
        "no intrinsics or exif",
        "no device profile",
        "pose uses exif-estimated intrinsics",
    )
    for index, warning in enumerate(dict.fromkeys(warnings)):
        lowered = warning.lower()
        if lowered.startswith((
            "auto smoothing radius",
            "accepted photo outline: bounded cleanup",
        )):
            checks.append(ReadinessCheck(
                code="cleanup.bounded",
                status="pass",
                source="cleanup",
                message=warning,
            ))
            continue
        if lowered.startswith((
            "auto smoothing skipped",
            "accepted photo outline: raw segmentation",
        )):
            checks.append(ReadinessCheck(
                code="cleanup.raw",
                status="review",
                source="cleanup",
                message=warning,
            ))
            continue
        confidence_match = re.search(r"confidence\s+([0-9]+(?:\.[0-9]+)?)", lowered)
        if confidence_match is None:
            confidence_match = re.search(r"\(sam\s+([0-9]+(?:\.[0-9]+)?)\)", lowered)
        if confidence_match is not None:
            confidence = float(confidence_match.group(1))
            checks.append(ReadinessCheck(
                code=f"segmentation.confidence.{index}",
                status="pass" if confidence >= 0.8 else "review",
                source="segmentation",
                message=warning,
                confidence=confidence,
            ))
            continue
        if lowered.startswith(informational) or lowered.startswith(already_structured):
            continue
        source: ReadinessSource = "segmentation"
        if "printer shrink" in lowered:
            source = "printer"
        elif any(word in lowered for word in ("overall height", "pocket", "mesh", "bin")):
            source = "generation"
        elif "corner" in lowered or "pose" in lowered or "intrinsic" in lowered:
            source = "calibration"
        checks.append(ReadinessCheck(
            code=f"{source}.warning.{index}",
            status="review",
            source=source,
            message=warning,
        ))
    return checks


def evaluate(
    *,
    calibration: Calibration | None,
    warnings: list[str] | None = None,
    outline: Poly | None = None,
    thickness_mm: float | None = None,
    thickness_source: str | None = None,
    require_calibration: bool = True,
    require_outline: bool = True,
    require_thickness: bool = True,
) -> ReadinessReport:
    """Classify all currently available evidence for one physical tool."""
    checks: list[ReadinessCheck] = []
    metrics: dict[str, float] = {}
    warnings = warnings or []

    if calibration is None:
        checks.append(ReadinessCheck(
            code="calibration.missing",
            status="block" if require_calibration else "review",
            source="calibration" if require_calibration else "provenance",
            message=(
                "calibration is missing"
                if require_calibration
                else "legacy tool has no capture calibration provenance"
            ),
        ))
    else:
        metrics["calibration.corners"] = float(calibration.n_corners)
        if math.isfinite(calibration.reproj_rms_px):
            metrics["calibration.reprojection_rms_px"] = float(calibration.reproj_rms_px)
        if (
            calibration.camera_height_mm is not None
            and math.isfinite(calibration.camera_height_mm)
        ):
            metrics["calibration.camera_height_mm"] = float(calibration.camera_height_mm)
        if calibration.tilt_deg is not None and math.isfinite(calibration.tilt_deg):
            metrics["calibration.tilt_deg"] = float(calibration.tilt_deg)
        calibration_blocked = False
        if calibration.n_corners < 8:
            calibration_blocked = True
            checks.append(ReadinessCheck(
                code="calibration.corners",
                status="block",
                source="calibration",
                message=(
                    f"only {calibration.n_corners} calibration corners detected; "
                    "at least 8 are required for a constrained mapping"
                ),
            ))
        elif calibration.n_corners < calibrate_mod.MIN_CORNERS:
            checks.append(ReadinessCheck(
                code="calibration.corners",
                status="review",
                source="calibration",
                message=(
                    f"only {calibration.n_corners} calibration corners detected; "
                    f"{calibrate_mod.MIN_CORNERS} or more are recommended"
                ),
            ))
        reprojection_rms = calibration.reproj_rms_px
        if not math.isfinite(reprojection_rms):
            calibration_blocked = True
            checks.append(ReadinessCheck(
                code="calibration.reprojection",
                status="block",
                source="calibration",
                message="reprojection RMS is not finite",
            ))
        elif reprojection_rms >= calibrate_mod.BLOCK_RMS_PX:
            calibration_blocked = True
            checks.append(ReadinessCheck(
                code="calibration.reprojection",
                status="block",
                source="calibration",
                message=(
                    f"reprojection RMS {reprojection_rms:.2f}px reaches the "
                    f"{calibrate_mod.BLOCK_RMS_PX:.2f}px hard limit"
                ),
            ))
        elif reprojection_rms > calibrate_mod.MAX_RMS_PX:
            checks.append(ReadinessCheck(
                code="calibration.reprojection",
                status="review",
                source="calibration",
                message=(
                    f"reprojection RMS {reprojection_rms:.2f}px exceeds the "
                    f"{calibrate_mod.MAX_RMS_PX:.2f}px target — inspect "
                    "calibration before printing"
                ),
            ))
        if calibration.camera_height_mm is None or calibration.nadir_xy_mm is None:
            calibration_blocked = True
            checks.append(ReadinessCheck(
                code="calibration.pose",
                status="block",
                source="calibration",
                message="camera pose is unavailable, so parallax cannot be corrected",
            ))
        if calibration.tilt_deg is None:
            if not any(check.code == "calibration.pose" for check in checks):
                calibration_blocked = True
                checks.append(ReadinessCheck(
                    code="calibration.tilt",
                    status="block",
                    source="calibration",
                    message="camera tilt could not be measured",
                ))
        elif calibration.tilt_deg > calibrate_mod.MAX_TILT_DEG:
            calibration_blocked = True
            checks.append(ReadinessCheck(
                code="calibration.tilt",
                status="block",
                source="calibration",
                message=(
                    f"camera tilt {calibration.tilt_deg:.1f}° exceeds "
                    f"{calibrate_mod.MAX_TILT_DEG:.1f}°"
                ),
            ))
        if not calibration_blocked:
            checks.append(ReadinessCheck(
                code="calibration.geometry",
                status="pass",
                source="calibration",
                message="calibration geometry passed",
            ))
        intrinsics_source = calibration.intrinsics_source or (
            "profile" if calibration.device_profile_id else "generic"
        )
        calibrated_intrinsics = (
            intrinsics_source == "profile"
            and calibration.device_profile_id is not None
        )
        thick_tool = (
            thickness_mm is not None
            and math.isfinite(thickness_mm)
            and thickness_mm > MAX_UNCALIBRATED_THICKNESS_MM
        )
        if calibrated_intrinsics:
            revision = (
                f" revision {calibration.device_profile_revision}"
                if calibration.device_profile_revision is not None
                else ""
            )
            checks.append(ReadinessCheck(
                code="calibration.intrinsics",
                status="pass",
                source="calibration",
                message=(
                    f"using device profile {calibration.device_profile_id}"
                    f"{revision}"
                ),
            ))
        else:
            estimate_label = {
                "exif": "EXIF-estimated",
                "generic": "generic estimated",
                "provided": "unversioned provided",
            }.get(intrinsics_source, "estimated")
            checks.append(ReadinessCheck(
                code="calibration.intrinsics",
                status="block" if thick_tool else "review",
                source="calibration",
                message=(
                    f"{estimate_label} camera intrinsics cannot safely correct "
                    f"a {thickness_mm:.1f}mm thick tool; calibrate this device, "
                    "lens, orientation, and zoom"
                    if thick_tool
                    else (
                        f"camera intrinsics are {estimate_label}; calibrate this "
                        "capture setup for distortion-corrected thick-tool traces"
                    )
                ),
            ))

    if require_outline or outline is not None:
        valid = False
        if outline is not None and len(outline.exterior) >= 3:
            try:
                shape = ShapelyPolygon(outline.exterior, outline.holes)
                valid = shape.is_valid and not shape.is_empty and shape.area > 0
            except (TypeError, ValueError):
                valid = False
        if valid:
            metrics["outline.area_mm2"] = float(shape.area)
        checks.append(ReadinessCheck(
            code="outline.geometry",
            status="pass" if valid else "block",
            source="outline",
            message=(
                "outline geometry passed"
                if valid
                else "tool outline is empty or geometrically invalid"
            ),
        ))

    if require_thickness or thickness_mm is not None:
        valid_thickness = (
            thickness_mm is not None
            and math.isfinite(thickness_mm)
            and thickness_mm > 0
        )
        source = thickness_source or "unknown"
        if valid_thickness:
            metrics["thickness.mm"] = float(thickness_mm)
        checks.append(ReadinessCheck(
            code="thickness.value",
            status="pass" if valid_thickness else "block",
            source="thickness",
            message=(
                f"positive {source} thickness recorded"
                if valid_thickness
                else "a positive tool thickness is required"
            ),
        ))

    checks.extend(_capture_warning_checks(warnings))
    for check in checks:
        if check.confidence is not None:
            metrics[check.code] = check.confidence
    return report(checks, metrics)


def blocking_message(value: ReadinessReport) -> str:
    messages = [
        check.message for check in value.checks if check.status == "block"
    ]
    return "; ".join(messages) or "artifact is not ready"
