"""Data contracts shared by every pipeline stage and both frontends.

All linear dimensions are millimetres unless the field name says otherwise.
Image coordinates are pixels with the origin at the top-left, x right, y down.
Mat-plane coordinates are millimetres on the printed mat, z = 0 on the paper.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Vec2 = tuple[float, float]

PAPER_MM: dict[str, tuple[float, float]] = {
    "a4": (210.0, 297.0),
    "a3": (297.0, 420.0),
    "letter": (215.9, 279.4),
}


def config_dir() -> Path:
    return Path(os.environ.get("GRIDSHOT_CONFIG_DIR", "~/.gridshot")).expanduser()


class Poly(BaseModel):
    """Polygon with optional interior holes (hollow tools)."""

    exterior: list[Vec2]
    holes: list[list[Vec2]] = Field(default_factory=list)


class MatSpec(BaseModel):
    """Geometry of a printable ChArUco calibration mat.

    square_mm defaults to 25.4 so a square is exactly 600 px at 600 DPI —
    the rendered pattern then has perfectly uniform squares, which keeps the
    printed geometry identical to the object points used in calibration.
    """

    paper: Literal["a4", "a3", "letter"] = "a4"
    squares_x: int = 7
    squares_y: int = 10
    square_mm: float = 25.4
    marker_mm: float = 19.05
    dict_name: str = "DICT_5X5_100"
    dpi: int = 600
    margin_mm: float = 10.0
    footer_mm: float = 26.0

    @property
    def board_w_mm(self) -> float:
        return self.squares_x * self.square_mm

    @property
    def board_h_mm(self) -> float:
        return self.squares_y * self.square_mm

    @property
    def span_squares(self) -> int:
        """Squares in the caliper-verification span (fits 150 mm calipers)."""
        return 4

    @property
    def span_mm(self) -> float:
        return self.span_squares * self.square_mm

    def mat_id(self) -> str:
        canon = json.dumps(self.model_dump(), sort_keys=True)
        digest = hashlib.sha1(canon.encode()).hexdigest()[:6]
        return f"{self.paper}-{self.squares_x}x{self.squares_y}-{digest}"


class MatProfile(BaseModel):
    """A physical printed mat: spec + caliper-measured print-scale correction."""

    mat_id: str
    spec: MatSpec
    created_at: str
    # printed size / nominal size, per axis; 1.0 until verified
    scale_x: float = 1.0
    scale_y: float = 1.0
    measured_span_x_mm: Optional[float] = None
    measured_span_y_mm: Optional[float] = None
    verified: bool = False


class PrinterSignature(BaseModel):
    """Printing process identity; compensation never crosses this boundary."""

    printer: str = "unspecified"
    material: str = "unspecified"
    nozzle_mm: float = Field(0.4, gt=0)
    process: str = "unspecified"


class PrinterProfile(BaseModel):
    """Immutable compensation revision for one printer/material/process."""

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_profile(cls, value):
        if isinstance(value, dict) and "schema_version" not in value:
            value = dict(value)
            value.update(
                schema_version="printer.v2",
                profile_id=value.get("profile_id", "legacy"),
                revision=value.get("revision", 1),
                signature=value.get("signature", {}),
                quality=value.get("quality", "legacy"),
            )
        return value

    schema_version: Literal["printer.v2"] = "printer.v2"
    profile_id: str = "legacy"
    revision: int = Field(1, ge=1)
    signature: PrinterSignature = Field(default_factory=PrinterSignature)
    created_at: str
    scale_x: float = 0.0  # fractional shrink along x (0.004 = 0.4%)
    scale_y: float = 0.0
    offset_mm: float = 0.0  # per-SIDE fixed cavity loss
    measurements: dict = Field(default_factory=dict)
    uncertainty: dict[str, float | int] = Field(default_factory=dict)
    quality: Literal["default", "legacy", "pass", "review"] = "legacy"


class CaptureSignature(BaseModel):
    """Identity of the pixel stream whose intrinsics are being used."""

    schema_version: Literal["capture.v1"] = "capture.v1"
    device_make: Optional[str] = None
    device_model: Optional[str] = None
    lens_model: Optional[str] = None
    image_size: tuple[int, int]
    orientation_deg: Literal[0, 90, 180, 270] = 0
    mirrored: bool = False
    focal_mm: Optional[float] = None
    focal_35mm: Optional[float] = None
    digital_zoom_ratio: Optional[float] = None


class DeviceProfile(BaseModel):
    """Per-phone-lens intrinsics from a one-time calibration (M2)."""

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_orientation(cls, value):
        if isinstance(value, dict) and "orientation_deg" not in value:
            image_size = value.get("image_size")
            if image_size and len(image_size) == 2:
                value = dict(value)
                value["orientation_deg"] = (
                    90 if image_size[1] > image_size[0] else 0
                )
        return value

    schema_version: Literal["device.v2"] = "device.v2"
    device_id: str
    revision: int = Field(1, ge=1)
    created_at: str = ""
    device_make: Optional[str] = None
    device_model: Optional[str] = None
    lens_model: Optional[str] = None
    image_size: tuple[int, int]
    orientation_deg: Literal[0, 90, 180, 270] = 0
    focal_mm: Optional[float] = None
    focal_35mm: Optional[float] = None
    digital_zoom_ratio: Optional[float] = None
    mat_id: Optional[str] = None
    n_views: Optional[int] = Field(None, ge=1)
    source_images: list[str] = Field(default_factory=list)
    K: list[list[float]]
    dist: list[float]
    reproj_rms_px: float


class Calibration(BaseModel):
    """Everything recovered from the mat in one photo."""

    mat_id: str
    device_profile_id: Optional[str] = None
    device_profile_revision: Optional[int] = None
    intrinsics_source: Optional[
        Literal["profile", "provided", "exif", "generic"]
    ] = None
    capture_signature: Optional[CaptureSignature] = None
    K: list[list[float]]
    dist: list[float] = Field(default_factory=list)
    H_img_to_mm: list[list[float]]
    rvec: Optional[list[float]] = None
    tvec: Optional[list[float]] = None
    camera_height_mm: Optional[float] = None
    nadir_xy_mm: Optional[Vec2] = None
    n_corners: int
    reproj_rms_px: float
    tilt_deg: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)


class FingerHole(BaseModel):
    center: Vec2
    shape: Literal["circle", "stadium"] = "circle"
    diameter_mm: float = 20.0
    length_mm: float = 0.0  # stadium only
    angle_deg: float = 0.0  # stadium only


class Tool(BaseModel):
    id: str
    label: str = ""
    polygon_raw: Poly
    # Compatibility alias for silhouette-driving vertical height.
    thickness_mm: float = 3.0
    silhouette_height_mm: Optional[float] = None
    full_height_mm: Optional[float] = None
    thickness_source: Literal["user", "vlm", "parallax", "default"] = "default"
    polygon_corrected: Optional[Poly] = None
    clearance_mm: float = 1.0
    smoothing: float = 0.0
    pocket_depth_mm: Optional[float] = None
    finger_holes: list[FingerHole] = Field(default_factory=list)
    mask_ref: Optional[str] = None
    on_mat_fraction: float = 1.0  # share of the outline on the calibrated board


class Trace(BaseModel):
    """One calibrated photo and the tools extracted from it."""

    image_ref: str
    calibration: Calibration
    tools: list[Tool] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Bin(BaseModel):
    tool_ids: list[str]
    grid_x: float = 1.0  # gridfinity units, 0.5 steps
    grid_y: float = 1.0
    height_u: int = 3
    lip: bool = True
    magnets: bool = False
    label_text: Optional[str] = None
    mesh_ref: Optional[str] = None


class Placement(BaseModel):
    bin_id: str
    col: float  # gridfinity-unit offset from the drawer's near-left corner
    row: float
    grid_x: float  # footprint AS PLACED (already rotated if `rotated`)
    grid_y: float
    rotated: bool = False  # tool + bin turned 90° from its natural orientation


class DrawerLayout(BaseModel):
    drawer_w_mm: float
    drawer_d_mm: float
    placed: list[Placement] = Field(default_factory=list)
    overflow: list[str] = Field(default_factory=list)
    used_cols: float = 0.0  # tight bounding footprint of the packing (cells)
    used_rows: float = 0.0
