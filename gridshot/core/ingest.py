"""Photo ingest: HEIC/JPEG/PNG → RGB pixels with EXIF orientation baked in.

Orientation is applied to the pixel data immediately so every downstream
stage sees the image exactly as the camera framed it — coordinates from
segmentation and calibration then always agree.  Full resolution is kept;
only copies handed to ML models get downscaled (never the geometry path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()


@dataclass
class SourceImage:
    pixels: np.ndarray  # HxWx3 uint8, RGB, orientation applied
    path: str
    exif: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.pixels.shape[1]

    @property
    def height(self) -> int:
        return self.pixels.shape[0]


_EXIF_FOCAL = 0x920A  # FocalLength
_EXIF_FOCAL_35MM = 0xA405  # FocalLengthIn35mmFilm
_EXIF_LENS_MODEL = 0xA434
_EXIF_DIGITAL_ZOOM = 0xA404
_EXIF_MAKE = 0x010F
_EXIF_MODEL = 0x0110
_EXIF_ORIENTATION = 0x0112

_ORIENTATION: dict[int, tuple[int, bool]] = {
    1: (0, False),
    2: (0, True),
    3: (180, False),
    4: (180, True),
    5: (90, True),
    6: (90, False),
    7: (270, True),
    8: (270, False),
}


def load(path: str | Path) -> SourceImage:
    with Image.open(path) as im:
        exif_raw = im.getexif()
        raw_width, raw_height = im.size
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert("RGB"))

    exif: dict = {}
    if exif_raw:
        make = exif_raw.get(_EXIF_MAKE)
        model = exif_raw.get(_EXIF_MODEL)
        if make:
            exif["device_make"] = str(make).strip()
        if model:
            exif["device_model"] = str(model).strip()

        orientation_value = int(exif_raw.get(_EXIF_ORIENTATION, 1))
        orientation_deg, mirrored = _ORIENTATION.get(
            orientation_value, (0, False)
        )
        # Some phones rotate stored pixels instead of setting EXIF. Treat an
        # already-portrait stream as the sensor's 90-degree view.
        if orientation_value == 1 and raw_height > raw_width:
            orientation_deg = 90
        exif["orientation_exif"] = orientation_value
        exif["orientation_deg"] = orientation_deg
        exif["orientation_mirrored"] = mirrored

        ifd = exif_raw.get_ifd(0x8769)  # Exif sub-IFD
        focal = ifd.get(_EXIF_FOCAL)
        if focal is not None:
            exif["focal_mm"] = float(focal)
        focal35 = ifd.get(_EXIF_FOCAL_35MM)
        if focal35 is not None:
            exif["focal_35mm"] = float(focal35)
        lens = ifd.get(_EXIF_LENS_MODEL)
        if lens:
            exif["lens_model"] = str(lens)
        zoom = ifd.get(_EXIF_DIGITAL_ZOOM)
        if zoom is not None:
            zoom_value = float(zoom)
            exif["digital_zoom_ratio"] = (
                zoom_value if zoom_value > 0 else 1.0
            )
    else:
        exif["orientation_deg"] = 90 if raw_height > raw_width else 0
        exif["orientation_mirrored"] = False

    return SourceImage(pixels=rgb, path=str(path), exif=exif)
