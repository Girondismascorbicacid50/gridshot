"""Printable ChArUco calibration mat: generation, PDF export, profile store.

The mat is the accuracy foundation: a full-field ChArUco board (tools occlude
the middle; ChArUco tolerates that) plus two caliper scale bars. Home printers
scale pages by ±1–2% ("fit to page"), which exceeds every other error term in
the pipeline, so a mat profile stays unverified — and refuses to calibrate —
until the printed bars have been measured and recorded with `mat verify`.

Rendering is pixel-exact: at the default 600 DPI a 25.4 mm square is exactly
600 px, so the pattern geometry matches the calibration object points with no
rounding inside the board.
"""

from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path

import cv2
import img2pdf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import PAPER_MM, MatProfile, MatSpec, config_dir

# ---------------------------------------------------------------------------
# board construction


def get_dictionary(name: str) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def build_board(spec: MatSpec) -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_mm,
        spec.marker_mm,
        get_dictionary(spec.dict_name),
    )


def default_spec(paper: str = "a4") -> MatSpec:
    """Largest board that fits the paper with margins and a footer."""
    page_w, page_h = PAPER_MM[paper]
    spec = MatSpec(paper=paper)  # type: ignore[arg-type]
    usable_w = page_w - 2 * spec.margin_mm
    usable_h = page_h - 2 * spec.margin_mm - spec.footer_mm
    squares_x = int(usable_w // spec.square_mm)
    squares_y = int(usable_h // spec.square_mm)
    return spec.model_copy(update={"squares_x": squares_x, "squares_y": squares_y})


# ---------------------------------------------------------------------------
# page rendering


def _ppmm(spec: MatSpec) -> float:
    return spec.dpi / 25.4


def render_page(spec: MatSpec) -> Image.Image:
    """Full printable page (grayscale) at spec.dpi."""
    ppmm = _ppmm(spec)
    page_w_mm, page_h_mm = PAPER_MM[spec.paper]
    page_px = (round(page_w_mm * ppmm), round(page_h_mm * ppmm))

    board = build_board(spec)
    board_px = (round(spec.board_w_mm * ppmm), round(spec.board_h_mm * ppmm))
    board_img = board.generateImage(board_px, marginSize=0, borderBits=1)

    page = Image.new("L", page_px, 255)
    origin_px = (round(spec.margin_mm * ppmm), round(spec.margin_mm * ppmm))
    page.paste(Image.fromarray(board_img), origin_px)

    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.load_default(size=round(2.6 * ppmm))
        small = ImageFont.load_default(size=round(2.0 * ppmm))
    except TypeError:  # very old Pillow
        font = small = ImageFont.load_default()

    span_px = round(spec.span_mm * ppmm)
    bar_thick = round(5.0 * ppmm)

    # X scale bar: solid black bar in the footer, exactly span_mm long
    x0 = origin_px[0]
    y_bar = round((PAPER_MM[spec.paper][1] - spec.footer_mm + 4.0) * ppmm)
    draw.rectangle([x0, y_bar, x0 + span_px - 1, y_bar + bar_thick], fill=0)

    # Y scale bar: solid black bar to the right of the board, exactly span_mm tall
    x_bar = origin_px[0] + round(spec.board_w_mm * ppmm) + round(4.0 * ppmm)
    y0 = origin_px[1]
    draw.rectangle([x_bar, y0, x_bar + bar_thick, y0 + span_px - 1], fill=0)

    text_x = x0 + span_px + round(4.0 * ppmm)
    lines = [
        f"GridShot mat {spec.mat_id()}  --  PRINT AT 100% / ACTUAL SIZE",
        f"Caliper each black bar (X below, Y right): expected {spec.span_mm:.2f} mm.",
        f"Record with: gridshot mat verify {spec.mat_id()} --measured-x <mm> --measured-y <mm>",
    ]
    y_text = y_bar
    for i, line in enumerate(lines):
        draw.text((text_x, y_text), line, fill=0, font=font if i == 0 else small)
        y_text += round(3.4 * ppmm)

    return page


def render_pdf(spec: MatSpec) -> bytes:
    page = render_page(spec)
    buf = io.BytesIO()
    page.save(buf, format="PNG", dpi=(spec.dpi, spec.dpi))
    return img2pdf.convert(buf.getvalue())


# ---------------------------------------------------------------------------
# profile store


def mats_dir() -> Path:
    d = config_dir() / "mats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(profile: MatProfile) -> Path:
    path = mats_dir() / f"{profile.mat_id}.json"
    path.write_text(profile.model_dump_json(indent=2))
    return path


def load_profile(mat_id: str) -> MatProfile:
    path = mats_dir() / f"{mat_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no mat profile '{mat_id}' in {mats_dir()}")
    return MatProfile.model_validate_json(path.read_text())


def list_profiles() -> list[MatProfile]:
    return [
        MatProfile.model_validate_json(p.read_text())
        for p in sorted(mats_dir().glob("*.json"))
    ]


def new_mat(spec: MatSpec, out_dir: Path) -> tuple[MatProfile, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_id = spec.mat_id()
    pdf_path = out_dir / f"gridshot-mat-{mat_id}.pdf"
    pdf_path.write_bytes(render_pdf(spec))
    profile = MatProfile(
        mat_id=mat_id,
        spec=spec,
        created_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )
    save_profile(profile)
    return profile, pdf_path


def reference_path(mat_id: str) -> Path:
    return mats_dir() / f"{mat_id}-ref.png"


def save_reference(mat_id: str, canonical_img) -> Path:
    from PIL import Image

    path = reference_path(mat_id)
    Image.fromarray(canonical_img).save(path)
    return path


def load_reference(mat_id: str):
    import numpy as np
    from PIL import Image

    path = reference_path(mat_id)
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB"))


MAX_SCALE_DEVIATION = 0.02  # ±2%: beyond this the print is suspect (wrong paper/fit mode)
MAX_ANISOTROPY = 0.003  # X and Y scales differing >0.3% → reprint


def verify_scale(
    profile: MatProfile,
    measured_x_mm: float,
    measured_y_mm: float,
    force: bool = False,
) -> tuple[MatProfile, list[str]]:
    """Record caliper measurements of the printed scale bars.

    A print more than MAX_SCALE_DEVIATION off nominal is refused (recorded but
    left unverified) unless force=True: a driver that rescaled the page once
    ("fit to printable area" behind a dialog that claims 100%) may do so
    non-uniformly or differently next time, so the right fix is reprinting
    with scaling truly off — not calibrating around it.
    """
    expected = profile.spec.span_mm
    scale_x = measured_x_mm / expected
    scale_y = measured_y_mm / expected

    warnings: list[str] = []
    for axis, scale in (("X", scale_x), ("Y", scale_y)):
        if abs(scale - 1.0) > MAX_SCALE_DEVIATION:
            warnings.append(
                f"{axis} scale off by {100 * (scale - 1):+.1f}% — the printer scaled "
                f"the page (likely 'fit to printable area'); reprint at true 100%"
            )
    if abs(scale_x - scale_y) > MAX_ANISOTROPY:
        warnings.append(
            f"anisotropic print: X {100 * (scale_x - 1):+.2f}% vs "
            f"Y {100 * (scale_y - 1):+.2f}% — reprint recommended"
        )

    acceptable = all("reprint at true 100%" not in w for w in warnings)
    updated = profile.model_copy(
        update={
            "scale_x": scale_x,
            "scale_y": scale_y,
            "measured_span_x_mm": measured_x_mm,
            "measured_span_y_mm": measured_y_mm,
            "verified": acceptable or force,
        }
    )
    if not updated.verified:
        warnings.append(
            "measurements recorded but mat left UNVERIFIED — reprint and re-measure, "
            "or pass --force to calibrate around this print anyway"
        )
    save_profile(updated)
    return updated, warnings
