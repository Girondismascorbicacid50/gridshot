"""Exports: STL + 3MF for slicing, SVG for 1:1 debug/bench measurement.

The 3MF writer is deliberately hand-rolled: the format is a small zip of XML,
and writing it directly (unit="millimeter" declared) avoids a heavyweight
dependency for M1.  Bambu Studio opens both; the SVG is the bench artifact —
raw vs corrected vs cleared outlines at true scale for caliper comparison.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from .contour import to_shapely
from .models import Poly

_3MF_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_3MF_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def stl_bytes(mesh) -> bytes:
    return mesh.export(file_type="stl")


def glb_bytes(mesh) -> bytes:
    """Binary glTF for the web 3D preview (compact, three.js loads it directly)."""
    return mesh.export(file_type="glb")


def threemf_bytes(mesh, name: str = "gridshot-bin") -> bytes:
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    v_xml = "".join(
        f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in verts
    )
    t_xml = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces
    )
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        f'<resources><object id="1" type="model" name="{name}">'
        f"<mesh><vertices>{v_xml}</vertices><triangles>{t_xml}</triangles></mesh>"
        "</object></resources>"
        '<build><item objectid="1"/></build></model>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _3MF_CONTENT_TYPES)
        zf.writestr("_rels/.rels", _3MF_RELS)
        zf.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


def _svg_path(poly: Poly) -> str:
    parts = []
    for ring in [poly.exterior, *poly.holes]:
        pts = " L ".join(f"{x:.3f} {y:.3f}" for x, y in ring)
        parts.append(f"M {pts} Z")
    return " ".join(parts)


def debug_svg(layers: list[tuple[str, str, Poly]]) -> str:
    """1:1-scale SVG of labelled outlines: [(label, css-colour, poly), ...]."""
    all_pts = [
        (x, y)
        for _, _, poly in layers
        for ring in [poly.exterior, *poly.holes]
        for x, y in ring
    ]
    xs, ys = zip(*all_pts)
    pad = 10.0
    minx, miny = min(xs) - pad, min(ys) - pad
    w, h = max(xs) - minx + pad, max(ys) - miny + pad

    paths = "\n".join(
        f'<path d="{_svg_path(poly)}" fill="none" stroke="{colour}" '
        f'stroke-width="0.15"><title>{label}</title></path>'
        for label, colour, poly in layers
    )
    legend = "\n".join(
        f'<text x="{minx + 2}" y="{miny + 5 + 4 * i}" font-size="3" '
        f'fill="{colour}">{label}</text>'
        for i, (label, colour, _) in enumerate(layers)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" '
        f'viewBox="{minx} {miny} {w} {h}">\n{paths}\n{legend}\n</svg>\n'
    )


def _poly_svg_path(poly: Poly, flip_y: float) -> str:
    """SVG path for a polygon, y mapped to print orientation (up = toward top)."""
    parts = []
    for ring in [poly.exterior, *poly.holes]:
        pts = " L ".join(f"{x:.3f} {flip_y - y:.3f}" for x, y in ring)
        parts.append(f"M {pts} Z")
    return " ".join(parts)


def layout_svg(
    gx: int,
    gy: int,
    height_u: int,
    pocket: Poly,
    tool: Poly | None,
    fingers: list[tuple[float, float, float]],
    clearance_mm: float,
    pitch: float = 42.0,
    bin_size: float = 41.5,
) -> str:
    """Top-down bin layout (print orientation): bin, grid cells, pocket, the
    tool outline inside it (gap = clearance), and finger scallops.  A standard
    per-trace artifact — open in any browser."""
    w = pitch * gx - (pitch - bin_size)
    d = pitch * gy - (pitch - bin_size)
    pad = 6.0
    vb_w, vb_h = w + 2 * pad, d + 2 * pad
    # bin-centred frame → SVG frame: shift by half-size + pad, flip y
    ox, oy = w / 2 + pad, d / 2 + pad

    def x(v):
        return v + ox

    flip = d / 2 + pad  # y_svg = flip - (y_bin - 0)  ... via _poly_svg_path with translate

    els = []
    # bin outline
    els.append(
        f'<rect x="{pad:.2f}" y="{pad:.2f}" width="{w:.2f}" height="{d:.2f}" '
        f'rx="3.75" fill="#fafafa" stroke="#555" stroke-width="0.6"/>'
    )
    # grid cell + foot guides
    for ix in range(gx):
        for iy in range(gy):
            cx = pad + (ix + 0.5) * pitch - (pitch - bin_size) / 2 if False else pad + ix * pitch + bin_size / 2
            cyv = pad + iy * pitch + bin_size / 2
            els.append(
                f'<rect x="{cx - bin_size/2:.2f}" y="{cyv - bin_size/2:.2f}" '
                f'width="{bin_size:.2f}" height="{bin_size:.2f}" fill="none" '
                f'stroke="#ddd" stroke-width="0.3"/>'
            )
    # translate group into (ox, flip) frame for polygons in bin-centred coords
    t = f'translate({ox:.3f},{oy:.3f}) scale(1,-1)'
    inner = []
    inner.append(
        f'<path d="{_poly_svg_path(pocket, 0.0)}" fill="#eaf1ff" '
        f'stroke="#c86e14" stroke-width="0.5"/>'
    )
    if tool is not None:
        inner.append(
            f'<path d="{_poly_svg_path(tool, 0.0)}" fill="none" '
            f'stroke="#2828dc" stroke-width="0.5"/>'
        )
    for fx, fy, dia in fingers:
        inner.append(
            f'<circle cx="{fx:.2f}" cy="{fy:.2f}" r="{dia/2:.2f}" fill="none" '
            f'stroke="#2aa02a" stroke-width="0.4" stroke-dasharray="1.5 1"/>'
        )
    els.append(f'<g transform="{t}">' + "".join(inner) + "</g>")

    title = (
        f'{gx}×{gy} units · {height_u}u · pocket +{clearance_mm}mm '
        f'(orange=pocket, blue=tool, green=finger)'
    )
    els.append(
        f'<text x="{pad:.2f}" y="{pad - 1.5:.2f}" font-size="2.6" '
        f'font-family="sans-serif" fill="#333">{title}</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{vb_w:.1f}mm" '
        f'height="{vb_h:.1f}mm" viewBox="0 0 {vb_w:.2f} {vb_h:.2f}">\n'
        + "\n".join(els)
        + "\n</svg>\n"
    )


def area_mm2(poly: Poly) -> float:
    return to_shapely(poly).area


def write_all(
    out_dir: Path,
    stem: str,
    mesh,
    svg: str | None = None,
    layout: str | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    stl = out_dir / f"{stem}.stl"
    stl.write_bytes(stl_bytes(mesh))
    written["stl"] = stl
    threemf = out_dir / f"{stem}.3mf"
    threemf.write_bytes(threemf_bytes(mesh, name=stem))
    written["3mf"] = threemf
    glb = out_dir / f"{stem}.glb"
    glb.write_bytes(glb_bytes(mesh))
    written["glb"] = glb
    if svg is not None:
        svg_path = out_dir / f"{stem}-outlines.svg"
        svg_path.write_text(svg)
        written["svg"] = svg_path
    if layout is not None:
        layout_path = out_dir / f"{stem}-layout.svg"
        layout_path.write_text(layout)
        written["layout"] = layout_path
    return written
