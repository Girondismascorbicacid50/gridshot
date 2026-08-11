"""Drawer nesting: pack tool bins into a gridfinity drawer, tightly.

Rotation-aware First-Fit-Decreasing-Height shelf packing
over a half-unit (21mm) cell grid. Long/thin tools are laid flat (long axis
along the drawer width) so many short shelves stack densely instead of a few
tall ones — the difference between "4 of 7 fit" and "all 7 fit". Each bin may
be turned 90° (its tool rotates with it); the packer picks the orientation
that keeps the footprint smallest.

Deterministic and pure: no GPU, no I/O. Overflow spills to a second drawer.
Upgrade path if it ever annoys: skyline / MaxRects.
"""

from __future__ import annotations

from .gridfinity import PITCH
from .models import DrawerLayout, Placement


def drawer_cells(drawer_w_mm: float, drawer_d_mm: float) -> tuple[int, int]:
    """Whole 42mm gridfinity cells that fit the drawer (cols, rows)."""
    return int(drawer_w_mm // PITCH), int(drawer_d_mm // PITCH)


def _half(u: float) -> int:
    """Footprint in units → half-cells (0.5u granularity), min one half-cell."""
    return max(1, round(u * 2))


def _pack_bias(
    bins: list[tuple[str, float, float]],
    width: int,
    height: int,
    allow_rotate: bool,
    bias: str,
) -> tuple[list[Placement], int, int, list[str]]:
    """One rotation-aware FFDH pass under a starting-orientation bias.

    bias 'flat' lays every bin landscape, 'tall' portrait, 'asis' keeps it; a
    bin still flips if that's the only way it fits the drawer width. Bins are
    placed tallest-first onto the first shelf with horizontal room, else a new
    shelf below. Returns (placements, used_width, used_height, overflow).
    """
    prepared: list[tuple[str, int, int, bool]] = []
    for bin_id, gx, gy in bins:
        w, h = _half(gx), _half(gy)
        rotated = False
        if allow_rotate and ((bias == "flat" and h > w) or (bias == "tall" and w > h)):
            w, h, rotated = h, w, True
        if w > width and h <= width and allow_rotate:  # won't fit flat → stand up
            w, h, rotated = h, w, not rotated
        prepared.append((bin_id, w, h, rotated))

    order = sorted(prepared, key=lambda t: (-t[2], -t[1], t[0]))  # h desc, w desc, id

    shelves: list[list[int]] = []  # [y, shelf_height, used_x] in half-cells
    placed: list[Placement] = []
    overflow: list[str] = []
    total_h = 0
    used_w = 0

    for bin_id, w, h, rotated in order:
        if w > width or h > height:
            overflow.append(bin_id)
            continue
        for shelf in shelves:
            if shelf[2] + w <= width:  # room on an existing (>= tall) shelf
                placed.append(_place(bin_id, shelf[2], shelf[0], w, h, rotated))
                shelf[2] += w
                used_w = max(used_w, shelf[2])
                break
        else:  # open a new shelf below
            if total_h + h <= height:
                shelves.append([total_h, h, w])
                placed.append(_place(bin_id, 0, total_h, w, h, rotated))
                total_h += h
                used_w = max(used_w, w)
            else:
                overflow.append(bin_id)

    return placed, used_w, total_h, overflow


def _best_pack(bins, width, height, allow_rotate):
    """Try each orientation bias; keep the arrangement that fits the most and
    then uses the least area. This is what makes the result 'minimal space':
    for a wide drawer, upright tools side-by-side may beat flat ones, and vice
    versa — so we pack both ways and let the smaller footprint win."""
    biases = ["flat", "tall", "asis"] if allow_rotate else ["asis"]
    best = None
    for bias in biases:
        placed, uw, uh, overflow = _pack_bias(bins, width, height, allow_rotate, bias)
        key = (len(overflow), uw * uh, uh, uw)  # fewest lost, then tightest, then short
        if best is None or key < best[0]:
            best = (key, placed, uw, uh, overflow)
    return best[1], best[2], best[3], best[4]


def _place(bin_id, x, y, w, h, rotated) -> Placement:
    return Placement(
        bin_id=bin_id,
        col=x / 2,
        row=y / 2,
        grid_x=w / 2,
        grid_y=h / 2,
        rotated=rotated,
    )


def nest(
    bins: list[tuple[str, float, float]],
    drawer_w_mm: float,
    drawer_d_mm: float,
    allow_rotate: bool = True,
) -> DrawerLayout:
    """Pack (bin_id, grid_x, grid_y) footprints into one drawer, tightly.

    grid_x/grid_y are gridfinity units (0.5 steps). Placement is returned in
    units with the as-placed footprint and a `rotated` flag; bins that don't
    fit go to `overflow`. `used_cols/used_rows` report the tight bounding
    footprint so you can see how little space the arrangement really needs.
    """
    cols, rows = drawer_cells(drawer_w_mm, drawer_d_mm)
    placed, used_w, used_h, overflow = _best_pack(bins, cols * 2, rows * 2, allow_rotate)
    return DrawerLayout(
        drawer_w_mm=drawer_w_mm,
        drawer_d_mm=drawer_d_mm,
        placed=placed,
        overflow=overflow,
        used_cols=used_w / 2,
        used_rows=used_h / 2,
    )


def smallest_drawer(
    bins: list[tuple[str, float, float]],
    max_cols: int = 24,
    allow_rotate: bool = True,
) -> tuple[int, int]:
    """Smallest whole-cell drawer (cols, rows) that holds every bin.

    Sweeps candidate widths and keeps the one giving the least bounding area
    (ties → squarer). Handy for 'what drawer do these tools need?'.
    """
    best: tuple[int, int, int] | None = None  # (area, cols, rows)
    lo = max(1, max((_half(gx) + 1) // 2 for _, gx, gy in bins) if not allow_rotate
             else max((min(_half(gx), _half(gy)) + 1) // 2 for _, gx, gy in bins))
    for cols in range(lo, max_cols + 1):
        _placed, _uw, used_h, overflow = _best_pack(bins, cols * 2, 10_000, allow_rotate)
        if overflow:
            continue
        rows = (used_h + 1) // 2
        area = cols * rows
        cand = (area, cols, rows)
        if best is None or cand < best:
            best = cand
    if best is None:
        raise ValueError("a bin is larger than max_cols in both orientations")
    return best[1], best[2]
