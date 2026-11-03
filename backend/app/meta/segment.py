"""Lasso selection: grid -> PNG, SAM call, threshold fallback, mask -> stats (PRD_BACKEND §4.16).

SAM only ever sees a PNG that this module renders from our own grid (fixed colormap and the
/meta colour range, land masked). Whatever it returns is intersected with `coverage == 1`, and
all numbers come from the numeric grid through `area_weighted_stats`, the same code as
/region-summary. SAM shapes are a visual aid; the fallback is a deterministic flood fill.
"""

from __future__ import annotations

import math
import struct
import zlib
from collections import deque
from dataclasses import dataclass

import numpy as np

# Cap on threshold selections (§4.16).
MAX_THRESHOLD_CELLS = 5_000
# The PNG sent to SAM is upscaled by an integer factor so its long side is at least this.
SAM_MIN_LONG_SIDE_PX = 512
EARTH_RADIUS_KM = 6371.0088

# Fixed colormap anchors (cold -> warm), linear between anchors. Land/no data is black.
_COLORMAP = np.array(
    [
        [49, 54, 149],
        [69, 117, 180],
        [116, 173, 209],
        [171, 217, 233],
        [254, 224, 144],
        [253, 174, 97],
        [244, 109, 67],
        [215, 48, 39],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class RenderedGrid:
    """A PNG of the grid plus how grid cells map to its pixels (north up, `scale` px a cell)."""

    png: bytes
    scale: int
    n_lat: int
    n_lon: int

    def pixel_of(self, lat_idx: int, lon_idx: int) -> tuple[int, int]:
        """Centre pixel (x, y) of a grid cell. Row 0 of the image is the northernmost lat."""
        x = lon_idx * self.scale + self.scale // 2
        y = (self.n_lat - 1 - lat_idx) * self.scale + self.scale // 2
        return x, y

    def mask_to_cells(self, mask: np.ndarray) -> np.ndarray:
        """Pixel mask (height, width) -> (lat, lon) cell mask, sampling each cell's centre pixel.

        A mask of the wrong size is treated as empty so a bad upstream reply falls back.
        """
        if mask.shape != (self.n_lat * self.scale, self.n_lon * self.scale):
            return np.zeros((self.n_lat, self.n_lon), dtype=bool)
        c = self.scale // 2
        centres = mask[c :: self.scale, c :: self.scale][: self.n_lat, : self.n_lon]
        return np.asarray(centres[::-1, :], dtype=bool)


def render_grid_png(
    values: np.ndarray, coverage: np.ndarray, vmin: float, vmax: float
) -> RenderedGrid:
    """Render the field at grid resolution with the fixed colormap, then upscale (nearest)."""
    n_lat, n_lon = values.shape
    scale = max(1, math.ceil(SAM_MIN_LONG_SIDE_PX / max(n_lat, n_lon)))
    span = vmax - vmin if vmax > vmin else 1.0
    t = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / span, 0.0, 1.0)
    pos = t * (len(_COLORMAP) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_COLORMAP) - 1)
    frac = (pos - lo)[..., None]
    rgb = _COLORMAP[lo] * (1 - frac) + _COLORMAP[hi] * frac
    ocean = (coverage == 1) & ~np.isnan(values)
    rgb[~ocean] = 0.0
    img = np.round(rgb).astype(np.uint8)[::-1]  # north up
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return RenderedGrid(png=_encode_png(img), scale=scale, n_lat=n_lat, n_lon=n_lon)


def _encode_png(rgb: np.ndarray) -> bytes:
    """Minimal 8-bit RGB PNG encoder (stdlib zlib only)."""
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def threshold_select(
    values: np.ndarray,
    valid: np.ndarray,
    seed: tuple[int, int],
    tolerance_c: float,
    max_cells: int = MAX_THRESHOLD_CELLS,
) -> tuple[np.ndarray, bool]:
    """4-connected valid cells within ±tolerance of the seed value, breadth-first from the seed.

    Returns (mask, capped). Stops at `max_cells`; the order is deterministic.
    """
    n_lat, n_lon = values.shape
    selected = np.zeros((n_lat, n_lon), dtype=bool)
    r0, c0 = seed
    if not valid[r0, c0]:
        return selected, False
    target = float(values[r0, c0])
    queue: deque[tuple[int, int]] = deque([seed])
    selected[r0, c0] = True
    count = 1
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_lat and 0 <= cc < n_lon and not selected[rr, cc] and valid[rr, cc]:
                if abs(float(values[rr, cc]) - target) <= tolerance_c:
                    if count >= max_cells:
                        return selected, True
                    selected[rr, cc] = True
                    count += 1
                    queue.append((rr, cc))
    return selected, False


def connected_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """The 4-connected part of `mask` that contains `seed` (empty if the seed is not in it)."""
    out = np.zeros_like(mask, dtype=bool)
    if not mask[seed]:
        return out
    n_lat, n_lon = mask.shape
    queue: deque[tuple[int, int]] = deque([seed])
    out[seed] = True
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n_lat and 0 <= cc < n_lon and mask[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                queue.append((rr, cc))
    return out


def mask_to_ranges(mask: np.ndarray) -> list[list[int]]:
    """Run-length rows: [lat_index, first_lon_index, last_lon_index] (inclusive), row-major."""
    ranges: list[list[int]] = []
    for r in range(mask.shape[0]):
        row = mask[r]
        if not row.any():
            continue
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1) - 1
        ranges.extend([r, int(s), int(e)] for s, e in zip(starts, ends))
    return ranges


def area_km2(mask: np.ndarray, lats: np.ndarray, res_deg: float) -> float:
    """Sum of spherical cell areas (R² · Δφ · Δλ · cos φ) over the selected cells."""
    res_rad = math.radians(res_deg)
    cell = (EARTH_RADIUS_KM * res_rad) ** 2 * np.cos(np.radians(lats))
    per_row = mask.sum(axis=1)
    return round(float(np.sum(per_row * cell)), 1)


def outline_polygon(
    mask: np.ndarray, lats: np.ndarray, lons: np.ndarray, res_deg: float
) -> list[list[float]]:
    """Outer boundary of the selection as a closed [[lon, lat], ...] ring along cell edges.

    Traces the cell-edge boundary of every selected cell, keeps the ring with the largest
    area (the outer edge; holes and detached pieces are not drawn) and drops collinear points.
    """
    if not mask.any():
        return []
    n_lat, n_lon = mask.shape
    # Directed edges, counter-clockwise around each selected cell, in corner coordinates
    # (row i = southern edge of lat row i, col j = western edge of lon column j).
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def sel(r: int, c: int) -> bool:
        return 0 <= r < n_lat and 0 <= c < n_lon and bool(mask[r, c])

    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if not sel(r - 1, c):  # south edge, west -> east
            edges.setdefault((r, c), []).append((r, c + 1))
        if not sel(r, c + 1):  # east edge, south -> north
            edges.setdefault((r, c + 1), []).append((r + 1, c + 1))
        if not sel(r + 1, c):  # north edge, east -> west
            edges.setdefault((r + 1, c + 1), []).append((r + 1, c))
        if not sel(r, c - 1):  # west edge, north -> south
            edges.setdefault((r + 1, c), []).append((r, c))

    rings: list[list[tuple[int, int]]] = []
    while edges:
        start = next(iter(edges))
        ring = [start]
        prev_dir: tuple[int, int] | None = None
        cur = start
        while True:
            outs = edges.get(cur)
            if not outs:
                break
            nxt = _pick_turn(cur, outs, prev_dir)
            outs.remove(nxt)
            if not outs:
                del edges[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start:
                break
            ring.append(cur)
        rings.append(ring)

    outer = max(rings, key=lambda ring: abs(_ring_area(ring)))
    simplified = _drop_collinear(outer)
    lat0 = float(lats[0]) - res_deg / 2
    lon0 = float(lons[0]) - res_deg / 2
    coords = [[round(lon0 + j * res_deg, 3), round(lat0 + i * res_deg, 3)] for i, j in simplified]
    coords.append(coords[0])
    return coords


def _pick_turn(
    cur: tuple[int, int], outs: list[tuple[int, int]], prev_dir: tuple[int, int] | None
) -> tuple[int, int]:
    """At a corner shared by two diagonal cells, turn left so each cell's ring stays separate."""
    if len(outs) == 1 or prev_dir is None:
        return outs[0]
    left = (prev_dir[1], -prev_dir[0])  # (dr, dc) rotated: for (row, col) axes this is a left turn
    for o in outs:
        if (o[0] - cur[0], o[1] - cur[1]) == left:
            return o
    return outs[0]


def _ring_area(ring: list[tuple[int, int]]) -> float:
    total = 0.0
    for k in range(len(ring)):
        r1, c1 = ring[k]
        r2, c2 = ring[(k + 1) % len(ring)]
        total += c1 * r2 - c2 * r1
    return total / 2.0


def _drop_collinear(ring: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    n = len(ring)
    for k in range(n):
        a, b, c = ring[k - 1], ring[k], ring[(k + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            out.append(b)
    return out or ring
