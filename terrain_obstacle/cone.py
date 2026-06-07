from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rasterio.warp import transform as _rasterio_transform  # noqa: E402

from .dem_io import DEMData

# ── Physical constants ─────────────────────────────────────────────────────────
CLEARANCE_M: float = 50.0   # vertical clearance above z_ground for each DAP (m)
FPA_DEG: float = 7.18       # maximum flight-path angle in degrees


@dataclass
class DeliveryZone:
    center_xy: Tuple[float, float]  # UTM easting/northing of the square centre (m)
    side_m: float                   # side length of the axis-aligned delivery square (m)
    z_ground_m: float               # ground elevation at the zone centre (m)


# ── Coordinate conversion ──────────────────────────────────────────────────────

def latlon_to_utm(lat: float, lon: float, crs: str) -> Tuple[float, float]:
    """Convert WGS-84 (lat, lon) to UTM (x, y) in the given CRS string."""
    xs, ys = _rasterio_transform("EPSG:4326", crs, [lon], [lat])
    return float(xs[0]), float(ys[0])


def delivery_zone_from_latlon(
    lat: float,
    lon: float,
    side_m: float,
    dem: DEMData,
) -> DeliveryZone:
    """
    Build a DeliveryZone by projecting (lat, lon) to UTM using dem.crs, then
    sampling the DEM for z_ground at the nearest grid cell.
    """
    cx, cy = latlon_to_utm(lat, lon, dem.crs)

    # Nearest pixel (col, row) to the UTM centre point
    t = dem.transform
    col = int(round((cx - t.c) / t.a - 0.5))
    row = int(round((cy - t.f) / t.e - 0.5))
    H, W = dem.elevation.shape
    col = max(0, min(W - 1, col))
    row = max(0, min(H - 1, row))

    z_ground = float(dem.elevation[row, col])
    if math.isnan(z_ground):
        # Fall back to nearest non-NaN neighbour in a small window
        r0, r1 = max(0, row - 5), min(H, row + 6)
        c0, c1 = max(0, col - 5), min(W, col + 6)
        patch = dem.elevation[r0:r1, c0:c1]
        valid = patch[~np.isnan(patch)]
        z_ground = float(valid.mean()) if len(valid) > 0 else 0.0

    return DeliveryZone(center_xy=(cx, cy), side_m=side_m, z_ground_m=z_ground)


# ── Core geometry ──────────────────────────────────────────────────────────────

def dist_to_square(
    X: np.ndarray,
    Y: np.ndarray,
    zone: DeliveryZone,
) -> np.ndarray:
    """
    Vectorised horizontal distance (m) from each grid point to the nearest
    point on zone's axis-aligned square.  Returns 0 for points inside the square.
    Shape: same as X and Y.
    """
    cx, cy = zone.center_xy
    half = zone.side_m / 2.0
    dx = np.maximum(0.0, np.abs(X - cx) - half)
    dy = np.maximum(0.0, np.abs(Y - cy) - half)
    return np.sqrt(dx * dx + dy * dy)


def cone_surface(
    X: np.ndarray,
    Y: np.ndarray,
    zone: DeliveryZone,
    clearance_m: float = CLEARANCE_M,
    fpa_deg: float = FPA_DEG,
) -> np.ndarray:
    """
    Minimum safe altitude surface for one delivery zone:

        z_cone(x, y) = (z_ground + clearance_m) + dist_to_square * tan(fpa_deg)

    The cone opens upward and outward from the DAP square in all 360 degrees.
    Shape: same as X and Y.
    """
    z_dap = zone.z_ground_m + clearance_m
    d = dist_to_square(X, Y, zone)
    return z_dap + d * math.tan(math.radians(fpa_deg))


# ── Pipeline entry point ───────────────────────────────────────────────────────

def unsafe_mask(
    dem: DEMData,
    zones: List[DeliveryZone],
    search_margin_m: float = 10000.0,
    clearance_m: float = CLEARANCE_M,
    fpa_deg: float = FPA_DEG,
) -> np.ndarray:
    """
    Return a boolean mask (H, W) that is True where terrain elevation exceeds
    the cone surface of *any* delivery zone.

    Only cells within the bounding box of all delivery squares ± search_margin_m
    are evaluated; the rest remain False.  Fully vectorised over DEM cells.
    """
    if not zones:
        return np.zeros(dem.elevation.shape, dtype=bool)

    X, Y = dem.xy_grids()

    # Bounding box of all zone squares + search margin
    all_cx = [z.center_xy[0] for z in zones]
    all_cy = [z.center_xy[1] for z in zones]
    all_half = [z.side_m / 2.0 for z in zones]

    x_lo = min(cx - h for cx, h in zip(all_cx, all_half)) - search_margin_m
    x_hi = max(cx + h for cx, h in zip(all_cx, all_half)) + search_margin_m
    y_lo = min(cy - h for cy, h in zip(all_cy, all_half)) - search_margin_m
    y_hi = max(cy + h for cy, h in zip(all_cy, all_half)) + search_margin_m

    region = (X >= x_lo) & (X <= x_hi) & (Y >= y_lo) & (Y <= y_hi)

    # Minimum cone surface over all zones (element-wise)
    min_cone = np.full(dem.elevation.shape, np.inf, dtype=np.float64)
    for zone in zones:
        z_c = cone_surface(X, Y, zone, clearance_m, fpa_deg)
        np.minimum(min_cone, z_c, out=min_cone)

    terrain = dem.elevation.astype(np.float64)
    return region & (terrain > min_cone) & ~np.isnan(terrain)
