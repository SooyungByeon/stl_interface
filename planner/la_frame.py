"""
Shared Los Angeles coordinate frame for the paper (Click-to-STL) examples.

All three paper examples live in one fixed local-km frame derived from the ALOS
DSM tiles in datasets/ (N033/N034 x W118/W119), covering lat 33-35, lon -119..-117.

Projection: WGS84 (EPSG:4326) -> UTM zone 11N (EPSG:32611), then
    local_km = (UTM_m - origin) / 1000,
with the origin at the combined-tile SW corner (lon -119, lat 33). This matches
the convention used by terrain_obstacle/planner_interface.py.

Terrain is SHOWCASE ONLY here (not turned into obstacles).

Fixed across ALL examples (the two "runways"):
    G1 = LAX               (33.95, -118.40)  -> (57.5, 106.2) km
    G2 = Ontario Intl      (34.06, -117.60)  -> (131.5, 117.7) km
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from pyproj import Transformer

from .geometry import Rect

# --------------------------------------------------------------------------
# Projection / frame
# --------------------------------------------------------------------------

_WGS84 = "EPSG:4326"
_UTM_11N = "EPSG:32611"
_TF = Transformer.from_crs(_WGS84, _UTM_11N, always_xy=True)

# Combined-tile corners: lat 33..35, lon -119..-117.
_TILE_LAT = (33.0, 35.0)
_TILE_LON = (-119.0, -117.0)

# Origin = SW corner of the combined tile region.
_ORIGIN_X, _ORIGIN_Y = _TF.transform(_TILE_LON[0], _TILE_LAT[0])
ORIGIN_UTM_M: Tuple[float, float] = (float(_ORIGIN_X), float(_ORIGIN_Y))


def latlon_to_km(lat: float, lon: float) -> Tuple[float, float]:
    """Convert WGS84 lat/lon to local planner km (UTM 11N, tile-SW origin)."""
    x, y = _TF.transform(float(lon), float(lat))
    return ((x - _ORIGIN_X) / 1000.0, (y - _ORIGIN_Y) / 1000.0)


def _workspace_bounds() -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Axis-aligned km bounding box over the four tile corners."""
    xs, ys = [], []
    for lat in _TILE_LAT:
        for lon in _TILE_LON:
            kx, ky = latlon_to_km(lat, lon)
            xs.append(kx)
            ys.append(ky)
    return ((min(xs), max(xs)), (min(ys), max(ys)))


WORKSPACE_X, WORKSPACE_Y = _workspace_bounds()  # ((0, 186.8), (0, 223.6))

# Fixed top-down view window (km), SHARED by every mission plot so the axes and
# aspect ratio are identical across examples. Contains G1/G2, H, all delivery
# POIs, and the obstacle corridor, with margin.
VIEW_XLIM: Tuple[float, float] = (40.0, 150.0)
VIEW_YLIM: Tuple[float, float] = (80.0, 150.0)

# --------------------------------------------------------------------------
# Fixed runways (G1, G2) and candidate delivery POIs
# --------------------------------------------------------------------------

G1_LATLON: Tuple[float, float] = (33.95, -118.40)   # LAX
G2_LATLON: Tuple[float, float] = (34.06, -117.60)   # Ontario Intl

G1_KM: Tuple[float, float] = latlon_to_km(*G1_LATLON)
G2_KM: Tuple[float, float] = latlon_to_km(*G2_LATLON)

# LA-basin airfields used as delivery points in Examples 2/3 (G1/G2 stay reserved).
DELIVERY_POIS_LATLON: Dict[str, Tuple[float, float]] = {
    "SMO": (34.0158, -118.4513),  # Santa Monica
    "HHR": (33.9228, -118.3353),  # Hawthorne
    "CPM": (33.8900, -118.2447),  # Compton
    "LGB": (33.8177, -118.1517),  # Long Beach
    "EMT": (34.0860, -118.0350),  # El Monte
    "FUL": (33.8720, -117.9799),  # Fullerton
    "BUR": (34.2007, -118.3585),  # Burbank
    "VNY": (34.2098, -118.4899),  # Van Nuys
}
DELIVERY_POIS_KM: Dict[str, Tuple[float, float]] = {
    name: latlon_to_km(lat, lon) for name, (lat, lon) in DELIVERY_POIS_LATLON.items()
}

# --------------------------------------------------------------------------
# Dynamics / bounds preset (low-altitude aircraft, LA workspace)
# --------------------------------------------------------------------------

DT: float = 60.0
V_MIN: float = 0.028   # km/s horizontal speed floor (stall avoidance)
V_MAX: float = 0.040   # km/s horizontal speed cap
CRUISE_Z_REF: float = 1.2
GRACE_LO: int = 0   # no initial speed-floor grace (floor active from step 0)
GRACE_HI: int = 3   # final grace for the landing flare


def make_bounds() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_min, x_max, u_min, u_max) for the LA workspace + low-alt limits."""
    x_min = np.array([WORKSPACE_X[0], WORKSPACE_Y[0], 0.0, -0.040, -0.040, -0.006])
    x_max = np.array([WORKSPACE_X[1], WORKSPACE_Y[1], 1.5,  0.040,  0.040,  0.006])
    u_min = np.array([-0.00020, -0.00020, -0.0001])
    u_max = np.array([ 0.00020,  0.00020,  0.0001])
    return x_min, x_max, u_min, u_max


def box(cx: float, cy: float, half: float) -> Rect:
    """Axis-aligned square region of half-side `half` km centred at (cx, cy)."""
    return (cx - half, cx + half, cy - half, cy + half)


__all__ = [
    "ORIGIN_UTM_M",
    "latlon_to_km",
    "WORKSPACE_X",
    "WORKSPACE_Y",
    "VIEW_XLIM",
    "VIEW_YLIM",
    "G1_LATLON",
    "G2_LATLON",
    "G1_KM",
    "G2_KM",
    "DELIVERY_POIS_LATLON",
    "DELIVERY_POIS_KM",
    "DT",
    "V_MIN",
    "V_MAX",
    "CRUISE_Z_REF",
    "GRACE_LO",
    "GRACE_HI",
    "make_bounds",
    "box",
]
