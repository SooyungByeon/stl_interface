"""
Backend-neutral loader for the LA DEM terrain background (showcase only).

Kept free of any matplotlib backend selection so it can be shared by the Agg
mission plots (planner/plot_mission.py) and the Qt authoring interface
(interface/app.py) without a backend conflict.

Returns (elev_2d float32, extent_km=[xmin,xmax,ymin,ymax]) in the shared
la_frame km coordinates, downsampled and cached to output/terrain/la_dem.npz.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import la_frame as la

_REPO = Path(__file__).resolve().parent.parent
_DEM_CACHE = _REPO / "output" / "terrain" / "la_dem.npz"
_LA_TILES = ["N033W118", "N033W119", "N034W118", "N034W119"]


def load_la_dem(max_px: int = 2000) -> Optional[Tuple[np.ndarray, List[float]]]:
    """Return (elev, extent_km) for the LA tiles, or None on failure.
    Downsampled to ~max_px per axis and cached to a .npz."""
    try:
        if _DEM_CACHE.exists():
            d = np.load(_DEM_CACHE)
            return d["elev"], [float(v) for v in d["extent"]]

        dsm = [_REPO / "datasets" / t / f"ALPSMLC30_{t}_DSM.tif" for t in _LA_TILES]
        msk = [_REPO / "datasets" / t / f"ALPSMLC30_{t}_MSK.tif" for t in _LA_TILES]
        if not all(p.exists() for p in dsm):
            return None

        from terrain_obstacle.dem_io import load_dem_tiles
        dem = load_dem_tiles(dsm, [p if p.exists() else None for p in msk])
        X, Y = dem.xy_grids()
        ox, oy = la.ORIGIN_UTM_M
        elev = dem.elevation.astype(np.float32)
        elev[elev < -100.0] = np.nan

        stride = max(1, max(elev.shape) // int(max_px))
        elev = elev[::stride, ::stride]
        Xk = (X[::stride, ::stride] - ox) / 1000.0
        Yk = (Y[::stride, ::stride] - oy) / 1000.0
        extent = [float(Xk.min()), float(Xk.max()), float(Yk.min()), float(Yk.max())]

        _DEM_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(_DEM_CACHE, elev=elev, extent=np.array(extent))
        return elev, extent
    except Exception as e:  # pragma: no cover - showcase background only
        print(f"[terrain_bg] DEM background unavailable: {e}")
        return None


__all__ = ["load_la_dem"]
