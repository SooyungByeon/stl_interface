from __future__ import annotations

import glob
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Allow imports from the repo root regardless of where the module is called from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from terrain_utils import (  # noqa: E402
    load_and_clean_terrain,
    load_stitched_terrain,
    identify_required_tiles,
)


@dataclass
class DEMData:
    elevation: np.ndarray  # (H, W) float32, metres; NaN where cloud-masked
    transform: object      # rasterio Affine: pixel (col, row) → UTM (x, y)
    crs: str               # e.g. "EPSG:32738"

    @property
    def resolution_m(self) -> float:
        """Pixel size in metres (assumes square, north-up pixels)."""
        return float(abs(self.transform.a))

    def xy_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X, Y) meshgrids of UTM pixel-centre coordinates, shape (H, W).

        Assumes a north-up raster (no rotation terms in the affine transform).
        Row 0 is the northernmost row; Y decreases with increasing row index.
        """
        H, W = self.elevation.shape
        # +0.5 shifts from pixel top-left corner to pixel centre
        cols = np.arange(W, dtype=np.float64) + 0.5
        rows = np.arange(H, dtype=np.float64) + 0.5
        x_1d = self.transform.c + cols * self.transform.a
        y_1d = self.transform.f + rows * self.transform.e
        X, Y = np.meshgrid(x_1d, y_1d)
        return X, Y


def load_dem_tiles(
    dsm_paths: List[Path],
    msk_paths: List[Optional[Path]],
) -> DEMData:
    """
    Load and reproject one or more DSM tiles into a single DEMData.
    Wraps terrain_utils.load_and_clean_terrain / load_stitched_terrain.
    """
    dsm_strs = [str(p) for p in dsm_paths]
    msk_strs = [str(p) if p is not None else None for p in msk_paths]

    if len(dsm_strs) == 1:
        elev, transform, crs = load_and_clean_terrain(dsm_strs[0], msk_strs[0])
    else:
        elev, transform, crs = load_stitched_terrain(dsm_strs, msk_strs)

    return DEMData(elevation=elev.astype(np.float32), transform=transform, crs=crs)


def load_dem_for_region(
    lat: float,
    lon: float,
    data_folder: Path,
    radius_m: float = 5000.0,
) -> DEMData:
    """
    Discover, stitch, and return a DEMData covering (lat, lon) ± radius_m.

    Uses identify_required_tiles to find which 1°×1° tiles are needed, then
    locates matching DSM/MSK files by the standard JAXA naming convention
    (e.g. S024E047/…_S024E047_DSM.tif).
    """
    needed = identify_required_tiles(lat, lon, radius_m)

    dsm_paths: List[Path] = []
    msk_paths: List[Optional[Path]] = []

    for t_lat, t_lon in needed:
        lat_prefix = "S" if t_lat < 0 else "N"
        lon_prefix = "W" if t_lon < 0 else "E"
        stem = f"{lat_prefix}{abs(t_lat):02d}{lon_prefix}{abs(t_lon):03d}"
        pattern = str(Path(data_folder) / "**" / f"*{stem}*DSM.tif")
        hits = glob.glob(pattern, recursive=True)
        for h in hits:
            dsm_p = Path(h)
            msk_p = Path(h.replace("DSM.tif", "MSK.tif"))
            dsm_paths.append(dsm_p)
            msk_paths.append(msk_p if msk_p.exists() else None)

    if not dsm_paths:
        raise FileNotFoundError(
            f"No tiles found for ({lat:.4f}, {lon:.4f}) with radius "
            f"{radius_m:.0f} m in {data_folder}"
        )

    return load_dem_tiles(dsm_paths, msk_paths)
