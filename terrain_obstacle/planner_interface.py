"""Terrain obstacle loader for the MICP planner (Examples 1 and 2).

Runs the full terrain pipeline once per example, converts polytopes to
axis-aligned OutsideRect predicates in local planner km coordinates, and
caches the results as JSON. Each example has its own zone-spec list and
its own cache file.

Cache guard (per example):
  - If cache_path exists  → load and return immediately.
  - If cache_path missing → run pipeline with the example's zone specs,
                            save JSON, then return.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Planner local origin: runway is at (150.0, 150.0) km in Example 1 and 2
_PLANNER_RUNWAY_KM = np.array([150.0, 150.0])

# Mission base coordinates (Farafangana) — shared by Ex1 and Ex2
_RUNWAY_LAT = -(22 + 48 / 60 + 18.70 / 3600)
_RUNWAY_LON =   47 + 49 / 60 + 13.90 / 3600

# Ex1: fixed Runway + D1..D4
_CACHE_PATH = _REPO_ROOT / "output" / "terrain" / "obstacles_ex1.json"
_ZONE_SPECS = [
    ("Runway", _RUNWAY_LAT,                          _RUNWAY_LON,                         300.0),
    ("D1",     -(23 + 27/60 + 47.30/3600),            47 + 16/60 +  3.50/3600,            300.0),
    ("D2",     -(23 + 39/60 + 36.80/3600),            47 + 12/60 +  4.50/3600,            300.0),
    ("D3",     -(23 + 49/60 + 21.87/3600),            46 + 59/60 +  4.26/3600,            300.0),
    ("D4",     -(23 + 35/60 + 24.74/3600),            47 +  1/60 +  5.09/3600,            300.0),
]

# Ex2: Runway + 15 EX2_POI_TABLE entries (Ex1's D1..D4 + 11 new POIs).
# Same Farafangana workspace as Ex1, but the unsafe_mask is conditioned on
# the full Ex2 zone set so any random 4-POI subset sees a terrain set that
# was computed accounting for ALL possible delivery sites.
_CACHE_PATH_EX2 = _REPO_ROOT / "output" / "terrain" / "obstacles_ex2.json"
_ZONE_SPECS_EX2 = [
    ("Runway",              _RUNWAY_LAT,                            _RUNWAY_LON,                            300.0),
    # Ex1's original D1..D4 (south of base)
    ("Ambongo",             -(23 + 27/60 + 47.30/3600),              47 + 16/60 +  3.50/3600,                300.0),
    ("Vatanato",            -(23 + 39/60 + 36.80/3600),              47 + 12/60 +  4.50/3600,                300.0),
    ("Befotaka",            -(23 + 49/60 + 21.87/3600),              46 + 59/60 +  4.26/3600,                300.0),
    ("Midongy_du_Sud",      -(23 + 35/60 + 24.74/3600),              47 +  1/60 +  5.09/3600,                300.0),
    # 11 new POIs added for Ex2
    ("Karimbary",           -(23 + 11/60 + 56.7/3600),               47 + 18/60 +  3.6/3600,                 300.0),
    ("Bevata",              -(23 + 18/60 + 55.0/3600),               47 + 16/60 + 47.5/3600,                 300.0),
    ("Ianakasy_Mahatsinjo", -(22 + 51/60 + 15.1/3600),               47 + 26/60 + 11.2/3600,                 300.0),
    ("Andremareo",          -(23 +  1/60 + 56.7/3600),               47 + 25/60 + 38.0/3600,                 300.0),
    ("Ambohimana",          -(22 + 35/60 + 12.2/3600),               47 + 18/60 + 37.1/3600,                 300.0),
    ("Mahasoa_I",           -(22 + 29/60 +  0.9/3600),               47 + 33/60 + 21.7/3600,                 300.0),
    ("Vohibe2",             -(22 + 40/60 +  6.8/3600),               47 + 24/60 + 59.2/3600,                 300.0),
    ("Antemanara",          -(23 +  0/60 + 24.8/3600),               47 + 23/60 + 59.3/3600,                 300.0),
    ("Manato",              -(22 + 24/60 +  3.2/3600),               47 + 30/60 + 34.3/3600,                 300.0),
    ("Ambaramafaitra",      -(22 + 30/60 + 58.9/3600),               47 + 20/60 + 21.0/3600,                 300.0),
    ("Bemandresy",          -(22 + 39/60 + 11.9/3600),               47 + 34/60 +  4.1/3600,                 300.0),
]

# Terrain pipeline parameters — must match terrain_obstacle/run.py real_data_example
_TILE_IDS       = ["S023E047", "S024E046", "S024E047"]
_MARGIN_M       = 8000.0
_MIN_SIZE       = 300
_MODE           = "8direction"
_CONVEX_DECOMP  = True
_CONVEXITY_THR  = 0.65
_MAX_K          = 3


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _run_pipeline(zone_specs: Sequence[Tuple[str, float, float, float]]):
    """Run terrain extraction with the given zone specs (first entry must be
    the runway). Returns (polytopes, runway_utm_m).
    """
    from terrain_obstacle.dem_io import load_dem_tiles
    from terrain_obstacle.cone import delivery_zone_from_latlon, unsafe_mask
    from terrain_obstacle.cluster import cluster_obstacles
    from terrain_obstacle.polytope import fit_polytopes

    datasets_dir = _REPO_ROOT / "datasets"
    dsm_paths = [datasets_dir / t / f"ALPSMLC30_{t}_DSM.tif" for t in _TILE_IDS]
    msk_paths = [datasets_dir / t / f"ALPSMLC30_{t}_MSK.tif" for t in _TILE_IDS]

    print(f"[terrain] Loading DEM tiles ... (n_zones={len(zone_specs)})")
    dem = load_dem_tiles(dsm_paths, msk_paths)

    zones = [delivery_zone_from_latlon(lat, lon, side_m, dem)
             for _, lat, lon, side_m in zone_specs]
    runway_utm_m = np.array(zones[0].center_xy)

    print("[terrain] Computing unsafe mask ...")
    mask = unsafe_mask(dem, zones, search_margin_m=_MARGIN_M)

    clusters = cluster_obstacles(mask, method="connected", min_size=_MIN_SIZE,
                                 resolution_m=dem.resolution_m)
    print(f"[terrain] {clusters.n_clusters} clusters found")

    polytopes = fit_polytopes(dem, clusters, mode=_MODE,
                              convex_decomp=_CONVEX_DECOMP,
                              convexity_threshold=_CONVEXITY_THR,
                              max_k=_MAX_K)
    print(f"[terrain] {len(polytopes)} polytopes fitted")
    return polytopes, runway_utm_m


def _poly_bbox_local_km(poly, utm_origin_m: np.ndarray):
    """
    Return axis-aligned bounding rect of the polytope in local km coords.

    Converts each vertex from UTM metres to local km, then takes min/max.
    Returns (xmin, xmax, ymin, ymax) — OutsideRect format.
    """
    verts_local = (poly.vertices - utm_origin_m) / 1000.0  # (V, 2)
    xmin = float(verts_local[:, 0].min())
    xmax = float(verts_local[:, 0].max())
    ymin = float(verts_local[:, 1].min())
    ymax = float(verts_local[:, 1].max())
    return (xmin, xmax, ymin, ymax)


def _build_terrain_cache(
    cache_path: Path,
    zone_specs: Sequence[Tuple[str, float, float, float]],
) -> List:
    """Shared cache-guarded loader. Loads from cache_path or runs the pipeline
    with `zone_specs` and writes a new cache. Returns a list of OutsideRect
    nodes.
    """
    from planner.stl import OutsideRect

    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        if data["obstacles"] and "A" in data["obstacles"][0]:
            # Old cache has polytope A/b format — force regeneration.
            cache_path.unlink()
            return _build_terrain_cache(cache_path, zone_specs)
    else:
        polytopes, runway_utm_m = _run_pipeline(zone_specs)
        utm_origin_m = runway_utm_m - _PLANNER_RUNWAY_KM * 1000.0
        obstacles = []
        for i, poly in enumerate(polytopes):
            rect = _poly_bbox_local_km(poly, utm_origin_m)
            obstacles.append({"name": f"O{i}", "rect": list(rect)})
        data = {
            "utm_origin_m": utm_origin_m.tolist(),
            "n_obstacles": len(obstacles),
            "obstacles": obstacles,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2))
        print(f"[terrain] Cached {len(obstacles)} obstacle rects → {cache_path}")

    return [
        OutsideRect(name=o["name"], rect=tuple(float(v) for v in o["rect"]))
        for o in data["obstacles"]
    ]


# ── Public API ─────────────────────────────────────────────────────────────────

def get_terrain_obstacles_ex1(cache_path: Path = _CACHE_PATH) -> List:
    """Ex1 terrain: cone clearance around Runway + Ex1's D1..D4.
    Cached at obstacles_ex1.json.
    """
    return _build_terrain_cache(cache_path, _ZONE_SPECS)


def get_terrain_obstacles_ex2(cache_path: Path = _CACHE_PATH_EX2) -> List:
    """Ex2 terrain: cone clearance around Runway + 15 EX2_POI candidates
    (Ex1's D1..D4 plus 11 new POIs). Cached at obstacles_ex2.json.
    """
    return _build_terrain_cache(cache_path, _ZONE_SPECS_EX2)
