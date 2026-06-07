"""
CLI entry point and synthetic end-to-end example.

Usage (synthetic, no real data needed):
    python -m terrain_obstacle.run --synthetic

Usage (real DEM):
    python -m terrain_obstacle.run \
        --dem path/to/DSM.tif path/to/MSK.tif \
        --zone -24.3 47.2 500 \
        --zone -24.5 47.4 300 \
        [--margin 10000] [--cluster connected|dbscan] \
        [--mode 8direction|axis_aligned] [--min-size 10] \
        [--output output/terrain/result.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from terrain_obstacle.dem_io import DEMData, load_dem_tiles
from terrain_obstacle.cone import (
    CLEARANCE_M, FPA_DEG,
    DeliveryZone, delivery_zone_from_latlon, unsafe_mask,
)
from terrain_obstacle.cluster import cluster_obstacles
from terrain_obstacle.polytope import fit_polytopes
from terrain_obstacle.visualize import plot_pipeline

_OUTPUT_DIR = _REPO_ROOT / "output" / "terrain"


# ── Synthetic example ──────────────────────────────────────────────────────────

def synthetic_example() -> None:
    """
    Build a 5 km × 5 km synthetic DEM with a central hill and a secondary peak,
    place two delivery zones, and run the full pipeline.  No real dataset needed.
    """
    import math
    from rasterio.transform import Affine

    res = 10.0                    # 10 m/pixel
    H, W = 500, 500               # 5 km × 5 km
    ox, oy = 400_000.0, 7_200_000.0  # arbitrary UTM origin (bottom-left corner)

    # Pixel-centre coordinate grids (for building the synthetic elevation)
    cols = np.arange(W, dtype=np.float64) + 0.5
    rows = np.arange(H, dtype=np.float64) + 0.5
    # top-left origin: f = oy + H*res  →  y decreases with row index
    x_1d = ox + cols * res
    y_1d = (oy + H * res) - rows * res
    X, Y = np.meshgrid(x_1d, y_1d)

    # Flat base at 200 m + two tall hills (600 m and 400 m above base).
    # Delivery zones are placed on flat ground at the corners; the hills
    # lie in the approach paths and poke above the cone surface.
    elev = (
        200.0
        + 600.0 * np.exp(-((X - ox - 2500) ** 2 + (Y - oy - 2500) ** 2) / 700 ** 2)
        + 400.0 * np.exp(-((X - ox - 1500) ** 2 + (Y - oy - 1500) ** 2) / 600 ** 2)
    ).astype(np.float32)

    transform = Affine(res, 0.0, ox, 0.0, -res, oy + H * res)
    dem = DEMData(elevation=elev, transform=transform, crs="EPSG:32738")

    # Delivery zones on flat ground at two corners (elevation ≈ 200 m).
    # The tall hills in between / along the approach path create unsafe regions.
    zones = [
        DeliveryZone(
            center_xy=(ox + 4400.0, oy + 4400.0),  # upper-right flat corner
            side_m=200.0,
            z_ground_m=200.0,
        ),
        DeliveryZone(
            center_xy=(ox + 600.0, oy + 600.0),    # lower-left flat corner
            side_m=200.0,
            z_ground_m=200.0,
        ),
    ]

    print("=" * 60)
    print("  Synthetic terrain obstacle extraction example")
    print(f"  DEM : {H} × {W} cells at {res:.0f} m resolution (5 km × 5 km)")
    print(f"  Clearance : {CLEARANCE_M:.0f} m   FPA : {FPA_DEG:.2f}°")
    print(f"  Delivery zones : {len(zones)}")
    print("=" * 60)

    mask = unsafe_mask(dem, zones, search_margin_m=3000.0)
    n_unsafe = int(mask.sum())
    print(f"Unsafe cells   : {n_unsafe:,}  ({100.0 * n_unsafe / mask.size:.1f}% of DEM)")

    clusters = cluster_obstacles(mask, method="connected", min_size=160,
                                 resolution_m=dem.resolution_m)
    print(f"Clusters found : {clusters.n_clusters}")

    polytopes = fit_polytopes(dem, clusters, mode="8direction")
    print(f"Polytopes      : {len(polytopes)}")
    for i, p in enumerate(polytopes):
        if len(p.vertices) == 0:
            print(f"  [{i}]  (no vertices)")
            continue
        cx, cy = p.vertices.mean(axis=0)
        span = float((p.vertices.max(axis=0) - p.vertices.min(axis=0)).max())
        print(f"  [{i}]  8-dir  centroid=({cx:.0f}, {cy:.0f})  ~{span:.0f} m span")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / "synthetic_example.png"
    fig = plot_pipeline(dem, zones, mask, clusters, polytopes,
                        title="Synthetic Example — Terrain Obstacle Polytopes")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


# ── Real-data example ─────────────────────────────────────────────────────────

def real_data_example() -> None:
    """
    Run the full pipeline on the actual Madagascar DEM tiles using the five
    mission waypoints (Runway + D1–D4).

    Tile coverage (JAXA ALPSMLC30, naming = south-edge lat):
        S023E047  →  lat 22–23°S, lon 47–48°E  (Runway)
        S024E046  →  lat 23–24°S, lon 46–47°E  (D3 Befotaka)
        S024E047  →  lat 23–24°S, lon 47–48°E  (D1, D2, D4)
    """
    datasets_dir = _REPO_ROOT / "datasets"

    tile_ids = ["S023E047", "S024E046", "S024E047"]
    dsm_paths = [datasets_dir / t / f"ALPSMLC30_{t}_DSM.tif" for t in tile_ids]
    msk_paths = [datasets_dir / t / f"ALPSMLC30_{t}_MSK.tif" for t in tile_ids]

    missing = [p for p in dsm_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing tiles: {missing}")

    print("Loading DEM tiles:", [t for t in tile_ids])
    dem = load_dem_tiles(dsm_paths, msk_paths)
    H, W = dem.elevation.shape
    print(f"  Stitched DEM : {H} × {W} cells, {dem.resolution_m:.1f} m/cell, CRS {dem.crs}")

    # Mission waypoints  (DMS → decimal, all in southern hemisphere / east longitude)
    #   Runway:  22°48'18.7"S 47°49'13.9"E
    #   D1 Ambongo:        23°27'47.3"S 47°16'03.5"E
    #   D2 Vatanato:       23°39'36.8"S 47°12'04.5"E
    #   D3 Befotaka:       23°49'21.87"S 46°59'04.26"E
    #   D4 Midongy du Sud: 23°35'24.74"S 47°01'05.09"E
    zone_specs = [
        ("Runway",  -(22 + 48/60 + 18.70/3600),  47 + 49/60 + 13.90/3600, 300.0),
        ("D1",      -(23 + 27/60 + 47.30/3600),  47 + 16/60 +  3.50/3600, 300.0),
        ("D2",      -(23 + 39/60 + 36.80/3600),  47 + 12/60 +  4.50/3600, 300.0),
        ("D3",      -(23 + 49/60 + 21.87/3600),  46 + 59/60 +  4.26/3600, 300.0),
        ("D4",      -(23 + 35/60 + 24.74/3600),  47 +  1/60 +  5.09/3600, 300.0),
    ]

    print("\nDelivery zones (UTM, EPSG:32738):")
    zones: List[DeliveryZone] = []
    labels: List[str] = []
    for name, lat, lon, side_m in zone_specs:
        z = delivery_zone_from_latlon(lat, lon, side_m, dem)
        zones.append(z)
        labels.append(name)
        print(f"  {name:8s}  utm=({z.center_xy[0]/1000:.1f}, {z.center_xy[1]/1000:.1f}) km"
              f"  z_ground={z.z_ground_m:.0f} m")

    print(f"\nComputing unsafe mask (clearance={CLEARANCE_M:.0f} m, FPA={FPA_DEG:.2f}°, margin=8 km)...")
    mask = unsafe_mask(dem, zones, search_margin_m=8000.0)
    n_unsafe = int(mask.sum())
    print(f"  Unsafe cells : {n_unsafe:,}  ({100.0 * n_unsafe / mask.size:.2f}% of full DEM)")

    clusters = cluster_obstacles(mask, method="connected", min_size=160,
                                 resolution_m=dem.resolution_m)
    print(f"  Clusters : {clusters.n_clusters}")

    polytopes = fit_polytopes(dem, clusters, mode="8direction", convex_decomp=True)
    print(f"  Polytopes: {len(polytopes)}")

    d2 = zones[2]
    d2_xy = np.array(d2.center_xy)

    print("\n  Polytope summary:")
    for i, p in enumerate(polytopes):
        if len(p.vertices) == 0:
            continue
        cx, cy = p.vertices.mean(axis=0) / 1000
        span = float((p.vertices.max(axis=0) - p.vertices.min(axis=0)).max()) / 1000
        inside = bool(np.all(p.A @ d2_xy <= p.b + 1e-3))
        flag = "  ← D2 INSIDE" if inside else ""
        print(f"    [{i}]  centroid=({cx:.1f}, {cy:.1f}) km  span≈{span:.2f} km{flag}")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig_full = plot_pipeline(
        dem, zones, mask, clusters, polytopes,
        title="Madagascar — Connected + Convex Decomposition",
        zone_labels=labels,
        zoom_to_zones=True,
        zoom_margin_km=15.0,
    )
    path_full = _OUTPUT_DIR / "result.png"
    fig_full.savefig(str(path_full), dpi=150, bbox_inches="tight")
    plt.close(fig_full)
    print(f"\n  Saved → {path_full}")

    fig_d2 = plot_pipeline(
        dem, zones, mask, clusters, polytopes,
        title="D2 zoom — Convex Decomposition",
        zone_labels=labels,
        zoom_to_zones=False,
    )
    km = 1e-3
    cx_d2, cy_d2 = d2.center_xy[0] * km, d2.center_xy[1] * km
    fig_d2.axes[0].set_xlim(cx_d2 - 5.0, cx_d2 + 5.0)
    fig_d2.axes[0].set_ylim(cy_d2 - 5.0, cy_d2 + 5.0)
    path_d2 = _OUTPUT_DIR / "result_d2zoom.png"
    fig_d2.savefig(str(path_d2), dpi=150, bbox_inches="tight")
    plt.close(fig_d2)
    print(f"  Saved → {path_d2}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m terrain_obstacle.run",
        description="Extract 2D obstacle polytopes from a DEM for STL trajectory planning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dem", nargs="+", metavar="PATH",
        help=(
            "DSM .tif path(s).  To pass matching MSK files, list all DSMs first "
            "then all MSKs in the same order (equal counts → paired automatically)."
        ),
    )
    parser.add_argument(
        "--zone", nargs=3, action="append", metavar=("LAT", "LON", "SIDE_M"),
        help="Delivery zone: latitude longitude side_length_m.  Repeat for multiple zones.",
    )
    parser.add_argument(
        "--margin", type=float, default=10_000.0,
        help="Search margin (m) added around all delivery zones (default: 10000).",
    )
    parser.add_argument(
        "--cluster", choices=["connected", "dbscan"], default="connected",
        help="Clustering method: connected-components (default) or DBSCAN.",
    )
    parser.add_argument(
        "--mode", choices=["8direction", "axis_aligned"], default="8direction",
        help="Polytope fitting mode (default: 8direction).",
    )
    parser.add_argument(
        "--convex-decomp", action="store_true",
        help="Split non-convex clusters into convex sub-pieces before fitting polytopes.",
    )
    parser.add_argument(
        "--convexity-threshold", type=float, default=0.65,
        help="Min filled-area/hull-area ratio to accept a cluster as convex (default: 0.65).",
    )
    parser.add_argument(
        "--max-k", type=int, default=6,
        help="Max k-means sub-clusters per non-convex cluster (default: 6).",
    )
    parser.add_argument(
        "--min-size", type=int, default=10,
        help="Minimum cluster size in cells; smaller clusters are discarded (default: 10).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output figure path (default: output/terrain/result.png).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run the built-in synthetic example and exit.",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Run the Madagascar real-data example (Runway + D1–D4) and exit.",
    )

    args = parser.parse_args(argv)

    if args.synthetic or (args.dem is None and args.zone is None and not args.real):
        synthetic_example()
        return

    if args.real:
        real_data_example()
        return

    if args.dem is None:
        parser.error("--dem is required (or use --synthetic).")
    if not args.zone:
        parser.error("At least one --zone is required.")

    # ── Load DEM ──────────────────────────────────────────────────────────────
    all_paths = [Path(p) for p in args.dem]
    n = len(all_paths)
    if n % 2 == 0:
        # Even count → first half DSMs, second half MSKs
        dsm_paths = all_paths[: n // 2]
        msk_paths = [p if p.exists() else None for p in all_paths[n // 2 :]]
    else:
        # Odd count → treat all as DSMs, no MSKs
        dsm_paths = all_paths
        msk_paths = [None] * n

    print(f"Loading DEM ({len(dsm_paths)} tile(s))...")
    dem = load_dem_tiles(dsm_paths, msk_paths)
    H, W = dem.elevation.shape
    print(f"  {H} × {W} cells, {dem.resolution_m:.1f} m/cell, CRS {dem.crs}")

    # ── Build delivery zones ──────────────────────────────────────────────────
    zones: List[DeliveryZone] = []
    for lat_s, lon_s, side_s in args.zone:
        z = delivery_zone_from_latlon(float(lat_s), float(lon_s), float(side_s), dem)
        zones.append(z)
        print(
            f"  Zone: utm=({z.center_xy[0]:.0f}, {z.center_xy[1]:.0f})  "
            f"side={z.side_m:.0f} m  z_ground={z.z_ground_m:.1f} m"
        )

    # ── Unsafe mask ───────────────────────────────────────────────────────────
    print(f"\nComputing unsafe mask (margin={args.margin:.0f} m, "
          f"clearance={CLEARANCE_M:.0f} m, FPA={FPA_DEG:.2f}°)...")
    mask = unsafe_mask(dem, zones, search_margin_m=args.margin)
    n_unsafe = int(mask.sum())
    print(f"  Unsafe cells : {n_unsafe:,}  ({100.0 * n_unsafe / mask.size:.2f}% of full DEM)")

    # ── Clustering ────────────────────────────────────────────────────────────
    print(f"\nClustering ({args.cluster}, min_size={args.min_size})...")
    clusters = cluster_obstacles(
        mask, method=args.cluster, min_size=args.min_size,
        resolution_m=dem.resolution_m,
    )
    print(f"  Clusters found : {clusters.n_clusters}")

    # ── Polytope fitting ──────────────────────────────────────────────────────
    print(f"\nFitting polytopes ({args.mode}, convex_decomp={args.convex_decomp})...")
    polytopes = fit_polytopes(
        dem, clusters, mode=args.mode,
        convex_decomp=args.convex_decomp,
        convexity_threshold=args.convexity_threshold,
        max_k=args.max_k,
    )
    print(f"  Polytopes : {len(polytopes)}")
    for i, p in enumerate(polytopes):
        if len(p.vertices) == 0:
            print(f"    [{i}]  (no vertices)")
            continue
        cx, cy = p.vertices.mean(axis=0)
        span = float((p.vertices.max(axis=0) - p.vertices.min(axis=0)).max())
        print(f"    [{i}]  {args.mode}  centroid=({cx:.0f}, {cy:.0f})  ~{span:.0f} m span")

    # ── Save figure ───────────────────────────────────────────────────────────
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output if args.output else _OUTPUT_DIR / "result.png"
    fig = plot_pipeline(dem, zones, mask, clusters, polytopes)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved figure → {out_path}")


if __name__ == "__main__":
    main()
