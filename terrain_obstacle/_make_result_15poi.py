"""One-off: regenerate output/terrain/result.png with ALL 15 Ex2 delivery POIs.

Mirrors get_terrain_obstacles_ex2 (same tiles, same pipeline params) so the
terrain obstacles shown match Example 2 exactly, but overlays Runway + the
full 15-POI delivery set used by EX2_POI_TABLE.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from terrain_obstacle.dem_io import load_dem_tiles
from terrain_obstacle.cone import delivery_zone_from_latlon, unsafe_mask, DeliveryZone
from terrain_obstacle.cluster import cluster_obstacles
from terrain_obstacle.polytope import fit_polytopes
from terrain_obstacle.visualize import plot_pipeline

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "output" / "terrain"

# Must match planner_interface._ZONE_SPECS_EX2 (Runway + 15 POIs) and the
# terrain pipeline parameters used for Ex2.
_TILE_IDS = ["S023E047", "S024E046", "S024E047"]
_ZONE_SPECS_EX2 = [
    ("Runway",              -(22 + 48/60 + 18.70/3600),  47 + 49/60 + 13.90/3600, 300.0),
    ("Ambongo",             -(23 + 27/60 + 47.30/3600),  47 + 16/60 +  3.50/3600, 300.0),
    ("Vatanato",            -(23 + 39/60 + 36.80/3600),  47 + 12/60 +  4.50/3600, 300.0),
    ("Befotaka",            -(23 + 49/60 + 21.87/3600),  46 + 59/60 +  4.26/3600, 300.0),
    ("Midongy_du_Sud",      -(23 + 35/60 + 24.74/3600),  47 +  1/60 +  5.09/3600, 300.0),
    ("Karimbary",           -(23 + 11/60 + 56.7/3600),   47 + 18/60 +  3.6/3600,  300.0),
    ("Bevata",              -(23 + 18/60 + 55.0/3600),   47 + 16/60 + 47.5/3600,  300.0),
    ("Ianakasy_Mahatsinjo", -(22 + 51/60 + 15.1/3600),   47 + 26/60 + 11.2/3600,  300.0),
    ("Andremareo",          -(23 +  1/60 + 56.7/3600),   47 + 25/60 + 38.0/3600,  300.0),
    ("Ambohimana",          -(22 + 35/60 + 12.2/3600),   47 + 18/60 + 37.1/3600,  300.0),
    ("Mahasoa_I",           -(22 + 29/60 +  0.9/3600),   47 + 33/60 + 21.7/3600,  300.0),
    ("Vohibe2",             -(22 + 40/60 +  6.8/3600),   47 + 24/60 + 59.2/3600,  300.0),
    ("Antemanara",          -(23 +  0/60 + 24.8/3600),   47 + 23/60 + 59.3/3600,  300.0),
    ("Manato",              -(22 + 24/60 +  3.2/3600),   47 + 30/60 + 34.3/3600,  300.0),
    ("Ambaramafaitra",      -(22 + 30/60 + 58.9/3600),   47 + 20/60 + 21.0/3600,  300.0),
    ("Bemandresy",          -(22 + 39/60 + 11.9/3600),   47 + 34/60 +  4.1/3600,  300.0),
]
_MARGIN_M, _MIN_SIZE, _MODE = 8000.0, 300, "8direction"
_CONVEX_DECOMP, _CONVEXITY_THR, _MAX_K = True, 0.65, 3


def main() -> None:
    datasets_dir = _REPO_ROOT / "datasets"
    dsm = [datasets_dir / t / f"ALPSMLC30_{t}_DSM.tif" for t in _TILE_IDS]
    msk = [datasets_dir / t / f"ALPSMLC30_{t}_MSK.tif" for t in _TILE_IDS]

    print(f"Loading DEM tiles {_TILE_IDS} ...")
    dem = load_dem_tiles(dsm, msk)

    zones: List[DeliveryZone] = []
    labels: List[str] = []
    for name, lat, lon, side_m in _ZONE_SPECS_EX2:
        zones.append(delivery_zone_from_latlon(lat, lon, side_m, dem))
        labels.append(name)

    print("Computing unsafe mask (15 POIs + Runway) ...")
    mask = unsafe_mask(dem, zones, search_margin_m=_MARGIN_M)
    clusters = cluster_obstacles(mask, method="connected", min_size=_MIN_SIZE,
                                 resolution_m=dem.resolution_m)
    polytopes = fit_polytopes(dem, clusters, mode=_MODE,
                              convex_decomp=_CONVEX_DECOMP,
                              convexity_threshold=_CONVEXITY_THR, max_k=_MAX_K)
    print(f"  {clusters.n_clusters} clusters, {len(polytopes)} polytopes")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full map (Runway + 15 POIs).
    fig = _draw_scene(
        dem, zones, labels, polytopes,
        title="Madagascar (Farafangana) — Runway + 15 Delivery POIs",
    )
    out = _OUTPUT_DIR / "result.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

    # ── Vatanato (D2) zoom — original real_data_example look (grid + scatter +
    # mask + colorbar via plot_pipeline). Reproduces the 5-zone pipeline
    # (min_size=160) so the figure matches the previously committed image.
    # Only two things are changed: legend "Delivery zone" -> "Mission point",
    # and the zoomed zone is annotated "Vatanato" instead of "D2".
    zones5 = zones[:5]
    labels5 = ["Runway", "D1", "Vatanato", "D3", "D4"]
    mask5 = unsafe_mask(dem, zones5, search_margin_m=8000.0)
    clusters5 = cluster_obstacles(mask5, method="connected", min_size=160,
                                  resolution_m=dem.resolution_m)
    polytopes5 = fit_polytopes(dem, clusters5, mode="8direction", convex_decomp=True)
    print(f"  [d2zoom] {clusters5.n_clusters} clusters, {len(polytopes5)} polytopes")

    fig_z = plot_pipeline(
        dem, zones5, mask5, clusters5, polytopes5,
        title="D2 zoom — Convex Decomposition",
        zone_labels=labels5,
        zoom_to_zones=False,
    )
    ax_z = fig_z.axes[0]
    cx_km, cy_km = zones5[2].center_xy[0] * 1e-3, zones5[2].center_xy[1] * 1e-3
    ax_z.set_xlim(cx_km - 5.0, cx_km + 5.0)
    ax_z.set_ylim(cy_km - 5.0, cy_km + 5.0)

    # Relabel legend: "Delivery zone" -> "Mission point" (keep other entries).
    ax_z.legend(handles=[
        mpatches.Patch(facecolor="red", alpha=0.50, label="Unsafe terrain"),
        mpatches.Patch(facecolor="red", alpha=0.30, edgecolor="red",
                       linewidth=2, label="Mission point"),
        mpatches.Patch(facecolor="none", edgecolor="#1a1a1a", linewidth=2,
                       label=f"Obstacles / bounding rect ({len(polytopes5)})"),
    ], loc="upper right", fontsize=8, framealpha=0.7)

    out_z = _OUTPUT_DIR / "result_d2zoom.png"
    fig_z.savefig(str(out_z), dpi=150, bbox_inches="tight")
    plt.close(fig_z)
    print(f"Saved -> {out_z}")


# ── Plot helper ──────────────────────────────────────────────────────────────
_KM = 1e-3
_OBS_COLOR = "magenta"   # high-contrast obstacle fill (clear, low transparency)


def _draw_scene(dem, zones, labels, polytopes, *, title, xlim=None, ylim=None):
    """DEM background + clear obstacle rects + red mission crosses/labels.

    When xlim/ylim are given (zoom), only mission points inside the view are
    drawn and labelled. Legend is identical across full and zoom figures.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    X, Y = dem.xy_grids()
    extent_km = [float((X * _KM).min()), float((X * _KM).max()),
                 float((Y * _KM).min()), float((Y * _KM).max())]

    elev = dem.elevation.astype(np.float64).copy()
    elev[elev < -100.0] = np.nan          # mask JAXA ocean/nodata
    cmap = plt.cm.terrain.copy()
    cmap.set_bad("lightgrey")
    vmin = float(np.nanpercentile(elev, 2))
    vmax = float(np.nanpercentile(elev, 98))
    im = ax.imshow(elev, origin="upper", extent=extent_km, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.65, pad=0.02)

    # Terrain obstacles — clear colour, low transparency, thick dark edge.
    for poly in polytopes:
        if len(poly.vertices) < 2:
            continue
        verts_km = poly.vertices * _KM
        xmin, ymin = verts_km.min(axis=0)
        xmax, ymax = verts_km.max(axis=0)
        ax.add_patch(mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2.0, edgecolor="black", facecolor=_OBS_COLOR,
            alpha=0.75, zorder=5,
        ))

    def _in_view(cx, cy):
        if xlim is None or ylim is None:
            return True
        return (xlim[0] <= cx <= xlim[1]) and (ylim[0] <= cy <= ylim[1])

    # Mission points — red crosses + name labels.
    for idx, zone in enumerate(zones):
        cx_km = zone.center_xy[0] * _KM
        cy_km = zone.center_xy[1] * _KM
        if not _in_view(cx_km, cy_km):
            continue
        ax.plot(cx_km, cy_km, "+", color="red", markersize=11,
                markeredgewidth=2.5, zorder=8)
        ax.annotate(labels[idx], xy=(cx_km, cy_km),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="red", zorder=9)

    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        cx_all = [z.center_xy[0] * _KM for z in zones]
        cy_all = [z.center_xy[1] * _KM for z in zones]
        m = 12.0
        ax.set_xlim(min(cx_all) - m, max(cx_all) + m)
        ax.set_ylim(min(cy_all) - m, max(cy_all) + m)

    ax.set_xlabel("UTM Easting (km)")
    ax.set_ylabel("UTM Northing (km)")
    ax.set_title(title)
    ax.legend(handles=[
        mlines.Line2D([], [], color="red", marker="+", linestyle="None",
                      markersize=11, markeredgewidth=2.5, label="Mission point"),
        mpatches.Patch(facecolor=_OBS_COLOR, alpha=0.75, edgecolor="black",
                       linewidth=2, label=f"Terrain obstacle ({len(polytopes)})"),
    ], loc="upper right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
