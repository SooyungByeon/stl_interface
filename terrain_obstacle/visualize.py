from __future__ import annotations

from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .cluster import ClusterResult
from .cone import DeliveryZone
from .dem_io import DEMData
from .polytope import Polytope

_DZ_COLOR   = "red"      # delivery zone borders and markers
_POLY_COLOR = "#1a1a1a" # polytope outline colour
_KM = 1e-3              # metres → kilometres for display


def plot_pipeline(
    dem: DEMData,
    zones: List[DeliveryZone],
    mask: np.ndarray,
    cluster_result: ClusterResult,
    polytopes: List[Polytope],
    ax: Optional[plt.Axes] = None,
    title: str = "Terrain Obstacle Polytopes",
    zone_labels: Optional[List[str]] = None,
    zoom_to_zones: bool = False,
    zoom_margin_km: float = 15.0,
) -> plt.Figure:
    """
    Single-panel overlay figure.  All axes are in kilometres.

    Layers (bottom → top):
      1. DEM elevation heatmap (terrain colourmap; JAXA ocean nodata masked to grey)
      2. Unsafe-cell mask (semi-transparent red)
      3. Cluster point scatter (tab10 colours)
      4. Polytope polygon outlines (Set1 colours)
      5. Delivery zone squares (cyan-green fill + outline)

    Parameters
    ----------
    zone_labels : Optional list of strings (same length as zones) drawn next to
                  each delivery zone centre.
    zoom_to_zones : If True, set axis limits to the bounding box of all zones
                    plus zoom_margin_km on each side.
    zoom_margin_km : Margin added around the zone bounding box when zoom_to_zones=True.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    else:
        fig = ax.get_figure()

    X, Y = dem.xy_grids()
    # All display coordinates in km
    X_km = X * _KM
    Y_km = Y * _KM
    extent_km = [float(X_km.min()), float(X_km.max()),
                 float(Y_km.min()), float(Y_km.max())]

    # ── 1. DEM heatmap ────────────────────────────────────────────────────────
    elev = dem.elevation.astype(np.float64).copy()
    # Mask JAXA ocean/nodata fill (-9999): anything below -100 m is not real terrain
    elev[elev < -100.0] = np.nan

    cmap = plt.cm.terrain.copy()
    cmap.set_bad("lightgrey")   # NaN / nodata shown as light grey

    vmin = float(np.nanpercentile(elev, 2))   # clip low outliers
    vmax = float(np.nanpercentile(elev, 98))  # clip high outliers

    im = ax.imshow(
        elev,
        origin="upper",
        extent=extent_km,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.65, pad=0.02)

    # ── 2. Unsafe-cell mask ───────────────────────────────────────────────────
    if mask.any():
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        mask_rgba[mask, 0] = 1.0   # R
        mask_rgba[mask, 3] = 0.50  # alpha
        ax.imshow(
            mask_rgba,
            origin="upper",
            extent=extent_km,
            aspect="equal",
            interpolation="nearest",
        )

    # ── 3. Cluster scatter ────────────────────────────────────────────────────
    if cluster_result.n_clusters > 0:
        cmap_c = plt.get_cmap("tab10")
        for lab in range(1, cluster_result.n_clusters + 1):
            cell_mask = cluster_result.labels == lab
            xs_km = X_km[cell_mask]
            ys_km = Y_km[cell_mask]
            color = cmap_c((lab - 1) % 10)
            ax.scatter(xs_km, ys_km, color=[color], s=2,
                       alpha=0.7, linewidths=0, zorder=3)

    # ── 4. Obstacle bounding rectangles ──────────────────────────────────────
    for poly in polytopes:
        if len(poly.vertices) < 2:
            continue
        verts_km = poly.vertices * _KM
        xmin, ymin = verts_km.min(axis=0)
        xmax, ymax = verts_km.max(axis=0)
        ax.add_patch(mpatches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1.5, edgecolor=_POLY_COLOR, facecolor=_POLY_COLOR,
            alpha=0.30, zorder=5,
        ))

    # ── 5. Delivery zone squares ──────────────────────────────────────────────
    for idx, zone in enumerate(zones):
        cx_km = zone.center_xy[0] * _KM
        cy_km = zone.center_xy[1] * _KM
        half_km = (zone.side_m / 2.0) * _KM

        ax.add_patch(mpatches.Rectangle(
            (cx_km - half_km, cy_km - half_km),
            2 * half_km, 2 * half_km,
            linewidth=2.5, edgecolor=_DZ_COLOR, facecolor=_DZ_COLOR, alpha=0.20, zorder=6,
        ))
        ax.plot(cx_km, cy_km, "+",
                color=_DZ_COLOR, markersize=10, markeredgewidth=2.5, zorder=8)

        if zone_labels and idx < len(zone_labels):
            ax.annotate(
                zone_labels[idx],
                xy=(cx_km, cy_km),
                xytext=(6, 4), textcoords="offset points",
                fontsize=8, fontweight="bold", color="red",
                zorder=9,
            )

    # ── Zoom to zones ─────────────────────────────────────────────────────────
    if zoom_to_zones and zones:
        cx_all = [z.center_xy[0] * _KM for z in zones]
        cy_all = [z.center_xy[1] * _KM for z in zones]
        m = zoom_margin_km
        ax.set_xlim(min(cx_all) - m, max(cx_all) + m)
        ax.set_ylim(min(cy_all) - m, max(cy_all) + m)

    # ── Labels & legend ───────────────────────────────────────────────────────
    ax.set_xlabel("UTM Easting (km)")
    ax.set_ylabel("UTM Northing (km)")
    ax.set_title(title)

    legend_handles = [
        mpatches.Patch(facecolor="red", alpha=0.50, label="Unsafe terrain"),
        mpatches.Patch(facecolor=_DZ_COLOR, alpha=0.30,
                       edgecolor=_DZ_COLOR, linewidth=2,
                       label="Delivery zone"),
    ]
    if polytopes:
        legend_handles.append(
            mpatches.Patch(facecolor="none", edgecolor=_POLY_COLOR,
                           linewidth=2, label=f"Obstacles / bounding rect ({len(polytopes)})")
        )
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8, framealpha=0.7)

    fig.tight_layout()
    return fig
