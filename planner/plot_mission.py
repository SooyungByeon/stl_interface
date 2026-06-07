"""
Mission visualization for the paper (LA) examples.

Mirrors the framework in planner/plot_result.py:
  - Left  (tall) : top-down xy map over the DEM terrain background, trajectory
                   coloured by mission time, with a self-contained legend.
  - Right-top    : altitude profile vs time.
  - Right-bottom : horizontal airspeed vs time (with v_min floor / v_max cap).

The top-down axes use the fixed la_frame.VIEW_XLIM/VIEW_YLIM with equal aspect,
so the axis ranges and ratio are identical for every mission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from . import la_frame as la
from .terrain_bg import load_la_dem as _load_la_dem


# Per-role styling for region rectangles. Runway is blue so it is clearly
# distinct from the salmon/red obstacles.
_ROLE_STYLE = {
    "keep":     dict(facecolor="gold",           edgecolor="goldenrod", ls="--", label="Keep region (loiter)"),
    "runway":   dict(facecolor="royalblue",      edgecolor="navy",      ls="-",  label="Runway (goal)"),
    "delivery": dict(facecolor="limegreen",      edgecolor="darkgreen", ls="--", label="Delivery point"),
    "key":      dict(facecolor="mediumseagreen", edgecolor="darkgreen", ls="--", label="Key"),
    "door":     dict(facecolor="orange",         edgecolor="darkorange", ls="--", label="Door (avoid until key)"),
}


def plot_mission(
    X: np.ndarray,
    T: int,
    mission_id: int,
    *,
    dt: float,
    regions: Sequence[Dict[str, Any]],   # each: {name, rect, role in keep|runway|delivery}
    obstacles: Sequence[Sequence[float]],
    v_min: float,
    v_max: float,
    cruise_z: float,
    png_path: str,
    landing_z: Optional[Sequence[float]] = None,    # (z_lo, z_hi) km landing band
    delivery_z: Optional[Sequence[float]] = None,   # (z_lo, z_hi) km delivery band
    subtitle: Optional[str] = None,
) -> None:
    px, py, pz = X[:, 0], X[:, 1], X[:, 2]
    vx, vy = X[:, 3], X[:, 4]
    v_horiz = np.hypot(vx, vy)
    steps = np.arange(T + 1)
    t_min = steps * float(dt) / 60.0

    # Compact double-column layout (minimal margins):
    #   top  : top-down map (wide) + two vertical colorbars on its right
    #   bottom-left  : altitude     bottom-right : horizontal speed
    fig = plt.figure(figsize=(7.6, 7.2))
    ax_map = fig.add_axes([0.085, 0.455, 0.66, 0.46])       # top map
    cax_ter = fig.add_axes([0.760, 0.455, 0.018, 0.46])     # terrain colorbar (vertical)
    cax_time = fig.add_axes([0.880, 0.455, 0.018, 0.46])    # time colorbar (vertical)
    ax_alt = fig.add_axes([0.095, 0.075, 0.37, 0.275])      # bottom-left
    ax_spd = fig.add_axes([0.595, 0.075, 0.37, 0.275])      # bottom-right

    ttl = f"Mission {mission_id}   —   T* = {T} steps ({int(T * dt // 60)} min)"
    if subtitle:
        ttl += f"   ·   {subtitle}"
    fig.suptitle(ttl, fontsize=12, fontweight="bold", y=0.975)

    # ── Left: top-down map ────────────────────────────────────────────────
    ax = ax_map
    bg = _load_la_dem()
    if bg is not None:
        elev, extent = bg
        cmap = plt.cm.terrain.copy()
        cmap.set_bad("lightgrey")
        vmin = float(np.nanpercentile(elev, 2))
        vmax = float(np.nanpercentile(elev, 98))
        im = ax.imshow(elev, origin="upper", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto",
                       interpolation="nearest", zorder=0)
        cb = fig.colorbar(im, cax=cax_ter, orientation="vertical")
        cb.set_label("Terrain elevation (m)", fontsize=9)
    else:
        cax_ter.axis("off")

    # obstacles
    for j, (x1, x2, y1, y2) in enumerate(obstacles):
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, facecolor="salmon", edgecolor="darkred",
            alpha=0.55, linewidth=1.2, zorder=3,
            label="Obstacle" if j == 0 else None))

    # mission regions (each carries either "rect" or "circle"=(cx,cy,r))
    seen_roles = set()
    for r in regions:
        role = r["role"]
        st = _ROLE_STYLE[role]
        lab = st["label"] if role not in seen_roles else None
        seen_roles.add(role)
        alpha = 0.45 if role == "runway" else 0.35
        lw = 2.2 if role == "runway" else 2.0
        if "circle" in r:
            cx, cy, rad = r["circle"]
            ax.add_patch(mpatches.Circle(
                (cx, cy), rad, facecolor=st["facecolor"], edgecolor=st["edgecolor"],
                alpha=alpha, linewidth=lw, linestyle=st["ls"], zorder=4, label=lab))
        else:
            x1, x2, y1, y2 = r["rect"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, facecolor=st["facecolor"],
                edgecolor=st["edgecolor"], alpha=alpha, linewidth=lw,
                linestyle=st["ls"], zorder=4, label=lab))
        # label at the region centre (with white halo for contrast)
        ax.text(cx, cy, r["name"], ha="center", va="center", fontsize=11,
                color="black", fontweight="bold", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    # trajectory coloured by time, white halo for contrast
    n = len(px)
    seg = plt.cm.plasma(np.linspace(0, 1, max(n - 1, 1)))
    for i in range(n - 1):
        ax.plot(px[i:i+2], py[i:i+2], color="white", linewidth=4.0, zorder=8)
        ax.plot(px[i:i+2], py[i:i+2], color=seg[i], linewidth=2.2, zorder=9)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                               norm=plt.Normalize(0, T * dt / 60.0))
    sm.set_array([])
    cb2 = fig.colorbar(sm, cax=cax_time, orientation="vertical")
    cb2.set_label("Mission time (min)", fontsize=9)

    ax.plot(px[0], py[0], "o", markersize=11, color="limegreen",
            markeredgecolor="black", markeredgewidth=1.2, zorder=10, label="Start")
    ax.plot(px[-1], py[-1], "s", markersize=11, color="red",
            markeredgecolor="black", markeredgewidth=1.2, zorder=10, label="End")

    ax.set_xlim(*la.VIEW_XLIM)
    ax.set_ylim(*la.VIEW_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East  px (km)", fontsize=11)
    ax.set_ylabel("North  py (km)", fontsize=11)
    ax.set_title("Trajectory over terrain (Los Angeles)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    # ── Right-top: altitude ───────────────────────────────────────────────
    ax = ax_alt
    ax.plot(t_min, pz, "-", color="steelblue", linewidth=1.8, label="altitude pz")
    ax.axhline(cruise_z, color="darkorange", linewidth=1.2, linestyle="--",
               label=f"cruise ref {cruise_z:.2f} km")
    if delivery_z is not None:
        ax.axhspan(float(delivery_z[0]), float(delivery_z[1]), alpha=0.25, color="limegreen",
                   label=f"delivery band [{float(delivery_z[0]):.2f}, {float(delivery_z[1]):.2f}] km")
    if landing_z is not None:
        ax.axhspan(float(landing_z[0]), float(landing_z[1]), alpha=0.30, color="red",
                   label=f"landing band [{float(landing_z[0]):.2f}, {float(landing_z[1]):.2f}] km")
    ax.set_xlabel("Mission time (min)", fontsize=10)
    ax.set_ylabel("Altitude (km)", fontsize=10)
    ax.set_title("Altitude profile", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    # ── Right-bottom: horizontal airspeed ─────────────────────────────────
    ax = ax_spd
    # The v_max cap is a 24-facet polygon circumscribing the speed circle, so the
    # achievable horizontal speed reaches v_max / cos(pi/24) at the corners.
    v_cap = v_max / float(np.cos(np.pi / 24))
    ax.plot(t_min, v_horiz, "-", color="darkorange", linewidth=1.8,
            label="horizontal speed ‖(vx,vy)‖")
    ax.axhline(v_min, color="firebrick", linestyle="-.", linewidth=1.3,
               label=f"v_min floor {v_min*1000:.0f} m/s")
    ax.axhline(v_cap, color="black", linestyle="--", linewidth=1.3,
               label=f"v_max cap {v_cap*1000:.1f} m/s")
    ax.set_ylim(0.0, v_cap * 1.18)
    ax.set_xlabel("Mission time (min)", fontsize=10)
    ax.set_ylabel("Speed (km/s)", fontsize=10)
    ax.set_title("Horizontal airspeed", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.4)

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_mission"]
