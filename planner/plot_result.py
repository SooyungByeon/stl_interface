from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def _load_dem_background(utm_origin_m: np.ndarray):
    """
    Load DEM tiles and return (elevation_array, extent_km) in local planner coords,
    where local km = (utm_m - utm_origin_m) / 1000.  Returns None on any failure.
    """
    try:
        from pathlib import Path
        _repo = Path(__file__).resolve().parent.parent
        tile_ids = ["S023E047", "S024E046", "S024E047"]
        dsm_paths = [_repo / "datasets" / t / f"ALPSMLC30_{t}_DSM.tif" for t in tile_ids]
        msk_paths = [_repo / "datasets" / t / f"ALPSMLC30_{t}_MSK.tif" for t in tile_ids]
        if not all(p.exists() for p in dsm_paths + msk_paths):
            return None
        from terrain_obstacle.dem_io import load_dem_tiles
        dem = load_dem_tiles(dsm_paths, msk_paths)
        X, Y = dem.xy_grids()
        X_km = (X - utm_origin_m[0]) / 1000.0
        Y_km = (Y - utm_origin_m[1]) / 1000.0
        extent_km = [float(X_km.min()), float(X_km.max()),
                     float(Y_km.min()), float(Y_km.max())]
        elev = dem.elevation.astype(float)
        elev[elev < -100.0] = np.nan
        return elev, extent_km
    except Exception:
        return None


def _verts_from_Ab(A, b):
    """Return ordered (N,2) vertex array for convex polytope Ax<=b, or None."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    pts = []
    for i, j in combinations(range(len(b)), 2):
        a1, a2 = A[i], A[j]
        denom = a1[0] * a2[1] - a1[1] * a2[0]
        if abs(denom) < 1e-10:
            continue
        x = (b[i] * a2[1] - b[j] * a1[1]) / denom
        y = (a1[0] * b[j] - a2[0] * b[i]) / denom
        if np.all(A @ np.array([x, y]) <= b + 1e-8):
            pts.append([x, y])
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    center = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]


def plot_result(
    X: np.ndarray,
    T_star: int,
    sample: Dict[str, Any],
    png_path: str,
) -> None:
    """
    Three-panel mission summary.
      Left  (full height) : xy top-down map
      Right-top            : altitude profile
      Right-bottom         : airspeed
    """
    env  = sample.get("env", {})
    meta = env.get("meta", {})
    ex   = int(sample.get("example_id", 0))

    obstacles = [tuple(r) for r in env.get("obstacles", [])]
    regions   = {k: tuple(v) for k, v in env.get("regions", {}).items()}

    runway_rect = regions.get("runway")
    target_rect = regions.get("target")

    steps  = np.arange(T_star + 1)
    px, py, pz = X[:, 0], X[:, 1], X[:, 2]
    vx, vy, vz = X[:, 3], X[:, 4], X[:, 5]
    v_air  = np.sqrt(vx**2 + vy**2 + vz**2)

    _dt   = int(meta.get("dt", 60))
    t_min = steps * _dt / 60.0          # step index → minutes

    v_land   = float(meta.get("v_land",      0.020))
    rwy_z_lo = float(meta.get("runway_z_lo", 0.0))
    rwy_z_hi = float(meta.get("runway_z_hi", 0.02))

    # Layout: explicit axes positions so right panels are decoupled from the
    # left map panel.  Left map gets a near-square allocation (aspect=equal
    # will further constrain it); right panels are short landscape rectangles.
    #   fig coords: [left, bottom, width, height]  (fractions of figure)
    fig = plt.figure(figsize=(16, 9))
    ax_map = fig.add_axes([0.03, 0.05, 0.53, 0.90])   # tall square-ish for map
    ax_alt = fig.add_axes([0.64, 0.55, 0.34, 0.38])   # right-top  (landscape)
    ax_spd = fig.add_axes([0.64, 0.07, 0.34, 0.38])   # right-bot  (landscape)

    fig.suptitle(
        f"Mission {ex}  —  T* = {T_star} steps  ({T_star * _dt // 60} min)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    # ── Left panel: xy top-down ───────────────────────────────────────────────────
    ax = ax_map

    # Terrain elevation background for Examples 1, 2, and 3 — same Farafangana
    # workspace + UTM origin. Each example has its own terrain cache, except
    # Example 3 which reuses Example 2's terrain (get_terrain_obstacles_ex2).
    if ex in (1, 2, 3):
        import json
        from pathlib import Path
        _terrain_ex = 2 if ex == 3 else ex
        _cache = Path(__file__).resolve().parent.parent / "output" / "terrain" / f"obstacles_ex{_terrain_ex}.json"
        if _cache.exists():
            _data = json.loads(_cache.read_text())
            _utm_origin = np.array(_data["utm_origin_m"])
            _bg = _load_dem_background(_utm_origin)
            if _bg is not None:
                _elev, _extent = _bg
                _cmap_dem = plt.cm.terrain.copy()
                _cmap_dem.set_bad("lightgrey")
                _vmin = float(np.nanpercentile(_elev, 2))
                _vmax = float(np.nanpercentile(_elev, 98))
                ax.imshow(_elev, origin="upper", extent=_extent, cmap=_cmap_dem,
                          vmin=_vmin, vmax=_vmax, aspect="auto",
                          interpolation="nearest", zorder=0)

    for j, (x1, x2, y1, y2) in enumerate(obstacles):
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor="salmon", edgecolor="darkred", alpha=0.50, linewidth=1,
            label="Obstacle" if j == 0 else None,
        ))

    if runway_rect is not None:
        x1, x2, y1, y2 = runway_rect
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor="royalblue", edgecolor="blue", alpha=0.25,
            linewidth=1.5, linestyle="--",
            label="Runway",
        ))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, "RWY",
                ha="center", va="center", fontsize=8, color="blue", fontweight="bold")

    if target_rect is not None:
        x1, x2, y1, y2 = target_rect
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor="limegreen", edgecolor="green", alpha=0.25,
            linewidth=1.5, linestyle="--",
            label="Target",
        ))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, "TGT",
                ha="center", va="center", fontsize=8, color="darkgreen", fontweight="bold")

    # Loiter zones T1/T2 (ex2)
    _loiter_colors = {"T1": ("gold", "goldenrod"), "T2": ("lightsalmon", "darkorange")}
    for rname, (fc, ec) in _loiter_colors.items():
        circle_info = meta.get(f"{rname}_circle")
        if circle_info is not None:
            cx, cy = float(circle_info["center"][0]), float(circle_info["center"][1])
            r = float(circle_info["radius"])
            ax.add_patch(mpatches.Circle(
                (cx, cy), r, facecolor=fc, edgecolor=ec, alpha=0.25, linewidth=1.5,
                linestyle="--",
            ))
            ax.text(cx, cy, rname, ha="center", va="center", fontsize=9,
                    color=ec, fontweight="bold")
        else:
            lrect = regions.get(rname)
            if lrect is not None:
                x1, x2, y1, y2 = lrect
                ax.add_patch(mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    facecolor=fc, edgecolor=ec, alpha=0.25, linewidth=1.5, linestyle="--",
                ))
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, rname,
                        ha="center", va="center", fontsize=9, color=ec, fontweight="bold")

    # Delivery zones D1–D4
    # Single bright gold on top-down (contrasts terrain greens/browns/blues);
    # per-zone colors used in altitude panel.
    _DLV_FC   = "#FFD700"   # gold fill (top-down)
    _DLV_EC   = "#B8860B"   # dark-goldenrod edge
    _DLV_COLORS = {         # per-zone colors for altitude shading
        "D1": "#e74c3c",    # red
        "D2": "#3498db",    # blue
        "D3": "#2ecc71",    # green
        "D4": "#f39c12",    # orange
    }
    _dlv_names = ["D1", "D2", "D3", "D4"]
    import matplotlib.patheffects as _pe
    for rname in _dlv_names:
        drect = regions.get(rname)
        if drect is not None:
            x1, x2, y1, y2 = drect
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            # Actual zone box (tiny at map scale, kept as reference)
            ax.add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                facecolor=_DLV_FC, edgecolor=_DLV_EC, alpha=0.70, linewidth=2.0,
                zorder=4,
            ))
            # Large star marker so it's visible at full-map scale
            ax.plot(cx, cy, "*", markersize=18, color=_DLV_FC,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=6)
            # Bold label with black stroke outline
            ax.text(cx, cy + 2.5, rname,
                    ha="center", va="bottom", fontsize=13, color="white",
                    fontweight="bold", zorder=7,
                    path_effects=[_pe.withStroke(linewidth=2.5, foreground="black")])

    # Trajectory colored by mission time — plasma colormap + white halo for visibility
    n = len(px)
    cmap_traj = plt.cm.plasma
    seg_colors = cmap_traj(np.linspace(0, 1, n - 1))
    for i in range(n - 1):
        ax.plot(px[i:i+2], py[i:i+2], color="white", linewidth=3.5, zorder=8)
        ax.plot(px[i:i+2], py[i:i+2], color=seg_colors[i], linewidth=2.0, zorder=9)

    sm = plt.cm.ScalarMappable(cmap=cmap_traj,
                               norm=plt.Normalize(vmin=0, vmax=T_star * _dt / 60))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mission time (min)", fontsize=8)

    ax.plot(px[0], py[0], "o", markersize=9, color="limegreen", zorder=5,
            markeredgecolor="darkgreen", markeredgewidth=1.2, label="Start")

    k_land = None
    if runway_rect is not None:
        rx1, rx2, ry1, ry2 = runway_rect
        for k in range(1, T_star + 1):
            if (rx1 <= px[k] <= rx2) and (ry1 <= py[k] <= ry2) and v_air[k] <= v_land * 1.5:
                k_land = k
                break
    if k_land is not None:
        ax.plot(px[k_land], py[k_land], "s", markersize=9, color="orange", zorder=5,
                markeredgecolor="darkorange", markeredgewidth=1.2, label="End")

    # Auto-fit square axes to all content
    _xs = list(px)
    _ys = list(py)
    for _r in list(obstacles) + list(regions.values()):
        _xs += [_r[0], _r[1]]
        _ys += [_r[2], _r[3]]
    _pad  = max((max(_xs) - min(_xs)) * 0.07, (max(_ys) - min(_ys)) * 0.07, 2.0)
    _xcen = (min(_xs) + max(_xs)) / 2.0
    _ycen = (min(_ys) + max(_ys)) / 2.0
    _half = max(max(_xs) - min(_xs), max(_ys) - min(_ys)) / 2.0 + _pad
    ax.set_xlim(_xcen - _half, _xcen + _half)
    ax.set_ylim(_ycen - _half, _ycen + _half)
    ax.set_aspect("equal", adjustable="box")

    path_km = float(np.sum(np.sqrt(np.diff(px)**2 + np.diff(py)**2)))
    ax.set_xlabel("px  (km)")
    ax.set_ylabel("py  (km)")
    ax.set_title(f"Top-down  (path={path_km:.0f} km, color=time)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── Right-top: altitude profile ───────────────────────────────────────────────
    ax = ax_alt
    ax.plot(t_min, pz, "-", color="steelblue", linewidth=1.8, label="pz")
    ax.fill_between(t_min, rwy_z_lo, rwy_z_hi, alpha=0.25, color="royalblue",
                    label=f"runway  z ∈ [{rwy_z_lo:.3f}, {rwy_z_hi:.3f}] km")

    cruise_z    = meta.get("cruise_z_ref")
    dlv_z_hi    = meta.get("delivery_z_hi")
    loiter_z_lo = meta.get("loiter_z_lo")
    loiter_z_hi = meta.get("loiter_z_hi")

    if cruise_z is not None:
        ax.axhline(float(cruise_z), color="darkorange", linewidth=1.2, linestyle="--",
                   label=f"cruise ref  {float(cruise_z):.2f} km")
    if loiter_z_lo is not None and loiter_z_hi is not None:
        ax.axhspan(float(loiter_z_lo), float(loiter_z_hi), alpha=0.30, color="gold",
                   label=f"loiter  z ∈ [{float(loiter_z_lo):.2f}, {float(loiter_z_hi):.2f}] km")
    if dlv_z_hi is not None:
        dlv_z_hi_f = float(dlv_z_hi)
        ax.axhline(dlv_z_hi_f, color="limegreen", linewidth=1.2, linestyle="--",
                   label=f"delivery ceiling  {dlv_z_hi_f:.3f} km")

    for dname in _dlv_names:
        drect = regions.get(dname)
        if drect is not None:
            dx1, dx2, dy1, dy2 = drect
            in_zone = [k for k in range(T_star + 1)
                       if dx1 <= px[k] <= dx2 and dy1 <= py[k] <= dy2]
            if in_zone:
                k_lo, k_hi = in_zone[0], in_zone[-1]
                _dcolor = _DLV_COLORS.get(dname, _DLV_FC)
                ax.axvspan(t_min[k_lo], t_min[k_hi], alpha=0.30, color=_dcolor,
                           label=f"in {dname}")
                ax.plot([t_min[k_lo], t_min[k_hi]], [pz[k_lo], pz[k_hi]], "D",
                        markersize=5, color=_dcolor, zorder=5)

    ax.set_xlabel("Mission time (min)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── Right-bottom: airspeed ────────────────────────────────────────────────────
    v_max      = float(meta.get("v_max", 0.040))
    _v_vert    = float(sample.get("bounds", {}).get("x_max", [0.0] * 6)[5])
    _v_L2_max  = math.sqrt((v_max / math.cos(math.pi / 12)) ** 2 + _v_vert ** 2)

    ax = ax_spd
    ax.plot(t_min, v_air, "-", color="darkorange", linewidth=1.8, label="‖v‖₂")
    ax.axhline(v_land, color="steelblue", linestyle="-.", linewidth=1.2,
               label=f"v_land={v_land*1000:.0f} m/s")
    ax.axhline(_v_L2_max, color="black", linestyle="--", linewidth=1.4,
               label=f"max ‖v‖₂={_v_L2_max*1000:.1f} m/s")
    ax.set_xlabel("Mission time (min)")
    ax.set_ylabel("‖v‖₂  (km/s)")
    ax.set_title("Airspeed  (3-D L2)")
    ax.legend(fontsize=7)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
