# trajectory_csv.py
"""
Export planned aircraft trajectories to universal CSV files.

Coordinate frame : local ENU — East = +x_m, North = +y_m, Up = +z_m
Heading convention: North = 0°, East = +90°, West = -90°, South = ±180°
Acceleration     : kinematic d²pos/dt² in m/s²  (gravity implicitly balanced)
Interpolation    : piecewise linear on planner position waypoints (60 s resolution);
                   velocity and acceleration are numerical gradients of the
                   interpolated positions.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

import numpy as np


_G_MPS2 = 9.80665

_DELIVERY_ZONES = ("D1", "D2", "D3")
_LOITER_ZONES   = ("T1", "T2")

CSV_HEADER: List[str] = [
    "time_s", "phase",
    "x_m", "y_m", "z_m",
    "vx_mps", "vy_mps", "vz_mps",
    "ax_mps2", "ay_mps2", "az_mps2",
    "heading_deg", "gamma_deg", "roll_deg",
    "yawrate_degps", "pitchrate_degps",
    "airspeed_mps", "horiz_speed_mps",
]


def _assign_phases(
    pos_km: np.ndarray,
    vel_kmps: np.ndarray,
    sample: Dict[str, Any],
) -> List[str]:
    env     = sample.get("env", {})
    meta    = env.get("meta", {}) or {}
    regions = {k: tuple(v) for k, v in env.get("regions", {}).items()}

    N  = len(pos_km)
    px, py, pz = pos_km[:, 0], pos_km[:, 1], pos_km[:, 2]
    vz    = vel_kmps[:, 2]
    v_air = np.linalg.norm(vel_kmps, axis=1)

    v_land   = float(meta.get("v_land",      0.020))
    cruise_z = float(meta.get("cruise_z_ref", 1.20))

    phases = np.full(N, "cruise", dtype=object)

    near_cruise = np.where(pz >= cruise_z * 0.90)[0]
    if len(near_cruise) > 0:
        phases[: near_cruise[0]] = "climb"

    is_default = (phases == "cruise") | (phases == "climb")
    phases[(vz < -1e-4) & (pz < cruise_z * 0.70) & is_default] = "approach"

    for dname in _DELIVERY_ZONES:
        rect = regions.get(dname)
        if rect is None:
            continue
        x1, x2, y1, y2 = rect
        phases[(x1 <= px) & (px <= x2) & (y1 <= py) & (py <= y2)] = f"delivery_{dname}"

    for lname in _LOITER_ZONES:
        circle = meta.get(f"{lname}_circle")
        rect   = regions.get(lname)
        if circle is not None:
            cx = float(circle["center"][0])
            cy = float(circle["center"][1])
            r  = float(circle["radius"])
            m  = (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2
        elif rect is not None:
            x1, x2, y1, y2 = rect
            m = (x1 <= px) & (px <= x2) & (y1 <= py) & (py <= y2)
        else:
            continue
        phases[m] = f"loiter_{lname}"

    target = regions.get("target")
    if target is not None:
        x1, x2, y1, y2 = target
        phases[(x1 <= px) & (px <= x2) & (y1 <= py) & (py <= y2)] = "target"

    runway = regions.get("runway")
    if runway is not None:
        rx1, rx2, ry1, ry2 = runway
        in_box = (rx1 <= px) & (px <= rx2) & (ry1 <= py) & (py <= ry2)
        phases[in_box & (v_air <= v_land * 1.5)] = "landing"

    return phases.tolist()


def export_trajectory_csv(
    X: np.ndarray,
    U: np.ndarray,
    T_star: int,
    sample: Dict[str, Any],
    csv_path: str,
    dt_out: float = 1.0,
) -> None:
    """
    Resample a planned trajectory to dt_out-second resolution and write CSV.

    Parameters
    ----------
    X        : (T_star+1, 6) state array in km / km·s⁻¹  [px, py, pz, vx, vy, vz]
    U        : (T_star, 3)   control array in km·s⁻²  (unused; accel derived from pos)
    T_star   : number of planner steps
    sample   : planning sample dict with env.regions and env.meta
    csv_path : output file path (parent dirs created automatically)
    dt_out   : output timestep in seconds (default 1.0)
    """
    env   = sample.get("env", {})
    meta  = env.get("meta", {}) or {}
    dt_pl = float(meta.get("dt", 60.0))

    t_raw  = np.arange(T_star + 1) * dt_pl
    t_fine = np.arange(0.0, t_raw[-1] + 1e-9, dt_out)

    pos_km    = np.column_stack([np.interp(t_fine, t_raw, X[:, i]) for i in range(3)])
    vel_kmps  = np.gradient(pos_km,  t_fine, axis=0)
    acc_kmps2 = np.gradient(vel_kmps, t_fine, axis=0)

    pos_m    = pos_km    * 1000.0
    vel_mps  = vel_kmps  * 1000.0
    acc_mps2 = acc_kmps2 * 1000.0

    vE, vN, vZ = vel_mps[:, 0],  vel_mps[:, 1],  vel_mps[:, 2]
    aE, aN, aZ = acc_mps2[:, 0], acc_mps2[:, 1], acc_mps2[:, 2]

    v_h2    = vE ** 2 + vN ** 2
    v_h_mps = np.sqrt(v_h2)
    airspeed = np.sqrt(v_h2 + vZ ** 2)

    heading_deg = np.degrees(np.arctan2(vE, vN))
    gamma_deg   = np.degrees(np.arctan2(vZ, np.maximum(v_h_mps, 1e-9)))

    yawrate_radps   = (vN * aE - vE * aN) / np.maximum(v_h2, 1e-12)
    yawrate_degps   = np.degrees(yawrate_radps)

    v_h_safe        = np.maximum(v_h_mps, 1e-9)
    dvh_dt          = (vE * aE + vN * aN) / v_h_safe
    pitchrate_radps = (v_h_safe * aZ - vZ * dvh_dt) / np.maximum(v_h2 + vZ ** 2, 1e-12)
    pitchrate_degps = np.degrees(pitchrate_radps)

    roll_deg = np.degrees(np.arctan2(v_h_mps * yawrate_radps, _G_MPS2))

    phases = _assign_phases(pos_km, vel_kmps, sample)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for i, t in enumerate(t_fine):
            w.writerow([
                f"{t:.3f}",
                phases[i],
                f"{pos_m[i, 0]:.3f}",    f"{pos_m[i, 1]:.3f}",    f"{pos_m[i, 2]:.3f}",
                f"{vel_mps[i, 0]:.4f}",  f"{vel_mps[i, 1]:.4f}",  f"{vel_mps[i, 2]:.4f}",
                f"{acc_mps2[i, 0]:.6f}", f"{acc_mps2[i, 1]:.6f}", f"{acc_mps2[i, 2]:.6f}",
                f"{heading_deg[i]:.3f}", f"{gamma_deg[i]:.3f}",   f"{roll_deg[i]:.3f}",
                f"{yawrate_degps[i]:.4f}", f"{pitchrate_degps[i]:.4f}",
                f"{airspeed[i]:.4f}",    f"{v_h_mps[i]:.4f}",
            ])
    print(f"  CSV saved → {csv_path}")