"""
Example factories and parameter presets.

Provides:
  - STL AST constructors for Examples 1–3 (aircraft missions)
  - Geometry constraints, validation, and perturbation sampler
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geometry import Rect
from .stl import (
    STLNode, And, Or, Always, Eventually,
    InRect, OutsideRect, OutsideVelocityRect, RunwayPredicate,
)


# ==============================================================================
# STL spec factories
# ==============================================================================

def make_example_1(
    T: int,
    T_land_lo: int,
    obstacles: Sequence[Rect],
    D1: Rect,
    D2: Rect,
    D3: Rect,
    D4: Rect,
    runway: RunwayPredicate,
    delivery_z_range: Optional[Tuple[float, float]] = None,
    v_min: float = 0.028,
    grace_lo: int = 3,
    grace_hi: int = 3,
) -> STLNode:
    """
    Example 1 — Farafangana 4-Drop Delivery with Return:

        phi_1 = □[0,T]              (∧_j ¬O_j)
              ∧ □[grace_lo,T-grace_hi]  OutsideVelocityRect(v_min)
              ∧ ◇[0,T]              InRect(D1)
              ∧ ◇[0,T]              InRect(D2)
              ∧ ◇[0,T]              InRect(D3)
              ∧ ◇[0,T]              InRect(D4)
              ∧ ◇[T_land_lo,T]      Runway

    T_land_lo is set late (e.g. T-20) to prevent early landing before
    all deliveries can plausibly be completed.

    Args:
        T:               horizon length
        T_land_lo:       earliest step at which landing is allowed
        obstacles:       sequence of obstacle rectangles
        D1..D4:          delivery zone rectangles
        runway:          RunwayPredicate
        delivery_z_range: optional (z_lo, z_hi) altitude band for InRect
        v_min:           minimum horizontal speed (km/s)
        grace_lo:        timesteps at start exempt from v_min
        grace_hi:        timesteps at end exempt from v_min
    """
    def _obs_node(j, o):
        return o if isinstance(o, STLNode) else OutsideRect(f"O{j}", o)
    avoid   = Always(0, T, And(tuple(_obs_node(j, o) for j, o in enumerate(obstacles))))
    v_floor = Always(grace_lo, T - grace_hi, OutsideVelocityRect("Vfloor", v_min))
    return And((
        avoid,
        v_floor,
        Eventually(0, T, InRect("D1", D1, delivery_z_range)),
        Eventually(0, T, InRect("D2", D2, delivery_z_range)),
        Eventually(0, T, InRect("D3", D3, delivery_z_range)),
        Eventually(0, T, InRect("D4", D4, delivery_z_range)),
        Eventually(T_land_lo, T, runway),
    ))


# Example 2 — Farafangana 4-Drop Random POI — reuses make_example_1.
# The STL spec is identical to Ex1; only the 4 delivery rects passed in are
# sampled per-sample (see planner.descriptor:sample_env, ex==2 branch).
# No separate `make_example_2` is needed.


def make_example_3(
    T: int,
    a1: int, b1: int,
    a2: int, b2: int,
    a3: int, b3: int,
    a4: int,
    obstacles: Sequence[Rect],
    D1: Rect,
    D2: Rect,
    D3: Rect,
    runway: RunwayPredicate,
    delivery_z_range: Optional[Tuple[float, float]] = None,
    v_min: float = 0.028,
    grace_lo: int = 3,
    grace_hi: int = 3,
) -> STLNode:
    """
    Example 3 — Multi-Drop Delivery with Return:

        phi_3 = □[0,T]    (∧_j ¬O_j)
              ∧ □[grace_lo,T-grace_hi]  OutsideVelocityRect(v_min)
              ∧ ◇[a1,b1]  InRect(D1)
              ∧ ◇[a2,b2]  InRect(D2)
              ∧ ◇[a3,b3]  InRect(D3)
              ∧ ◇[a4,T]   Runway

    Time windows [ai,bi] and [a4,T] control the degree of ordering
    flexibility: non-overlapping windows fix delivery order; overlapping
    windows allow the solver to reorder deliveries.

    Args:
        T:               horizon length
        a1,b1,...,a4:    delivery time-window endpoints (a4 is landing open)
        obstacles:       sequence of obstacle rectangles
        D1,D2,D3:        delivery zone rectangles
        runway:          RunwayPredicate
        delivery_z_range: optional (z_lo, z_hi) altitude band for all InRect
        v_min:           minimum horizontal speed (km/s)
        grace_lo:        timesteps at start exempt from v_min
        grace_hi:        timesteps at end exempt from v_min
    """
    avoid   = Always(0, T, And(tuple(
        OutsideRect(f"O{j}", r) for j, r in enumerate(obstacles)
    )))
    v_floor = Always(grace_lo, T - grace_hi, OutsideVelocityRect("Vfloor", v_min))
    return And((
        avoid,
        v_floor,
        Eventually(a1, b1, InRect("D1", D1, delivery_z_range)),
        Eventually(a2, b2, InRect("D2", D2, delivery_z_range)),
        Eventually(a3, b3, InRect("D3", D3, delivery_z_range)),
        Eventually(a4, T,  runway),
    ))


# ==============================================================================
# Geometry / validation / perturbation
# ==============================================================================

def shift_rect(r: Rect, dx: float, dy: float) -> Rect:
    x1, x2, y1, y2 = r
    return (float(x1) + float(dx), float(x2) + float(dx),
            float(y1) + float(dy), float(y2) + float(dy))


def shift_x0(x0: np.ndarray, dx: float, dy: float) -> np.ndarray:
    x0p = np.array(x0, dtype=float).copy()
    x0p[0] += float(dx)
    x0p[1] += float(dy)
    return x0p


@dataclass(frozen=True)
class GeometryConstraints:
    min_goal_obstacle_clearance: float = 0.0
    min_start_obstacle_clearance: float = 0.0
    min_start_goal_clearance: float = 0.0
    min_obstacle_obstacle_clearance: float = 0.0
    require_rects_in_bounds: bool = True
    require_start_in_bounds: bool = True


def rect_expand(r: Rect, margin: float) -> Rect:
    xmin, xmax, ymin, ymax = r
    m = float(margin)
    return (float(xmin) - m, float(xmax) + m, float(ymin) - m, float(ymax) + m)


def rect_intersects(a: Rect, b: Rect) -> bool:
    axmin, axmax, aymin, aymax = a
    bxmin, bxmax, bymin, bymax = b
    return not (axmax < bxmin or axmin > bxmax or aymax < bymin or aymin > bymax)


def rect_contains_point(r: Rect, x: float, y: float, margin: float = 0.0) -> bool:
    xmin, xmax, ymin, ymax = rect_expand(r, margin)
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def rect_within_bounds(r: Rect, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bool:
    xmin, xmax, ymin, ymax = r
    return (xmin >= xlim[0]) and (xmax <= xlim[1]) and (ymin >= ylim[0]) and (ymax <= ylim[1])


def start_within_bounds(x0: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bool:
    px, py = float(x0[0]), float(x0[1])
    return (xlim[0] <= px <= xlim[1]) and (ylim[0] <= py <= ylim[1])


def validate_geometry(
    *,
    x0: np.ndarray,
    obstacles: Sequence[Rect],
    regions: Mapping[str, Rect],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    rules: GeometryConstraints,
) -> Tuple[bool, str]:
    """
    Check geometry validity:
      - start in bounds (optional)
      - obstacles and regions in bounds (optional)
      - obstacles must not overlap each other
      - regions must not overlap any obstacle
      - start must not be inside / too close to any obstacle
      - start-region clearance (optional)
    """
    obs  = list(obstacles)
    regs = dict(regions)

    if rules.require_start_in_bounds and not start_within_bounds(x0, xlim, ylim):
        return False, "start_out_of_bounds"

    if rules.require_rects_in_bounds:
        for j, Oj in enumerate(obs):
            if not rect_within_bounds(Oj, xlim, ylim):
                return False, f"obstacle_{j}_out_of_bounds"
        for name, R in regs.items():
            if not rect_within_bounds(R, xlim, ylim):
                return False, f"region_{name}_out_of_bounds"

    m_obs = float(rules.min_obstacle_obstacle_clearance)
    for i in range(len(obs)):
        Oi = rect_expand(obs[i], m_obs)
        for j in range(i + 1, len(obs)):
            if rect_intersects(Oi, rect_expand(obs[j], m_obs)):
                return False, f"obstacle_{i}_overlaps_obstacle_{j}"

    m_goal = float(rules.min_goal_obstacle_clearance)
    for j, Oj in enumerate(obs):
        Oe = rect_expand(Oj, m_goal)
        for name, R in regs.items():
            if rect_intersects(Oe, R):
                return False, f"region_{name}_overlaps_obstacle_{j}"

    m_start = float(rules.min_start_obstacle_clearance)
    for j, Oj in enumerate(obs):
        if rect_contains_point(Oj, float(x0[0]), float(x0[1]), margin=m_start):
            return False, f"start_inside_or_too_close_to_obstacle_{j}"

    if rules.min_start_goal_clearance > 0.0:
        m = float(rules.min_start_goal_clearance)
        for name, R in regs.items():
            if rect_contains_point(R, float(x0[0]), float(x0[1]), margin=m):
                return False, f"start_inside_or_too_close_to_region_{name}"

    return True, "ok"


def sample_valid_perturbation(
    *,
    x0_base: np.ndarray,
    obstacles_base: Sequence[Rect],
    regions_base: Mapping[str, Rect],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    dx0_range: Tuple[float, float],
    dy0_range: Tuple[float, float],
    dO_range: Tuple[float, float, float, float],
    dR_range: Tuple[float, float, float, float],
    rules: GeometryConstraints,
    rng: np.random.Generator,
    max_tries: int = 200,
    fixed_obstacle_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """
    Rejection-sample perturbations of x0, obstacles, and regions.

    Returns dict: x0, obstacles, regions, deltas, tries.
    """
    obs0 = list(obstacles_base)
    if len(obs0) < 1:
        raise ValueError("obstacles_base must contain at least one obstacle.")

    fixed: List[bool]
    if fixed_obstacle_mask is None:
        fixed = [False] * len(obs0)
    else:
        fixed = list(fixed_obstacle_mask)
        if len(fixed) != len(obs0):
            raise ValueError(
                f"fixed_obstacle_mask length mismatch: got {len(fixed)}, expected {len(obs0)}"
            )

    dx_min, dx_max = float(dO_range[0]), float(dO_range[1])
    dy_min, dy_max = float(dO_range[2]), float(dO_range[3])
    rx_min, rx_max = float(dR_range[0]), float(dR_range[1])
    ry_min, ry_max = float(dR_range[2]), float(dR_range[3])
    region_names = list(regions_base.keys())

    for t in range(1, int(max_tries) + 1):
        dx0 = float(rng.uniform(dx0_range[0], dx0_range[1]))
        dy0 = float(rng.uniform(dy0_range[0], dy0_range[1]))
        x0 = shift_x0(x0_base, dx0, dy0)

        obstacles: List[Rect] = []
        dOx: List[float] = []
        dOy: List[float] = []
        for i, Oi in enumerate(obs0):
            if fixed[i]:
                obstacles.append(Oi)
                dOx.append(0.0)
                dOy.append(0.0)
            else:
                ox = float(rng.uniform(dx_min, dx_max))
                oy = float(rng.uniform(dy_min, dy_max))
                obstacles.append(shift_rect(Oi, ox, oy))
                dOx.append(ox)
                dOy.append(oy)

        regions: Dict[str, Rect] = {}
        dRx: Dict[str, float] = {}
        dRy: Dict[str, float] = {}
        for name in region_names:
            r0 = regions_base[name]
            rx = float(rng.uniform(rx_min, rx_max))
            ry = float(rng.uniform(ry_min, ry_max))
            regions[name] = shift_rect(r0, rx, ry)
            dRx[name] = rx
            dRy[name] = ry

        ok, _ = validate_geometry(
            x0=x0, obstacles=obstacles, regions=regions,
            xlim=xlim, ylim=ylim, rules=rules,
        )
        if ok:
            return dict(
                x0=x0,
                obstacles=obstacles,
                regions=regions,
                deltas=dict(dx0=dx0, dy0=dy0, dOx=dOx, dOy=dOy,
                            fixed=fixed, dRx=dRx, dRy=dRy),
                tries=t,
            )

    raise RuntimeError(f"Failed to sample valid perturbation after max_tries={max_tries}.")