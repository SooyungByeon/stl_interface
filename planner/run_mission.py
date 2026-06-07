"""
Build, solve, and plot a paper mission (1, 2, or 3) on the shared LA frame.

    Mission 1  keep H, then land:        □H ∧ ◇(G1∨G2)
    Mission 2  timed deliveries, land:   ⋀ ◇P_i ∧ ◇(G1∨G2)
    Mission 3  door–key (until), land:   ⋀(¬D_i U K_i) ∧ ◇(G1∨G2)

Environment (start, 5 obstacles, speed floor) is supplied automatically with the
same sampler as the interface. Minimum horizon T* is found by bisection.

Run from the project root:
    PYTHONPATH=. conda run --no-capture-output -n stl_gnn python -m planner.run_mission --example {1,2,3} [--seed N]
or use ./run_mission.sh {1,2,3} [--seed N]
"""

from __future__ import annotations

import argparse

import numpy as np

from . import la_frame as la
from .paper_examples import (
    make_paper_example_1, make_paper_example_2, make_paper_example_3,
)
from .examples import GeometryConstraints
from .descriptor import _place_random_obstacles
from .dynamics import AircraftLTI
from .micp import PlanningProblem, solve_with_bisection
from .stl import InCircle
from .plot_mission import plot_mission

LANDING_Z = (0.0, 0.02)
TIME_LIMIT = 30.0
BIG_M = 300.0

G1 = la.box(*la.G1_KM, 2.0)
G2 = la.box(*la.G2_KM, 2.0)
RUNWAYS = [("G1", G1), ("G2", G2)]
RUNWAY_PLOT = [{"name": "G1", "role": "runway", "rect": G1},
               {"name": "G2", "role": "runway", "rect": G2}]


def make_x0() -> np.ndarray:
    sx, sy = 118.0, 100.0
    d = np.array([95.0 - sx, 112.0 - sy])
    v = d / (np.linalg.norm(d) + 1e-9) * 0.035
    return np.array([sx, sy, 1.2, v[0], v[1], 0.0])


# ----------------------------------------------------------------------------
# per-mission geometry: returns (plot_regions, avoid_rects, ast_factory, T_range)
#   ast_factory(obstacles) -> ast_fn(T)
# ----------------------------------------------------------------------------

def geometry(ex: int):
    if ex == 1:
        H_center, H_radius = (90.0, 110.0), 5.0
        plot = [{"name": "H", "role": "keep", "circle": (H_center[0], H_center[1], H_radius)}] + RUNWAY_PLOT
        avoid = {"H": la.box(*H_center, H_radius), "G1": G1, "G2": G2}

        def factory(obstacles):
            def ast_fn(T):
                return make_paper_example_1(
                    T=T, t1=20, t2=28, t3=30, t4=T,
                    H_pred=InCircle("H", H_center, H_radius),
                    G1=G1, G2=G2, obstacles=obstacles,
                    v_min=la.V_MIN, grace_lo=la.GRACE_LO, grace_hi=la.GRACE_HI,
                    goal_z_range=LANDING_Z)
            return ast_fn
        return plot, avoid, factory, (32, 48)

    if ex == 2:
        # three delivery points (LA-basin airfields), then land at a runway
        P = {"P1": la.box(63.4, 103.1, 2.0),   # Hawthorne
             "P2": la.box(91.4, 120.9, 2.0),   # El Monte
             "P3": la.box(96.2, 97.1, 2.0)}    # Fullerton
        plot = [{"name": n, "role": "delivery", "rect": r} for n, r in P.items()] + RUNWAY_PLOT
        avoid = {**P, "G1": G1, "G2": G2}

        def factory(obstacles):
            def ast_fn(T):
                return make_paper_example_2(
                    T=T,
                    deliveries=[(n, r, None) for n, r in P.items()],
                    goal_rects=RUNWAYS, goal_window=None, obstacles=obstacles,
                    v_min=la.V_MIN, grace_lo=la.GRACE_LO, grace_hi=la.GRACE_HI,
                    goal_z_range=LANDING_Z)
            return ast_fn
        return plot, avoid, factory, (12, 72)

    if ex == 3:
        # two door–key pairs (keys near the SE start, doors guarding the west
        # approach), then land. Reach each key before its door is allowed.
        K1 = la.box(108.0, 110.0, 2.0); D1 = la.box(76.0, 80.0, 109.0, 113.0)
        K2 = la.box(101.0, 95.0, 2.0);  D2 = la.box(82.0, 84.0, 99.0, 103.0)
        dk = [("D1", D1, "K1", K1), ("D2", D2, "K2", K2)]
        plot = ([{"name": "K1", "role": "key", "rect": K1},
                 {"name": "K2", "role": "key", "rect": K2},
                 {"name": "D1", "role": "door", "rect": D1},
                 {"name": "D2", "role": "door", "rect": D2}] + RUNWAY_PLOT)
        avoid = {"K1": K1, "K2": K2, "D1": D1, "D2": D2, "G1": G1, "G2": G2}

        def factory(obstacles):
            def ast_fn(T):
                return make_paper_example_3(
                    T=T,
                    doorkeys=[(dn, dr, kn, kr, None) for dn, dr, kn, kr in dk],
                    goal_rects=RUNWAYS, goal_window=None, obstacles=obstacles,
                    v_min=la.V_MIN, grace_lo=la.GRACE_LO, grace_hi=la.GRACE_HI,
                    goal_z_range=LANDING_Z)
            return ast_fn
        return plot, avoid, factory, (12, 72)

    raise ValueError(f"unknown example {ex}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ex = args.example

    x0 = make_x0()
    bounds = la.make_bounds()
    x_min, x_max, u_min, u_max = bounds
    plot_regions, avoid, factory, (T_min, T_max) = geometry(ex)

    # obstacles avoid the authored regions + runways + start
    rng = np.random.default_rng(args.seed)
    rules = GeometryConstraints(min_goal_obstacle_clearance=4.0,
                                min_start_obstacle_clearance=4.0,
                                require_rects_in_bounds=True, require_start_in_bounds=True)
    obstacles = _place_random_obstacles(
        rng=rng, x0=x0, regions=avoid, n_obstacles=5,
        obstacle_w_min=6.0, obstacle_w_max=10.0, obstacle_h_min=6.0, obstacle_h_max=12.0,
        xlim=(x_min[0], x_max[0]), ylim=(x_min[1], x_max[1]), rules=rules, max_tries=8000,
        sample_xlim=(60.0, 135.0), sample_ylim=(95.0, 135.0))
    print(f"Mission {ex}  seed={args.seed}  x0=({x0[0]:.0f},{x0[1]:.0f})  obstacles={len(obstacles)}")

    ast_fn = factory(obstacles)

    def prob_fn(T):
        return PlanningProblem(
            T=T, dynamics=AircraftLTI(dt=la.DT), x0=x0,
            x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max,
            Q=1e-2 * np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), R=1e-2 * np.eye(3),
            bigM=BIG_M, cruise_z_ref=la.CRUISE_Z_REF, v_max=la.V_MAX)

    T_star, X, U, _strat, t_total = solve_with_bisection(
        ast_fn, prob_fn, T_min, T_max, output_flag=0, verbose=True, time_limit=TIME_LIMIT)
    print(f"T* = {T_star}   total solve time {t_total:.1f}s")

    out = f"output/paper/ex{ex}.png"
    plot_mission(X, T_star, mission_id=ex, dt=la.DT, regions=plot_regions,
                 obstacles=obstacles, v_min=la.V_MIN, v_max=la.V_MAX,
                 cruise_z=la.CRUISE_Z_REF, landing_z=LANDING_Z, png_path=out)
    print(f"plot saved -> {out}")


if __name__ == "__main__":
    main()
