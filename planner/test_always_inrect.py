"""
Standalone check for the new Always(InRect) ("keep") encoder branch.

Builds a minimal MICP whose only spec is  G[0,T] InRect(H),  solves it with the
existing solver, and verifies the trajectory stays inside H at every timestep.

Run from the project root (parent of planner/):
    PYTHONPATH=. conda run --no-capture-output -n stl_gnn python -m planner.test_always_inrect
"""

from __future__ import annotations

import numpy as np

from planner.dynamics import AircraftLTI
from planner.micp import PlanningProblem, solve_micp_with_strategy
from planner.stl import Always, InRect


def main() -> None:
    T = 10
    H = (20.0, 40.0, 20.0, 40.0)  # 20x20 km keep box

    spec = Always(0, T, InRect("H", H))

    dyn = AircraftLTI(dt=60.0)
    prob = PlanningProblem(
        T=T,
        dynamics=dyn,
        x0=np.array([30.0, 30.0, 1.2, 0.01, 0.0, 0.0]),  # start at box center
        x_min=np.array([0.0, 0.0, 0.0, -0.040, -0.040, -0.006]),
        x_max=np.array([60.0, 60.0, 1.5, 0.040, 0.040, 0.006]),
        u_min=np.array([-0.00020, -0.00020, -0.0001]),
        u_max=np.array([0.00020, 0.00020, 0.0001]),
        Q=1e-2 * np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        R=1e-2 * np.eye(3),
        bigM=1e3,
        v_max=0.040,
    )

    X, U, strategy, t_sec = solve_micp_with_strategy(spec, prob, output_flag=0)

    xmin, xmax, ymin, ymax = H
    inside = np.all(
        (X[:, 0] >= xmin - 1e-6) & (X[:, 0] <= xmax + 1e-6)
        & (X[:, 1] >= ymin - 1e-6) & (X[:, 1] <= ymax + 1e-6)
    )

    print(f"solved in {t_sec:.3f}s   binaries={len(strategy.binaries)} (expect 0)")
    print(f"x range: x∈[{X[:,0].min():.2f},{X[:,0].max():.2f}]  "
          f"y∈[{X[:,1].min():.2f},{X[:,1].max():.2f}]")
    print(f"keep H satisfied at all {T + 1} steps: {bool(inside)}")

    assert len(strategy.binaries) == 0, "Always(InRect) must add no binaries"
    assert inside, "trajectory left the keep region H"
    print("PASS")


if __name__ == "__main__":
    main()
