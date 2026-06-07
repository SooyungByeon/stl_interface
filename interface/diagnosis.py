"""
Plan a GestureProgram, or diagnose infeasibility back to gestures.

Planning searches for the minimum feasible horizon T by bisection (so a spec is
not falsely reported infeasible just because a fixed horizon was too short). Only
when the spec is infeasible even at the largest horizon do we extract the solver's
Irreducible Inconsistent Subsystem (IIS) and map it back to the responsible
gestures (paper Section 4).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from planner.dynamics import AircraftLTI
from planner.micp import PlanningProblem, MICPBuilder, solve_with_bisection
from planner.stl import STLBinaryExtractor
from .gestures import GestureProgram, IISReport


def plan(
    program: GestureProgram,
    *,
    x0: np.ndarray,
    bounds,                      # (x_min, x_max, u_min, u_max)
    dt: float,
    v_max: float,
    cruise_z: float,
    bigM: float = 300.0,
    T_max: int = 64,
    time_limit: float = 20.0,
) -> Dict[str, Any]:
    """Return {feasible, T, X?, U?, stl, iis?:IISReport, iis_names?}."""
    x_min, x_max, u_min, u_max = bounds
    x0 = np.asarray(x0, float)

    # T must be >= every fixed window end so those windows fit in the horizon.
    fixed_ends = [r.window[1] for r in program.requirements if r.window is not None]
    T_min = max(fixed_ends + [12])
    T_hi = max(T_min, int(T_max))

    def ast_fn(T: int):
        return program.translate(T)

    def prob_fn(T: int) -> PlanningProblem:
        return PlanningProblem(
            T=T, dynamics=AircraftLTI(dt=dt), x0=x0,
            x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max,
            Q=1e-2 * np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            R=1e-2 * np.eye(3), bigM=bigM, cruise_z_ref=cruise_z, v_max=v_max,
        )

    # 1) Try to find the minimum feasible horizon.
    try:
        T_star, X, U, _strat, _t = solve_with_bisection(
            ast_fn, prob_fn, T_min, T_hi,
            output_flag=0, verbose=False, time_limit=time_limit,
        )
        return {"feasible": True, "T": int(T_star), "X": X, "U": U,
                "stl": program.to_text(T_star)}
    except (RuntimeError, ValueError):
        pass  # infeasible across [T_min, T_hi] -> diagnose at T_hi

    # 2) Genuinely infeasible: build at the largest horizon and extract the IIS.
    T = T_hi
    ast = program.translate(T)
    prob = prob_fn(T)
    descs = STLBinaryExtractor(T).extract(ast)
    builder = MICPBuilder(prob, ast)
    model = builder.build(descs, output_flag=0)
    model.Params.TimeLimit = float(time_limit)
    model.optimize()

    if model.SolCount > 0 and model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        X = np.array([[builder.x[k, i].X for i in range(6)] for k in range(T + 1)])
        U = np.array([[builder.u[k, i].X for i in range(3)] for k in range(T)])
        return {"feasible": True, "T": T, "X": X, "U": U, "stl": program.to_text(T)}

    model.computeIIS()
    iis_names = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
    iis_names += [g.GenConstrName for g in model.getGenConstrs() if g.IISGenConstr]
    report: IISReport = program.diagnose(iis_names)
    return {"feasible": False, "T": T, "stl": program.to_text(T),
            "iis": report, "iis_names": iis_names}


__all__ = ["plan"]
