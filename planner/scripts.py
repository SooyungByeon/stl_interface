# scripts.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from .dynamics import AircraftLTI
from .examples import GeometryConstraints
from .descriptor import get_descriptor
from .micp import (
    PlanningProblem,
    generate_dataset,
    solve_with_bisection,
    write_jsonl_gz,
    ast_to_dict,
    descs_to_dicts,
    canonicalize_strategy,
)
from .plot_result import plot_result
from .stl import STLBinaryExtractor
from .trajectory_csv import export_trajectory_csv


def build_samples(
    *,
    n: int,
    seed: int,
    example_id: int,
    dx0_range: Tuple[float, float],
    dy0_range: Tuple[float, float],
    dO_range: Tuple[float, float, float, float],
    dR_range: Tuple[float, float, float, float],
    rules: GeometryConstraints,
    max_tries: int,
    n_obstacles: int = 3,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    desc = get_descriptor(int(example_id), n_obstacles=int(n_obstacles))

    bounds = {
        "x_min": desc.bounds.x_min,
        "x_max": desc.bounds.x_max,
        "u_min": desc.bounds.u_min,
        "u_max": desc.bounds.u_max,
    }

    xlim = (float(bounds["x_min"][0]), float(bounds["x_max"][0]))
    ylim = (float(bounds["x_min"][1]), float(bounds["x_max"][1]))

    samples: List[Dict[str, Any]] = []
    for _ in range(n):
        env = desc.sample_env(
            rng=rng,
            xlim=xlim,
            ylim=ylim,
            dx0_range=dx0_range,
            dy0_range=dy0_range,
            dO_range=dO_range,
            dR_range=dR_range,
            rules=rules,
            max_tries=max_tries,
            fixed_obstacle_mask=None,
        )

        samples.append({
            "example_id": int(example_id),
            "T": int(desc.T),
            "bounds": {
                "x_min": bounds["x_min"].tolist(),
                "x_max": bounds["x_max"].tolist(),
                "u_min": bounds["u_min"].tolist(),
                "u_max": bounds["u_max"].tolist(),
            },
            "env": env,
        })

    return samples


def build_ast_fn(sample: Dict[str, Any]):
    ex = int(sample.get("example_id", 1))
    desc = get_descriptor(ex, n_obstacles=len(sample["env"]["obstacles"]))
    return desc.build_ast_from_sample(sample)


def build_problem_fn(sample: Dict[str, Any]) -> PlanningProblem:
    T   = int(sample["T"])
    b   = sample["bounds"]
    env = sample["env"]
    meta = env.get("meta", {})

    dt  = float(meta.get("dt", 60.0))
    dyn = AircraftLTI(dt=dt)

    x0    = np.array(env["x0"],   dtype=float)
    x_min = np.array(b["x_min"], dtype=float)
    x_max = np.array(b["x_max"], dtype=float)
    u_min = np.array(b["u_min"], dtype=float)
    u_max = np.array(b["u_max"], dtype=float)

    # 6-state cost: penalise velocities (indices 3,4,5); positions left to STL.
    # Cruise altitude 1.2 km is a soft target encoded via a linear term added
    # to the objective inside MICPBuilder using meta["cruise_z_ref"].
    Q = 1e-2 * np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    R = 1e-2 * np.eye(3)

    v_max = meta.get("v_max")

    # Soft runway-tracking weight (Example 3). Absent -> 0.0 -> identical
    # objective to Examples 1/2 (no behavior change for those).
    q_runway = float(meta.get("Q_runway", 0.0))

    return PlanningProblem(
        T=T,
        dynamics=dyn,
        x0=x0,
        x_min=x_min,
        x_max=x_max,
        u_min=u_min,
        u_max=u_max,
        Q=Q,
        R=R,
        bigM=1e3,
        cruise_z_ref=float(meta.get("cruise_z_ref", 1.2)),
        v_max=float(v_max) if v_max is not None else None,
        Q_runway=q_runway,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        type=str, default="output/micp/dataset.jsonl.gz")
    parser.add_argument("--example_id", type=int, default=1)
    parser.add_argument("--n",          type=int, default=200)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--gurobi_out", type=int, default=0)

    parser.add_argument("--dx0", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--dy0", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--dO",  type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--dR",  type=float, nargs=4, default=[-5.0, 5.0, -5.0, 5.0])

    parser.add_argument("--goal_obs_clear",  type=float, default=5.0)
    parser.add_argument("--start_obs_clear", type=float, default=5.0)
    parser.add_argument("--max_tries",       type=int,   default=1000)
    parser.add_argument("--n_obstacles",     type=int,   default=3)

    parser.add_argument("--bisect", action="store_true",
                        help="Find minimum T* via bisection instead of using a fixed horizon.")
    parser.add_argument("--T_min", type=int, default=None,
                        help="Bisection lower bound (default: from descriptor meta).")
    parser.add_argument("--T_max", type=int, default=None,
                        help="Bisection upper bound (default: from descriptor meta).")
    parser.add_argument("--bisect_timeout", type=float, default=120.0,
                        help="Per-solve wall-clock timeout in seconds (default: 120). "
                             "Set to 0 to disable.")
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="Per-solve wall-clock timeout for fixed-T (non-bisect) solver, "
                             "in seconds. 0 means no limit (default).")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip saving trajectory PNG plots during bisection.")
    parser.add_argument("--csv_dir", type=str, default=None,
                        help="Directory for trajectory CSV files. "
                             "Default: output/micp/csv/. Pass '' to disable.")
    parser.add_argument("--dt_out", type=float, default=1.0,
                        help="CSV output timestep in seconds (default: 1.0). "
                             "Trajectory is resampled from 60 s planner steps via linear interp.")
    parser.add_argument("--compact", dest="compact", action="store_true", default=True,
                        help="Emit compact strategy schema with canonical max-margin face "
                             "labels and no per-binary descs (default: True).")
    parser.add_argument("--no-compact", dest="compact", action="store_false",
                        help="Emit the legacy schema (raw binary dict + full descs).")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rules = GeometryConstraints(
        min_goal_obstacle_clearance=float(args.goal_obs_clear),
        min_start_obstacle_clearance=float(args.start_obs_clear),
        min_start_goal_clearance=0.0,
        require_rects_in_bounds=True,
        require_start_in_bounds=True,
    )

    samples = build_samples(
        n=int(args.n),
        seed=int(args.seed),
        example_id=int(args.example_id),
        dx0_range=(float(args.dx0[0]), float(args.dx0[1])),
        dy0_range=(float(args.dy0[0]), float(args.dy0[1])),
        dO_range=(float(args.dO[0]), float(args.dO[1]),
                  float(args.dO[2]), float(args.dO[3])),
        dR_range=(float(args.dR[0]), float(args.dR[1]),
                  float(args.dR[2]), float(args.dR[3])),
        rules=rules,
        max_tries=int(args.max_tries),
        n_obstacles=int(args.n_obstacles),
    )

    if args.bisect:
        bisect_timeout = float(args.bisect_timeout) if args.bisect_timeout > 0 else None

        plot_dir = os.path.join(os.path.dirname(args.out) or ".", "plots")
        if not args.no_plot:
            os.makedirs(plot_dir, exist_ok=True)

        _csv_enabled = (args.csv_dir != "")
        _csv_dir = (
            args.csv_dir if (args.csv_dir is not None and args.csv_dir != "")
            else os.path.join(os.path.dirname(args.out) or ".", "csv")
        )
        _dt_out = float(args.dt_out)

        records: List[Dict[str, Any]] = []
        solve_times: List[float] = []
        n_skipped = 0

        for idx, s in enumerate(samples):
            env_meta = s.get("env", {}).get("meta", {}) or {}
            t_lo = int(args.T_min) if args.T_min is not None else int(env_meta.get("T_min", 40))
            t_hi = int(args.T_max) if args.T_max is not None else int(env_meta.get("T_max", 60))

            print(f"\n[Sample {idx + 1}/{len(samples)}]")

            def _ast_fn(T: int, _s: Dict[str, Any] = s) -> Any:
                _s2 = dict(_s); _s2["T"] = T
                return build_ast_fn(_s2)

            def _prob_fn(T: int, _s: Dict[str, Any] = s) -> PlanningProblem:
                _s2 = dict(_s); _s2["T"] = T
                return build_problem_fn(_s2)

            try:
                T_star, X, U, strategy, t_total = solve_with_bisection(
                    _ast_fn, _prob_fn, t_lo, t_hi,
                    output_flag=int(args.gurobi_out),
                    verbose=True,
                    time_limit=bisect_timeout,
                )
            except (RuntimeError, ValueError) as e:
                print(f"  [WARN] Bisection failed (skipping): {e}")
                n_skipped += 1
                continue

            solve_times.append(float(t_total))
            n_done = len(solve_times)
            t_mean = float(np.mean(solve_times))
            t_std  = float(np.std(solve_times, ddof=1)) if n_done > 1 else 0.0
            print(f"  Solve time: {t_total:.1f}s  |  running stats ({n_done} samples): "
                  f"mean={t_mean:.1f}s  std={t_std:.1f}s")

            # Reconstruct AST and descriptors at T_star so graph_builder has
            # the correct horizon, predicate nodes, and binary descriptor ids.
            final_sample = dict(s)
            final_sample["T"] = T_star
            ast_obj = build_ast_fn(final_sample)
            descs = STLBinaryExtractor(int(T_star)).extract(ast_obj)

            if args.compact:
                records.append({
                    "id": int(idx),
                    "T_star": int(T_star),
                    "sample": s,
                    "ast": ast_to_dict(ast_obj, geometry=False),
                    "micp": {
                        "time_sec": float(t_total),
                        "meta": strategy.meta,
                        "X": X.tolist(),
                        "U": U.tolist(),
                        "strategy": canonicalize_strategy(
                            descs=descs,
                            strategy_binaries=strategy.binaries,
                            X=X,
                            ast=ast_obj,
                        ),
                    },
                })
            else:
                records.append({
                    "id": idx,
                    "T_star": int(T_star),
                    "sample": s,
                    "ast": ast_to_dict(ast_obj),
                    "micp": {
                        "time_sec": float(t_total),
                        "X": X.tolist(),
                        "U": U.tolist(),
                        "strategy": strategy.binaries,
                        "meta": strategy.meta,
                        "descs": descs_to_dicts(descs),
                    },
                })

            if not args.no_plot:
                png_path = os.path.join(plot_dir, f"ex{int(args.example_id)}_{idx:03d}.png")
                plot_result(X, T_star, s, png_path)
                print(f"  Plot saved → {png_path}")

            if _csv_enabled:
                csv_path = os.path.join(_csv_dir, f"ex{int(args.example_id)}_{idx:03d}.csv")
                export_trajectory_csv(X, U, T_star, s, csv_path, dt_out=_dt_out)

        write_jsonl_gz(str(args.out), records)
        print(f"\nWrote {len(records)} bisection records to: {args.out}")

        if solve_times:
            t_mean = float(np.mean(solve_times))
            t_std  = float(np.std(solve_times, ddof=1)) if len(solve_times) > 1 else 0.0
            print(f"\n{'='*60}")
            print(f" Solve-time summary  ({len(solve_times)} solved, {n_skipped} skipped)")
            print(f"   mean : {t_mean:.1f} s")
            print(f"   std  : {t_std:.1f} s")
            print(f"   min  : {float(np.min(solve_times)):.1f} s")
            print(f"   max  : {float(np.max(solve_times)):.1f} s")
            print(f"   total: {sum(solve_times):.1f} s  ({sum(solve_times)/3600:.2f} h)")
            print(f"{'='*60}")

            _out_base = str(args.out)
            for _sfx in (".jsonl.gz", ".jsonl", ".gz"):
                if _out_base.endswith(_sfx):
                    _out_base = _out_base[: -len(_sfx)]
                    break
            _timing_path = _out_base + "_timing.json"
            with open(_timing_path, "w") as _f:
                json.dump({
                    "solver": "micp_bisection",
                    "example_id": int(args.example_id),
                    "n_solved": len(solve_times),
                    "n_skipped": int(n_skipped),
                    "time_mean": t_mean,
                    "time_std": t_std,
                    "time_min": float(np.min(solve_times)),
                    "time_max": float(np.max(solve_times)),
                    "time_total": float(sum(solve_times)),
                }, _f, indent=2)
            print(f"Timing stats saved to: {_timing_path}")
    else:
        fixed_timeout = float(args.timeout) if float(args.timeout) > 0.0 else None

        plot_dir = os.path.join(os.path.dirname(args.out) or ".", "plots")
        if not args.no_plot:
            os.makedirs(plot_dir, exist_ok=True)

        def _fixed_plot_fn(*, idx: int, X: np.ndarray, T: int, sample: dict) -> None:
            png_path = os.path.join(plot_dir, f"ex{int(args.example_id)}_{idx:03d}.png")
            plot_result(X, T, sample, png_path)
            print(f"  Plot saved → {png_path}")

        generate_dataset(
            out_path=str(args.out),
            samples=samples,
            build_ast_fn=build_ast_fn,
            build_problem_fn=build_problem_fn,
            output_flag=int(args.gurobi_out),
            time_limit=fixed_timeout,
            plot_fn=None if args.no_plot else _fixed_plot_fn,
            compact=bool(args.compact),
        )


if __name__ == "__main__":
    main()