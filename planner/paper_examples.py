"""
STL spec factories for the paper (Click-to-STL) examples, on the shared LA frame.

These mirror the paper's gesture->STL translation (Table 1). Each factory returns
a plain STL AST that the existing MICPBuilder / solver consume unchanged.

Environment-supplied conjuncts (obstacles, speed floor) are added here exactly as
the paper describes: the human authors only the mission; obstacles and the stall
speed floor are conjoined automatically.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .geometry import Rect
from .stl import (
    STLNode, And, Or, Always, Eventually, Until,
    InRect, OutsideRect, OutsideVelocityRect,
)

Window = Optional[Tuple[int, int]]


def _goal_pred(goal_rects, T, goal_z_range):
    """Single InRect or Or(InRect…) for the landing goal (runways)."""
    preds = [InRect(n, tuple(r), goal_z_range) for n, r in goal_rects]
    return preds[0] if len(preds) == 1 else Or(tuple(preds))


def _env_conjuncts(
    T: int,
    obstacles: Sequence[Rect],
    v_min: float,
    grace_lo: int,
    grace_hi: int,
) -> Tuple[STLNode, ...]:
    """Environment-supplied (not authored) conjuncts: obstacle avoidance over the
    full horizon and the always-on min-speed (stall) floor with end graces."""
    conj: list[STLNode] = []
    if obstacles:
        conj.append(Always(0, T, And(tuple(
            OutsideRect(f"O{j}", tuple(o)) for j, o in enumerate(obstacles)
        ))))
    conj.append(Always(grace_lo, T - grace_hi, OutsideVelocityRect("Vfloor", v_min)))
    return tuple(conj)


def make_paper_example_1(
    *,
    T: int,
    t1: int, t2: int,        # keep window  (early)
    t3: int, t4: int,        # reach window (later)
    H_pred: STLNode,         # keep / loiter region predicate (InRect or InCircle, name "H")
    G1: Rect,                # runway 1 (LAX)
    G2: Rect,                # runway 2 (Ontario)
    obstacles: Sequence[Rect],
    v_min: float,
    grace_lo: int,
    grace_hi: int,
    goal_z_range: Optional[Tuple[float, float]] = None,
) -> STLNode:
    """
    Example 1 (paper eq. (7), keep-then-reach order):

        phi_1 = []_[t1,t2] H  /\\  <>_[t3,t4] (G1 \\/ G2)
                /\\ []_[0,T] (/\\_j !O_j)            (env obstacles)
                /\\ []_[grace_lo, T-grace_hi] Vfloor (env speed floor)

    Authored part: keep H over [t1,t2], then reach either runway over [t3,t4].
    `H_pred` is the keep-region predicate (InRect or InCircle) built by the caller.

    `goal_z_range` constrains altitude at the reached runway (e.g. (0, 0.02) km)
    so the trajectory must descend to land; H carries no altitude constraint and
    stays at cruise.
    """
    authored = (
        Always(t1, t2, H_pred),
        Eventually(t3, t4, Or((
            InRect("G1", tuple(G1), goal_z_range),
            InRect("G2", tuple(G2), goal_z_range),
        ))),
    )
    env = _env_conjuncts(T, obstacles, v_min, grace_lo, grace_hi)
    return And(authored + env)


def make_paper_example_2(
    *,
    T: int,
    deliveries: Sequence[Tuple[str, Rect, Window]],   # (name, rect, window)
    goal_rects: Sequence[Tuple[str, Rect]],           # runways (Or if >1)
    goal_window: Window,
    obstacles: Sequence[Rect],
    v_min: float, grace_lo: int, grace_hi: int,
    goal_z_range: Optional[Tuple[float, float]] = None,
    delivery_z_range: Optional[Tuple[float, float]] = (0.0, 0.05),
) -> STLNode:
    """
    Example 2 — timed multi-delivery, then land:

        φ_2 = ⋀_i ◇_[wi] P_i  ∧  ◇_[wg] (G1∨G2)  ∧ env

    Each delivery is a reach with its own window; the aircraft must descend into
    `delivery_z_range` (~0-50 m) over each drop, as in planner.examples. The final
    reach lands at a runway.
    """
    authored = []
    for name, rect, win in deliveries:
        t1, t2 = win if win is not None else (0, T)
        authored.append(Eventually(t1, t2, InRect(name, tuple(rect), delivery_z_range)))
    gt1, gt2 = goal_window if goal_window is not None else (0, T)
    authored.append(Eventually(gt1, gt2, _goal_pred(goal_rects, T, goal_z_range)))
    env = _env_conjuncts(T, obstacles, v_min, grace_lo, grace_hi)
    return And(tuple(authored) + env)


def make_paper_example_3(
    *,
    T: int,
    doorkeys: Sequence[Tuple[str, Rect, str, Rect, Window]],  # (Dname,Drect,Kname,Krect,window)
    goal_rects: Sequence[Tuple[str, Rect]],
    goal_window: Window,
    obstacles: Sequence[Rect],
    v_min: float, grace_lo: int, grace_hi: int,
    goal_z_range: Optional[Tuple[float, float]] = None,
) -> STLNode:
    """
    Example 3 — Door–Key (until), then land:

        φ_3 = ⋀_i (¬D_i U_[wi] K_i)  ∧  ◇_[wg] (G1∨G2)  ∧ env

    ¬D_i U K_i: avoid door D_i until key K_i is reached (encoder Until(OutsideRect,
    InRect)). Demonstrates the until operator end to end.
    """
    authored = []
    for dname, drect, kname, krect, win in doorkeys:
        t1, t2 = win if win is not None else (0, T)
        authored.append(Until(t1, t2, OutsideRect(dname, tuple(drect)),
                              InRect(kname, tuple(krect))))
    gt1, gt2 = goal_window if goal_window is not None else (0, T)
    authored.append(Eventually(gt1, gt2, _goal_pred(goal_rects, T, goal_z_range)))
    env = _env_conjuncts(T, obstacles, v_min, grace_lo, grace_hi)
    return And(tuple(authored) + env)


__all__ = ["make_paper_example_1", "make_paper_example_2", "make_paper_example_3"]
