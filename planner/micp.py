# micp.py
"""
Data generation module for neurosymbolic path planning.

Implements:
- STL -> strategic binaries (STLBinaryExtractor)
- MICP (mixed-integer convex programming) construction for Gurobi.
- Strategy extraction (save strategic binaries)
- Dataset generation for GNN training (jsonl.gz)

Current project requirement (patched):
- Ground-truth supervision is the raw strategic binaries `strategy.binaries`
  keyed by BinaryDescriptor.id for ALL descriptors extracted from the AST.
- The dataset must store:
  (1) raw strategic binaries (id -> 0/1)  [authoritative GT for training]
  (2) binary descriptors (id semantics: role/node_tag/time/meta)
  (3) AST (dict) and sample env
  (4) optional solution trajectory (X,U) for analysis/debugging

PATCH NOTES (graph correctness):
- ast_to_dict now includes a deterministic per-node uid:
    uid = f"{node.key()}@{path}"  where root path is "0"
  This matches stl._uid_for / stl.compute_uid_maps and enables unambiguous
  operator-instance attachment in graph_builder without overwrite risk.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .geometry import Rect, rect_in_constraints, rect_outside_bigM, polytope_outside_bigM
from .dynamics import AircraftLTI
from .stl import (
    STLNode,
    InRect,
    InCircle,
    OutsideRect,
    OutsidePolytope,
    OutsideVelocityRect,
    RunwayPredicate,
    And,
    Or,
    Eventually,
    Always,
    Until,
    BinaryDescriptor,
    STLBinaryExtractor,
    STLStrategy,
)


# ============================================================
# 1) Planning problem definition
# ============================================================

@dataclass
class PlanningProblem:
    T: int
    dynamics: AircraftLTI
    x0: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    bigM: float = 1e3
    until_margin: float = 0.5  # km; door of an Until is avoided inflated by this,
                               # so a region cannot be grazed (closed "reach K" vs
                               # open "avoid D" at a shared boundary) — keeps the
                               # ordering of chained untils correct.
    cruise_z_ref: float = 1.2
    v_max: Optional[float] = None  # horizontal speed cap: n=12 circumscribed polygon, faces tangent to L2 circle at v_max
    Q_runway: float = 0.0  # weight on (xy) position-tracking toward the runway center (0 => disabled)


def runway_center_from_ast(ast: STLNode) -> Tuple[float, float]:
    """
    Walk the AST and return the (xy) center of the single RunwayPredicate's
    pos_rect. Used to add a soft position-tracking term toward the runway
    WITHOUT hard-coding it. Raises if there is not exactly one RunwayPredicate.
    """
    found: List[Tuple[float, float, float, float]] = []

    def walk(node: STLNode) -> None:
        if isinstance(node, RunwayPredicate):
            found.append(tuple(node.pos_rect))  # type: ignore[arg-type]
        for c in node.children():
            walk(c)

    walk(ast)

    if len(found) == 0:
        raise ValueError(
            "runway_center_from_ast: no RunwayPredicate found in AST; "
            "cannot infer the runway center for the tracking term."
        )
    if len(found) > 1:
        raise ValueError(
            f"runway_center_from_ast: expected exactly one RunwayPredicate, "
            f"found {len(found)}; cannot infer a unique runway center."
        )
    xmin, xmax, ymin, ymax = found[0]
    return (0.5 * (float(xmin) + float(xmax)), 0.5 * (float(ymin) + float(ymax)))


# ============================================================
# 2) Strategy extraction (ALL strategic binaries)
# ============================================================

class StrategyExtractor:
    """
    Extract ALL binary variables corresponding to ALL BinaryDescriptors.

    Rationale:
      - Training will supervise all binary nodes using BCE.
      - Exact key match: BinaryDescriptor.id <-> strategy.binaries key
    """

    @staticmethod
    def from_model(
        T: int,
        descs: List[BinaryDescriptor],
        bin_vars: Dict[str, gp.Var],
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> STLStrategy:
        binaries: Dict[str, int] = {}
        for d in descs:
            if d.id not in bin_vars:
                raise RuntimeError(f"StrategyExtractor: missing binary var for id={d.id}")
            binaries[d.id] = int(round(bin_vars[d.id].X))
        m = dict(meta) if isinstance(meta, dict) else {}
        m.setdefault("n_descs_total", int(len(descs)))
        return STLStrategy(T=int(T), binaries=binaries, meta=m)


# ============================================================
# 2.5) Descriptor serialization (dataset logging)
# ============================================================

def descs_to_dicts(descs: List[BinaryDescriptor]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in descs:
        out.append(
            {
                "id": str(d.id),
                "role": str(d.role),
                "node_tag": str(d.node_tag),
                "time": (None if d.time is None else int(d.time)),
                "meta": dict(d.meta),
            }
        )
    return out


# ============================================================
# 2.6) Canonical-strategy serialization (compact schema)
# ============================================================
#
# The compact schema replaces the per-binary {id: 0/1} dict + per-binary
# descriptor list with:
#
#   strategy = {
#       "event_time":   { op_uid: k_int },                           # one int per Eventually
#       "event_opt":    { op_uid: opt_int },                         # only when Or-Eventually
#       "outside_face": { ent_name: {"t0": int, "faces": [int8]} },  # max-margin face per (ent, t)
#       "always_or":    { op_uid:  {"t0": int, "opts":  [int]   } }, # only when present
#   }
#
# - outside_face labels are the deterministic max-margin face of X[t] against
#   the entity's rect (uniform rule at every t, no t=0 special case). This
#   removes the ~91% halfplane-tie ambiguity in the raw Gurobi assignment.
# - event_time / event_opt / always_or read from the raw assignment (one-hot
#   by construction in the MICP encoding).

def _max_margin_face(px: float, py: float, rect: Tuple[float, float, float, float]) -> int:
    """Argmax of the four signed halfplane margins (x1-px, px-x2, y1-py, py-y2)."""
    x1, x2, y1, y2 = rect
    margins = (x1 - px, px - x2, y1 - py, py - y2)
    best, best_m = 0, margins[0]
    for h in range(1, 4):
        if margins[h] > best_m:
            best, best_m = h, margins[h]
    return best


def _build_outside_entity_lookup(ast: STLNode) -> Dict[str, Dict[str, Any]]:
    """Walk AST and collect every OutsideRect / OutsideVelocityRect leaf,
    keyed by `.name`. Each value carries the rect and the state-index pair
    needed to evaluate max-margin against X[t].
    """
    lookup: Dict[str, Dict[str, Any]] = {}

    def walk(n: STLNode) -> None:
        if isinstance(n, OutsideRect):
            lookup[str(n.name)] = {
                "rect": (float(n.rect[0]), float(n.rect[1]), float(n.rect[2]), float(n.rect[3])),
                "state_idx": (0, 1),
            }
        elif isinstance(n, OutsideVelocityRect):
            v = float(n.v_min)
            lookup[str(n.name)] = {
                "rect": (-v, v, -v, v),
                "state_idx": (3, 4),
            }
        elif isinstance(n, OutsidePolytope):
            raise NotImplementedError(
                "canonicalize_strategy: OutsidePolytope canonicalization not yet supported."
            )
        if isinstance(n, (And, Or)):
            for c in n.args:
                walk(c)
        elif isinstance(n, (Always, Eventually)):
            walk(n.phi)
        elif isinstance(n, Until):
            walk(n.phi)
            walk(n.psi)

    walk(ast)
    return lookup


def canonicalize_strategy(
    *,
    descs: List[BinaryDescriptor],
    strategy_binaries: Dict[str, int],
    X: Any,
    ast: STLNode,
) -> Dict[str, Any]:
    """Build the compact, canonical strategy dict from solved MICP outputs.

    - outside_face: max-margin face of X[t] vs. the entity rect (uniform at all t).
    - event_time / event_opt / always_or: read one-hot from `strategy_binaries`.
    """
    if X is None:
        raise ValueError("canonicalize_strategy: X is required.")

    ent_lookup = _build_outside_entity_lookup(ast)

    of_t_range: Dict[str, Tuple[int, int]] = {}                            # ent -> (t_min, t_max)
    et_groups: Dict[str, List[BinaryDescriptor]] = {}                      # op_uid -> [descs]
    ao_groups: Dict[Tuple[str, int], List[BinaryDescriptor]] = {}          # (op_uid, k) -> [descs]

    for d in descs:
        if d.role == "outside_face":
            ent = str(d.meta.get("ent_name") or d.meta.get("pred") or "")
            if ent == "":
                continue
            t = int(d.meta.get("t", d.meta.get("k", -1)))
            if t < 0:
                continue
            lo, hi = of_t_range.get(ent, (t, t))
            of_t_range[ent] = (min(lo, t), max(hi, t))
        elif d.role == "event_time":
            et_groups.setdefault(str(d.node_tag), []).append(d)
        elif d.role == "always_or_choice":
            k = int(d.meta.get("k", -1))
            if k < 0:
                continue
            ao_groups.setdefault((str(d.node_tag), k), []).append(d)

    outside_face: Dict[str, Dict[str, Any]] = {}
    for ent, (t_min, t_max) in of_t_range.items():
        info = ent_lookup.get(ent)
        if info is None:
            raise RuntimeError(
                f"canonicalize_strategy: entity {ent!r} has outside_face binaries "
                f"but no matching OutsideRect/OutsideVelocityRect in AST."
            )
        rect = info["rect"]
        i_px, i_py = info["state_idx"]
        if t_max >= len(X):
            raise RuntimeError(
                f"canonicalize_strategy: entity {ent!r} needs X[{t_max}] but |X|={len(X)}."
            )
        faces: List[int] = []
        for t in range(t_min, t_max + 1):
            xt = X[t]
            px = float(xt[i_px])
            py = float(xt[i_py])
            faces.append(int(_max_margin_face(px, py, rect)))
        outside_face[ent] = {"t0": int(t_min), "faces": faces}

    event_time: Dict[str, int] = {}
    event_opt:  Dict[str, int] = {}
    for op_uid, group in et_groups.items():
        active = [d for d in group if int(strategy_binaries.get(d.id, 0)) == 1]
        if len(active) == 0:
            raise RuntimeError(
                f"canonicalize_strategy: event_time group {op_uid!r} has no active binary."
            )
        if len(active) > 1:
            active.sort(key=lambda dd: (int(dd.meta.get("k", 0)), int(dd.meta.get("opt", 0))))
        d = active[0]
        event_time[op_uid] = int(d.meta["k"])
        if "opt" in d.meta:
            event_opt[op_uid] = int(d.meta["opt"])

    always_or: Dict[str, Dict[str, Any]] = {}
    if ao_groups:
        per_op: Dict[str, Dict[int, int]] = {}
        for (op_uid, k), group in ao_groups.items():
            active = [d for d in group if int(strategy_binaries.get(d.id, 0)) == 1]
            if len(active) == 0:
                raise RuntimeError(
                    f"canonicalize_strategy: always_or_choice group ({op_uid!r}, k={k}) "
                    "has no active binary."
                )
            if len(active) > 1:
                active.sort(key=lambda dd: int(dd.meta.get("opt", 0)))
            per_op.setdefault(op_uid, {})[k] = int(active[0].meta["opt"])
        for op_uid, k_map in per_op.items():
            ks = sorted(k_map.keys())
            k_min, k_max = ks[0], ks[-1]
            opts = [int(k_map[k]) for k in range(k_min, k_max + 1)]
            always_or[op_uid] = {"t0": int(k_min), "opts": opts}

    out: Dict[str, Any] = {
        "event_time": event_time,
        "outside_face": outside_face,
    }
    if event_opt:
        out["event_opt"] = event_opt
    if always_or:
        out["always_or"] = always_or
    return out


# ============================================================
# 3) MICP builder
# ============================================================

class MICPBuilder:
    def __init__(self, problem: PlanningProblem, ast: STLNode):
        self.p = problem
        self.ast = ast
        self.model: Optional[gp.Model] = None
        self.x = None
        self.u = None
        self.bin_vars: Dict[str, gp.Var] = {}

        self._outside_face_id: Dict[Tuple[str, int, int], str] = {}
        self._event_time_id: Dict[Tuple[str, int, Optional[int]], str] = {}
        self._always_or_id: Dict[Tuple[str, int, int], str] = {}

    def build(self, descs: List[BinaryDescriptor], *, output_flag: int = 0) -> gp.Model:
        T = int(self.p.T)
        A, B = self.p.dynamics.matrices()

        m = gp.Model("stl_micp")
        m.Params.OutputFlag = int(output_flag)
        m.Params.Threads = 8  # small integers for preventing segmentation faults (observed empirically)
        self.model = m

        self.x = m.addVars(T + 1, 6, lb=-GRB.INFINITY, name="x")
        self.u = m.addVars(T, 3, lb=-GRB.INFINITY, name="u")

        for k in range(T + 1):
            for i in range(6):
                m.addConstr(self.x[k, i] >= float(self.p.x_min[i]), name=f"xmin[{k},{i}]")
                m.addConstr(self.x[k, i] <= float(self.p.x_max[i]), name=f"xmax[{k},{i}]")
        for k in range(T):
            for i in range(3):
                m.addConstr(self.u[k, i] >= float(self.p.u_min[i]), name=f"umin[{k},{i}]")
                m.addConstr(self.u[k, i] <= float(self.p.u_max[i]), name=f"umax[{k},{i}]")

        if self.p.v_max is not None:
            _v_max = float(self.p.v_max)
            _n_hp = 24
            _angles = [2.0 * math.pi * _ki / _n_hp for _ki in range(_n_hp)]
            for k in range(T + 1):
                for _ki, _ang in enumerate(_angles):
                    m.addConstr(
                        math.cos(_ang) * self.x[k, 3] + math.sin(_ang) * self.x[k, 4] <= _v_max,
                        name=f"v_max_hp{_ki}[{k}]",
                    )

        x0 = np.asarray(self.p.x0, dtype=float).reshape(-1)
        if x0.shape[0] != 6:
            raise ValueError("x0 must have shape (6,)")
        for i in range(6):
            m.addConstr(self.x[0, i] == float(x0[i]), name=f"x0[{i}]")

        for k in range(T):
            for i in range(6):
                m.addConstr(
                    self.x[k + 1, i]
                    == gp.quicksum(float(A[i, j]) * self.x[k, j] for j in range(6))
                    + gp.quicksum(float(B[i, j]) * self.u[k, j] for j in range(3)),
                    name=f"dyn[{k},{i}]",
                )

        # Create ALL binary variables for ALL descriptors.
        for bd in descs:
            if bd.id not in self.bin_vars:
                # name=... must be unique; sanitize ":" for Gurobi's name display only
                self.bin_vars[bd.id] = m.addVar(vtype=GRB.BINARY, name=bd.id.replace(":", "_"))
        m.update()

        self._outside_face_id = {}
        self._event_time_id = {}
        self._always_or_id = {}

        for d in descs:
            if d.role == "outside_face":
                pred = d.meta.get("pred", d.meta.get("rect"))
                if pred is None:
                    raise RuntimeError(f"outside_face descriptor missing pred/rect in meta: {d.meta}")

                t = int(d.meta["t"]) if "t" in d.meta else int(d.meta["k"])
                face = int(d.meta["face"]) if "face" in d.meta else int(d.meta["h"])

                key = (str(pred), t, face)
                self._outside_face_id[key] = d.id

            elif d.role == "event_time":
                pred = d.meta.get("pred")
                k = int(d.meta["k"])
                opt = d.meta.get("opt", None)
                opt = int(opt) if opt is not None else None
                key = (str(pred), k, opt)
                self._event_time_id[key] = d.id

            elif d.role == "always_or_choice":
                pred = d.meta.get("pred")
                k = int(d.meta["k"])
                opt = int(d.meta["opt"])
                key = (str(pred), k, opt)
                self._always_or_id[key] = d.id

        self._add_stl_constraints(self.ast)

        Q = 0.5 * (np.asarray(self.p.Q, dtype=float) + np.asarray(self.p.Q, dtype=float).T)
        R = 0.5 * (np.asarray(self.p.R, dtype=float) + np.asarray(self.p.R, dtype=float).T)

        obj = gp.QuadExpr()

        for k in range(T + 1):
            for i in range(6):
                xi = self.x[k, i]
                for j in range(6):
                    qij = float(Q[i, j])
                    if qij != 0.0:
                        obj += qij * xi * self.x[k, j]

        for k in range(T):
            for i in range(3):
                ui = self.u[k, i]
                for j in range(3):
                    rij = float(R[i, j])
                    if rij != 0.0:
                        obj += rij * ui * self.u[k, j]

        # Soft cruise altitude penalty: encourage pz ≈ cruise_z_ref
        cruise_q = 1e-3
        z_ref = float(self.p.cruise_z_ref)
        for k in range(T + 1):
            zk = self.x[k, 2]
            obj += cruise_q * zk * zk - 2.0 * cruise_q * z_ref * zk

        # Soft (xy) position-tracking term: pull the trajectory toward the
        # runway center inferred from the STL spec. With Q_runway == 0 this adds
        # nothing and the objective reproduces the original Q/R behavior exactly.
        # A small weight keeps deliveries (hard STL) intact while parking the
        # aircraft at the runway once the mission is done.
        q_runway = float(getattr(self.p, "Q_runway", 0.0))
        if q_runway != 0.0:
            rwx, rwy = runway_center_from_ast(self.ast)
            for k in range(T + 1):
                dpx = self.x[k, 0] - rwx
                dpy = self.x[k, 1] - rwy
                obj += q_runway * (dpx * dpx + dpy * dpy)

        m.setObjective(obj, GRB.MINIMIZE)
        return m

    def _add_stl_constraints(self, node: STLNode) -> None:
        if isinstance(node, And):
            for c in node.args:
                self._add_stl_constraints(c)
            return

        if isinstance(node, Always):
            self._add_always(node.a, node.b, node.phi)
            return

        if isinstance(node, Eventually):
            self._add_eventually(node.a, node.b, node.phi)
            return

        if isinstance(node, Until):
            self._add_until(node.a, node.b, node.phi, node.psi)
            return

        raise NotImplementedError("Top-level predicate must be under temporal operator in this pipeline.")

    def _add_always(self, a: int, b: int, phi: STLNode) -> None:
        assert self.model is not None

        # G[a,b](InRect) — "keep": stay inside a region over the whole window.
        # Pure conjunction of halfplanes at every timestep; no binaries needed.
        if isinstance(phi, InRect):
            name = phi.name
            for k in range(a, b + 1):
                rect_in_constraints(
                    self.model,
                    self.x[k, 0],
                    self.x[k, 1],
                    phi.rect,
                    prefix=f"always_in_{name}[{k}]",
                )
                if phi.z_range is not None:
                    self.model.addConstr(self.x[k, 2] >= float(phi.z_range[0]), name=f"always_in_{name}[{k}]_zmin")
                    self.model.addConstr(self.x[k, 2] <= float(phi.z_range[1]), name=f"always_in_{name}[{k}]_zmax")
            return

        # G[a,b](InCircle) — "keep" in a circular region (inscribed n-gon).
        # Conjunction of the n-gon halfplanes at every timestep; no binaries.
        if isinstance(phi, InCircle):
            name = phi.name
            hps = phi.halfplanes()
            for k in range(a, b + 1):
                for fi, (ah, bh, ch) in enumerate(hps):
                    self.model.addConstr(
                        ah * self.x[k, 0] + bh * self.x[k, 1] <= ch,
                        name=f"always_incircle_{name}[{k}]_hp{fi}",
                    )
                if phi.z_range is not None:
                    self.model.addConstr(self.x[k, 2] >= float(phi.z_range[0]), name=f"always_incircle_{name}[{k}]_zmin")
                    self.model.addConstr(self.x[k, 2] <= float(phi.z_range[1]), name=f"always_incircle_{name}[{k}]_zmax")
            return

        if isinstance(phi, OutsidePolytope):
            poly_name = phi.name
            poly_A = phi.A
            poly_b = phi.b
            K = phi.n_faces
            for k in range(a, b + 1):
                face = []
                for h in range(K):
                    bid = self._outside_face_id.get((poly_name, k, h))
                    if bid is None:
                        raise RuntimeError(
                            f"Missing outside_face binary for (poly={poly_name}, time={k}, h={h})"
                        )
                    face.append(self.bin_vars[bid])
                polytope_outside_bigM(
                    self.model,
                    self.x[k, 0], self.x[k, 1],
                    poly_A, poly_b, face,
                    prefix=f"always_out_{poly_name}[{k}]",
                    M=self.p.bigM,
                    rhs=1.0,
                )
            return

        if isinstance(phi, OutsideRect):
            rect = phi.rect
            name = phi.name
            for k in range(a, b + 1):
                face = []
                for h in range(4):
                    bid = self._outside_face_id.get((name, k, h))
                    if bid is None:
                        raise RuntimeError(f"Missing outside_face binary for (rect={name}, time={k}, h={h})")
                    face.append(self.bin_vars[bid])
                rect_outside_bigM(
                    self.model,
                    self.x[k, 0],
                    self.x[k, 1],
                    rect,
                    face,
                    prefix=f"always_out_{name}[{k}]",
                    M=self.p.bigM,
                    rhs=1.0,
                )
            return

        if isinstance(phi, OutsideVelocityRect):
            v_min = float(phi.v_min)
            v_rect: Rect = (-v_min, v_min, -v_min, v_min)
            name = phi.name
            for k in range(a, b + 1):
                face = []
                for h in range(4):
                    bid = self._outside_face_id.get((name, k, h))
                    if bid is None:
                        raise RuntimeError(f"Missing outside_face binary for (pred={name}, time={k}, h={h})")
                    face.append(self.bin_vars[bid])
                rect_outside_bigM(
                    self.model,
                    self.x[k, 3],
                    self.x[k, 4],
                    v_rect,
                    face,
                    prefix=f"always_vfloor_{name}[{k}]",
                    M=self.p.bigM,
                    rhs=1.0,
                )
            return

        if isinstance(phi, Or) and all(isinstance(arg, (InRect, InCircle)) for arg in phi.args):
            args = list(phi.args)
            names = [arg.name for arg in args]
            for k in range(a, b + 1):
                sel = []
                for i, nm in enumerate(names):
                    bid = self._always_or_id.get((nm, k, i))
                    if bid is None:
                        raise RuntimeError(f"Missing always_or_choice binary for (pred={nm}, time={k}, opt={i})")
                    sel.append(self.bin_vars[bid])

                self.model.addConstr(gp.quicksum(sel) >= 1, name=f"always_or_atleast1[k{k}]")

                for i, arg in enumerate(args):
                    if isinstance(arg, InRect):
                        xmin, xmax, ymin, ymax = arg.rect
                        self.model.addGenConstrIndicator(sel[i], True, self.x[k, 0] >= xmin, name=f"sel_in_{names[i]}[{k}]_xmin")
                        self.model.addGenConstrIndicator(sel[i], True, self.x[k, 0] <= xmax, name=f"sel_in_{names[i]}[{k}]_xmax")
                        self.model.addGenConstrIndicator(sel[i], True, self.x[k, 1] >= ymin, name=f"sel_in_{names[i]}[{k}]_ymin")
                        self.model.addGenConstrIndicator(sel[i], True, self.x[k, 1] <= ymax, name=f"sel_in_{names[i]}[{k}]_ymax")
                    else:  # InCircle
                        for fi, (ah, bh, ch) in enumerate(arg.halfplanes()):
                            self.model.addGenConstrIndicator(
                                sel[i], True, ah * self.x[k, 0] + bh * self.x[k, 1] <= ch,
                                name=f"sel_incircle_{names[i]}[{k}]_hp{fi}")
            return

        if isinstance(phi, And):
            for c in phi.args:
                self._add_always(a, b, c)
            return

        raise NotImplementedError("Always supports OutsideRect, OutsideVelocityRect, Or(InRect,...), or And of those.")

    def _add_eventually(self, a: int, b: int, phi: STLNode) -> None:
        assert self.model is not None

        # F[a,b](Or(Always(a2,b2, InRect|InCircle), ...)) — loiter dwell (Ex 2)
        if (isinstance(phi, Or)
                and all(isinstance(arg, Always) and isinstance(arg.phi, (InRect, InCircle))
                        for arg in phi.args)):
            alw_args = list(phi.args)
            a2 = int(alw_args[0].a)
            b2 = int(alw_args[0].b)
            for arg in alw_args:
                if int(arg.a) != a2 or int(arg.b) != b2:
                    raise NotImplementedError(
                        "Eventually(Or(Always(...))) requires identical Always intervals across options."
                    )

            bvars = []
            for k in range(a, b + 1):
                for i, alw in enumerate(alw_args):
                    pred = alw.phi
                    nm = pred.name

                    bid = self._event_time_id.get((nm, k, i))
                    if bid is None:
                        raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k}, opt={i})")
                    bk = self.bin_vars[bid]
                    bvars.append(bk)

                    t_lo = k + a2
                    t_hi = k + b2
                    if t_lo < 0 or t_hi > self.p.T:
                        raise ValueError(
                            f"Dwell window out of bounds: k={k}, [a2,b2]=[{a2},{b2}], T={self.p.T}"
                        )

                    if isinstance(pred, InRect):
                        xmin, xmax, ymin, ymax = pred.rect
                        for t in range(t_lo, t_hi + 1):
                            self.model.addGenConstrIndicator(bk, True, self.x[t, 0] >= xmin, name=f"evt_dwell_{nm}[k{k},t{t}]_xmin")
                            self.model.addGenConstrIndicator(bk, True, self.x[t, 0] <= xmax, name=f"evt_dwell_{nm}[k{k},t{t}]_xmax")
                            self.model.addGenConstrIndicator(bk, True, self.x[t, 1] >= ymin, name=f"evt_dwell_{nm}[k{k},t{t}]_ymin")
                            self.model.addGenConstrIndicator(bk, True, self.x[t, 1] <= ymax, name=f"evt_dwell_{nm}[k{k},t{t}]_ymax")
                            if pred.z_range is not None:
                                self.model.addGenConstrIndicator(bk, True, self.x[t, 2] >= float(pred.z_range[0]), name=f"evt_dwell_{nm}[k{k},t{t}]_zmin")
                                self.model.addGenConstrIndicator(bk, True, self.x[t, 2] <= float(pred.z_range[1]), name=f"evt_dwell_{nm}[k{k},t{t}]_zmax")

                    elif isinstance(pred, InCircle):
                        halfplanes = pred.halfplanes()
                        for t in range(t_lo, t_hi + 1):
                            for fi, (a_hp, b_hp, c_hp) in enumerate(halfplanes):
                                self.model.addGenConstrIndicator(
                                    bk, True,
                                    a_hp * self.x[t, 0] + b_hp * self.x[t, 1] <= c_hp,
                                    name=f"evt_dwell_{nm}[k{k},t{t}]_hp{fi}"
                                )
                            if pred.z_range is not None:
                                self.model.addGenConstrIndicator(bk, True, self.x[t, 2] >= float(pred.z_range[0]), name=f"evt_dwell_{nm}[k{k},t{t}]_zmin")
                                self.model.addGenConstrIndicator(bk, True, self.x[t, 2] <= float(pred.z_range[1]), name=f"evt_dwell_{nm}[k{k},t{t}]_zmax")

            self.model.addConstr(gp.quicksum(bvars) == 1, name="evt_select_or_always")
            return

        # F[a,b](InRect)
        if isinstance(phi, InRect):
            rect = phi.rect
            nm = phi.name
            bvars = []
            for k in range(a, b + 1):
                bid = self._event_time_id.get((nm, k, None))
                if bid is None:
                    raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k})")
                bk = self.bin_vars[bid]
                bvars.append(bk)

                xmin, xmax, ymin, ymax = rect
                self.model.addGenConstrIndicator(bk, True, self.x[k, 0] >= xmin, name=f"evt_in_{nm}[{k}]_xmin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 0] <= xmax, name=f"evt_in_{nm}[{k}]_xmax")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 1] >= ymin, name=f"evt_in_{nm}[{k}]_ymin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 1] <= ymax, name=f"evt_in_{nm}[{k}]_ymax")
                if phi.z_range is not None:
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 2] >= float(phi.z_range[0]), name=f"evt_in_{nm}[{k}]_zmin")
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 2] <= float(phi.z_range[1]), name=f"evt_in_{nm}[{k}]_zmax")

            self.model.addConstr(gp.quicksum(bvars) == 1, name=f"evt_select_{nm}")
            return

        # F[a,b](InCircle) — reach a circular region (inscribed n-gon halfplanes)
        if isinstance(phi, InCircle):
            nm = phi.name
            hps = phi.halfplanes()
            bvars = []
            for k in range(a, b + 1):
                bid = self._event_time_id.get((nm, k, None))
                if bid is None:
                    raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k})")
                bk = self.bin_vars[bid]
                bvars.append(bk)
                for fi, (ah, bh, ch) in enumerate(hps):
                    self.model.addGenConstrIndicator(
                        bk, True, ah * self.x[k, 0] + bh * self.x[k, 1] <= ch,
                        name=f"evt_incircle_{nm}[{k}]_hp{fi}")
                if phi.z_range is not None:
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 2] >= float(phi.z_range[0]), name=f"evt_incircle_{nm}[{k}]_zmin")
                    self.model.addGenConstrIndicator(bk, True, self.x[k, 2] <= float(phi.z_range[1]), name=f"evt_incircle_{nm}[{k}]_zmax")
            self.model.addConstr(gp.quicksum(bvars) == 1, name=f"evt_select_{nm}")
            return

        # F[a,b](RunwayPredicate)
        if isinstance(phi, RunwayPredicate):
            nm = phi.name
            pos_rect = phi.pos_rect
            z_lo, z_hi = float(phi.z_bounds[0]), float(phi.z_bounds[1])
            v_land = float(phi.v_land)
            theta = float(phi.theta_rwy)
            heading_bound = math.sqrt(2.0) * v_land * math.sin(float(phi.delta_theta))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            bvars = []
            for k in range(a, b + 1):
                bid = self._event_time_id.get((nm, k, None))
                if bid is None:
                    raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k})")
                bk = self.bin_vars[bid]
                bvars.append(bk)

                xmin, xmax, ymin, ymax = pos_rect
                # 1. Position in pos_rect
                self.model.addGenConstrIndicator(bk, True, self.x[k, 0] >= xmin, name=f"rwy_{nm}[{k}]_xmin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 0] <= xmax, name=f"rwy_{nm}[{k}]_xmax")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 1] >= ymin, name=f"rwy_{nm}[{k}]_ymin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 1] <= ymax, name=f"rwy_{nm}[{k}]_ymax")
                # 2. Altitude
                self.model.addGenConstrIndicator(bk, True, self.x[k, 2] >= z_lo, name=f"rwy_{nm}[{k}]_zmin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 2] <= z_hi, name=f"rwy_{nm}[{k}]_zmax")
                # 3. Speed (infinity norm on all axes)
                self.model.addGenConstrIndicator(bk, True, self.x[k, 3] >= -v_land, name=f"rwy_{nm}[{k}]_vxmin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 3] <= v_land,  name=f"rwy_{nm}[{k}]_vxmax")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 4] >= -v_land, name=f"rwy_{nm}[{k}]_vymin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 4] <= v_land,  name=f"rwy_{nm}[{k}]_vymax")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 5] >= -v_land, name=f"rwy_{nm}[{k}]_vzmin")
                self.model.addGenConstrIndicator(bk, True, self.x[k, 5] <= v_land,  name=f"rwy_{nm}[{k}]_vzmax")
                # 4. Heading alignment: |vy*cos(theta) - vx*sin(theta)| <= bound
                self.model.addGenConstrIndicator(
                    bk, True,
                    self.x[k, 4] * cos_t - self.x[k, 3] * sin_t <= heading_bound,
                    name=f"rwy_{nm}[{k}]_hdg_pos"
                )
                self.model.addGenConstrIndicator(
                    bk, True,
                    self.x[k, 4] * cos_t - self.x[k, 3] * sin_t >= -heading_bound,
                    name=f"rwy_{nm}[{k}]_hdg_neg"
                )

            self.model.addConstr(gp.quicksum(bvars) == 1, name=f"evt_select_{nm}")
            return

        # F[a,b](Or(InRect|InCircle, ...))
        if isinstance(phi, Or) and all(isinstance(arg, (InRect, InCircle)) for arg in phi.args):
            args = list(phi.args)
            names = [arg.name for arg in args]
            bvars = []
            for k in range(a, b + 1):
                for i, arg in enumerate(args):
                    nm = names[i]
                    bid = self._event_time_id.get((nm, k, i))
                    if bid is None:
                        raise RuntimeError(f"Missing event_time binary for (pred={nm}, time={k}, opt={i})")
                    bk = self.bin_vars[bid]
                    bvars.append(bk)

                    if isinstance(arg, InRect):
                        xmin, xmax, ymin, ymax = arg.rect
                        self.model.addGenConstrIndicator(bk, True, self.x[k, 0] >= xmin, name=f"evt_or_in_{nm}[k{k}]_xmin")
                        self.model.addGenConstrIndicator(bk, True, self.x[k, 0] <= xmax, name=f"evt_or_in_{nm}[k{k}]_xmax")
                        self.model.addGenConstrIndicator(bk, True, self.x[k, 1] >= ymin, name=f"evt_or_in_{nm}[k{k}]_ymin")
                        self.model.addGenConstrIndicator(bk, True, self.x[k, 1] <= ymax, name=f"evt_or_in_{nm}[k{k}]_ymax")
                        zr = arg.z_range
                        if zr is not None:
                            self.model.addGenConstrIndicator(bk, True, self.x[k, 2] >= float(zr[0]), name=f"evt_or_in_{nm}[k{k}]_zmin")
                            self.model.addGenConstrIndicator(bk, True, self.x[k, 2] <= float(zr[1]), name=f"evt_or_in_{nm}[k{k}]_zmax")
                    else:  # InCircle
                        for fi, (ah, bh, ch) in enumerate(arg.halfplanes()):
                            self.model.addGenConstrIndicator(
                                bk, True, ah * self.x[k, 0] + bh * self.x[k, 1] <= ch,
                                name=f"evt_or_incircle_{nm}[k{k}]_hp{fi}")
                        if arg.z_range is not None:
                            self.model.addGenConstrIndicator(bk, True, self.x[k, 2] >= float(arg.z_range[0]), name=f"evt_or_incircle_{nm}[k{k}]_zmin")
                            self.model.addGenConstrIndicator(bk, True, self.x[k, 2] <= float(arg.z_range[1]), name=f"evt_or_incircle_{nm}[k{k}]_zmax")

            self.model.addConstr(gp.quicksum(bvars) == 1, name="evt_select_or_inrect")
            return

        raise NotImplementedError("Eventually supports InRect, InCircle, RunwayPredicate, Or(InRect|InCircle,...), or Or(Always(InRect|InCircle),...).")

    def _add_until(self, a: int, b: int, phi: STLNode, psi: STLNode) -> None:
        assert self.model is not None

        if not isinstance(phi, OutsideRect):
            raise NotImplementedError("Until left side must be OutsideRect(D) in this project.")
        if not isinstance(psi, InRect):
            raise NotImplementedError("Until right side must be InRect(K) in this project.")

        # Inflate the door by until_margin so it cannot be grazed on its boundary
        # (closed "reach K" vs open "avoid D" would otherwise coincide at an edge
        # when a region is both a key and a door in a chain of untils).
        _m = float(self.p.until_margin)
        d_rect = (phi.rect[0] - _m, phi.rect[1] + _m, phi.rect[2] - _m, phi.rect[3] + _m)
        d_name = phi.name
        k_rect = psi.rect
        k_name = psi.name

        # ------------------------------------------------------------
        # 1) Trigger time selection: choose exactly one t in [a,b]
        #    such that x[t] ∈ K (psi).
        # ------------------------------------------------------------
        b_by_t: Dict[int, gp.Var] = {}
        for t in range(a, b + 1):
            bid = self._event_time_id.get((k_name, t, None))
            if bid is None:
                raise RuntimeError(f"Missing event_time binary for (pred={k_name}, time={t})")
            bt = self.bin_vars[bid]
            b_by_t[int(t)] = bt

            xmin, xmax, ymin, ymax = k_rect
            self.model.addGenConstrIndicator(bt, True, self.x[t, 0] >= xmin, name=f"until_goal_{k_name}[{t}]_xmin")
            self.model.addGenConstrIndicator(bt, True, self.x[t, 0] <= xmax, name=f"until_goal_{k_name}[{t}]_xmax")
            self.model.addGenConstrIndicator(bt, True, self.x[t, 1] >= ymin, name=f"until_goal_{k_name}[{t}]_ymin")
            self.model.addGenConstrIndicator(bt, True, self.x[t, 1] <= ymax, name=f"until_goal_{k_name}[{t}]_ymax")

        self.model.addConstr(gp.quicksum(b_by_t.values()) == 1, name=f"until_select_trigger[{k_name}]")

        # ------------------------------------------------------------
        # 2) Safety BEFORE trigger: enforce x[t] outside D only for t < t*
        # ------------------------------------------------------------
        for t in range(0, self.p.T + 1):
            face: List[gp.Var] = []
            for h in range(4):
                bid = self._outside_face_id.get((d_name, t, h))
                if bid is None:
                    raise RuntimeError(f"Missing outside_face binary for (rect={d_name}, time={t}, h={h})")
                face.append(self.bin_vars[bid])

            if t < a:
                pre_t = 1.0
            elif t >= b:
                pre_t = 0.0
            else:
                pre_t = gp.quicksum(b_by_t[tau] for tau in range(t + 1, b + 1))

            rect_outside_bigM(
                self.model,
                self.x[t, 0],
                self.x[t, 1],
                d_rect,
                face,
                prefix=f"until_safe_{d_name}[{t}]",
                M=self.p.bigM,
                rhs=pre_t,
            )


def solve_micp_with_strategy(
    ast: STLNode,
    prob: PlanningProblem,
    *,
    output_flag: int = 0,
    time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, STLStrategy, float]:
    T = int(prob.T)

    descs = STLBinaryExtractor(T).extract(ast)

    builder = MICPBuilder(prob, ast)
    model = builder.build(descs, output_flag=output_flag)

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    t0 = time.time()
    model.optimize()
    t_sec = time.time() - t0

    has_solution = model.SolCount > 0
    if model.Status == GRB.OPTIMAL:
        pass
    elif model.Status == GRB.TIME_LIMIT and has_solution:
        pass
    else:
        raise RuntimeError(
            f"Gurobi status={model.Status}, SolCount={model.SolCount}. No feasible solution."
        )

    X = np.zeros((T + 1, 6), dtype=float)
    U = np.zeros((T, 3), dtype=float)

    for k in range(T + 1):
        for i in range(6):
            X[k, i] = float(builder.x[k, i].X)
    for k in range(T):
        for i in range(3):
            U[k, i] = float(builder.u[k, i].X)

    strategy = StrategyExtractor.from_model(T, descs, builder.bin_vars, meta={"gurobi_status": int(model.Status)})
    return X, U, strategy, float(t_sec)


# ============================================================
# Bisection solver
# ============================================================

def _is_feasible(
    ast: STLNode,
    prob: PlanningProblem,
    *,
    output_flag: int = 0,
    time_limit: Optional[float] = None,
) -> bool:
    """Return True iff the MICP admits a feasible solution within the time budget."""
    descs = STLBinaryExtractor(int(prob.T)).extract(ast)
    builder = MICPBuilder(prob, ast)
    model = builder.build(descs, output_flag=output_flag)
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        return True
    if model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        return True
    return False


def solve_with_bisection(
    ast_fn,
    prob_fn,
    T_min: int,
    T_max: int,
    *,
    output_flag: int = 0,
    verbose: bool = False,
    time_limit: Optional[float] = None,
) -> Tuple[int, np.ndarray, np.ndarray, Any, float]:
    """
    Find the minimum T* in [T_min, T_max] such that MICP(T*) is feasible,
    then return its full solution.

    A timed-out feasibility check with no solution is treated as INFEASIBLE
    (conservative).  The final recovery solve accepts TIME_LIMIT-with-solution.

    Returns: (T*, X, U, strategy, total_solve_time_sec)
    Raises:  RuntimeError if no feasible T exists in [T_min, T_max].
    """
    import math as _math
    import sys as _sys

    T_lo, T_hi = int(T_min), int(T_max)
    if T_lo > T_hi:
        raise ValueError(f"T_min={T_min} > T_max={T_max}")

    n_steps = _math.ceil(_math.log2(T_hi - T_lo + 1)) if T_hi > T_lo else 1
    step = 0
    t_total = 0.0

    if verbose:
        tl_str = f", timeout={time_limit}s/solve" if time_limit is not None else ""
        print(f"  Bisect  T ∈ [{T_lo}, {T_hi}]  (≤ {n_steps} feasibility solves{tl_str})")

    while T_lo < T_hi:
        T_mid = (T_lo + T_hi) // 2
        step += 1
        if verbose:
            print(f"  [{step}/{n_steps}] T_mid={T_mid:3d}  solving ...", flush=True)

        t0 = time.time()
        feasible = _is_feasible(
            ast_fn(T_mid), prob_fn(T_mid),
            output_flag=output_flag, time_limit=time_limit,
        )
        dt = time.time() - t0
        t_total += dt

        timed_out = time_limit is not None and dt >= time_limit * 0.99
        if verbose:
            tag = "FEASIBLE  " if feasible else "INFEASIBLE"
            note = "  [TIMEOUT→INFEASIBLE]" if (timed_out and not feasible) else ""
            new_lo = T_lo if feasible else T_mid + 1
            new_hi = T_mid if feasible else T_hi
            result = (f"  [{step}/{n_steps}] T_mid={T_mid:3d}  {tag}"
                      f"  ({dt:.1f}s)  → range [{new_lo}, {new_hi}]{note}")
            _sys.stdout.write(f"\033[1A\033[2K{result}\n")
            _sys.stdout.flush()

        if feasible:
            T_hi = T_mid
        else:
            T_lo = T_mid + 1

    T_star = T_lo
    while True:
        if verbose:
            print(f"  T* = {T_star}  → final solve for X, U ...")
        t0 = time.time()
        try:
            X, U, strategy, _ = solve_micp_with_strategy(
                ast_fn(T_star), prob_fn(T_star),
                output_flag=output_flag, time_limit=time_limit,
            )
            t_total += time.time() - t0
            break
        except RuntimeError as e:
            t_total += time.time() - t0
            if T_star >= T_max:
                raise RuntimeError(
                    f"Bisection final solve failed at T={T_star} (T_max reached): {e}"
                ) from e
            if verbose:
                print(f"  [WARN] Final solve failed at T={T_star} ({e}); retrying at T={T_star+1} ...")
            T_star += 1

    if verbose:
        print(f"  Done. total wall time = {t_total:.1f}s")

    return T_star, X, U, strategy, t_total


# ============================================================
# 5) AST serialization
# ============================================================

def ast_to_dict(node: STLNode, *, _path: str = "0", geometry: bool = True) -> Dict[str, Any]:
    """Serialize STL AST to a JSON-friendly dict, including deterministic per-node uid.

    uid scheme matches stl.compute_uid_maps / stl._uid_for:
      uid = f"{node.key()}@{path}"
    where `path` is a structural path (root "0").

    geometry=True : include rect / pos_rect / center / radius / A / b geometry
                    on predicate leaves (the original / legacy form).
    geometry=False: strip those geometric fields. The reader joins predicates
                    back to environment geometry via `name` (matching
                    sample.env.obstacles[j] for "O{j}", sample.env.regions[k]
                    for region names). z_range / z_bounds and scalar predicate
                    fields (v_min, v_land, theta_rwy, delta_theta, n_sides) are
                    STL semantics and are kept regardless of `geometry`.
    """
    uid = f"{node.key()}@{_path}"

    if isinstance(node, InRect):
        out: Dict[str, Any] = {"type": "InRect", "uid": uid, "name": node.name}
        if geometry:
            out["rect"] = list(node.rect)
        if node.z_range is not None:
            out["z_range"] = list(node.z_range)
        return out
    if isinstance(node, InCircle):
        out = {
            "type": "InCircle", "uid": uid, "name": node.name,
            "n_sides": int(node.n_sides),
        }
        if geometry:
            out["center"] = list(node.center)
            out["radius"] = float(node.radius)
        if node.z_range is not None:
            out["z_range"] = list(node.z_range)
        return out
    if isinstance(node, OutsideRect):
        out = {"type": "OutsideRect", "uid": uid, "name": node.name}
        if geometry:
            out["rect"] = list(node.rect)
        return out
    if isinstance(node, OutsidePolytope):
        out = {"type": "OutsidePolytope", "uid": uid, "name": node.name}
        if geometry:
            out["A"] = [list(row) for row in node.A]
            out["b"] = list(node.b)
        return out
    if isinstance(node, OutsideVelocityRect):
        return {"type": "OutsideVelocityRect", "uid": uid, "name": node.name, "v_min": float(node.v_min)}
    if isinstance(node, RunwayPredicate):
        out = {
            "type": "RunwayPredicate", "uid": uid, "name": node.name,
            "z_bounds": list(node.z_bounds),
            "v_land": float(node.v_land),
            "theta_rwy": float(node.theta_rwy),
            "delta_theta": float(node.delta_theta),
        }
        if geometry:
            out["pos_rect"] = list(node.pos_rect)
        return out
    if isinstance(node, And):
        return {"type": "And", "uid": uid, "args": [ast_to_dict(a, _path=f"{_path}/{i}", geometry=geometry) for i, a in enumerate(node.args)]}
    if isinstance(node, Or):
        return {"type": "Or", "uid": uid, "args": [ast_to_dict(a, _path=f"{_path}/{i}", geometry=geometry) for i, a in enumerate(node.args)]}
    if isinstance(node, Eventually):
        return {"type": "Eventually", "uid": uid, "a": int(node.a), "b": int(node.b), "phi": ast_to_dict(node.phi, _path=f"{_path}/0", geometry=geometry)}
    if isinstance(node, Always):
        return {"type": "Always", "uid": uid, "a": int(node.a), "b": int(node.b), "phi": ast_to_dict(node.phi, _path=f"{_path}/0", geometry=geometry)}
    if isinstance(node, Until):
        return {
            "type": "Until",
            "uid": uid,
            "a": int(node.a),
            "b": int(node.b),
            "phi": ast_to_dict(node.phi, _path=f"{_path}/0", geometry=geometry),
            "psi": ast_to_dict(node.psi, _path=f"{_path}/1", geometry=geometry),
        }
    raise NotImplementedError(f"ast_to_dict: unsupported node type {type(node)}")


# ============================================================
# 6) Dataset writer (jsonl.gz)
# ============================================================

def write_jsonl_gz(path: str, records: List[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _infer_T(sample: Dict[str, Any], X: Any) -> int:
    if "T" in sample:
        return int(sample["T"])
    return int(len(X) - 1)


def generate_dataset(
    *,
    out_path: str,
    samples: List[Dict[str, Any]],
    build_ast_fn,
    build_problem_fn,
    output_flag: int = 0,
    time_limit: Optional[float] = None,
    plot_fn=None,
    compact: bool = True,
) -> None:
    """
    Dataset schema (per record):

    Compact form (compact=True, default):
      - id: int
      - T_star: int
      - sample: original sample dict (env carries obstacle/region geometry)
      - ast: ast_to_dict(ast, geometry=False)    (topology + STL semantics only)
      - micp:
          - time_sec
          - meta
          - X, U
          - strategy: canonicalize_strategy(...) ->
              { "event_time":   { op_uid: k_int },
                "event_opt":    { op_uid: opt_int },          # if present
                "outside_face": { ent_name: {"t0": int, "faces": [int]} },
                "always_or":    { op_uid:  {"t0": int, "opts":  [int]} }, # if present
              }

    Legacy form (compact=False):
      - id, sample, ast: ast_to_dict(ast)
      - micp:
          - time_sec, meta, X, U
          - strategy: Dict[str,int] keyed by BinaryDescriptor.id
          - descs:    List[dict]   (serialized BinaryDescriptor list)
    """
    records: List[Dict[str, Any]] = []
    solve_times: List[float] = []
    n_skipped = 0

    for idx, s in enumerate(samples):
        ast = build_ast_fn(s)
        prob = build_problem_fn(s)

        try:
            X, U, strategy, t_sec = solve_micp_with_strategy(
                ast, prob, output_flag=output_flag, time_limit=time_limit
            )
        except RuntimeError as e:
            print(f"[WARN] MICP failed (skipping sample): {e}")
            n_skipped += 1
            continue
        except Exception as e:
            print(f"[WARN] Unexpected error (skipping sample): {type(e).__name__}: {e}")
            n_skipped += 1
            continue

        T = _infer_T(s, X)

        # Extract descriptors again (must match the ids in strategy.binaries exactly)
        descs = STLBinaryExtractor(int(T)).extract(ast)

        strategy_binaries: Dict[str, int] = strategy.binaries

        # Optional sanity: ensure every desc id exists in strategy dict
        # (strict, because training will match ids exactly)
        for d in descs:
            if d.id not in strategy_binaries:
                raise RuntimeError(f"Dataset generation: strategy missing id={d.id} (role={d.role})")

        if compact:
            records.append(
                {
                    "id": int(idx),
                    "T_star": int(T),
                    "sample": s,
                    "ast": ast_to_dict(ast, geometry=False),
                    "micp": {
                        "time_sec": float(t_sec),
                        "meta": strategy.meta,
                        "X": X.tolist(),
                        "U": U.tolist(),
                        "strategy": canonicalize_strategy(
                            descs=descs,
                            strategy_binaries=strategy_binaries,
                            X=X,
                            ast=ast,
                        ),
                    },
                }
            )
        else:
            records.append(
                {
                    "id": int(idx),
                    "sample": s,
                    "ast": ast_to_dict(ast),
                    "micp": {
                        "time_sec": float(t_sec),
                        "X": X.tolist(),
                        "U": U.tolist(),
                        "strategy": strategy_binaries,
                        "meta": strategy.meta,
                        "descs": descs_to_dicts(descs),
                    },
                }
            )

        if plot_fn is not None:
            plot_fn(idx=idx, X=X, T=T, sample=s)

        solve_times.append(float(t_sec))
        n_done = len(solve_times)
        t_mean = float(np.mean(solve_times))
        t_std  = float(np.std(solve_times, ddof=1)) if n_done > 1 else 0.0
        print(f"[{idx+1}/{len(samples)}] solve={t_sec:.3f}s  |  "
              f"running ({n_done} solved): mean={t_mean:.1f}s  std={t_std:.1f}s")

    write_jsonl_gz(out_path, records)
    print(f"Wrote {len(records)} samples to: {out_path}")

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

        _out_base = out_path
        for _sfx in (".jsonl.gz", ".jsonl", ".gz"):
            if _out_base.endswith(_sfx):
                _out_base = _out_base[: -len(_sfx)]
                break
        _example_id = int(samples[0].get("example_id", 0)) if samples else 0
        _timing_path = _out_base + "_timing.json"
        with open(_timing_path, "w") as _f:
            json.dump({
                "solver": "micp_fixed",
                "example_id": _example_id,
                "n_solved": len(solve_times),
                "n_skipped": int(n_skipped),
                "time_mean": t_mean,
                "time_std": t_std,
                "time_min": float(np.min(solve_times)),
                "time_max": float(np.max(solve_times)),
                "time_total": float(sum(solve_times)),
            }, _f, indent=2)
        print(f"Timing stats saved to: {_timing_path}")


__all__ = [
    "PlanningProblem",
    "MICPBuilder",
    "StrategyExtractor",
    "solve_micp_with_strategy",
    "solve_with_bisection",
    "generate_dataset",
    "ast_to_dict",
    "descs_to_dicts",
    "canonicalize_strategy",
    "write_jsonl_gz",
    "runway_center_from_ast",
]