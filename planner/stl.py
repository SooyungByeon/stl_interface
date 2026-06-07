# stl.py
"""
Signal Temporal Logic (STL) utilities.

Predicates : InRect, InCircle, OutsideRect, OutsideVelocityRect, RunwayPredicate
Boolean     : And, Or
Temporal    : Eventually (F), Always (G), Until (U)

Also includes:
- Plotting helpers for rectangles / circles and trajectories.
- Strategy representation (strategic binaries only) and JSON persistence.
- Binary extractor for the supported fragment.

Node identity:
  Every AST node carries a deterministic uid = f"{node.key()}@{path}" where
  path is the structural root-to-node address ("0", "0/1", "0/1/2", …).
  BinaryDescriptor.node_tag is the uid of the enclosing temporal operator
  (or the predicate uid when no temporal operator encloses it).
  BinaryDescriptor.meta always carries op_uid, pred_uid, ent_name.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt

from .geometry import Rect


# ============================================================
# 1) STL AST nodes
# ============================================================

class STLNode:
    def children(self) -> List["STLNode"]:
        return []

    def key(self) -> str:
        raise NotImplementedError


# ============================================================
# AST unique-identity helpers
# ============================================================

def _uid_for(node: "STLNode", path: str) -> str:
    return f"{node.key()}@{path}"


def compute_uid_maps(root: "STLNode") -> Tuple[Dict[int, str], Dict[int, str]]:
    """Return (id->uid, id->path) for every node in the AST."""
    uid_map: Dict[int, str] = {}
    path_map: Dict[int, str] = {}

    def walk(n: STLNode, path: str) -> None:
        uid_map[id(n)] = _uid_for(n, path)
        path_map[id(n)] = path

        if isinstance(n, (And, Or)):
            for j, c in enumerate(n.args):
                walk(c, f"{path}/{j}")
            return
        if isinstance(n, (Always, Eventually)):
            walk(n.phi, f"{path}/0")
            return
        if isinstance(n, Until):
            walk(n.phi, f"{path}/0")
            walk(n.psi, f"{path}/1")
            return
        # leaf predicates have no children

    walk(root, "0")
    return uid_map, path_map


# ============================================================
# Predicate nodes
# ============================================================

@dataclass(frozen=True)
class InRect(STLNode):
    name: str
    rect: Rect
    z_range: Optional[Tuple[float, float]] = None  # (z_lo, z_hi) for pz; None = unconstrained

    def key(self) -> str:
        return f"InRect({self.name})"


@dataclass(frozen=True)
class OutsideRect(STLNode):
    name: str
    rect: Rect

    def key(self) -> str:
        return f"OutsideRect({self.name})"


@dataclass(frozen=True)
class OutsideVelocityRect(STLNode):
    """
    Minimum horizontal speed predicate in velocity space (vx, vy).

    Requires max(|vx|, |vy|) >= v_min, encoded as:
        (vx, vy) outside the dead-zone square [-v_min, v_min]^2.

    Encoded identically to OutsideRect but on state indices 3 (vx) and 4 (vy).
    vz (index 5) is NOT constrained — this is a horizontal speed floor only.
    """
    name: str
    v_min: float

    def key(self) -> str:
        return f"OutsideVelocityRect({self.name})"


@dataclass(frozen=True)
class OutsidePolytope(STLNode):
    """
    Convex polytope obstacle avoidance: (px, py) NOT in {x : A @ x <= b}.

    A : K rows of (a0, a1) outward normals in the planner km coordinate system.
    b : K right-hand-side values.
    At each constrained time step, at least one face h must satisfy
    A[h,0]*px + A[h,1]*py > b[h]  (aircraft outside the polytope).
    """
    name: str
    A: Tuple[Tuple[float, float], ...]   # (K, 2)
    b: Tuple[float, ...]                 # (K,)

    def key(self) -> str:
        return f"OutsidePolytope({self.name})"

    @property
    def n_faces(self) -> int:
        return len(self.b)


@dataclass(frozen=True)
class InCircle(STLNode):
    """
    Circular region predicate in the xy plane: (px-cx)^2 + (py-cy)^2 <= radius^2.

    Encoded in MICP as an inscribed regular n-gon (polygon inner approximation).
    The inradius of the n-gon is radius * cos(pi/n), making the region
    slightly smaller than the true disk (conservative).

    z_range: optional (z_lo, z_hi) altitude band on pz; None = unconstrained.
    """
    name: str
    center: Tuple[float, float]
    radius: float
    z_range: Optional[Tuple[float, float]] = None
    n_sides: int = 12

    def key(self) -> str:
        return f"InCircle({self.name})"

    def halfplanes(self) -> List[Tuple[float, float, float]]:
        """
        Return n halfplanes (a, b, c) with a*px + b*py <= c defining the
        inscribed regular n-gon.  Face i has outward normal at angle pi*(2i+1)/n.
        """
        cx, cy = self.center
        inradius = self.radius * math.cos(math.pi / self.n_sides)
        result: List[Tuple[float, float, float]] = []
        for i in range(self.n_sides):
            angle = math.pi * (2 * i + 1) / self.n_sides
            a = math.cos(angle)
            b = math.sin(angle)
            c = a * cx + b * cy + inradius
            result.append((a, b, c))
        return result


@dataclass(frozen=True)
class RunwayPredicate(STLNode):
    """
    Compound landing predicate for AircraftLTI (6-state: [px,py,pz,vx,vy,vz]).

    Encodes four simultaneous linear conditions:
      1. (px, py) in pos_rect
      2. pz in [z_lo, z_hi]
      3. ||(vx, vy, vz)||_inf <= v_land
      4. |vy*cos(theta_rwy) - vx*sin(theta_rwy)| <= sqrt(2)*v_land*sin(delta_theta)
         (linearised heading-alignment condition)

    All sub-conditions are linear in the state — no additional binary variables.
    The single event_time binary (landing timestep selector) from the enclosing
    Eventually is shared across all four sub-conditions.
    """
    name: str
    pos_rect: Rect
    z_bounds: Tuple[float, float]
    v_land: float
    theta_rwy: float
    delta_theta: float

    def key(self) -> str:
        return f"Runway({self.name})"


# ============================================================
# Boolean and temporal nodes
# ============================================================

@dataclass(frozen=True)
class And(STLNode):
    args: Tuple[STLNode, ...]

    def children(self) -> List[STLNode]:
        return list(self.args)

    def key(self) -> str:
        return "And"


@dataclass(frozen=True)
class Or(STLNode):
    args: Tuple[STLNode, ...]

    def children(self) -> List[STLNode]:
        return list(self.args)

    def key(self) -> str:
        return "Or"


@dataclass(frozen=True)
class Eventually(STLNode):
    a: int
    b: int
    phi: STLNode

    def children(self) -> List[STLNode]:
        return [self.phi]

    def key(self) -> str:
        return f"F[{self.a},{self.b}]"


@dataclass(frozen=True)
class Always(STLNode):
    a: int
    b: int
    phi: STLNode

    def children(self) -> List[STLNode]:
        return [self.phi]

    def key(self) -> str:
        return f"G[{self.a},{self.b}]"


@dataclass(frozen=True)
class Until(STLNode):
    a: int
    b: int
    phi: STLNode   # left (safety before trigger)
    psi: STLNode   # right (trigger / goal)

    def children(self) -> List[STLNode]:
        return [self.phi, self.psi]

    def key(self) -> str:
        return f"U[{self.a},{self.b}]"


# ============================================================
# 2) Plot helpers
# ============================================================

def plot_environment(ax, ast: STLNode) -> None:
    """Draw spatial predicates from the AST."""

    def walk(node: STLNode) -> None:
        if isinstance(node, RunwayPredicate):
            xmin, xmax, ymin, ymax = node.pos_rect
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                "b--", linewidth=1.5,
            )
            return
        if isinstance(node, InCircle):
            theta = [2 * math.pi * i / 64 for i in range(65)]
            cx, cy = node.center
            r = node.radius
            ax.plot(
                [cx + r * math.cos(t) for t in theta],
                [cy + r * math.sin(t) for t in theta],
                "k--", linewidth=1.5,
            )
            return
        if isinstance(node, (InRect, OutsideRect)):
            xmin, xmax, ymin, ymax = node.rect
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                "k--" if isinstance(node, InRect) else "k-",
                linewidth=1.5,
            )
        for c in node.children():
            walk(c)

    walk(ast)


def plot_trajectory(
    X,
    ast: STLNode,
    filename: str,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_environment(ax, ast)
    ax.plot(X[:, 0], X[:, 1], "-o", linewidth=2)
    ax.plot(X[0, 0], X[0, 1], "go", label="start")
    ax.plot(X[-1, 0], X[-1, 1], "ro", label="end")

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ============================================================
# 3) Binary descriptors + strategy container
# ============================================================

@dataclass
class BinaryDescriptor:
    id: str
    role: str               # "event_time", "outside_face", "always_or_choice"
    node_tag: str           # uid of enclosing temporal op (or predicate uid if none)
    time: Optional[int]
    meta: Dict[str, Any]


@dataclass
class STLStrategy:
    T: int
    binaries: Dict[str, int]
    meta: Dict[str, Any]

    def save(self, path: str) -> None:
        obj = {"T": int(self.T), "binaries": self.binaries, "meta": self.meta}
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    @staticmethod
    def load(path: str) -> "STLStrategy":
        with open(path, "r") as f:
            obj = json.load(f)
        return STLStrategy(
            T=int(obj["T"]),
            binaries={k: int(v) for k, v in obj["binaries"].items()},
            meta=obj.get("meta", {}),
        )


# ============================================================
# 4) Strategic binary extractor
# ============================================================

class STLBinaryExtractor:
    """
    Extract strategic binaries needed to replay an MICP solution as a QP.

    Supported patterns:
      Always(a, b, InRect)               -> no binaries (pure halfplane "keep")
      Always(a, b, InCircle)             -> no binaries (n-gon halfplane "keep")
      Always(a, b, OutsideRect)          -> outside_face per timestep in [0, T]
      Always(a, b, OutsideVelocityRect)  -> outside_face per timestep in [a, b]
      Always(a, b, Or(InRect, ...))      -> always_or_choice per timestep in [a, b]
      Eventually(a, b, InRect)           -> event_time per k in [a, b]
      Eventually(a, b, RunwayPredicate)  -> event_time per k in [a, b]
      Eventually(a, b, Or(InRect, ...))  -> event_time per (k, opt) in [a,b] x options
      Eventually(a, b,
        Or(Always(a2,b2,InRect|InCircle),
           Always(a2,b2,InRect|InCircle))) -> event_time per (k, opt) in [a,b] x options
                                             (loiter dwell pattern — Ex 2)
      Until(a, b, OutsideRect, InRect)   -> event_time for trigger + outside_face for left
    """

    def __init__(self, T: int):
        self.T = int(T)
        self.descs: List[BinaryDescriptor] = []
        self._uid_map: Dict[int, str] = {}
        self._path_map: Dict[int, str] = {}

    def extract(self, node: STLNode) -> List[BinaryDescriptor]:
        self.descs = []
        self._uid_map, self._path_map = compute_uid_maps(node)
        self._walk(node, context={"enclosing_op_uid": None}, path="0")
        return self.descs

    def _uid(self, n: STLNode, path: str) -> str:
        return self._uid_map.get(id(n), _uid_for(n, path))

    def _walk(self, node: STLNode, context: Dict[str, Any], path: str) -> None:

        # ------------------------------------------------------------------
        # Boolean connectives: recurse, pass context through
        # ------------------------------------------------------------------
        if isinstance(node, And):
            for j, c in enumerate(node.args):
                self._walk(c, context, f"{path}/{j}")
            return

        if isinstance(node, Or):
            for j, c in enumerate(node.args):
                self._walk(c, context, f"{path}/{j}")
            return

        # ------------------------------------------------------------------
        # Always
        # ------------------------------------------------------------------
        if isinstance(node, Always):
            op_uid = self._uid(node, path)
            phi = node.phi

            # Always(a, b, Or(InRect|InCircle, ...)) — per-timestep zone choice
            if isinstance(phi, Or) and all(isinstance(a, (InRect, InCircle)) for a in phi.args):
                or_path = f"{path}/0"
                for k in range(int(node.a), int(node.b) + 1):
                    for i, pred in enumerate(phi.args):
                        pred_uid = self._uid(pred, f"{or_path}/{i}")
                        ent_name = str(pred.name)
                        bid = f"always_or:{op_uid}:k{k}:opt{i}:{ent_name}"
                        self.descs.append(BinaryDescriptor(
                            id=bid, role="always_or_choice", node_tag=op_uid, time=k,
                            meta={"k": int(k), "opt": int(i), "pred": ent_name,
                                  "pred_uid": pred_uid, "ent_name": ent_name, "op_uid": op_uid},
                        ))
                return

            # Recurse into other Always children, passing always_a/b context
            self._walk(phi,
                       context={"enclosing_op_uid": op_uid,
                                "always_a": int(node.a),
                                "always_b": int(node.b)},
                       path=f"{path}/0")
            return

        # ------------------------------------------------------------------
        # Eventually
        # ------------------------------------------------------------------
        if isinstance(node, Eventually):
            op_uid = self._uid(node, path)
            a, b = int(node.a), int(node.b)

            # F[a,b](Or(Always(a2,b2, InRect|InCircle), ...)) — loiter dwell
            if (isinstance(node.phi, Or)
                    and all(isinstance(c, Always)
                            and isinstance(c.phi, (InRect, InCircle))
                            for c in node.phi.args)):
                a2 = int(node.phi.args[0].a)
                b2 = int(node.phi.args[0].b)
                for c in node.phi.args:
                    if int(c.a) != a2 or int(c.b) != b2:
                        raise NotImplementedError(
                            "F[a,b](Or(Always(...))) requires identical Always intervals."
                        )
                or_path = f"{path}/0"
                for k in range(a, b + 1):
                    for i, alw in enumerate(node.phi.args):
                        pred = alw.phi
                        pred_path = (f"{or_path}/{i}/0")
                        pred_uid = self._uid(pred, pred_path)
                        ent_name = str(pred.name)
                        bid = f"event:{op_uid}:k{k}:opt{i}:{ent_name}"
                        self.descs.append(BinaryDescriptor(
                            id=bid, role="event_time", node_tag=op_uid, time=k,
                            meta={"k": int(k), "opt": int(i), "pred": ent_name,
                                  "pred_uid": pred_uid, "ent_name": ent_name,
                                  "dwell_a": int(a2), "dwell_b": int(b2), "op_uid": op_uid},
                        ))
                return

            # F[a,b](Or(InRect|InCircle, ...))
            if isinstance(node.phi, Or) and all(isinstance(c, (InRect, InCircle)) for c in node.phi.args):
                or_path = f"{path}/0"
                for k in range(a, b + 1):
                    for i, pred in enumerate(node.phi.args):
                        pred_uid = self._uid(pred, f"{or_path}/{i}")
                        ent_name = str(pred.name)
                        bid = f"event:{op_uid}:k{k}:opt{i}:{ent_name}"
                        self.descs.append(BinaryDescriptor(
                            id=bid, role="event_time", node_tag=op_uid, time=k,
                            meta={"k": int(k), "opt": int(i), "pred": ent_name,
                                  "pred_uid": pred_uid, "ent_name": ent_name, "op_uid": op_uid},
                        ))
                return

            # F[a,b](InRect)
            if isinstance(node.phi, InRect):
                pred = node.phi
                pred_uid = self._uid(pred, f"{path}/0")
                ent_name = str(pred.name)
                for k in range(a, b + 1):
                    bid = f"event:{op_uid}:k{k}:{ent_name}"
                    self.descs.append(BinaryDescriptor(
                        id=bid, role="event_time", node_tag=op_uid, time=k,
                        meta={"k": int(k), "pred": ent_name, "pred_uid": pred_uid,
                              "ent_name": ent_name, "op_uid": op_uid},
                    ))
                return

            # F[a,b](InCircle)
            if isinstance(node.phi, InCircle):
                pred = node.phi
                pred_uid = self._uid(pred, f"{path}/0")
                ent_name = str(pred.name)
                for k in range(a, b + 1):
                    bid = f"event:{op_uid}:k{k}:{ent_name}"
                    self.descs.append(BinaryDescriptor(
                        id=bid, role="event_time", node_tag=op_uid, time=k,
                        meta={"k": int(k), "pred": ent_name, "pred_uid": pred_uid,
                              "ent_name": ent_name, "op_uid": op_uid},
                    ))
                return

            # F[a,b](RunwayPredicate)
            if isinstance(node.phi, RunwayPredicate):
                pred = node.phi
                pred_uid = self._uid(pred, f"{path}/0")
                ent_name = str(pred.name)
                for k in range(a, b + 1):
                    bid = f"event:{op_uid}:k{k}:{ent_name}"
                    self.descs.append(BinaryDescriptor(
                        id=bid, role="event_time", node_tag=op_uid, time=k,
                        meta={"k": int(k), "pred": ent_name, "pred_uid": pred_uid,
                              "ent_name": ent_name, "op_uid": op_uid},
                    ))
                return

            raise NotImplementedError(
                "Eventually supports: InRect, RunwayPredicate, Or(InRect,...), "
                "Or(Always(InRect|InCircle),...)."
            )

        # ------------------------------------------------------------------
        # Until — left: OutsideRect, right: InRect
        # ------------------------------------------------------------------
        if isinstance(node, Until):
            if not isinstance(node.phi, OutsideRect):
                raise NotImplementedError("Until left side must be OutsideRect.")
            if not isinstance(node.psi, InRect):
                raise NotImplementedError("Until right side must be InRect.")

            op_uid = self._uid(node, path)

            psi = node.psi
            psi_uid = self._uid(psi, f"{path}/1")
            ent_name_psi = str(psi.name)
            for k in range(int(node.a), int(node.b) + 1):
                bid = f"event:{op_uid}:k{k}:{ent_name_psi}"
                self.descs.append(BinaryDescriptor(
                    id=bid, role="event_time", node_tag=op_uid, time=k,
                    meta={"k": int(k), "pred": ent_name_psi, "pred_uid": psi_uid,
                          "ent_name": ent_name_psi, "until": True, "op_uid": op_uid},
                ))

            phi = node.phi
            phi_uid = self._uid(phi, f"{path}/0")
            ent_name_phi = str(phi.name)
            for t in range(0, self.T + 1):
                for h in range(4):
                    bid = f"outside_face:{op_uid}:t{t}:h{h}:{ent_name_phi}"
                    self.descs.append(BinaryDescriptor(
                        id=bid, role="outside_face", node_tag=op_uid, time=t,
                        meta={"t": int(t), "face": int(h), "pred": ent_name_phi,
                              "pred_uid": phi_uid, "ent_name": ent_name_phi,
                              "until_support": True, "op_uid": op_uid},
                    ))
            return

        # ------------------------------------------------------------------
        # OutsidePolytope — K-face obstacle avoidance, all timesteps
        # ------------------------------------------------------------------
        if isinstance(node, OutsidePolytope):
            pred_uid = self._uid(node, path)
            ent_name = str(node.name)
            enclosing_op_uid = context.get("enclosing_op_uid", None)
            node_tag = str(enclosing_op_uid) if enclosing_op_uid is not None else pred_uid
            always_a = int(context.get("always_a", 0))
            always_b = int(context.get("always_b", self.T))

            for k in range(0, self.T + 1):
                for h in range(node.n_faces):
                    bid = f"outside_face:{node_tag}:t{k}:h{h}:{ent_name}"
                    self.descs.append(BinaryDescriptor(
                        id=bid, role="outside_face", node_tag=node_tag, time=k,
                        meta={"t": int(k), "face": int(h), "pred": ent_name,
                              "pred_uid": pred_uid, "ent_name": ent_name,
                              "op_uid": (str(enclosing_op_uid) if enclosing_op_uid is not None else None),
                              "always_a": always_a, "always_b": always_b},
                    ))
            return

        # ------------------------------------------------------------------
        # OutsideRect — obstacle avoidance faces, all timesteps
        # ------------------------------------------------------------------
        if isinstance(node, OutsideRect):
            pred_uid = self._uid(node, path)
            ent_name = str(node.name)
            enclosing_op_uid = context.get("enclosing_op_uid", None)
            node_tag = str(enclosing_op_uid) if enclosing_op_uid is not None else pred_uid
            always_a = int(context.get("always_a", 0))
            always_b = int(context.get("always_b", self.T))

            for k in range(0, self.T + 1):
                for h in range(4):
                    bid = f"outside_face:{node_tag}:t{k}:h{h}:{ent_name}"
                    self.descs.append(BinaryDescriptor(
                        id=bid, role="outside_face", node_tag=node_tag, time=k,
                        meta={"t": int(k), "face": int(h), "pred": ent_name,
                              "pred_uid": pred_uid, "ent_name": ent_name,
                              "op_uid": (str(enclosing_op_uid) if enclosing_op_uid is not None else None),
                              "always_a": always_a, "always_b": always_b},
                    ))
            return

        # ------------------------------------------------------------------
        # OutsideVelocityRect — speed floor faces, restricted to Always window
        # ------------------------------------------------------------------
        if isinstance(node, OutsideVelocityRect):
            pred_uid = self._uid(node, path)
            ent_name = str(node.name)
            enclosing_op_uid = context.get("enclosing_op_uid", None)
            node_tag = str(enclosing_op_uid) if enclosing_op_uid is not None else pred_uid
            t_start = int(context.get("always_a", 0))
            t_end   = int(context.get("always_b", self.T))

            for k in range(t_start, t_end + 1):
                for h in range(4):
                    bid = f"outside_face:{node_tag}:t{k}:h{h}:{ent_name}"
                    self.descs.append(BinaryDescriptor(
                        id=bid, role="outside_face", node_tag=node_tag, time=k,
                        meta={"t": int(k), "face": int(h), "pred": ent_name,
                              "pred_uid": pred_uid, "ent_name": ent_name,
                              "op_uid": (str(enclosing_op_uid) if enclosing_op_uid is not None else None),
                              "always_a": t_start, "always_b": t_end},
                    ))
            return

        # ------------------------------------------------------------------
        # Leaf predicates that carry no strategic binaries themselves
        # ------------------------------------------------------------------
        if isinstance(node, (InRect, InCircle, RunwayPredicate)):
            return

        raise NotImplementedError(f"Unsupported STL node type: {type(node)}")


__all__ = [
    # AST — predicates
    "STLNode",
    "InRect",
    "InCircle",
    "OutsideRect",
    "OutsidePolytope",
    "OutsideVelocityRect",
    "RunwayPredicate",
    # AST — boolean / temporal
    "And",
    "Or",
    "Eventually",
    "Always",
    "Until",
    # uid helpers
    "compute_uid_maps",
    # plot
    "plot_environment",
    "plot_trajectory",
    # strategy
    "BinaryDescriptor",
    "STLStrategy",
    "STLBinaryExtractor",
]