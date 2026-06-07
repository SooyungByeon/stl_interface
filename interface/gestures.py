"""
Gesture model and gesture->STL translation for the authoring interface
(paper Sections 3 and 4).

A GestureProgram is a first-class, editable list of gestures with STABLE ids:
  - Region    : a named rectangle referent (drawn, or fixed runways G1/G2).
  - Requirement (reach | keep), over one region (or several => Or), with a window.

translate(T) is the compositional map tau: gestures -> STL (a homomorphism),
conjoining one subformula per requirement plus the environment conjuncts
(obstacles + min-speed floor), exactly as the paper describes.

For the IIS->gesture trace, every MICP constraint name carries the predicate /
operator name, and each predicate name belongs to exactly one gesture. iis_owners()
returns the (name-substring -> owner) signatures used to map an IIS back to the
responsible gestures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from planner.geometry import Rect
from planner.stl import (
    STLNode, And, Or, Always, Eventually, InRect, InCircle,
    OutsideRect, OutsideVelocityRect,
)


@dataclass
class Region:
    gid: str
    name: str                 # display name, unique (H, G1, G2, R1, ...)
    rect: Rect                # axis-aligned bounding box (also the rect for shape="rect")
    is_runway: bool = False    # fixed runways are landing goals (descend on reach)
    shape: str = "rect"        # "rect" | "circle"
    circle: Optional[Tuple[float, float, float]] = None   # (cx, cy, r) for shape="circle"

    def predicate(self, z_range=None) -> STLNode:
        """The STL predicate for this region (InRect or InCircle)."""
        if self.shape == "circle" and self.circle is not None:
            cx, cy, r = self.circle
            return InCircle(self.name, (cx, cy), r, z_range)
        return InRect(self.name, self.rect, z_range)


@dataclass
class Requirement:
    gid: str
    kind: str                  # "reach" | "keep"
    region_names: List[str]    # >1 => Or (reach only)
    window: Optional[Tuple[int, int]] = None   # None => full horizon [0,T]
    window_gid: Optional[str] = None           # the Window gesture that set it


@dataclass
class IISReport:
    feasible: bool
    requirements: List[Requirement] = field(default_factory=list)  # implicated
    region_names: List[str] = field(default_factory=list)          # to highlight
    environment: bool = False     # an env predicate (obstacle / speed floor) is implicated
    structural: bool = False      # dynamics / bounds conflict
    text: str = ""


class GestureProgram:
    """Editable gesture list + environment, with stable ids and translation."""

    def __init__(
        self,
        *,
        v_min: float = 0.028,
        grace_lo: int = 0,
        grace_hi: int = 3,
        landing_z: Tuple[float, float] = (0.0, 0.02),
        obstacles: Optional[List[Rect]] = None,
    ) -> None:
        self.regions: Dict[str, Region] = {}
        self.requirements: List[Requirement] = []
        self.v_min = v_min
        self.grace_lo = grace_lo
        self.grace_hi = grace_hi
        self.landing_z = landing_z
        self.obstacles: List[Rect] = list(obstacles or [])
        self._counters: Dict[str, int] = {}

    # ---- id allocation ---------------------------------------------------
    def _gid(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}{self._counters[prefix]}"

    # ---- editing ---------------------------------------------------------
    def add_region(self, rect: Rect, *, name: Optional[str] = None,
                   is_runway: bool = False, shape: str = "rect",
                   circle: Optional[Tuple[float, float, float]] = None) -> Region:
        gid = self._gid("reg")
        if name is None:
            name = f"R{self._counters['reg']}"
        if name in self.regions:
            raise ValueError(f"duplicate region name {name!r}")
        r = Region(gid=gid, name=name, rect=tuple(rect), is_runway=is_runway,
                   shape=shape, circle=circle)
        self.regions[name] = r
        return r

    def add_circle_region(self, cx: float, cy: float, r: float, *,
                          name: Optional[str] = None) -> Region:
        rect = (cx - r, cx + r, cy - r, cy + r)
        return self.add_region(rect, name=name, shape="circle", circle=(cx, cy, r))

    def add_requirement(self, kind: str, region_names: List[str],
                        window: Optional[Tuple[int, int]] = None) -> Requirement:
        if kind not in ("reach", "keep"):
            raise ValueError(kind)
        if not region_names:
            raise ValueError("a requirement needs at least one region")
        for n in region_names:
            if n not in self.regions:
                raise ValueError(f"unknown region {n!r}")
        req = Requirement(gid=self._gid("req"), kind=kind,
                          region_names=list(region_names), window=window)
        self.requirements.append(req)
        return req

    def can_merge(self, a: Requirement, b: Requirement) -> bool:
        """Two requirements may be OR-combined iff same operator and same window
        (◇(A∨B) ≡ ◇A∨◇B and □(A∨B) only when the windows coincide)."""
        return a is not b and a.kind == b.kind and a.window == b.window

    def merge_or(self, anchor: Requirement, other: Requirement) -> Requirement:
        """Fold `other` into `anchor` as a disjunction and drop `other`."""
        for n in other.region_names:
            if n not in anchor.region_names:
                anchor.region_names.append(n)
        if other in self.requirements:
            self.requirements.remove(other)
        return anchor

    def set_window(self, req: Requirement, t1: int, t2: int) -> None:
        req.window = (int(t1), int(t2))
        if req.window_gid is None:
            req.window_gid = self._gid("win")

    def max_window_end(self) -> int:
        return max([(r.window or (0, 0))[1] for r in self.requirements] + [0])

    # ---- translation: gestures -> STL -----------------------------------
    def translate(self, T: int) -> STLNode:
        authored: List[STLNode] = []
        for req in self.requirements:
            t1, t2 = req.window if req.window is not None else (0, T)
            if req.kind == "keep":
                # keep over a disjunction = □(π1∨…∨πn): stay in the union (K1).
                preds = [self.regions[n].predicate() for n in req.region_names]
                phi = preds[0] if len(preds) == 1 else Or(tuple(preds))
                authored.append(Always(t1, t2, phi))
            else:  # reach: ◇(π1∨…∨πn)  ≡  ◇π1∨…∨◇πn
                preds = []
                for n in req.region_names:
                    R = self.regions[n]
                    z = self.landing_z if R.is_runway else None
                    preds.append(R.predicate(z))
                phi = preds[0] if len(preds) == 1 else Or(tuple(preds))
                authored.append(Eventually(t1, t2, phi))

        env: List[STLNode] = []
        if self.obstacles:
            env.append(Always(0, T, And(tuple(
                OutsideRect(f"O{j}", tuple(o)) for j, o in enumerate(self.obstacles)))))
        env.append(Always(self.grace_lo, T - self.grace_hi,
                          OutsideVelocityRect("Vfloor", self.v_min)))
        return And(tuple(authored + env))

    def to_text(self, T: Optional[int] = None) -> str:
        """Human-readable STL specification (authored part + environment)."""
        def win(req):
            if req.window is None:
                return "[0,T]"
            return f"[{req.window[0]},{req.window[1]}]"
        lines: List[str] = []
        for req in self.requirements:
            body = (req.region_names[0] if len(req.region_names) == 1
                    else "(" + " ∨ ".join(req.region_names) + ")")
            op = "□" if req.kind == "keep" else "◇"
            lines.append(f"{op}_{win(req)} {body}      ({req.gid})")
        authored = "\n∧ ".join(lines) if lines else "(no requirements yet)"
        env = ["□_[0,T] ¬O_j   (obstacles)"] if self.obstacles else []
        env.append(f"□_[{self.grace_lo},T-{self.grace_hi}] (‖v‖ ≥ {self.v_min})   (speed floor)")
        return "φ =\n  " + authored + "\n∧ " + "\n∧ ".join(env)

    # ---- IIS -> gesture ownership ---------------------------------------
    def iis_owners(self) -> List[Tuple[str, Optional[Requirement], str]]:
        """List of (constraint-name substring, owning requirement | None, label).

        A constraint whose name contains the substring is attributed to that owner.
        Region names are unique, and each op-kind prefix (always_in / evt_in /
        evt_or_in) is contributed by exactly one requirement, so the match is
        one-to-one (paper P4/P5).
        """
        sigs: List[Tuple[str, Optional[Requirement], str]] = []
        for req in self.requirements:
            single = len(req.region_names) == 1
            label = f"{req.kind} " + (req.region_names[0] if single
                                      else "(" + "∨".join(req.region_names) + ")")
            for R in req.region_names:
                circ = self.regions[R].shape == "circle"
                if req.kind == "keep":
                    pre = ("always_incircle" if circ else "always_in") if single \
                          else ("sel_incircle" if circ else "sel_in")
                else:
                    pre = ("evt_incircle" if circ else "evt_in") if single \
                          else ("evt_or_incircle" if circ else "evt_or_in")
                sigs.append((f"{pre}_{R}[", req, label))
        # environment
        sigs.append(("always_vfloor_Vfloor[", None, "environment: min-speed floor"))
        sigs.append(("always_out_O", None, "environment: obstacle"))
        return sigs

    def diagnose(self, iis_names: List[str]) -> IISReport:
        sigs = self.iis_owners()
        reqs: List[Requirement] = []
        regions: List[str] = []
        env = False
        struct = False
        seen_req = set()
        for nm in iis_names:
            matched = False
            for substr, owner, _label in sigs:
                if substr in nm:
                    matched = True
                    if owner is None:
                        env = True
                    elif owner.gid not in seen_req:
                        seen_req.add(owner.gid)
                        reqs.append(owner)
                        regions.extend(owner.region_names)
            if not matched:
                # dynamics / bounds / x0 / speed-cap: structural
                if any(k in nm for k in ("dyn[", "x0[", "xmin", "xmax",
                                          "umin", "umax", "v_max_hp")):
                    struct = True

        # report text
        parts: List[str] = []
        if reqs:
            parts.append("Conflicting gestures:")
            for r in reqs:
                w = "[0,T]" if r.window is None else f"[{r.window[0]},{r.window[1]}]"
                tgt = (r.region_names[0] if len(r.region_names) == 1
                       else "(" + "∨".join(r.region_names) + ")")
                wn = f"  + window {r.window_gid}" if r.window_gid else ""
                parts.append(f"  • {r.kind} {tgt}  {w}   ({r.gid}{wn})")
        if env:
            parts.append("• environment (obstacle / speed floor) — a requirement "
                         "cannot be met in this map.")
        # The IIS always contains the dynamics constraints that connect the
        # implicated gestures; only report a structural cause when NO gesture or
        # environment predicate is at fault (a pure dynamics/bounds conflict).
        if struct and not reqs and not env:
            parts.append("• dynamics / bounds limits (no single gesture is at fault).")
        if not parts:
            parts.append("Infeasible, but the IIS did not map to a gesture.")
        else:
            parts.insert(0, "No satisfying trajectory at any horizon. Edit a "
                            "highlighted gesture (move/resize a region, or change a "
                            "window so they no longer compete).")
        return IISReport(feasible=False, requirements=reqs,
                         region_names=sorted(set(regions)), environment=env,
                         structural=struct, text="\n".join(parts))


__all__ = ["Region", "Requirement", "IISReport", "GestureProgram"]
