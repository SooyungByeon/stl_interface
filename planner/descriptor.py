from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geometry import Rect
from .stl import RunwayPredicate
from .examples import (
    GeometryConstraints,
    validate_geometry,
    make_example_1,
)


# ==============================================================================
# Example 2 (random-POI variant) — POI table
# ==============================================================================
# 15 real-world delivery candidate POIs around the Farafangana base — the 4
# Ex1 deliveries PLUS 11 additional sites from the POI list (Ikongo excluded
# because it lands outside the 200×200 km workspace). Coords in workspace km
# via the same lat/lon → km projection used for Ex1's D1..D4 (verified to
# reproduce Ex1's hard-coded D1..D4 within ±0.2 km).
#
# Each sample picks N_DELIVERIES distinct entries via the env-sampling rng;
# the chosen rects are labelled D1..D4 in the pick order. With 4 southern
# entries (Ambongo/Vatanato/Befotaka/Midongy) the pool covers both the
# southern terrain corridor (Ex1's natural problem region) and the new
# northern sites for trajectory-shape diversity.
EX2_POI_TABLE: List[Tuple[float, float, str]] = [
    # ── Ex1's original D1..D4 (south of base; 4 entries) ──
    ( 93.3,  76.9, "Ambongo"),
    ( 86.5,  54.9, "Vatanato"),
    ( 64.3,  36.9, "Befotaka"),
    ( 67.8,  62.7, "Midongy_du_Sud"),
    # ── New additions from POI list (11 entries) ──
    ( 96.7, 106.2, "Karimbary"),
    ( 94.5,  93.2, "Bevata"),
    (110.6, 144.6, "Ianakasy_Mahatsinjo"),
    (109.6, 124.7, "Andremareo"),
    ( 97.6, 174.3, "Ambohimana"),
    (122.9, 185.8, "Mahasoa_I"),
    (108.5, 165.2, "Vohibe2"),
    (106.8, 127.5, "Antemanara"),
    (118.1, 195.0, "Manato"),
    (100.6, 182.2, "Ambaramafaitra"),
    (124.1, 166.9, "Bemandresy"),
]
EX2_POI_HALF_SIDE: float = 0.8  # 1.6×1.6 km delivery box (matches Ex1)
EX2_N_DELIVERIES: int = 4


# ==============================================================================
# Private geometry helpers
# ==============================================================================

def _rect_inflate(r: Rect, eps: float) -> Rect:
    x1, x2, y1, y2 = r
    return (x1 - eps, x2 + eps, y1 - eps, y2 + eps)


def _rects_overlap(a: Rect, b: Rect) -> bool:
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    return (min(ax2, bx2) > max(ax1, bx1)) and (min(ay2, by2) > max(ay1, by1))


def _point_in_rect(x: float, y: float, r: Rect) -> bool:
    x1, x2, y1, y2 = r
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def _sample_rect_uniform(
    *,
    rng: np.random.Generator,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    w: float,
    h: float,
) -> Rect:
    x_min, x_max = float(xlim[0]), float(xlim[1])
    y_min, y_max = float(ylim[0]), float(ylim[1])
    if (x_max - x_min) <= w or (y_max - y_min) <= h:
        raise ValueError("Workspace too small for requested rectangle size.")
    x1 = float(rng.uniform(x_min, x_max - w))
    y1 = float(rng.uniform(y_min, y_max - h))
    return (x1, x1 + w, y1, y1 + h)


def _place_random_obstacles(
    *,
    rng: np.random.Generator,
    x0: np.ndarray,
    regions: Mapping[str, Rect],
    n_obstacles: int,
    obstacle_w_min: float,
    obstacle_h_min: float,
    obstacle_w_max: float,
    obstacle_h_max: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    rules: GeometryConstraints,
    max_tries: int,
    sample_xlim: Optional[Tuple[float, float]] = None,
    sample_ylim: Optional[Tuple[float, float]] = None,
) -> List[Rect]:
    """
    Place n_obstacles randomly, each with independently sampled size in
    [w_min, w_max] x [h_min, h_max].  sample_xlim/ylim restrict where
    candidates are drawn from; validate_geometry always uses full xlim/ylim.
    """
    _sxlim = sample_xlim if sample_xlim is not None else xlim
    _sylim = sample_ylim if sample_ylim is not None else ylim
    obstacles: List[Rect] = []
    for i in range(int(n_obstacles)):
        placed = False
        for _ in range(int(max_tries)):
            w = float(rng.uniform(obstacle_w_min, obstacle_w_max))
            h = float(rng.uniform(obstacle_h_min, obstacle_h_max))
            cand = _sample_rect_uniform(rng=rng, xlim=_sxlim, ylim=_sylim, w=w, h=h)
            ok, _ = validate_geometry(
                x0=x0, obstacles=obstacles + [cand], regions=regions,
                xlim=xlim, ylim=ylim, rules=rules,
            )
            if ok:
                obstacles.append(cand)
                placed = True
                break
        if not placed:
            raise RuntimeError(f"Failed to place obstacle {i} within max_tries={max_tries}.")
    return obstacles


# ==============================================================================
# Bounds and descriptor dataclasses
# ==============================================================================

@dataclass(frozen=True)
class BoundsSpec:
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray


def _bounds_aircraft() -> BoundsSpec:
    """Base bounds for the high-altitude aircraft schema (300x300 km workspace)."""
    return BoundsSpec(
        x_min=np.array([0.0,   0.0,   5.0,  -0.20, -0.20, -0.050], dtype=float),
        x_max=np.array([300.0, 300.0, 12.0,  0.20,  0.20,  0.050], dtype=float),
        u_min=np.array([-0.001, -0.001, -0.003], dtype=float),
        u_max=np.array([ 0.001,  0.001,  0.003], dtype=float),
    )


def _bounds_aircraft_lowalt() -> BoundsSpec:
    """
    Bounds for the low-altitude aircraft schema (60x60 km workspace).

    Horizontal workspace : 60x60 km
    Max horizontal speed : 40 m/s  (0.040 km/s)
    Min horizontal speed : 28 m/s  (0.028 km/s) — enforced via STL OutsideVelocityRect
    Altitude             : 0.0–1.5 km  (runway ~0 km, cruise ~1.2 km)
    Max vertical speed   : 6 m/s   (0.006 km/s,  ~8.5 deg flight-path angle at 40 m/s)
    Horiz accel          : 0.20 mm/s^2  (0.00020 km/s^2, ~25 deg heading change/step at v_min)
    Vert  accel          : 0.10 mm/s^2  (0.0001  km/s^2  = vz_max/dt)
    """
    b = _bounds_aircraft()
    x_min = b.x_min.copy()
    x_max = b.x_max.copy()
    u_min = b.u_min.copy()
    u_max = b.u_max.copy()
    x_max[0] = 60.0
    x_max[1] = 60.0
    x_min[3] = -0.040;  x_max[3] = 0.040
    x_min[4] = -0.040;  x_max[4] = 0.040
    x_min[2] = 0.0;     x_max[2] = 1.5
    x_min[5] = -0.006;  x_max[5] = 0.006
    u_min[0] = -0.00020; u_max[0] = 0.00020
    u_min[1] = -0.00020; u_max[1] = 0.00020
    u_min[2] = -0.0001;  u_max[2] = 0.0001
    return BoundsSpec(x_min=x_min, x_max=x_max, u_min=u_min, u_max=u_max)


def _bounds_aircraft_farafangana() -> BoundsSpec:
    """
    Bounds for the Farafangana 4-drop mission (200×200 km workspace).
    Velocity/altitude limits are identical to _bounds_aircraft_lowalt.
    """
    b = _bounds_aircraft_lowalt()
    x_min = b.x_min.copy()
    x_max = b.x_max.copy()
    x_max[0] = 200.0
    x_max[1] = 200.0
    return BoundsSpec(x_min=x_min, x_max=x_max, u_min=b.u_min.copy(), u_max=b.u_max.copy())


@dataclass(frozen=True)
class ExampleDescriptor:
    example_id: int
    T: int
    x0_base: np.ndarray
    bounds: BoundsSpec
    obstacles_base: List[Rect]
    regions_base: Dict[str, Rect]
    meta_base: Dict[str, Any]
    obstacle_nodes_base: Optional[List] = None  # List[OutsideRect] when terrain obstacles are loaded

    # ------------------------------------------------------------------
    # sample_env
    # ------------------------------------------------------------------

    def sample_env(
        self,
        *,
        rng: np.random.Generator,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        dx0_range: Tuple[float, float],
        dy0_range: Tuple[float, float],
        dO_range: Tuple[float, float, float, float],
        dR_range: Tuple[float, float, float, float],
        rules: GeometryConstraints,
        max_tries: int,
        fixed_obstacle_mask: Optional[Sequence[bool]] = None,
    ) -> Dict[str, Any]:

        ex = int(self.example_id)
        meta = self.meta_base or {}

        # ------------------------------------------------------------------
        # Example 1 — Farafangana 4-Drop Delivery with Return
        # All geometry (delivery zones, runway, obstacle) is fixed.
        # ------------------------------------------------------------------
        if ex == 1:
            x0_now = self.x0_base.copy()
            regions_now: Dict[str, Rect] = {
                k: tuple(v) for k, v in self.regions_base.items()  # type: ignore[misc]
            }
            placement_xlim = (float(meta["obstacle_xlim_lo"]), float(meta["obstacle_xlim_hi"]))
            placement_ylim = (float(meta["obstacle_ylim_lo"]), float(meta["obstacle_ylim_hi"]))
            obstacles_now = _place_random_obstacles(
                rng=rng,
                x0=x0_now,
                regions=regions_now,
                n_obstacles=len(self.obstacles_base),
                obstacle_w_min=float(meta["obstacle_w_min"]),
                obstacle_h_min=float(meta["obstacle_h_min"]),
                obstacle_w_max=float(meta["obstacle_w_max"]),
                obstacle_h_max=float(meta["obstacle_h_max"]),
                xlim=xlim,
                ylim=ylim,
                rules=rules,
                max_tries=max_tries,
                sample_xlim=placement_xlim,
                sample_ylim=placement_ylim,
            )
            env: Dict[str, Any] = {
                "x0": x0_now.tolist(),
                "obstacles": [list(r) for r in obstacles_now],
                "regions": {k: list(v) for k, v in regions_now.items()},
                "meta": dict(meta),
                "deltas": {
                    "dx0": 0.0, "dy0": 0.0,
                    "dOx": [], "dOy": [],
                    "fixed": [False] * len(obstacles_now),
                    "dRx": {}, "dRy": {},
                },
                "tries": 1,
            }
            if self.obstacle_nodes_base:
                env["obstacles"] += [list(n.rect) for n in self.obstacle_nodes_base]
                env["deltas"]["fixed"] += [True] * len(self.obstacle_nodes_base)
            return env

        # ------------------------------------------------------------------
        # Example 2 — Farafangana 4-Drop Random POI (variant of Ex1).
        # Same dynamics / bounds / STL spec as Ex1, but D1..D4 are 4 distinct
        # POIs sampled from EX2_POI_TABLE via the env-sampling rng (so the
        # seed reproduces the choice). Obstacles random in same corridor.
        # ------------------------------------------------------------------
        # Example 3 shares Example 2's environment exactly (random POIs +
        # random obstacles + terrain); only the MICP objective differs.
        if ex in (2, 3):
            x0_now = self.x0_base.copy()
            runway_rect: Rect = tuple(self.regions_base["runway"])  # type: ignore[assignment]

            # Pick EX2_N_DELIVERIES distinct POIs using the shared rng so the
            # selection is tied to the dataset seed.
            poi_idx = rng.choice(
                len(EX2_POI_TABLE),
                size=int(EX2_N_DELIVERIES),
                replace=False,
            )
            chosen = [EX2_POI_TABLE[int(i)] for i in poi_idx]
            h = float(EX2_POI_HALF_SIDE)
            delivery_rects: Dict[str, Rect] = {}
            for k, (cx, cy, _name) in enumerate(chosen, start=1):
                delivery_rects[f"D{k}"] = (cx - h, cx + h, cy - h, cy + h)

            regions_now: Dict[str, Rect] = dict(delivery_rects)
            regions_now["runway"] = runway_rect

            placement_xlim = (float(meta["obstacle_xlim_lo"]), float(meta["obstacle_xlim_hi"]))
            placement_ylim = (float(meta["obstacle_ylim_lo"]), float(meta["obstacle_ylim_hi"]))
            obstacles_now = _place_random_obstacles(
                rng=rng,
                x0=x0_now,
                regions=regions_now,
                n_obstacles=len(self.obstacles_base),
                obstacle_w_min=float(meta["obstacle_w_min"]),
                obstacle_h_min=float(meta["obstacle_h_min"]),
                obstacle_w_max=float(meta["obstacle_w_max"]),
                obstacle_h_max=float(meta["obstacle_h_max"]),
                xlim=xlim,
                ylim=ylim,
                rules=rules,
                max_tries=max_tries,
                sample_xlim=placement_xlim,
                sample_ylim=placement_ylim,
            )

            env: Dict[str, Any] = {
                "x0": x0_now.tolist(),
                "obstacles": [list(r) for r in obstacles_now],
                "regions": {k: list(v) for k, v in regions_now.items()},
                "meta": dict(meta),
                "deltas": {
                    "dx0": 0.0, "dy0": 0.0,
                    "dOx": [], "dOy": [],
                    "fixed": [False] * len(obstacles_now),
                    "dRx": {}, "dRy": {},
                    # Diagnostic: which physical POIs got picked, in D1..D4 order.
                    "chosen_pois": [str(name) for (_, _, name) in chosen],
                },
                "tries": 1,
            }
            if self.obstacle_nodes_base:
                env["obstacles"] += [list(n.rect) for n in self.obstacle_nodes_base]
                env["deltas"]["fixed"] += [True] * len(self.obstacle_nodes_base)
            return env

        raise ValueError(f"Unsupported example_id={ex}")

    # ------------------------------------------------------------------
    # build_ast_from_sample
    # ------------------------------------------------------------------

    def build_ast_from_sample(self, sample: Dict[str, Any]):
        T   = int(sample["T"])
        env = sample["env"]
        obstacles = [tuple(r) for r in env["obstacles"]]
        regions   = {k: tuple(v) for k, v in env["regions"].items()}
        meta      = env.get("meta", {})
        ex        = int(sample.get("example_id", self.example_id))

        v_min     = float(meta.get("v_min",          0.028))
        grace_lo  = int(meta.get("v_min_grace_lo",   3))
        grace_hi  = int(meta.get("v_min_grace_hi",   3))

        runway = RunwayPredicate(
            name="runway",
            pos_rect=tuple(regions["runway"]),  # type: ignore[arg-type]
            z_bounds=(
                float(meta.get("runway_z_lo", 0.0)),
                float(meta.get("runway_z_hi", 0.02)),
            ),
            v_land=float(meta.get("v_land", 0.020)),
            theta_rwy=float(meta.get("theta_rwy", 0.0)),
            delta_theta=float(meta.get("delta_theta", 0.1745)),  # pi/18
        )

        # ------------------------------------------------------------------
        # Example 1
        # ------------------------------------------------------------------
        if ex == 1:
            T_land_lo = max(int(T) - 20, 0)
            delivery_z_range = None
            if "delivery_z_lo" in meta and "delivery_z_hi" in meta:
                delivery_z_range = (float(meta["delivery_z_lo"]), float(meta["delivery_z_hi"]))

            return make_example_1(
                T, T_land_lo,
                obstacles,
                regions["D1"], regions["D2"], regions["D3"], regions["D4"],
                runway,
                delivery_z_range=delivery_z_range,
                v_min=v_min, grace_lo=grace_lo, grace_hi=grace_hi,
            )

        # ------------------------------------------------------------------
        # Examples 2 & 3 — Farafangana 4-Drop Random POI (variant of Ex1).
        # Identical STL spec to Ex1; only the four D-rects are picked per
        # sample. Example 3 differs from Example 2 ONLY in the MICP objective
        # (a soft runway-tracking term), which lives in build_problem_fn /
        # MICPBuilder, not in the STL AST.
        # ------------------------------------------------------------------
        if ex in (2, 3):
            T_land_lo = max(int(T) - 20, 0)
            delivery_z_range = None
            if "delivery_z_lo" in meta and "delivery_z_hi" in meta:
                delivery_z_range = (float(meta["delivery_z_lo"]), float(meta["delivery_z_hi"]))

            return make_example_1(
                T, T_land_lo,
                obstacles,
                regions["D1"], regions["D2"], regions["D3"], regions["D4"],
                runway,
                delivery_z_range=delivery_z_range,
                v_min=v_min, grace_lo=grace_lo, grace_hi=grace_hi,
            )

        raise ValueError(f"Unsupported example_id: {ex}")


# ==============================================================================
# get_descriptor
# ==============================================================================

def get_descriptor(
    example_id: int,
    *,
    n_obstacles: int = 3,
) -> ExampleDescriptor:
    ex = int(example_id)
    n_obstacles = int(n_obstacles)
    if n_obstacles < 1:
        raise ValueError(f"n_obstacles must be >= 1, got {n_obstacles}")

    # ------------------------------------------------------------------
    # Example 1 — Farafangana 4-Drop Delivery with Return
    # ------------------------------------------------------------------
    if ex == 1:
        T = 140

        # Start just east of runway, heading east
        v0 = 0.040
        x0 = np.array([152.0, 150.0, 0.01, v0, 0.0, 0.0], dtype=float)

        # Obstacle positions are randomised in sample_env; this list sets the count.
        obstacles: List[Rect] = [(105.0, 115.0, 85.0, 105.0)] * n_obstacles

        # Delivery zones: 1.6×1.6 km boxes centred at real-world coordinates
        # Runway:  22°48'18.7"S 47°49'13.9"E  → (150, 150) km
        # D1 Ambongo:        23°27'47.3"S 47°16'03.5"E  → (93.3,  76.9) km
        # D2 Vatanato:       23°39'36.8"S 47°12'04.5"E  → (86.5,  54.9) km
        # D3 Befotaka:       23°49'21.87"S 46°59'04.26"E → (64.3,  36.9) km
        # D4 Midongy du Sud: 23°35'24.74"S 47°01'05.09"E → (67.8,  62.7) km
        D1: Rect = (92.5, 94.1,  76.1,  77.7)
        D2: Rect = (85.7, 87.3,  54.1,  55.7)
        D3: Rect = (63.5, 65.1,  36.1,  37.7)
        D4: Rect = (67.0, 68.6,  61.9,  63.5)

        # Runway: 2×0.4 km, E-W axis (theta_rwy=0), centred at (150, 150)
        runway_pos: Rect = (149.0, 151.0, 149.8, 150.2)

        regions: Dict[str, Rect] = {
            "D1": D1,
            "D2": D2,
            "D3": D3,
            "D4": D4,
            "runway": runway_pos,
        }

        # Landing allowed only in last 20 steps to prevent early return
        T_land_lo = T - 20

        meta: Dict[str, Any] = {
            "dt": 60.0,
            "T_min": 140,
            "T_max": 240,
            # RunwayPredicate
            "runway_z_lo":    0.0,
            "runway_z_hi":    0.02,
            "cruise_z_ref":   1.2,
            "v_land":         0.020,
            "theta_rwy":      0.0,
            "delta_theta":    float(np.pi / 18),
            # Delivery altitude band
            "delivery_z_lo":  0.0,
            "delivery_z_hi":  0.05,
            # Speed floor
            "v_max":          0.040,
            "v_min":          0.028,
            "v_min_grace_lo": 3,
            "v_min_grace_hi": 3,
            # Landing window lower bound
            "T_land_lo":      int(T_land_lo),
            # Random obstacle placement corridor (between delivery zones and runway)
            "obstacle_w_min": 8.0,  "obstacle_w_max": 12.0,
            "obstacle_h_min": 10.0, "obstacle_h_max": 20.0,
            "obstacle_xlim_lo": 70.0, "obstacle_xlim_hi": 135.0,
            "obstacle_ylim_lo": 40.0, "obstacle_ylim_hi": 120.0,
        }
        b = _bounds_aircraft_farafangana()
        from terrain_obstacle.planner_interface import get_terrain_obstacles_ex1
        obstacle_nodes = get_terrain_obstacles_ex1()
        return ExampleDescriptor(ex, T, x0, b, list(obstacles), dict(regions), dict(meta),
                                 obstacle_nodes_base=list(obstacle_nodes))

    # ------------------------------------------------------------------
    # Example 2 — Farafangana 4-Drop Random POI (Ex1 variant)
    # Same dynamics, bounds, runway, terrain obstacles, and STL spec as Ex1.
    # The only difference: D1..D4 rectangles are sampled per sample from
    # EX2_POI_TABLE (11 candidates) via the env-sampling rng. Wider T window
    # because random POI mixes can be much easier or harder than Ex1's
    # canonical 4-delivery loop.
    # ------------------------------------------------------------------
    if ex == 2:
        T = 160  # default horizon (wider than Ex1's 140 to fit hard POI mixes)

        # Start just east of runway, heading east — same as Ex1.
        v0 = 0.040
        x0 = np.array([152.0, 150.0, 0.01, v0, 0.0, 0.0], dtype=float)

        # Obstacle count placeholder; positions overwritten in sample_env.
        obstacles: List[Rect] = [(105.0, 115.0, 85.0, 105.0)] * n_obstacles

        # Runway: same as Ex1.
        runway_pos: Rect = (149.0, 151.0, 149.8, 150.2)

        # regions_base carries only the fixed runway. D1..D4 are added per
        # sample in sample_env from the POI table.
        regions: Dict[str, Rect] = {"runway": runway_pos}

        T_land_lo = T - 20

        meta: Dict[str, Any] = {
            "dt": 60.0,
            # Bisect window — wider than Ex1's because POI difficulty varies
            # widely; some 4-POI mixes are very short, others very long.
            "T_min": 100,
            "T_max": 160,
            # RunwayPredicate
            "runway_z_lo":    0.0,
            "runway_z_hi":    0.02,
            "cruise_z_ref":   1.2,
            "v_land":         0.020,
            "theta_rwy":      0.0,
            "delta_theta":    float(np.pi / 18),
            # Delivery altitude band
            "delivery_z_lo":  0.0,
            "delivery_z_hi":  0.05,
            # Speed floor
            "v_max":          0.040,
            "v_min":          0.028,
            "v_min_grace_lo": 3,
            "v_min_grace_hi": 3,
            # Landing window lower bound (recomputed per-sample in build_ast)
            "T_land_lo":      int(T_land_lo),
            # Random obstacle placement corridor — widened from Ex1 to cover
            # the POI spread: EX2_POI_TABLE x ∈ [94, 125], y ∈ [93, 195].
            # Corridor wraps the base→POI paths from the south + west + north.
            "obstacle_w_min": 8.0,  "obstacle_w_max": 12.0,
            "obstacle_h_min": 10.0, "obstacle_h_max": 20.0,
            "obstacle_xlim_lo": 80.0, "obstacle_xlim_hi": 145.0,
            "obstacle_ylim_lo": 80.0, "obstacle_ylim_hi": 200.0,
        }
        b = _bounds_aircraft_farafangana()
        # Ex2 terrain: own cache at obstacles_ex2.json, cone-clearance
        # conditioned on Runway + all 15 EX2 POI candidates (Ex1's D1..D4
        # + 11 new POIs).
        from terrain_obstacle.planner_interface import get_terrain_obstacles_ex2
        obstacle_nodes = get_terrain_obstacles_ex2()
        return ExampleDescriptor(ex, T, x0, b, list(obstacles), dict(regions), dict(meta),
                                 obstacle_nodes_base=list(obstacle_nodes))

    # ------------------------------------------------------------------
    # Example 3 — Farafangana 4-Drop Random POI with runway-tracking cost.
    # Environment is IDENTICAL to Example 2 (random POIs, 3 random obstacles,
    # 5 terrain obstacles, same bounds / dynamics / STL spec). The ONLY
    # difference is the MICP objective: a soft 2D (xy) quadratic term pulling
    # the trajectory toward the runway center (meta["Q_runway"]), so the
    # aircraft parks at the runway once the mission is complete.
    # ------------------------------------------------------------------
    if ex == 3:
        T = 160  # same default horizon as Ex2

        # Start just east of runway, heading east — same as Ex1/Ex2.
        v0 = 0.040
        x0 = np.array([152.0, 150.0, 0.01, v0, 0.0, 0.0], dtype=float)

        # Obstacle count placeholder; positions overwritten in sample_env.
        obstacles: List[Rect] = [(105.0, 115.0, 85.0, 105.0)] * n_obstacles

        # Runway: same as Ex1/Ex2.
        runway_pos: Rect = (149.0, 151.0, 149.8, 150.2)

        # regions_base carries only the fixed runway. D1..D4 are added per
        # sample in sample_env from the POI table (shared ex in (2,3) branch).
        regions: Dict[str, Rect] = {"runway": runway_pos}

        T_land_lo = T - 20

        meta: Dict[str, Any] = {
            "dt": 60.0,
            "T_min": 100,
            "T_max": 160,
            # RunwayPredicate
            "runway_z_lo":    0.0,
            "runway_z_hi":    0.02,
            "cruise_z_ref":   1.2,
            "v_land":         0.020,
            "theta_rwy":      0.0,
            "delta_theta":    float(np.pi / 18),
            # Delivery altitude band
            "delivery_z_lo":  0.0,
            "delivery_z_hi":  0.05,
            # Speed floor
            "v_max":          0.040,
            "v_min":          0.028,
            "v_min_grace_lo": 3,
            "v_min_grace_hi": 3,
            # Landing window lower bound (recomputed per-sample in build_ast)
            "T_land_lo":      int(T_land_lo),
            # Random obstacle placement corridor — same as Ex2.
            "obstacle_w_min": 8.0,  "obstacle_w_max": 12.0,
            "obstacle_h_min": 10.0, "obstacle_h_max": 20.0,
            "obstacle_xlim_lo": 80.0, "obstacle_xlim_hi": 145.0,
            "obstacle_ylim_lo": 80.0, "obstacle_ylim_hi": 200.0,
            # NEW for Ex3: soft (xy) runway-tracking weight in the MICP objective.
            # Read by build_problem_fn -> PlanningProblem.Q_runway. The v_min
            # speed floor keeps the aircraft moving mid-mission, so this term
            # mainly keeps it parked at the runway in the post-landing tail.
            "Q_runway": 1e-3,
        }
        b = _bounds_aircraft_farafangana()
        # Reuse Ex2 terrain obstacles (same workspace / POI conditioning).
        from terrain_obstacle.planner_interface import get_terrain_obstacles_ex2
        obstacle_nodes = get_terrain_obstacles_ex2()
        return ExampleDescriptor(ex, T, x0, b, list(obstacles), dict(regions), dict(meta),
                                 obstacle_nodes_base=list(obstacle_nodes))

    raise ValueError(f"Unsupported example_id={ex}")