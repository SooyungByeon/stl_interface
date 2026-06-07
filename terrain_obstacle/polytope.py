from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial import ConvexHull

from .dem_io import DEMData
from .cluster import ClusterResult


@dataclass
class Polytope:
    A: np.ndarray        # (K, 2)  outward normals; K=4 (axis-aligned) or 8 (8-dir)
    b: np.ndarray        # (K,)    A @ x <= b  defines the obstacle polytope
    vertices: np.ndarray # (V, 2)  polygon vertices in CCW order (for plotting)


# ── Vertex computation ─────────────────────────────────────────────────────────

def _vertices_from_halfplanes(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute vertices of the polytope {x : A @ x <= b} by intersecting each
    consecutive pair of boundary hyperplanes, then keeping only the candidates
    that satisfy all constraints.  Returns a (V, 2) array sorted CCW.
    """
    n = len(b)
    candidates = []
    for k in range(n):
        k2 = (k + 1) % n
        A2 = A[[k, k2]]
        b2 = b[[k, k2]]
        try:
            v = np.linalg.solve(A2, b2)
        except np.linalg.LinAlgError:
            continue
        # Keep only if it satisfies all K constraints (with a small tolerance)
        if np.all(A @ v <= b + 1e-6):
            candidates.append(v)

    if not candidates:
        return np.zeros((0, 2))

    verts = np.array(candidates)
    # Remove near-duplicates
    unique = [verts[0]]
    for v in verts[1:]:
        if np.all(np.linalg.norm(np.array(unique) - v, axis=1) > 1e-3):
            unique.append(v)
    verts = np.array(unique)

    # Sort CCW by angle around centroid
    c = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    return verts[np.argsort(angles)]


# ── Fitting functions ──────────────────────────────────────────────────────────

def fit_axis_aligned(points: np.ndarray) -> Polytope:
    """
    Fit a 4-constraint axis-aligned bounding rectangle around *points* (N, 2).

    Half-planes:  -x <= -xmin,  x <= xmax,  -y <= -ymin,  y <= ymax
    Convention matches planner/geometry.py Rect = (xmin, xmax, ymin, ymax).
    """
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)

    A = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    b = np.array([-xmin, xmax, -ymin, ymax])
    vertices = np.array([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax],
    ])
    return Polytope(A=A, b=b, vertices=vertices)


def fit_8direction(points: np.ndarray) -> Polytope:
    """
    Fit an 8-direction bounding polytope around *points* (N, 2).

    Outward normals at 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°.
    For each normal n_k:  b_k = max(n_k @ points.T).
    All 8 constraints are kept even when some are redundant for elongated clusters.
    """
    angles_deg = np.arange(8) * 45.0
    normals = np.column_stack([
        np.cos(np.radians(angles_deg)),
        np.sin(np.radians(angles_deg)),
    ])  # (8, 2)

    b = (normals @ points.T).max(axis=1)  # (8,)
    vertices = _vertices_from_halfplanes(normals, b)
    return Polytope(A=normals, b=b, vertices=vertices)


# ── Convex decomposition helpers ───────────────────────────────────────────────

def _convexity_ratio(points: np.ndarray, cell_area: float) -> float:
    """Fraction of the convex hull area actually filled by cluster points."""
    if len(points) < 4:
        return 1.0
    try:
        hull = ConvexHull(points)
        filled = len(points) * cell_area
        return min(1.0, filled / hull.volume)  # .volume == area in 2-D
    except Exception:
        return 1.0


def _kmeans_split(points: np.ndarray, k: int) -> List[np.ndarray]:
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(points)
    return [points[labels == i] for i in range(k) if (labels == i).sum() >= 2]


def _decompose_convex(
    points: np.ndarray,
    cell_area: float,
    threshold: float,
    max_k: int,
) -> List[np.ndarray]:
    """
    Return a list of sub-point-sets that are each convex enough (ratio >= threshold).
    Increases k-means k until satisfied or max_k is reached.
    """
    if len(points) < 20 or _convexity_ratio(points, cell_area) >= threshold:
        return [points]

    for k in range(2, max_k + 1):
        pieces = _kmeans_split(points, k)
        if all(
            _convexity_ratio(p, cell_area) >= threshold
            for p in pieces if len(p) >= 4
        ):
            return pieces

    return _kmeans_split(points, max_k)


# ── Pipeline entry point ───────────────────────────────────────────────────────

def fit_polytopes(
    dem: DEMData,
    cluster_result: ClusterResult,
    mode: str = "8direction",
    convex_decomp: bool = False,
    convexity_threshold: float = 0.65,
    max_k: int = 6,
) -> List[Polytope]:
    """
    For each retained cluster (labels 1..n_clusters), extract its UTM (x, y)
    point set and fit a polytope in the chosen mode.

    Parameters
    ----------
    mode : "8direction" (default) or "axis_aligned".
    convex_decomp : If True, split non-convex clusters into convex sub-pieces
                    before fitting, producing more polytopes but tighter bounds.
    convexity_threshold : Minimum filled-area / convex-hull-area ratio to accept
                          a cluster as convex (default 0.65).
    max_k : Maximum number of k-means sub-clusters per original cluster (default 6).
    """
    if mode not in ("axis_aligned", "8direction"):
        raise ValueError(f"Unknown mode {mode!r}. Use 'axis_aligned' or '8direction'.")

    X, Y = dem.xy_grids()
    cell_area = dem.resolution_m ** 2
    fit_fn = fit_axis_aligned if mode == "axis_aligned" else fit_8direction

    polytopes: List[Polytope] = []
    for lab in range(1, cluster_result.n_clusters + 1):
        cell_mask = cluster_result.labels == lab
        pts = np.column_stack([X[cell_mask], Y[cell_mask]])

        if convex_decomp:
            pieces = _decompose_convex(pts, cell_area, convexity_threshold, max_k)
        else:
            pieces = [pts]

        for sub_pts in pieces:
            polytopes.append(fit_fn(sub_pts))

    return polytopes
