from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage


@dataclass
class ClusterResult:
    labels: np.ndarray  # (H, W) int: 0=background, 1..K=cluster id, -1=too small
    n_clusters: int     # number of retained clusters (labelled 1..n_clusters)


def cluster_obstacles(
    mask: np.ndarray,
    method: str = "connected",
    min_size: int = 10,
    dbscan_eps_m: float = 50.0,
    dbscan_min_samples: int = 5,
    resolution_m: float = 30.0,
) -> ClusterResult:
    """
    Cluster the True cells in *mask* into spatially connected groups.

    Parameters
    ----------
    mask : (H, W) bool array of unsafe DEM cells.
    method : "connected" — 8-connectivity connected-components (default).
             "dbscan"    — DBSCAN on pixel-space coordinates.
    min_size : Clusters with fewer cells than this are labelled -1 (discarded).
    dbscan_eps_m : DBSCAN neighbourhood radius in metres.
    dbscan_min_samples : DBSCAN core-point threshold.
    resolution_m : DEM cell size in metres (used to convert dbscan_eps_m to pixels).
    """
    if method == "connected":
        return _connected(mask, min_size)
    elif method == "dbscan":
        return _dbscan(mask, min_size, dbscan_eps_m, dbscan_min_samples, resolution_m)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'connected' or 'dbscan'.")


# ── Implementations ────────────────────────────────────────────────────────────

def _connected(mask: np.ndarray, min_size: int) -> ClusterResult:
    struct = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
    labeled, n_raw = ndimage.label(mask, structure=struct)

    out = np.zeros_like(labeled)
    kept = 0
    for lab in range(1, n_raw + 1):
        cell_mask = labeled == lab
        size = int(cell_mask.sum())
        if size >= min_size:
            kept += 1
            out[cell_mask] = kept
        else:
            out[cell_mask] = -1

    return ClusterResult(labels=out, n_clusters=kept)


def _dbscan(
    mask: np.ndarray,
    min_size: int,
    eps_m: float,
    min_samples: int,
    resolution_m: float,
) -> ClusterResult:
    from sklearn.cluster import DBSCAN

    rows, cols = np.where(mask)
    out = np.zeros(mask.shape, dtype=int)

    if len(rows) == 0:
        return ClusterResult(labels=out, n_clusters=0)

    eps_pixels = eps_m / resolution_m
    points = np.column_stack([rows, cols]).astype(np.float64)
    raw = DBSCAN(eps=eps_pixels, min_samples=min_samples).fit(points).labels_

    kept = 0
    for lab in sorted(set(raw)):
        if lab < 0:
            continue
        idx = np.where(raw == lab)[0]
        if len(idx) >= min_size:
            kept += 1
            out[rows[idx], cols[idx]] = kept
        else:
            out[rows[idx], cols[idx]] = -1

    # DBSCAN noise points → -1
    noise_idx = np.where(raw == -1)[0]
    out[rows[noise_idx], cols[noise_idx]] = -1

    return ClusterResult(labels=out, n_clusters=kept)
