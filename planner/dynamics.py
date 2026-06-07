"""
Dynamics models used by the planning and learning pipeline.

Aircraft LTI (discrete-time):
    state x = [px, py, pz, vx, vy, vz]   (6-state double integrator)
    control u = [ax, ay, az]

    Heading is NOT a separate state; derived as arctan2(vy, vx).
    Units: km for position, km/s for velocity, km/s^2 for acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class AircraftLTI:
    """
    Discrete-time LTI model for a fixed-wing UAV.

    State:  x = [px, py, pz, vx, vy, vz]  (6,)
    Input:  u = [ax, ay, az]               (3,)

    Dynamics: x_{k+1} = A x_k + B u_k

    A (6x6): position integrates velocity via dt; velocities unchanged by A.
    B (6x3): velocity integrates acceleration via dt; position unchanged by B.

    Units: km, km/s, km/s^2.  dt in seconds.
    """

    dt: float

    def matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        dt = float(self.dt)
        A = np.array(
            [
                [1, 0, 0, dt, 0,  0 ],
                [0, 1, 0, 0,  dt, 0 ],
                [0, 0, 1, 0,  0,  dt],
                [0, 0, 0, 1,  0,  0 ],
                [0, 0, 0, 0,  1,  0 ],
                [0, 0, 0, 0,  0,  1 ],
            ],
            dtype=float,
        )
        B = np.array(
            [
                [0,  0,  0 ],
                [0,  0,  0 ],
                [0,  0,  0 ],
                [dt, 0,  0 ],
                [0,  dt, 0 ],
                [0,  0,  dt],
            ],
            dtype=float,
        )
        return A, B

    @property
    def n_x(self) -> int:
        return 6

    @property
    def n_u(self) -> int:
        return 3

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Roll out dynamics given initial state and control sequence.

        Args:
            x0: Initial state, shape (6,)
            U:  Control sequence, shape (T, 3)

        Returns:
            X: State trajectory, shape (T+1, 6) where X[0] = x0
        """
        x0 = np.asarray(x0, dtype=float).reshape(self.n_x)
        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != self.n_u:
            raise ValueError(
                f"Control sequence U must have shape (T, {self.n_u}), got {U.shape}"
            )

        A, B = self.matrices()
        T = int(U.shape[0])

        X = np.zeros((T + 1, self.n_x), dtype=float)
        X[0] = x0
        for k in range(T):
            X[k + 1] = A @ X[k] + B @ U[k]
        return X


__all__ = [
    "AircraftLTI",
]