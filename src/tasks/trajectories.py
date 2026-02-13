# src/tasks/trajectories.py
import numpy as np
from typing import Callable, Tuple


def make_circle_trajectory(
    center: np.ndarray,
    radius: float,
    omega: float,
) -> Callable[[float], Tuple[np.ndarray, np.ndarray]]:
    center = np.asarray(center, dtype=float).copy()

    def traj(t: float):
        theta = omega * t
        p = center.copy()
        p[0] += radius * np.cos(theta)
        p[1] += radius * np.sin(theta)

        v = np.zeros(3)
        v[0] = -radius * omega * np.sin(theta)
        v[1] =  radius * omega * np.cos(theta)
        return p, v

    return traj
