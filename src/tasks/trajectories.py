# src/tasks/trajectories.py

import numpy as np


def circle_trajectory(
    center: np.ndarray,
    radius: float,
    omega: float,
    normal_axis: int = 2,
):
    """
    Returns a function traj(t) that outputs:
        p_des(t), v_des(t)

    Circular motion in XY plane by default.
    """

    center = np.asarray(center, dtype=float)

    def traj(t: float):
        theta = omega * t

        # Position
        p = center.copy()
        p[0] += radius * np.cos(theta)
        p[1] += radius * np.sin(theta)

        # Velocity
        v = np.zeros(3)
        v[0] = -radius * omega * np.sin(theta)
        v[1] =  radius * omega * np.cos(theta)

        return p, v

    return traj
