from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def make_approach_then_circle(
    center: np.ndarray,
    radius: float,
    omega: float,
    z_contact: float,
    t_approach: float = 2.0,
    ee_start: np.ndarray | None = None,
    z_pre: float | None = None,
    t_pre: float = 0.0,
) -> Callable[[float], Tuple[np.ndarray, np.ndarray, bool]]:
    """
    Generate a staged EE trajectory:
      - optional pre-approach phase (0..t_pre): move to z_pre above circle start
      - approach phase (t_pre..t_pre+t_approach): descend to contact start point
      - contact phase: circle at constant z_contact

    Returns:
      traj(t) -> (p_ref[3], v_ref[3], surface_mode_bool)
    """
    center = np.asarray(center, dtype=float).reshape(3).copy()
    radius = float(radius)
    omega = float(omega)
    z_contact = float(z_contact)
    t_approach = max(float(t_approach), 1.0e-6)
    t_pre = max(float(t_pre), 0.0)

    p_contact_start = center.copy()
    p_contact_start[0] += radius
    p_contact_start[2] = z_contact

    if ee_start is None:
        p_start = p_contact_start.copy()
        p_start[2] += 0.08
    else:
        p_start = np.asarray(ee_start, dtype=float).reshape(3).copy()

    if z_pre is None:
        z_pre = max(z_contact + 0.05, p_start[2])
    z_pre = float(z_pre)

    p_pre = p_contact_start.copy()
    p_pre[2] = z_pre

    def smoothstep(s: float) -> float:
        s = np.clip(s, 0.0, 1.0)
        return s * s * (3.0 - 2.0 * s)

    def dsmoothstep_ds(s: float) -> float:
        s = np.clip(s, 0.0, 1.0)
        return 6.0 * s * (1.0 - s)

    def blend(p0: np.ndarray, p1: np.ndarray, tau: float, T: float) -> Tuple[np.ndarray, np.ndarray]:
        s_lin = tau / T
        s = smoothstep(s_lin)
        dsdt = dsmoothstep_ds(s_lin) / T
        dp = p1 - p0
        p = (1.0 - s) * p0 + s * p1
        v = dsdt * dp
        return p, v

    def traj(t: float) -> Tuple[np.ndarray, np.ndarray, bool]:
        t = float(t)
        if t_pre > 0.0 and t < t_pre:
            p, v = blend(p_start, p_pre, t, t_pre)
            return p, v, False

        if t < t_pre + t_approach:
            p0 = p_pre if t_pre > 0.0 else p_start
            p, v = blend(p0, p_contact_start, t - t_pre, t_approach)
            return p, v, False

        tt = t - (t_pre + t_approach)
        th = omega * tt

        p = center.copy()
        p[0] += radius * np.cos(th)
        p[1] += radius * np.sin(th)
        p[2] = z_contact

        v = np.zeros(3, dtype=float)
        v[0] = -radius * omega * np.sin(th)
        v[1] = radius * omega * np.cos(th)
        v[2] = 0.0
        return p, v, True

    return traj
