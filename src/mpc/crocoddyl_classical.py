from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import crocoddyl
import numpy as np
import pinocchio as pin
from example_robot_data import load as load_erd


@dataclass
class ClassicalMPCConfig:
    # timing
    horizon: int = 20
    dt: float = 0.01  # should match sim.dt

    # free-space tracking (approach phase)
    w_ee_pos: float = 2.0e2
    w_ee_ori: float = 1.0e1

    # regularization
    w_posture: float = 5.0e-1
    w_v: float = 2.5e-1
    w_tau: float = 1.0e-3
    w_tau_smooth: float = 5.0e-2  # used in command filter

    # contact phase objectives
    z_contact: float = 0.35
    z_press: float = 0.0070
    w_plane_z: float = 6.0e3
    w_vz: float = 6.0e2
    w_tangent_pos: float = 2.0e2
    w_tangent_vel: float = 1.0e2

    # surface detection
    fn_contact_on: float = 2.0
    fn_contact_off: float = 0.5
    z_contact_band: float = 0.01

    # command safety filtering
    tau_limits: np.ndarray = field(default_factory=lambda: np.array([87, 87, 87, 87, 12, 12, 12], dtype=float))
    tau_rate_limit: np.ndarray = field(
        default_factory=lambda: np.array([450, 450, 450, 450, 180, 180, 180], dtype=float)
    )  # Nm/s
    tau_trust_inf: float = 40.0
    tau_smoothing_alpha: float = 0.35  # blend factor toward new command

    # joint-specific velocity damping weights
    v_damp_weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4]))

    # solver
    max_iters: int = 20
    verbose: bool = False
    debug_every: int = 25


class ClassicalCrocoddylMPC:
    """
    Classical EE MPC baseline (no force-feedback state augmentation):
      - Crocoddyl FDDP with free dynamics
      - Two modes driven by trajectory/surface latch:
        free-space approach and contact-plane circle tracking
      - MuJoCo references are converted to Pinocchio world before building costs
      - Pure MPC torque output (no extra external posture/bias torques injected)
    """

    def __init__(
        self,
        sim,
        traj_fn: Callable[[float], Tuple[np.ndarray, np.ndarray, bool]],
        config: ClassicalMPCConfig = ClassicalMPCConfig(),
        lock_fingers: bool = True,
        pin_ee_frame: str = "panda_link8",
    ):
        self.sim = sim
        self.traj_fn = traj_fn
        self.cfg = config
        self._k = 0

        robot_full = load_erd("panda")
        model_full: pin.Model = robot_full.model
        self.model: pin.Model = self._make_arm_only_model(model_full) if lock_fingers else model_full

        if not self.model.existFrame(pin_ee_frame):
            raise ValueError(f"Pinocchio frame '{pin_ee_frame}' does not exist in panda model.")
        self.ee_fid = self.model.getFrameId(pin_ee_frame)

        self.data: pin.Data = self.model.createData()
        self.state = crocoddyl.StateMultibody(self.model)
        self.actuation = crocoddyl.ActuationModelFull(self.state)  # nu = nv

        # Pinocchio panda world differs from MuJoCo panda world by a fixed 180deg yaw.
        # p_mj = R_mj_from_pin @ p_pin
        self.R_mj_from_pin = np.diag([-1.0, -1.0, 1.0])

        obs0 = self.sim.get_observation(with_ee=True, with_jacobian=False)
        self.q_nom = np.asarray(obs0.q, dtype=float).copy()
        self.R_site_from_pin_ee = self._calibrate_site_rotation(obs0)
        self.R_des = self._rot_mj_to_pin(self._make_vertical_down_rotation_mj())

        self.xs: Optional[List[np.ndarray]] = None
        self.us: Optional[List[np.ndarray]] = None
        self._tau_prev = np.asarray(obs0.tau_bias, dtype=float).copy()

        # Surface mode latch (hysteresis).
        self._surface_latched = False

    def _make_arm_only_model(self, model_full: pin.Model) -> pin.Model:
        lock_joint_ids = []
        for j in range(1, model_full.njoints):
            if "finger" in model_full.names[j]:
                lock_joint_ids.append(j)
        if not lock_joint_ids:
            raise RuntimeError("Could not find finger joints to lock in Pinocchio model.")
        q0 = pin.neutral(model_full)
        return pin.buildReducedModel(model_full, lock_joint_ids, q0)

    def _calibrate_site_rotation(self, obs0) -> np.ndarray:
        q0 = np.asarray(obs0.q, dtype=float)
        pin.forwardKinematics(self.model, self.data, q0, np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model, self.data)
        R_pin_ee = self.data.oMf[self.ee_fid].rotation.copy()

        if getattr(obs0, "ee_quat", None) is not None:
            R_mj_site = self._quat_wxyz_to_R(np.asarray(obs0.ee_quat, dtype=float))
        else:
            R_mj_site = self.sim.data.site_xmat[self.sim.ee_site_id].reshape(3, 3).copy()

        # R_mj_site = R_mj_from_pin @ R_pin_ee @ R_site_from_pin_ee
        return R_pin_ee.T @ self.R_mj_from_pin.T @ R_mj_site

    @staticmethod
    def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(4)
        q = q / (np.linalg.norm(q) + 1e-12)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def _make_vertical_down_rotation_mj(self) -> np.ndarray:
        z = np.array([0.0, 0.0, -1.0], dtype=float)
        x = np.array([1.0, 0.0, 0.0], dtype=float)
        y = np.cross(z, x)
        y /= np.linalg.norm(y) + 1e-12
        x = np.cross(y, z)
        x /= np.linalg.norm(x) + 1e-12
        return np.column_stack([x, y, z])

    def _pos_mj_to_pin(self, p_mj: np.ndarray) -> np.ndarray:
        return self.R_mj_from_pin.T @ np.asarray(p_mj, dtype=float)

    def _vel_mj_to_pin(self, v_mj: np.ndarray) -> np.ndarray:
        return self.R_mj_from_pin.T @ np.asarray(v_mj, dtype=float)

    def _rot_mj_to_pin(self, R_mj_site: np.ndarray) -> np.ndarray:
        return self.R_mj_from_pin.T @ np.asarray(R_mj_site, dtype=float) @ self.R_site_from_pin_ee.T

    def _safe_tau(self, tau_target: np.ndarray) -> np.ndarray:
        tau_target = np.asarray(tau_target, dtype=float).copy()
        if not np.all(np.isfinite(tau_target)):
            tau_target = self._tau_prev.copy()

        tau_target = np.clip(tau_target, -self.cfg.tau_limits, self.cfg.tau_limits)

        # Trust region and slew limit around last command.
        d = tau_target - self._tau_prev
        d = np.clip(d, -self.cfg.tau_trust_inf, self.cfg.tau_trust_inf)

        dt = float(getattr(self.sim, "dt", self.cfg.dt))
        max_step = np.asarray(self.cfg.tau_rate_limit, dtype=float) * dt
        d = np.clip(d, -max_step, max_step)
        tau_limited = self._tau_prev + d

        alpha = float(np.clip(self.cfg.tau_smoothing_alpha, 0.0, 1.0))
        tau_cmd = (1.0 - alpha) * self._tau_prev + alpha * tau_limited
        tau_cmd = np.clip(tau_cmd, -self.cfg.tau_limits, self.cfg.tau_limits)

        self._tau_prev = tau_cmd.copy()
        return tau_cmd

    def _detect_surface(self, obs, t: float) -> bool:
        _, _, surf_hint = self.traj_fn(t)

        fn = float(getattr(obs, "f_contact_normal", 0.0))
        ee_z = float(obs.ee_pos[2]) if getattr(obs, "ee_pos", None) is not None else 1e9
        z_target = float(self.cfg.z_contact) - float(self.cfg.z_press)
        near_plane = abs(ee_z - z_target) < float(self.cfg.z_contact_band)

        if self._surface_latched:
            if (fn < self.cfg.fn_contact_off) and (not near_plane) and (not surf_hint):
                self._surface_latched = False
        else:
            if (fn > self.cfg.fn_contact_on) or (near_plane and surf_hint):
                self._surface_latched = True

        return self._surface_latched

    def compute_control(self, obs, t: float) -> np.ndarray:
        self._k += 1

        q = np.asarray(obs.q, dtype=float)
        v = np.asarray(obs.dq, dtype=float)
        x0 = np.concatenate([q, v])

        surface_now = self._detect_surface(obs, t)
        problem = self._build_problem(t0=t, x0=x0, surface_now=surface_now)
        solver = crocoddyl.SolverFDDP(problem)
        if self.cfg.verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])

        N = self.cfg.horizon
        if self.xs is None or self.us is None:
            xs_init = [x0.copy() for _ in range(N + 1)]
            us_init = [self._tau_prev.copy() for _ in range(N)]
        else:
            xs_init = [self.xs[i].copy() if i < len(self.xs) else x0.copy() for i in range(1, N + 1)]
            xs_init.insert(0, x0.copy())
            us_init = [self.us[i].copy() if i < len(self.us) else self._tau_prev.copy() for i in range(1, N)]
            us_init.append(self.us[-1].copy() if len(self.us) > 0 else self._tau_prev.copy())

        ok = solver.solve(xs_init, us_init, self.cfg.max_iters, False)

        tau_raw = self._tau_prev.copy()
        if len(solver.us) > 0:
            u0 = np.asarray(solver.us[0], dtype=float).copy()
            if np.all(np.isfinite(u0)):
                tau_raw = u0
                self.xs = [np.asarray(xi, dtype=float).copy() for xi in solver.xs]
                self.us = [np.asarray(ui, dtype=float).copy() for ui in solver.us]

        tau_cmd = self._safe_tau(tau_raw)

        if (self._k % self.cfg.debug_every) == 0:
            cost = float(getattr(solver, "cost", np.nan))
            iters = int(getattr(solver, "iter", -1))
            fn = float(getattr(obs, "f_contact_normal", 0.0))
            print(
                f"[MPC] t={t:6.3f} ok={ok} cost={cost:.2e} iters={iters:2d} "
                f"|tau_raw|∞={np.max(np.abs(tau_raw)):.2f} |tau_cmd|∞={np.max(np.abs(tau_cmd)):.2f} "
                f"surf={int(surface_now)} fn={fn:.2f} ee_z={obs.ee_pos[2]:.4f}"
            )

        return tau_cmd

    def _build_problem(self, t0: float, x0: np.ndarray, surface_now: bool) -> crocoddyl.ShootingProblem:
        running = []
        for k in range(self.cfg.horizon):
            tk = t0 + k * self.cfg.dt
            p_ref_mj, v_ref_mj, surf_hint = self.traj_fn(tk)
            p_ref_pin = self._pos_mj_to_pin(np.asarray(p_ref_mj, dtype=float))
            v_ref_pin = self._vel_mj_to_pin(np.asarray(v_ref_mj, dtype=float))

            surf_k = bool(surf_hint) or bool(surface_now)
            dam = self._make_dam(
                p_ref=p_ref_pin,
                v_ref=v_ref_pin,
                surface_mode=surf_k,
                terminal=False,
            )
            running.append(crocoddyl.IntegratedActionModelEuler(dam, self.cfg.dt))

        pT_mj, vT_mj, surfT = self.traj_fn(t0 + self.cfg.horizon * self.cfg.dt)
        pT_pin = self._pos_mj_to_pin(np.asarray(pT_mj, dtype=float))
        vT_pin = self._vel_mj_to_pin(np.asarray(vT_mj, dtype=float))
        terminal_dam = self._make_dam(
            p_ref=pT_pin,
            v_ref=vT_pin,
            surface_mode=bool(surfT) or bool(surface_now),
            terminal=True,
        )
        terminal = crocoddyl.IntegratedActionModelEuler(terminal_dam, self.cfg.dt)
        return crocoddyl.ShootingProblem(x0, running, terminal)

    def _make_dam(self, p_ref: np.ndarray, v_ref: np.ndarray, surface_mode: bool, terminal: bool):
        costs = crocoddyl.CostModelSum(self.state, self.actuation.nu)

        if surface_mode:
            # Contact mode: tangential tracking + normal depth/velocity shaping.
            r_xy = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_ref, self.actuation.nu)
            act_xy = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 0.0], dtype=float))
            costs.addCost("ee_xy", crocoddyl.CostModelResidual(self.state, act_xy, r_xy), self.cfg.w_tangent_pos)

            m_ref_xy = pin.Motion(np.array([v_ref[0], v_ref[1], 0.0], dtype=float), np.zeros(3, dtype=float))
            r_vxy = crocoddyl.ResidualModelFrameVelocity(
                self.state, self.ee_fid, m_ref_xy, pin.LOCAL_WORLD_ALIGNED, self.actuation.nu
            )
            act_vxy = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
            costs.addCost("ee_vxy", crocoddyl.CostModelResidual(self.state, act_vxy, r_vxy), self.cfg.w_tangent_vel)

            z_target = float(self.cfg.z_contact) - float(self.cfg.z_press)
            p_ref_z = p_ref.copy()
            p_ref_z[2] = z_target
            r_z = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_ref_z, self.actuation.nu)
            act_z = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0], dtype=float))
            costs.addCost("z_track", crocoddyl.CostModelResidual(self.state, act_z, r_z), self.cfg.w_plane_z)

            m_vz = pin.Motion(np.zeros(3, dtype=float), np.zeros(3, dtype=float))
            r_vz = crocoddyl.ResidualModelFrameVelocity(
                self.state, self.ee_fid, m_vz, pin.LOCAL_WORLD_ALIGNED, self.actuation.nu
            )
            act_vz = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float))
            costs.addCost("vz_damp", crocoddyl.CostModelResidual(self.state, act_vz, r_vz), self.cfg.w_vz)
        else:
            r_pos = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_ref, self.actuation.nu)
            act_pos = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 1.5], dtype=float))
            costs.addCost("ee_pos", crocoddyl.CostModelResidual(self.state, act_pos, r_pos), self.cfg.w_ee_pos)

        r_rot = crocoddyl.ResidualModelFrameRotation(self.state, self.ee_fid, self.R_des, self.actuation.nu)
        costs.addCost("ee_ori", crocoddyl.CostModelResidual(self.state, r_rot), self.cfg.w_ee_ori)

        x_ref = np.concatenate([self.q_nom, np.zeros(self.model.nv)])
        r_x = crocoddyl.ResidualModelState(self.state, x_ref, self.actuation.nu)
        costs.addCost("posture", crocoddyl.CostModelResidual(self.state, r_x), self.cfg.w_posture)

        x_zero = np.zeros(self.model.nq + self.model.nv)
        r_v = crocoddyl.ResidualModelState(self.state, x_zero, self.actuation.nu)
        act_v = crocoddyl.ActivationModelWeightedQuad(
            np.concatenate([np.zeros(self.model.nq), np.asarray(self.cfg.v_damp_weights, dtype=float)])
        )
        costs.addCost("v_damp", crocoddyl.CostModelResidual(self.state, act_v, r_v), self.cfg.w_v)

        if not terminal:
            r_tau = crocoddyl.ResidualModelControl(self.state, self.actuation.nu)
            costs.addCost("tau_reg", crocoddyl.CostModelResidual(self.state, r_tau), self.cfg.w_tau)

        return crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, costs)
