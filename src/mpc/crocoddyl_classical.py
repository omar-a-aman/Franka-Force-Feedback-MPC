from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

import crocoddyl
import numpy as np
import pinocchio as pin
from example_robot_data import load as load_erd


@dataclass
class ClassicalMPCConfig:
    # timing
    horizon: int = 20
    dt: float = 0.01  # control rate (dt_mpc), should match sim.dt
    dt_ocp: Optional[float] = None  # OCP integration step; defaults to dt when None

    # free-space tracking (approach phase)
    w_ee_pos: float = 2.0e2
    w_ee_ori: float = 1.0e1
    # Weights on rotation-log error components (not Euler roll/pitch/yaw).
    ori_weights: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 0.15], dtype=float))

    # regularization
    w_posture: float = 5.0e-1
    w_v: float = 2.5e-1
    w_tau: float = 1.0e-3
    w_tau_smooth: float = 5.0e-2  # used in command filter
    # State regularization reference mode:
    # - "x0": current measured state at MPC solve (benchmark-oriented Eq. 19)
    # - "q_nom": fixed nominal posture
    posture_ref_mode: str = "x0"
    # Torque regularization target mode:
    # - "gravity_x0": tau_ref = tau_g(q0) each MPC update
    # - "gravity_qnom": tau_ref = tau_g(q_nom)
    # - "zero": tau_ref = 0
    torque_ref_mode: str = "gravity_x0"
    w_tau_soft_limits: float = 0.0
    tau_soft_limit_margin: float = 0.2  # Nm kept away from hard bounds
    w_q_soft_limits: float = 0.0
    q_soft_limit_margin: float = 0.05  # rad

    # contact phase objectives
    z_contact: float = 0.35
    z_press: float = 0.0020
    # Keep these at zero for benchmark-oriented classical baseline; the normal behavior
    # should come from contact dynamics + normal-force objective, not extra z shaping.
    w_plane_z: float = 0.0
    w_vz: float = 0.0
    w_tangent_pos: float = 2.0e2
    w_tangent_vel: float = 1.0e2

    # contact modeling (Crocoddyl contact dynamics)
    contact_name: str = "ee_contact"
    # "normal_1d" keeps unilateral normal contact and allows tangential sliding
    # (required for circle-on-surface tasks). "point3d" enforces no-slip point contact.
    contact_model: str = "normal_1d"
    mu: float = 0.6                  # friction coefficient
    friction_margin: float = 1e-3    # slack
    w_friction_cone: float = 2.0e2
    w_unilateral: float = 5.0e1
    contact_gains: np.ndarray = field(default_factory=lambda: np.array([0.0, 60.0], dtype=float))
    contact_inv_damping: float = 1.0e-8
    strict_force_residual_dim: bool = True

    # optional: classic baseline "desired normal force" (ONLY after contact modeling works)
    fn_des: float = 8.0
    w_fn: float = 2.0e1

    # orientation stabilization (extra damping)
    w_wdamp: float = 2.0e1           # angular velocity damping weight
    # Angular-rate damping weights [wx, wy, wz]; keep wz lighter to allow yaw motion.
    w_wdamp_weights: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 0.2], dtype=float))


    # surface detection
    # phase_source:
    # - "trajectory": use traj_fn surf_hint only (benchmark-oriented baseline)
    # - "force_latch": use measured-force latch logic (engineering helper)
    phase_source: str = "trajectory"
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
    apply_command_filter: bool = False

    # joint-specific velocity damping weights
    v_damp_weights: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0, 0.4, 0.4, 0.4]))

    # solver
    max_iters: int = 20
    use_box_fddp: bool = True
    mpc_update_steps: int = 1  # solve OCP every N control steps (>=1)
    use_feedback_policy: bool = True  # apply u = u_ff + K * dx between MPC solves
    feedback_gain_scale: float = 1.0
    verbose: bool = False
    debug_every: int = 25
    # numerical safety guard for solver blow-ups
    max_solver_cost: float = 1.0e8
    max_tau_raw_inf: float = 3.0e2
    fallback_dq_damping: float = 5.0
    contact_release_steps: int = 25


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
        self._warned_keys: Set[str] = set()

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
        self.p_site_minus_frame_pin = self._calibrate_site_position_offset(obs0)
        self.R_des = self._rot_mj_to_pin(self._make_vertical_down_rotation_mj())

        self.xs: Optional[List[np.ndarray]] = None
        self.us: Optional[List[np.ndarray]] = None
        self.Ks: Optional[List[np.ndarray]] = None
        self.ks: Optional[List[np.ndarray]] = None
        self._tau_prev = np.asarray(obs0.tau_bias, dtype=float).copy()
        self._last_solve_step = -1_000_000_000
        self._last_solve_ok = False
        self._last_solve_cost = np.nan
        self._last_solve_iters = -1

        # Surface mode latch (hysteresis).
        self._surface_latched = False
        self._contact_loss_count = 0
        self._prev_surface_mode: Optional[bool] = None
        self.last_info = {
            "ok": False,
            "cost": np.nan,
            "iters": -1,
            "tau_raw_inf": np.nan,
            "tau_cmd_inf": np.nan,
            "surface_mode": False,
            "unstable": False,
            "fn_pred": np.nan,
        }

    @property
    def _dt_ocp(self) -> float:
        dt_ocp = self.cfg.dt if self.cfg.dt_ocp is None else float(self.cfg.dt_ocp)
        return float(max(dt_ocp, 1.0e-6))

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

    def _calibrate_site_position_offset(self, obs0) -> np.ndarray:
        q0 = np.asarray(obs0.q, dtype=float)
        pin.forwardKinematics(self.model, self.data, q0, np.zeros(self.model.nv))
        pin.updateFramePlacements(self.model, self.data)
        p_pin_ee = np.asarray(self.data.oMf[self.ee_fid].translation, dtype=float).copy()

        if getattr(obs0, "ee_pos", None) is not None:
            p_mj_site = np.asarray(obs0.ee_pos, dtype=float).reshape(3)
        else:
            p_mj_site = np.asarray(self.sim.data.site_xpos[self.sim.ee_site_id], dtype=float).reshape(3)

        p_pin_site = self.R_mj_from_pin.T @ p_mj_site
        return p_pin_site - p_pin_ee

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
        # Map MuJoCo EE-site position target to the Pinocchio EE-frame target.
        return self.R_mj_from_pin.T @ np.asarray(p_mj, dtype=float) - self.p_site_minus_frame_pin

    def _vel_mj_to_pin(self, v_mj: np.ndarray) -> np.ndarray:
        return self.R_mj_from_pin.T @ np.asarray(v_mj, dtype=float)

    def _rot_mj_to_pin(self, R_mj_site: np.ndarray) -> np.ndarray:
        return self.R_mj_from_pin.T @ np.asarray(R_mj_site, dtype=float) @ self.R_site_from_pin_ee.T

    def _safe_tau(self, tau_target: np.ndarray) -> np.ndarray:
        tau_target = np.asarray(tau_target, dtype=float).copy()
        if not np.all(np.isfinite(tau_target)):
            tau_target = self._tau_prev.copy()

        tau_target = np.clip(tau_target, -self.cfg.tau_limits, self.cfg.tau_limits)
        if not bool(self.cfg.apply_command_filter):
            self._tau_prev = tau_target.copy()
            return tau_target

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

    def _detect_surface(self, obs, t: float, surf_hint: bool) -> bool:
        fn = float(getattr(obs, "f_contact_normal", 0.0))
        ee_z = float(obs.ee_pos[2]) if getattr(obs, "ee_pos", None) is not None else float("inf")
        near_surface = np.isfinite(ee_z) and (ee_z <= float(self.cfg.z_contact) + float(self.cfg.z_contact_band))

        if self._surface_latched:
            lost_contact = fn < self.cfg.fn_contact_off
            self._contact_loss_count = self._contact_loss_count + 1 if lost_contact else 0
            if self._contact_loss_count >= int(self.cfg.contact_release_steps):
                self._surface_latched = False
                self._contact_loss_count = 0
        else:
            # Allow a robust latch either from measured force or from proximity once contact phase starts.
            if (fn > self.cfg.fn_contact_on) or (surf_hint and near_surface):
                self._surface_latched = True
                self._contact_loss_count = 0

        return self._surface_latched

    def compute_control(self, obs, t: float) -> np.ndarray:
        self._k += 1

        q = np.asarray(obs.q, dtype=float)
        v = np.asarray(obs.dq, dtype=float)
        x0 = np.concatenate([q, v])

        _, _, surf_hint_now = self.traj_fn(t)
        phase_mode = str(self.cfg.phase_source).strip().lower()
        if phase_mode == "force_latch":
            surface_now = self._detect_surface(obs, t, surf_hint_now)
        else:
            # Scheduled mode: contact phase is provided by the reference trajectory.
            surface_now = bool(surf_hint_now)

        if self._prev_surface_mode is None:
            self._prev_surface_mode = bool(surface_now)
        elif bool(surface_now) != bool(self._prev_surface_mode):
            # Mode switches invalidate the shifted warm start.
            self.xs = None
            self.us = None
            self.Ks = None
            self.ks = None
            self._last_solve_step = -1_000_000_000
            self._prev_surface_mode = bool(surface_now)

        solve_period = max(1, int(self.cfg.mpc_update_steps))
        need_solve = (
            (self.us is None)
            or (self.xs is None)
            or ((self._k - self._last_solve_step) >= solve_period)
        )

        solved_now = False
        ok = self._last_solve_ok
        cost = float(self._last_solve_cost)
        iters = int(self._last_solve_iters)
        fn_pred = float(self.last_info.get("fn_pred", np.nan))

        if need_solve:
            # Build (or rebuild) the shooting problem for this MPC update.
            problem = self._build_problem(t0=t, x0=x0, surface_now=surface_now)

            # (Recommended) reuse solver object if possible
            # Reuse solver if the Python API supports updating the problem; otherwise recreate it.
            if not hasattr(self, "_solver") or self._solver is None:
                self._solver = self._make_solver(problem)
                if self.cfg.verbose:
                    self._solver.setCallbacks([crocoddyl.CallbackVerbose()])
            else:
                if hasattr(self._solver, "setProblem"):
                    self._solver.setProblem(problem)
                else:
                    # Older Crocoddyl python bindings: no setProblem() -> rebuild solver each tick
                    self._solver = self._make_solver(problem)
                    if self.cfg.verbose:
                        self._solver.setCallbacks([crocoddyl.CallbackVerbose()])

            solver = self._solver
            N = self.cfg.horizon
            xs_init, us_init = self._shift_guess(x0, N)

            ok = solver.solve(xs_init, us_init, self.cfg.max_iters, False)
            cost = float(getattr(solver, "cost", np.nan))
            iters = int(getattr(solver, "iter", -1))
            fn_pred = self._extract_predicted_normal_force(problem) if surface_now else np.nan
            solved_now = True

            self._last_solve_step = self._k
            self._last_solve_ok = bool(ok)
            self._last_solve_cost = float(cost)
            self._last_solve_iters = int(iters)

            # Store policy around the nominal trajectory for high-rate execution between MPC solves.
            if len(solver.us) > 0:
                u0 = np.asarray(solver.us[0], dtype=float).copy()
                if np.all(np.isfinite(u0)):
                    self.xs = [np.asarray(xi, dtype=float).copy() for xi in solver.xs]
                    self.us = [np.asarray(ui, dtype=float).copy() for ui in solver.us]
                    if hasattr(solver, "K"):
                        self.Ks = [np.asarray(Ki, dtype=float).copy() for Ki in solver.K]
                    else:
                        self.Ks = None
                    self.ks = None

        tau_raw, policy_idx = self._policy_control(x0)

        # Hard safety fallback if solver diverges numerically.
        tau_raw_inf = float(np.max(np.abs(tau_raw)))
        unstable = (not np.isfinite(cost)) or (cost > float(self.cfg.max_solver_cost)) or (
            tau_raw_inf > float(self.cfg.max_tau_raw_inf)
        )
        if unstable:
            tau_raw = np.asarray(obs.tau_bias, dtype=float) - float(self.cfg.fallback_dq_damping) * v
            self.xs = None
            self.us = None
            self.Ks = None
            self.ks = None
            self._last_solve_step = -1_000_000_000

        tau_cmd = self._safe_tau(tau_raw)
        tau_cmd_inf = float(np.max(np.abs(tau_cmd)))
        self.last_info = {
            "ok": bool(ok),
            "cost": float(cost),
            "iters": iters,
            "tau_raw_inf": tau_raw_inf,
            "tau_cmd_inf": tau_cmd_inf,
            "surface_mode": bool(surface_now),
            "unstable": bool(unstable),
            "fn_pred": float(fn_pred) if np.isfinite(fn_pred) else np.nan,
            "solved_now": bool(solved_now),
            "policy_idx": int(policy_idx),
        }

        if (self._k % self.cfg.debug_every) == 0:
            fn = float(getattr(obs, "f_contact_normal", 0.0))
            ee_z = float(obs.ee_pos[2]) if getattr(obs, "ee_pos", None) is not None else np.nan
            print(
                f"[MPC] t={t:6.3f} ok={ok} cost={cost:.2e} iters={iters:2d} "
                f"|tau_raw|∞={np.max(np.abs(tau_raw)):.2f} |tau_cmd|∞={np.max(np.abs(tau_cmd)):.2f} "
                f"surf={int(surface_now)} fn={fn:.2f} fn_pred={fn_pred:.2f} ee_z={ee_z:.4f} "
                f"solve={int(solved_now)} i={int(policy_idx)} unstable={int(unstable)}"
            )

        # Receding-horizon rollout: shift stored policy between MPC solves.
        if (not solved_now) and (self.us is not None) and (self.xs is not None):
            if len(self.us) > 1:
                self.us = self.us[1:] + [self.us[-1]]
            if len(self.xs) > 1:
                self.xs = self.xs[1:] + [self.xs[-1]]
            if (self.Ks is not None) and (len(self.Ks) > 1):
                self.Ks = self.Ks[1:] + [self.Ks[-1]]
            self.ks = None

        return tau_cmd

    def _make_solver(self, problem: crocoddyl.ShootingProblem):
        if self.cfg.use_box_fddp and hasattr(crocoddyl, "SolverBoxFDDP"):
            return crocoddyl.SolverBoxFDDP(problem)
        return crocoddyl.SolverFDDP(problem)

    def _gravity_torque(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(self.model.nq)
        v0 = np.zeros(self.model.nv, dtype=float)
        a0 = np.zeros(self.model.nv, dtype=float)
        return np.asarray(pin.rnea(self.model, self.data, q, v0, a0), dtype=float).reshape(self.actuation.nu)

    def _compute_tau_reference(self, q_now: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.torque_ref_mode).strip().lower()
        if mode == "zero":
            return np.zeros(self.actuation.nu, dtype=float)
        if mode == "gravity_qnom":
            return self._gravity_torque(self.q_nom)
        # Default/benchmark-oriented: gravity centered at the current OCP linearization point.
        return self._gravity_torque(q_now)

    def _compute_posture_reference(self, x0: np.ndarray) -> np.ndarray:
        mode = str(self.cfg.posture_ref_mode).strip().lower()
        if mode == "q_nom":
            return np.concatenate([self.q_nom, np.zeros(self.model.nv, dtype=float)])
        return np.asarray(x0, dtype=float).reshape(self.state.nx).copy()

    def _make_control_residual(self, u_ref: np.ndarray):
        u_ref = np.asarray(u_ref, dtype=float).reshape(self.actuation.nu)
        try:
            return crocoddyl.ResidualModelControl(self.state, u_ref, self.actuation.nu)
        except TypeError:
            try:
                return crocoddyl.ResidualModelControl(self.state, u_ref)
            except TypeError:
                return crocoddyl.ResidualModelControl(self.state, self.actuation.nu)

    def _make_tau_soft_limit_activation(self):
        tau_lim = np.asarray(self.cfg.tau_limits, dtype=float).reshape(self.actuation.nu)
        margin = float(max(self.cfg.tau_soft_limit_margin, 0.0))
        margin = min(margin, float(np.min(tau_lim) - 1.0e-6))
        lb = -tau_lim + margin
        ub = tau_lim - margin
        bounds = crocoddyl.ActivationBounds(lb, ub, 1.0)
        return crocoddyl.ActivationModelQuadraticBarrier(bounds)

    def _make_q_soft_limit_cost(self):
        q_lb = np.asarray(self.model.lowerPositionLimit, dtype=float).reshape(self.model.nq)
        q_ub = np.asarray(self.model.upperPositionLimit, dtype=float).reshape(self.model.nq)
        finite_lb = np.isfinite(q_lb)
        finite_ub = np.isfinite(q_ub)
        finite_both = finite_lb & finite_ub

        q_ref = self.q_nom.copy()
        q_ref[finite_both] = 0.5 * (q_lb[finite_both] + q_ub[finite_both])

        margin = float(max(self.cfg.q_soft_limit_margin, 0.0))
        q_lb_shrunk = q_lb.copy()
        q_ub_shrunk = q_ub.copy()
        q_lb_shrunk[finite_lb] = q_lb[finite_lb] + margin
        q_ub_shrunk[finite_ub] = q_ub[finite_ub] - margin
        invalid = finite_both & (q_lb_shrunk > q_ub_shrunk)
        if np.any(invalid):
            mid = 0.5 * (q_lb[invalid] + q_ub[invalid])
            q_lb_shrunk[invalid] = mid - 1.0e-3
            q_ub_shrunk[invalid] = mid + 1.0e-3

        lb_q_res = np.full(self.model.nq, -np.inf, dtype=float)
        ub_q_res = np.full(self.model.nq, np.inf, dtype=float)
        lb_q_res[finite_lb] = q_lb_shrunk[finite_lb] - q_ref[finite_lb]
        ub_q_res[finite_ub] = q_ub_shrunk[finite_ub] - q_ref[finite_ub]

        lb = np.concatenate([lb_q_res, np.full(self.model.nv, -np.inf, dtype=float)])
        ub = np.concatenate([ub_q_res, np.full(self.model.nv, np.inf, dtype=float)])

        x_ref = np.concatenate([q_ref, np.zeros(self.model.nv, dtype=float)])
        residual = crocoddyl.ResidualModelState(self.state, x_ref, self.actuation.nu)
        activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub, 1.0))
        return crocoddyl.CostModelResidual(self.state, activation, residual)

    def _build_problem(self, t0: float, x0: np.ndarray, surface_now: bool) -> crocoddyl.ShootingProblem:
        dt_ocp = self._dt_ocp
        tau_ref = self._compute_tau_reference(x0[: self.model.nq])
        x_reg_ref = self._compute_posture_reference(x0)
        running = []
        for k in range(self.cfg.horizon):
            tk = t0 + k * dt_ocp
            p_ref_mj, v_ref_mj, _ = self.traj_fn(tk)
            p_ref_pin = self._pos_mj_to_pin(np.asarray(p_ref_mj, dtype=float))
            v_ref_pin = self._vel_mj_to_pin(np.asarray(v_ref_mj, dtype=float))

            # Use one model per MPC tick: free dynamics until surface is latched.
            surf_k = bool(surface_now)
            dam = self._make_dam(
                p_ref=p_ref_pin,
                v_ref=v_ref_pin,
                surface_mode=surf_k,
                terminal=False,
                tau_ref=tau_ref,
                x_reg_ref=x_reg_ref,
            )
            running.append(crocoddyl.IntegratedActionModelEuler(dam, dt_ocp))

        pT_mj, vT_mj, _ = self.traj_fn(t0 + self.cfg.horizon * dt_ocp)
        pT_pin = self._pos_mj_to_pin(np.asarray(pT_mj, dtype=float))
        vT_pin = self._vel_mj_to_pin(np.asarray(vT_mj, dtype=float))
        terminal_dam = self._make_dam(
            p_ref=pT_pin,
            v_ref=vT_pin,
            surface_mode=bool(surface_now),
            terminal=True,
            tau_ref=tau_ref,
            x_reg_ref=x_reg_ref,
        )
        terminal = crocoddyl.IntegratedActionModelEuler(terminal_dam, dt_ocp)
        return crocoddyl.ShootingProblem(x0, running, terminal)

    def _make_dam(
        self,
        p_ref: np.ndarray,
        v_ref: np.ndarray,
        surface_mode: bool,
        terminal: bool,
        tau_ref: np.ndarray,
        x_reg_ref: np.ndarray,
    ):
        costs = crocoddyl.CostModelSum(self.state, self.actuation.nu)

        # --- common state regularization ---
        x_ref = np.asarray(x_reg_ref, dtype=float).reshape(self.state.nx)
        r_x = crocoddyl.ResidualModelState(self.state, x_ref, self.actuation.nu)
        costs.addCost("posture", crocoddyl.CostModelResidual(self.state, r_x), self.cfg.w_posture)

        x_zero = np.zeros(self.model.nq + self.model.nv)
        r_v = crocoddyl.ResidualModelState(self.state, x_zero, self.actuation.nu)
        act_v = crocoddyl.ActivationModelWeightedQuad(
            np.concatenate([np.zeros(self.model.nq), np.asarray(self.cfg.v_damp_weights, dtype=float)])
        )
        costs.addCost("v_damp", crocoddyl.CostModelResidual(self.state, act_v, r_v), self.cfg.w_v)

        if self.cfg.w_q_soft_limits > 0.0:
            costs.addCost("q_soft_limits", self._make_q_soft_limit_cost(), self.cfg.w_q_soft_limits)

        # --- orientation stabilization: pose + angular velocity damping ---
        r_rot = crocoddyl.ResidualModelFrameRotation(self.state, self.ee_fid, self.R_des, self.actuation.nu)
        act_rot = crocoddyl.ActivationModelWeightedQuad(np.asarray(self.cfg.ori_weights, dtype=float))
        costs.addCost("ee_ori", crocoddyl.CostModelResidual(self.state, act_rot, r_rot), self.cfg.w_ee_ori)

        # angular velocity damping (LOCAL_WORLD_ALIGNED; penalize only rotational part)
        m_w0 = pin.Motion(np.zeros(3), np.zeros(3))
        r_w = crocoddyl.ResidualModelFrameVelocity(
            self.state, self.ee_fid, m_w0, pin.LOCAL_WORLD_ALIGNED, self.actuation.nu
        )
        ww = np.asarray(self.cfg.w_wdamp_weights, dtype=float).reshape(3)
        act_w = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 0.0, ww[0], ww[1], ww[2]], dtype=float))
        costs.addCost("w_damp", crocoddyl.CostModelResidual(self.state, act_w, r_w), self.cfg.w_wdamp)

        if not terminal:
            r_tau = self._make_control_residual(np.asarray(tau_ref, dtype=float))
            costs.addCost("tau_reg", crocoddyl.CostModelResidual(self.state, r_tau), self.cfg.w_tau)

            if self.cfg.w_tau_soft_limits > 0.0:
                r_tau_lim = self._make_control_residual(np.zeros(self.actuation.nu, dtype=float))
                act_tau_lim = self._make_tau_soft_limit_activation()
                costs.addCost(
                    "tau_soft_limits",
                    crocoddyl.CostModelResidual(self.state, act_tau_lim, r_tau_lim),
                    self.cfg.w_tau_soft_limits,
                )

        # ===========================
        # FREE SPACE (no contact)
        # ===========================
        if not surface_mode:
            r_pos = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_ref, self.actuation.nu)
            act_pos = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 2.5], dtype=float))
            costs.addCost("ee_pos", crocoddyl.CostModelResidual(self.state, act_pos, r_pos), self.cfg.w_ee_pos)

            dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, costs)
            dam.u_lb = -np.asarray(self.cfg.tau_limits, dtype=float)
            dam.u_ub = np.asarray(self.cfg.tau_limits, dtype=float)
            return dam
        # ===========================
        # CONTACT MODE (benchmark-oriented)
        # ===========================
        contacts = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)

        # Desired contact point for the EE frame in Pinocchio WORLD coordinates.
        # p_ref is already the frame-consistent target converted from MuJoCo site refs.
        z_target = float(p_ref[2]) - float(self.cfg.z_press)
        p_contact = p_ref.copy()
        p_contact[2] = z_target

        # NOTE: 3D rigid point contact enforces no-slip and therefore blocks tangential motion.
        # For "draw circle while maintaining table contact", the physically consistent baseline is
        # normal-only contact (unilateral normal force + tangential tracking costs).
        contact_mode = str(self.cfg.contact_model).strip().lower()
        if contact_mode in ("point3d", "3d", "rigid3d", "route_a_3d"):
            contact_model = self._make_contact_model_3d(p_contact)
            nc = 3
        else:
            contact_model = self._make_contact_model_1d(z_target)
            nc = 1

        # Add contact (name is for bookkeeping).
        contacts.addContact(self.cfg.contact_name, contact_model)
        # In Crocoddyl residuals, this id is the contact frame id (not an index in ContactModelMultiple).
        contact_frame_id = self._contact_frame_id(contact_model)

        # ----- Tangential tracking costs (x,y) -----
        r_xy = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_ref, self.actuation.nu)
        act_xy = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 0.0], dtype=float))
        costs.addCost("ee_xy", crocoddyl.CostModelResidual(self.state, act_xy, r_xy), self.cfg.w_tangent_pos)

        # Tangential velocity tracking (x,y)
        m_ref_xy = pin.Motion(np.array([v_ref[0], v_ref[1], 0.0], dtype=float), np.zeros(3, dtype=float))
        r_vxy = crocoddyl.ResidualModelFrameVelocity(
            self.state, self.ee_fid, m_ref_xy, pin.LOCAL_WORLD_ALIGNED, self.actuation.nu
        )
        act_vxy = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float))
        costs.addCost("ee_vxy", crocoddyl.CostModelResidual(self.state, act_vxy, r_vxy), self.cfg.w_tangent_vel)

        # Optional soft vertical shaping around the pressed contact height.
        if self.cfg.w_plane_z > 0.0:
            r_z = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_contact, self.actuation.nu)
            act_z = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0], dtype=float))
            costs.addCost("plane_z", crocoddyl.CostModelResidual(self.state, act_z, r_z), self.cfg.w_plane_z)

        if self.cfg.w_vz > 0.0:
            m_vz = pin.Motion(np.zeros(3, dtype=float), np.zeros(3, dtype=float))
            r_vz = crocoddyl.ResidualModelFrameVelocity(
                self.state, self.ee_fid, m_vz, pin.LOCAL_WORLD_ALIGNED, self.actuation.nu
            )
            act_vz = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float))
            costs.addCost("vz_damp", crocoddyl.CostModelResidual(self.state, act_vz, r_vz), self.cfg.w_vz)

        # Friction cone + unilateral feasibility (soft constraints via barrier activation).
        if (nc == 3) and (self.cfg.w_friction_cone > 0.0):
            cone = self._make_friction_cone()
            r_fc = self._make_contact_friction_residual(contact_frame_id, cone)
            act_fc = self._make_friction_barrier_activation(cone)
            costs.addCost(
                "friction_cone",
                crocoddyl.CostModelResidual(self.state, act_fc, r_fc),
                self.cfg.w_friction_cone,
            )

        # Explicit unilateral normal-force barrier (Fn >= 0).
        if self.cfg.w_unilateral > 0.0:
            if nc == 1:
                r_f0 = self._make_contact_force_residual(contact_frame_id, np.array([0.0], dtype=float), nc)
            else:
                r_f0 = self._make_contact_force_residual(contact_frame_id, pin.Force(np.zeros(6, dtype=float)), nc)
            self._check_force_residual_dimension(r_f0, nc, context="unilateral")
            if nc == 1:
                lb_uni = np.array([0.0 + float(self.cfg.friction_margin)], dtype=float)
                ub_uni = np.array([np.inf], dtype=float)
            else:
                lb_uni = np.array([-np.inf, -np.inf, 0.0 + float(self.cfg.friction_margin)], dtype=float)
                ub_uni = np.array([np.inf, np.inf, np.inf], dtype=float)
            act_uni = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb_uni, ub_uni, 1.0))
            costs.addCost("unilateral", crocoddyl.CostModelResidual(self.state, act_uni, r_f0), self.cfg.w_unilateral)

        # Classical baseline normal-force objective (predicted force, no feedback injection).
        if self.cfg.w_fn > 0.0:
            if nc == 1:
                r_fn = self._make_contact_force_residual(
                    contact_frame_id, np.array([float(self.cfg.fn_des)], dtype=float), nc
                )
            else:
                fref = pin.Force(np.array([0.0, 0.0, float(self.cfg.fn_des), 0.0, 0.0, 0.0], dtype=float))
                r_fn = self._make_contact_force_residual(contact_frame_id, fref, nc)
            self._check_force_residual_dimension(r_fn, nc, context="fn_track")
            if nc == 1:
                act_fz = crocoddyl.ActivationModelWeightedQuad(np.array([1.0], dtype=float))
            else:
                act_fz = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0], dtype=float))
            costs.addCost("fn_track", crocoddyl.CostModelResidual(self.state, act_fz, r_fn), self.cfg.w_fn)

        # ----- Contact forward dynamics model -----
        # NOTE: This signature also varies; keep your current one unless it errors.
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contacts, costs, 0.0, True
        )
        dam.JMinvJt_damping = float(self.cfg.contact_inv_damping)
        dam.u_lb = -np.asarray(self.cfg.tau_limits, dtype=float)
        dam.u_ub = np.asarray(self.cfg.tau_limits, dtype=float)
        return dam



    
    def _shift_guess(self, x0: np.ndarray, N: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Proper receding-horizon warm start:
        xs_init = [x0] + xs_prev[1:] (clipped to horizon)
        us_init = us_prev[1:] + [us_prev[-1]]
        Falls back to holding x0 / tau_prev if no previous solution.
        """
        if self.xs is None or self.us is None or len(self.us) < N:
            xs_init = [x0.copy() for _ in range(N + 1)]
            us_init = [self._tau_prev.copy() for _ in range(N)]
            return xs_init, us_init

        xs_prev = self.xs
        us_prev = self.us

        xs_init = [x0.copy()]
        xs_init += [xs_prev[i].copy() for i in range(1, min(len(xs_prev), N + 1))]
        while len(xs_init) < (N + 1):
            xs_init.append(xs_prev[-1].copy())

        us_init = [us_prev[i].copy() for i in range(1, min(len(us_prev), N))]
        while len(us_init) < N:
            us_init.append(us_prev[-1].copy())

        return xs_init, us_init

    def _policy_control(self, x_now: np.ndarray) -> tuple[np.ndarray, int]:
        if self.us is None or len(self.us) == 0:
            return self._tau_prev.copy(), -1

        i = 0
        u = np.asarray(self.us[i], dtype=float).copy()
        # Use the solved nominal control plus local state feedback.
        # Do not add solver.k here: solver.us is already the optimized nominal sequence.

        if (
            self.cfg.use_feedback_policy
            and self.Ks is not None
            and self.xs is not None
            and i < len(self.Ks)
            and i < len(self.xs)
        ):
            dx = np.asarray(x_now - self.xs[i], dtype=float)
            K = np.asarray(self.Ks[i], dtype=float)
            u += float(self.cfg.feedback_gain_scale) * (K @ dx)

        return u, i
    
    def _make_contact_friction_residual(self, contact_id: int, cone):
        try:
            return crocoddyl.ResidualModelContactFrictionCone(self.state, contact_id, cone, self.actuation.nu, True)
        except TypeError:
            try:
                return crocoddyl.ResidualModelContactFrictionCone(self.state, contact_id, cone, self.actuation.nu)
            except TypeError:
                return crocoddyl.ResidualModelContactFrictionCone(self.state, contact_id, cone)

    def _contact_frame_id(self, contact_model) -> int:
        """
        Return contact frame id used by Crocoddyl contact-force/friction residuals.
        In current Crocoddyl bindings, these residuals take a frame id.
        """
        cid = int(self.ee_fid)
        if hasattr(contact_model, "id"):
            try:
                cid_model = int(contact_model.id)
                if cid_model != cid:
                    self._warn_once(
                        "contact_frame_id_mismatch",
                        f"Contact model frame id={cid_model} differs from ee_fid={cid}; using model id.",
                    )
                cid = cid_model
            except Exception:
                pass
        return cid

    def _make_contact_force_residual(self, contact_id: int, fref, nc: int):
        if int(nc) == 1:
            return self._make_contact_force_residual_1d(contact_id, fref)
        return self._make_contact_force_residual_generic(contact_id, fref, nc)

    def _make_contact_force_residual_generic(self, contact_id: int, fref, nc: int):
        for args in (
            (self.state, contact_id, fref, nc, self.actuation.nu, True),
            (self.state, contact_id, fref, nc, self.actuation.nu),
            (self.state, contact_id, fref, nc),
        ):
            try:
                return crocoddyl.ResidualModelContactForce(*args)
            except TypeError:
                continue
        raise TypeError("Could not build ResidualModelContactForce with available overloads.")

    def _extract_normal_force_scalar(self, fref) -> float:
        if isinstance(fref, pin.Force):
            if hasattr(fref, "linear"):
                lin = np.asarray(fref.linear, dtype=float).reshape(-1)
                if lin.size >= 3:
                    return float(lin[2])
            arr = np.asarray(fref, dtype=float).reshape(-1)
            if arr.size >= 3:
                return float(arr[2])
            return float(arr[0])
        arr = np.asarray(fref, dtype=float).reshape(-1)
        if arr.size == 1:
            return float(arr[0])
        if arr.size >= 3:
            return float(arr[2])
        return float(arr[0])

    def _make_contact_force_residual_1d(self, contact_id: int, fref):
        fn_ref = self._extract_normal_force_scalar(fref)
        refs = [
            np.array([fn_ref], dtype=float),
            float(fn_ref),
            pin.Force(np.array([0.0, 0.0, fn_ref, 0.0, 0.0, 0.0], dtype=float)),
        ]
        best_residual = None
        best_nr = None
        for ref in refs:
            try:
                residual = self._make_contact_force_residual_generic(contact_id, ref, 1)
            except TypeError:
                continue
            nr = int(getattr(residual, "nr", -1))
            if nr == 1:
                return residual
            if best_residual is None:
                best_residual = residual
                best_nr = nr

        if best_residual is not None:
            msg = (
                f"1D contact-force residual returned nr={best_nr} (expected 1). "
                "Reference semantics are likely inconsistent with nc=1."
            )
            if self.cfg.strict_force_residual_dim:
                raise RuntimeError(msg)
            self._warn_once("force_nr_1d", msg)
            return best_residual

        raise TypeError("Failed to create 1D ResidualModelContactForce with scalar or spatial references.")

    def _check_force_residual_dimension(self, residual, expected_nr: int, context: str):
        nr = int(getattr(residual, "nr", expected_nr))
        if nr == int(expected_nr):
            return
        msg = f"{context}: residual dimension mismatch nr={nr}, expected={int(expected_nr)}."
        if self.cfg.strict_force_residual_dim:
            raise RuntimeError(msg)
        self._warn_once(f"force_nr_{context}", msg)

    def _warn_once(self, key: str, msg: str):
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(f"[MPC][warn] {msg}")

    def _make_friction_barrier_activation(self, cone):
        lb = np.asarray(cone.lb, dtype=float).copy()
        ub = np.asarray(cone.ub, dtype=float).copy()

        eps = max(float(self.cfg.friction_margin), 0.0)
        if eps > 0.0:
            finite_lb = np.isfinite(lb)
            finite_ub = np.isfinite(ub)
            lb[finite_lb] += eps
            ub[finite_ub] -= eps

        bounds = crocoddyl.ActivationBounds(lb, ub, 1.0)
        return crocoddyl.ActivationModelQuadraticBarrier(bounds)

    def _extract_predicted_normal_force(self, problem) -> float:
        try:
            if problem is None or len(problem.runningDatas) == 0:
                return np.nan

            rd0 = problem.runningDatas[0]
            dd0 = getattr(rd0, "differential", None)
            if dd0 is None:
                return np.nan

            mb = getattr(dd0, "multibody", None)
            if mb is None or not hasattr(mb, "contacts"):
                return np.nan

            contacts = mb.contacts.contacts
            contacts_dict = contacts.todict() if hasattr(contacts, "todict") else {}
            if len(contacts_dict) == 0:
                return np.nan

            cdata = contacts_dict.get(self.cfg.contact_name, next(iter(contacts_dict.values())))
            if not hasattr(cdata, "f"):
                return np.nan

            f_obj = cdata.f
            if hasattr(f_obj, "linear"):
                f_lin = np.asarray(f_obj.linear, dtype=float).reshape(-1)
                if f_lin.size >= 3:
                    # Contact normal is world +z in LOCAL_WORLD_ALIGNED for the table task.
                    return float(f_lin[2])

            f_arr = np.asarray(f_obj, dtype=float).reshape(-1)
            if f_arr.size == 1:
                return float(f_arr[0])
            if f_arr.size >= 3:
                return float(f_arr[2])
            return np.nan
        except Exception:
            return np.nan

    def _make_contact_model_3d(self, p_contact: np.ndarray):
        gains = np.asarray(self.cfg.contact_gains, dtype=float).reshape(2)
        try:
            return crocoddyl.ContactModel3D(
                self.state, self.ee_fid, p_contact, pin.LOCAL_WORLD_ALIGNED, self.actuation.nu, gains
            )
        except TypeError:
            try:
                return crocoddyl.ContactModel3D(
                    self.state, self.ee_fid, p_contact, pin.LOCAL_WORLD_ALIGNED, gains
                )
            except TypeError:
                try:
                    return crocoddyl.ContactModel3D(
                        self.state, self.ee_fid, p_contact, pin.LOCAL_WORLD_ALIGNED, self.actuation.nu
                    )
                except TypeError:
                    try:
                        return crocoddyl.ContactModel3D(
                            self.state, self.ee_fid, p_contact, pin.LOCAL_WORLD_ALIGNED
                        )
                    except TypeError:
                        return crocoddyl.ContactModel3D(self.state, self.ee_fid, p_contact)

    def _make_contact_model_1d(self, z_target: float):
        gains = np.asarray(self.cfg.contact_gains, dtype=float).reshape(2)
        R = np.eye(3)
        try:
            return crocoddyl.ContactModel1D(
                self.state,
                self.ee_fid,
                float(z_target),
                pin.LOCAL_WORLD_ALIGNED,
                R,
                self.actuation.nu,
                gains,
            )
        except TypeError:
            try:
                return crocoddyl.ContactModel1D(
                    self.state,
                    self.ee_fid,
                    float(z_target),
                    pin.LOCAL_WORLD_ALIGNED,
                    self.actuation.nu,
                    gains,
                )
            except TypeError:
                try:
                    return crocoddyl.ContactModel1D(
                        self.state, self.ee_fid, float(z_target), pin.LOCAL_WORLD_ALIGNED, gains
                    )
                except TypeError:
                    return crocoddyl.ContactModel1D(self.state, self.ee_fid, float(z_target), pin.LOCAL_WORLD_ALIGNED)

    def _make_friction_cone(self):
        R = np.eye(3)
        mu = float(self.cfg.mu)
        nf = 4
        inner = False

        # Try the most common signature first: (R, mu, nf, inner)
        try:
            return crocoddyl.FrictionCone(R, mu, nf, inner)
        except Exception:
            pass

        # Some builds: (R, mu, nf)
        try:
            return crocoddyl.FrictionCone(R, mu, nf)
        except Exception:
            pass

        # Some older builds: (mu, nf, inner)
        return crocoddyl.FrictionCone(mu, nf, inner)
