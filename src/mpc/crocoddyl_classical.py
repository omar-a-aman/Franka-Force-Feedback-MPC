# src/mpc/crocoddyl_classical.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List

import numpy as np
import pinocchio as pin
import crocoddyl
from example_robot_data import load as load_erd


@dataclass
class ClassicalMPCConfig:
    horizon: int = 20
    dt: float = 0.02

    # Costs
    w_ee_pos: float = 2.0e3
    w_ee_ori: float = 8.0e2      # <<< NEW: EE orientation cost
    w_posture: float = 2.0
    w_tau: float = 1.0e-3

    # Limits
    tau_limits: np.ndarray = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

    # Solver
    max_iters: int = 15
    verbose: bool = False

    # Joint-limit avoidance (soft barrier)
    w_joint_limits: float = 5.0e2
    joint_margin: float = 0.10      # [rad] keep this distance away from limits
    vmax_for_barrier: float = 10.0  # [rad/s] just to bound velocity part of barrier



class ClassicalCrocoddylMPC:
    """
    Classical MPC baseline using Crocoddyl FDDP:
      x = (q, v)  with nq=7, nv=7
      u = tau     with nu=7

    Costs:
      - EE position tracking
      - EE orientation tracking (vertical-down tool)
      - posture regularization
      - torque regularization
    """

    def __init__(
        self,
        sim,
        traj_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
        config: ClassicalMPCConfig = ClassicalMPCConfig(),
        ee_frame_candidates: Optional[List[str]] = None,
        q_nominal: Optional[np.ndarray] = None,
        lock_fingers: bool = True,
    ):
        self.sim = sim
        self.traj_fn = traj_fn
        self.cfg = config

        # --- Load Panda model (Pinocchio) ---
        robot_full = load_erd("panda")
        model_full: pin.Model = robot_full.model

        if lock_fingers:
            model = self._make_arm_only_model(model_full)
        else:
            model = model_full

        self.model: pin.Model = model
        self.data: pin.Data = self.model.createData()

        self.state = crocoddyl.StateMultibody(self.model)
        self.actuation = crocoddyl.ActuationModelFull(self.state)

        if self.model.nq != 7 or self.model.nv != 7:
            raise RuntimeError(
                f"Expected arm-only nq=7,nv=7 but got nq={self.model.nq}, nv={self.model.nv}. "
                f"lock_fingers={lock_fingers} might have failed (check finger joint names)."
            )

        # EE frame selection
        if ee_frame_candidates is None:
            ee_frame_candidates = ["panda_hand", "panda_link8", "panda_grasptarget", "panda_tcp"]
        self.ee_frame = self._pick_frame(ee_frame_candidates)
        self.ee_fid = self.model.getFrameId(self.ee_frame)

        # nominal posture
        if q_nominal is None:
            q_nominal = np.array([0.0, -0.758, 0.0, -2.22, 0.0, 1.43, 0.0], dtype=float)
        self.q_nom = q_nominal.copy()

        # Desired EE orientation (world): z-axis down, x-axis aligned with +x
        self.R_des = self._make_vertical_down_rotation()
        
        # -------------------------
        # Joint limits (arm-only model)
        # -------------------------
        self.q_min = self.model.lowerPositionLimit[: self.model.nq].copy()
        self.q_max = self.model.upperPositionLimit[: self.model.nq].copy()

        # Build a "midpoint" reference to make bounds symmetric in the residual space
        self.q_mid = 0.5 * (self.q_min + self.q_max)

        # Apply safety margin (avoid getting too close)
        margin = float(self.cfg.joint_margin)
        self.q_lo_safe = self.q_min + margin
        self.q_hi_safe = self.q_max - margin

        # Sanity clamp in case margin is too large for a joint range
        self.q_lo_safe = np.minimum(self.q_lo_safe, self.q_mid - 1e-3)
        self.q_hi_safe = np.maximum(self.q_hi_safe, self.q_mid + 1e-3)

        # State reference for the barrier residual
        self.x_lim_ref = np.concatenate([self.q_mid, np.zeros(self.model.nv)])

        # Bounds are expressed in the *tangent space* at x_lim_ref.
        # For revolute joints, this is effectively (q - q_mid).
        q_lb = self.q_lo_safe - self.q_mid
        q_ub = self.q_hi_safe - self.q_mid

        v_lb = -np.ones(self.model.nv) * float(self.cfg.vmax_for_barrier)
        v_ub =  np.ones(self.model.nv) * float(self.cfg.vmax_for_barrier)

        self.x_lim_lb = np.concatenate([q_lb, v_lb])
        self.x_lim_ub = np.concatenate([q_ub, v_ub])

        # warm start storage
        self.xs = None
        self.us = None

    # -------------------------
    # Model helpers
    # -------------------------
    def _make_arm_only_model(self, model_full: pin.Model) -> pin.Model:
        finger_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
            "panda_joint_finger_1",
            "panda_joint_finger_2",
        ]

        lock_joint_ids = []
        for jn in finger_joint_names:
            if model_full.existJointName(jn):
                lock_joint_ids.append(model_full.getJointId(jn))

        if len(lock_joint_ids) == 0:
            for j in range(1, model_full.njoints):
                if "finger" in model_full.names[j]:
                    lock_joint_ids.append(j)

        if len(lock_joint_ids) == 0:
            raise RuntimeError(
                "Could not find finger joints to lock in Pinocchio model. "
                "Print model.names to see actual joint names."
            )

        q0_full = pin.neutral(model_full)
        model_reduced = pin.buildReducedModel(model_full, lock_joint_ids, q0_full)
        return model_reduced

    def _pick_frame(self, candidates: List[str]) -> str:
        for name in candidates:
            if self.model.existFrame(name):
                return name
        frame_names = [f.name for f in self.model.frames]
        raise ValueError(
            "Could not find any EE frame among candidates: "
            f"{candidates}\n\nAvailable frames include e.g.:\n"
            + "\n".join(frame_names[:80])
            + ("\n... (truncated)" if len(frame_names) > 80 else "")
        )

    def _make_vertical_down_rotation(self) -> np.ndarray:
        """
        World-frame desired rotation:
          - EE z-axis points DOWN: [0,0,-1]
          - EE x-axis aligned with world +x: [1,0,0]
        """
        z = np.array([0.0, 0.0, -1.0])
        x = np.array([1.0, 0.0,  0.0])
        y = np.cross(z, x)
        y = y / (np.linalg.norm(y) + 1e-12)
        x = np.cross(y, z)
        x = x / (np.linalg.norm(x) + 1e-12)
        R = np.column_stack([x, y, z])
        return R

    # -------------------------
    # MPC public API
    # -------------------------
    def compute_control(self, obs, t: float) -> np.ndarray:
        q = np.asarray(obs.q, dtype=float).copy()
        v = np.asarray(obs.dq, dtype=float).copy()
        x0 = np.concatenate([q, v])

        problem = self._build_problem(t0=t, x0=x0)

        solver = crocoddyl.SolverFDDP(problem)
        solver.setCallbacks([crocoddyl.CallbackVerbose()] if self.cfg.verbose else [])

        if self.xs is None or self.us is None:
            xs_init = [x0.copy() for _ in range(self.cfg.horizon + 1)]
            us_init = [np.zeros(self.model.nv) for _ in range(self.cfg.horizon)]
        else:
            xs_init = self.xs
            us_init = self.us

        solver.solve(xs_init, us_init, self.cfg.max_iters, False)

        self.xs = solver.xs
        self.us = solver.us

        tau = np.asarray(solver.us[0], dtype=float).copy()
        tau = np.clip(tau, -self.cfg.tau_limits, self.cfg.tau_limits)
        return tau

    # -------------------------
    # Build Crocoddyl OCP
    # -------------------------
    def _build_problem(self, t0: float, x0: np.ndarray) -> crocoddyl.ShootingProblem:
        running = []
        for k in range(self.cfg.horizon):
            tk = t0 + k * self.cfg.dt
            p_ref, _ = self.traj_fn(tk)

            # mirror x to match your Pinocchio-vs-MuJoCo convention
            p_ref = np.asarray(p_ref, dtype=float).copy()
            p_ref[0] *= -1.0

            dam = self._make_dam(p_ref=p_ref, terminal=False)
            running.append(crocoddyl.IntegratedActionModelEuler(dam, self.cfg.dt))

        pT, _ = self.traj_fn(t0 + self.cfg.horizon * self.cfg.dt)
        pT = np.asarray(pT, dtype=float).copy()
        pT[0] *= -1.0

        terminal_dam = self._make_dam(p_ref=pT, terminal=True)
        terminal = crocoddyl.IntegratedActionModelEuler(terminal_dam, 0.0)

        return crocoddyl.ShootingProblem(x0, running, terminal)

    def _make_dam(self, p_ref: np.ndarray, terminal: bool) -> crocoddyl.DifferentialActionModelAbstract:
        costs = crocoddyl.CostModelSum(self.state, self.model.nv)

        # --- EE position tracking ---
        r_pos = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_ref, self.model.nv)
        c_pos = crocoddyl.CostModelResidual(self.state, r_pos)
        costs.addCost("ee_pos", c_pos, self.cfg.w_ee_pos)

        # --- EE orientation tracking (rotation-only) ---
        # Use FramePlacement residual (6D), but activate only rotation components (last 3)
        # ref placement uses desired rotation + current p_ref (translation ignored by activation weights anyway)
        pref_se3 = pin.SE3(self.R_des, p_ref)

        r_place = crocoddyl.ResidualModelFramePlacement(self.state, self.ee_fid, pref_se3, self.model.nv)
        act_rot_only = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
        c_ori = crocoddyl.CostModelResidual(self.state, act_rot_only, r_place)
        costs.addCost("ee_ori", c_ori, self.cfg.w_ee_ori)

        # --- posture regularization ---
        x_ref = np.concatenate([self.q_nom, np.zeros(self.model.nv)])
        r_x = crocoddyl.ResidualModelState(self.state, x_ref, self.model.nv)
        c_x = crocoddyl.CostModelResidual(self.state, r_x)
        costs.addCost("posture", c_x, self.cfg.w_posture)

        # --- joint-limit avoidance (soft barrier on state) ---
        # residual is state difference around x_lim_ref, barrier enforces bounds
        r_lim = crocoddyl.ResidualModelState(self.state, self.x_lim_ref, self.model.nv)
        act_lim = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(self.x_lim_lb, self.x_lim_ub)
        )
        c_lim = crocoddyl.CostModelResidual(self.state, act_lim, r_lim)
        costs.addCost("joint_limits", c_lim, self.cfg.w_joint_limits)

        # --- torque regularization ---
        if not terminal:
            r_u = crocoddyl.ResidualModelControl(self.state, self.model.nv)
            c_u = crocoddyl.CostModelResidual(self.state, r_u)
            costs.addCost("tau_reg", c_u, self.cfg.w_tau)

        dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, costs)
        return dam
