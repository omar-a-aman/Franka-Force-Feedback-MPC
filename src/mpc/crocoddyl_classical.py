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
    w_posture: float = 2.0
    w_tau: float = 1.0e-3

    # Limits
    tau_limits: np.ndarray = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

    # Solver
    max_iters: int = 15
    verbose: bool = False


class ClassicalCrocoddylMPC:
    """
    Classical MPC baseline using Crocoddyl FDDP:
      x = (q, v)  with nq=7, nv=7
      u = tau     with nu=7

    Costs:
      - EE position tracking
      - posture regularization
      - torque regularization

    NOTE:
      This is a baseline WITHOUT explicit contact constraints/forces yet.
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
        self.actuation = crocoddyl.ActuationModelFull(self.state)  # tau on all nv

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

        # warm start storage
        self.xs = None
        self.us = None

    # -------------------------
    # Model helpers
    # -------------------------
    def _make_arm_only_model(self, model_full: pin.Model) -> pin.Model:
        """
        Reduce example_robot_data panda (9DoF: 7 arm + 2 fingers)
        into arm-only (7DoF) by locking finger joints.
        """
        finger_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
            # sometimes named like this in some models:
            "panda_joint_finger_1",
            "panda_joint_finger_2",
        ]

        lock_joint_ids = []
        for jn in finger_joint_names:
            if model_full.existJointName(jn):
                lock_joint_ids.append(model_full.getJointId(jn))

        # fallback: lock anything that contains "finger"
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

    # -------------------------
    # MPC public API
    # -------------------------
    def compute_control(self, obs, t: float) -> np.ndarray:
        q = np.asarray(obs.q, dtype=float).copy()
        v = np.asarray(obs.dq, dtype=float).copy()
        x0 = np.concatenate([q, v])

        problem = self._build_problem(t0=t, x0=x0)

        solver = crocoddyl.SolverFDDP(problem)
        if self.cfg.verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])
        else:
            solver.setCallbacks([])

        # warm start
        if self.xs is None or self.us is None:
            xs_init = [x0.copy() for _ in range(self.cfg.horizon + 1)]
            us_init = [np.zeros(self.model.nv) for _ in range(self.cfg.horizon)]
        else:
            xs_init = self.xs
            us_init = self.us

        solver.solve(xs_init, us_init, self.cfg.max_iters, False)

        # store warm start (shift will happen naturally because next x0 changes)
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
            dam = self._make_dam(p_ref=p_ref, terminal=False)
            iam = crocoddyl.IntegratedActionModelEuler(dam, self.cfg.dt)
            running.append(iam)

        pT, _ = self.traj_fn(t0 + self.cfg.horizon * self.cfg.dt)
        terminal_dam = self._make_dam(p_ref=pT, terminal=True)
        terminal = crocoddyl.IntegratedActionModelEuler(terminal_dam, 0.0)

        return crocoddyl.ShootingProblem(x0, running, terminal)

    def _make_dam(self, p_ref: np.ndarray, terminal: bool) -> crocoddyl.DifferentialActionModelAbstract:
        # --- Costs container FIRST (required by Crocoddyl constructor) ---
        costs = crocoddyl.CostModelSum(self.state, self.model.nv)

        # EE translation tracking: p(frame) - p_ref
        r_ee = crocoddyl.ResidualModelFrameTranslation(self.state, self.ee_fid, p_ref, self.model.nv)
        c_ee = crocoddyl.CostModelResidual(self.state, r_ee)
        costs.addCost("ee_pos", c_ee, self.cfg.w_ee_pos)

        # posture regularization: (q,v) -> (q_nom, 0)
        x_ref = np.concatenate([self.q_nom, np.zeros(self.model.nv)])
        r_x = crocoddyl.ResidualModelState(self.state, x_ref, self.model.nv)
        c_x = crocoddyl.CostModelResidual(self.state, r_x)
        costs.addCost("posture", c_x, self.cfg.w_posture)

        # torque regularization
        if not terminal:
            r_u = crocoddyl.ResidualModelControl(self.state, self.model.nv)
            c_u = crocoddyl.CostModelResidual(self.state, r_u)
            costs.addCost("tau_reg", c_u, self.cfg.w_tau)

        # --- Correct constructor call (THIS FIXES YOUR ERROR) ---
        dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, costs)
        return dam
