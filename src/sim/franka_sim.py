from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import mujoco


@dataclass
class Observation:
    q: np.ndarray                 # (7,)
    dq: np.ndarray                # (7,)
    tau_meas: np.ndarray          # (7,) proxy (actuator generalized forces)
    f_contact_world: np.ndarray   # (3,) total contact force on ee_collision, world frame (force ON EE)
    f_contact_normal: float       # scalar normal component (>=0) (force ON EE)
    ee_pos: Optional[np.ndarray] = None   # (3,)
    ee_quat: Optional[np.ndarray] = None  # (4,) (w, x, y, z)
    J_pos: Optional[np.ndarray] = None    # (3,7)
    J_rot: Optional[np.ndarray] = None    # (3,7)


class FrankaMujocoSim:
    """
    Minimal MuJoCo wrapper for Franka Panda tasks.

    API:
      - reset(keyframe="neutral") -> Observation
      - step(u) -> Observation
      - get_observation(with_ee=True, with_jacobian=False) -> Observation

    command_type:
      - "pos": uses data.ctrl as position targets
      - "torque": applies joint torques via data.qfrc_applied on the 7 arm DoFs
    """

    def __init__(
        self,
        scene_xml: str | Path,
        command_type: str = "pos",
        n_substeps: int = 1,
        ee_site_name: str = "ee_site",
        ee_collision_geom_name: str = "ee_collision",
        arm_joint_names: Optional[list[str]] = None,
        arm_actuator_names: Optional[list[str]] = None,
    ):
        self.scene_xml = str(scene_xml)
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        if command_type not in ("pos", "torque"):
            raise ValueError("command_type must be 'pos' or 'torque'")
        self.command_type = command_type
        self.n_substeps = int(n_substeps)

        # ---- Resolve joint DoFs for the 7 arm joints ----
        if arm_joint_names is None:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_names = arm_joint_names

        self.jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        if any(j < 0 for j in self.jnt_ids):
            missing = [n for n, j in zip(self.joint_names, self.jnt_ids) if j < 0]
            raise ValueError(f"Missing joint(s) in XML: {missing}")

        self.qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self.jnt_ids]
        self.dof_adr = [int(self.model.jnt_dofadr[j]) for j in self.jnt_ids]

        # ---- Resolve actuators for the 7 arm joints ----
        if arm_actuator_names is None:
            arm_actuator_names = [f"actuator{i}" for i in range(1, 8)]
        self.actuator_names = arm_actuator_names

        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.actuator_names]
        if any(a < 0 for a in self.act_ids):
            missing = [n for n, a in zip(self.actuator_names, self.act_ids) if a < 0]
            raise ValueError(f"Missing actuator(s) in XML: {missing}")

        # ---- EE site + collision geom ----
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"Missing site '{ee_site_name}' (needed for EE pose/Jacobian)")

        self.ee_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ee_collision_geom_name)
        if self.ee_geom_id < 0:
            raise ValueError(f"Missing geom '{ee_collision_geom_name}' (needed for EE contact force)")

        # If torque mode: disable actuator contribution (so ctrl doesn't create forces)
        if self.command_type == "torque":
            self.model.actuator_gainprm[:, :] = 0.0
            self.model.actuator_biasprm[:, :] = 0.0

        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.n_substeps)

    def reset(self, keyframe: str = "neutral") -> Observation:
        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kf_id < 0:
            raise ValueError(f"Keyframe '{keyframe}' not found in model keyframes.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation(with_ee=True, with_jacobian=True)

    def step(self, u: np.ndarray) -> Observation:
        u = np.asarray(u, dtype=np.float64).copy()

        if self.command_type == "pos":
            if u.shape == (7,):
                self.data.ctrl[:] = 0.0
                self.data.ctrl[self.act_ids] = u
            elif u.shape == (self.model.nu,):
                self.data.ctrl[:] = u
            else:
                raise ValueError(f"For command_type='pos', u must be shape (7,) or (nu,), got {u.shape}")

            self.data.qfrc_applied[:] = 0.0

        else:  # "torque"
            if u.shape != (7,):
                raise ValueError(f"For command_type='torque', u must be shape (7,), got {u.shape}")

            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.act_ids] = self.data.qpos[self.qpos_adr]

            self.data.qfrc_applied[:] = 0.0
            for k, dof in enumerate(self.dof_adr):
                self.data.qfrc_applied[dof] = u[k]

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation(with_ee=True, with_jacobian=True)

    def get_observation(self, with_ee: bool = True, with_jacobian: bool = False) -> Observation:
        q = self.data.qpos[self.qpos_adr].copy()
        dq = self.data.qvel[self.dof_adr].copy()
        tau_meas = self.data.qfrc_actuator[self.dof_adr].copy()

        f_world, f_normal = self._ee_contact_force_world()

        ee_pos = None
        ee_quat = None
        J_pos = None
        J_rot = None

        if with_ee:
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            xmat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            ee_quat = self._mat_to_quat_wxyz(xmat)

        if with_jacobian:
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            J_pos = jacp[:, self.dof_adr].copy()
            J_rot = jacr[:, self.dof_adr].copy()

        return Observation(
            q=q,
            dq=dq,
            tau_meas=tau_meas,
            f_contact_world=f_world,
            f_contact_normal=float(f_normal),
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            J_pos=J_pos,
            J_rot=J_rot,
        )

    def bias_torque(self) -> np.ndarray:
        return self.data.qfrc_bias[self.dof_adr].copy()

    def _ee_contact_force_world(self) -> Tuple[np.ndarray, float]:
        """
        Returns:
        f_world_total: (3,) total contact force ON ee_collision geom in world coords
        f_normal_total: scalar normal force magnitude (>=0) accumulated over contacts

        Notes:
        - MuJoCo contact.frame normal points from geom1 -> geom2.
        - mj_contactForce gives components in the contact frame; we reconstruct a world force vector.
        - The returned vector from reconstruction is treated as the force ON geom2.
            Therefore:
            - if EE is geom2 -> force_on_ee = +f_w
            - if EE is geom1 -> force_on_ee = -f_w
        - For the scalar normal force, your controller wants a magnitude: use abs(fn).
        """
        f_world_total = np.zeros(3, dtype=np.float64)
        f_normal_total = 0.0

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 != self.ee_geom_id) and (c.geom2 != self.ee_geom_id):
                continue

            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, cf)

            fr = np.asarray(c.frame, dtype=np.float64)
            n_world  = fr[0:3]
            t1_world = fr[3:6]
            t2_world = fr[6:9]

            ft1 = float(cf[0])
            ft2 = float(cf[1])
            fn  = float(cf[2])  # normal component along n_world (geom1->geom2)

            # Reconstruct world force (interpreted as force ON geom2)
            f_w_on_geom2 = t1_world * ft1 + t2_world * ft2 + n_world * fn

            # Flip if EE is geom1
            if c.geom2 == self.ee_geom_id:
                f_w_on_ee = f_w_on_geom2
            else:
                f_w_on_ee = -f_w_on_geom2

            f_world_total += f_w_on_ee
            f_normal_total += abs(fn)

        return f_world_total, f_normal_total


    @staticmethod
    def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S

        q = np.array([w, x, y, z], dtype=np.float64)
        return q / (np.linalg.norm(q) + 1e-12)
