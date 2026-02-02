from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import mujoco


@dataclass
class Observation:
    q: np.ndarray              # (7,)
    dq: np.ndarray             # (7,)
    tau_meas: np.ndarray       # (7,)  proxy
    f_contact_world: np.ndarray  # (3,) total contact force on ee_collision, world frame
    f_contact_normal: float      # scalar normal component (>=0 typically)
    ee_pos: Optional[np.ndarray] = None   # (3,)
    ee_quat: Optional[np.ndarray] = None  # (4,) (w, x, y, z) in MuJoCo convention
    J_pos: Optional[np.ndarray] = None    # (3,7)
    J_rot: Optional[np.ndarray] = None    # (3,7)


class FrankaMujocoSim:
    """
    Minimal MuJoCo wrapper for Franka Panda tasks.
    Exposes only:
      - reset(keyframe="neutral")
      - step(u)
      - get_observation()

    Supports command_type:
      - "pos": uses data.ctrl as position targets (fits mujoco-menagerie 'general' actuators)
      - "torque": applies joint torques via data.qfrc_applied on the 7 arm DoFs
    """

    def __init__(
        self,
        scene_xml: str | Path,
        command_type: str = "pos",
        n_substeps: int = 1,
        ee_site_name: str = "ee_site",
        ee_collision_geom_name: str = "ee_collision",
    ):
        self.scene_xml = str(scene_xml)
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        assert command_type in ("pos", "torque"), "command_type must be 'pos' or 'torque'"
        self.command_type = command_type
        self.n_substeps = int(n_substeps)

        # ---- Resolve joint DoFs (qpos/qvel indices) for the 7 arm joints ----
        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        if any(j < 0 for j in self.jnt_ids):
            missing = [n for n, j in zip(self.joint_names, self.jnt_ids) if j < 0]
            raise ValueError(f"Missing joint(s) in XML: {missing}")

        # qpos and qvel addresses
        self.qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self.jnt_ids]
        self.dof_adr = [int(self.model.jnt_dofadr[j]) for j in self.jnt_ids]  # hinge -> 1 DoF each

        # ---- EE site + collision geom ----
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"Missing site '{ee_site_name}' (needed for EE pose/Jacobian)")

        self.ee_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ee_collision_geom_name)
        if self.ee_geom_id < 0:
            raise ValueError(f"Missing geom '{ee_collision_geom_name}' (needed for EE contact force)")

        # ---- Sanity for ctrl dimension ----
        # For "pos", you'll provide u shape (nu,) or at least (7,) and we map to first 7 actuators by default.
        # For "torque", u shape must be (7,)
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------
    # Public API
    # ---------------------------

    def reset(self, keyframe: str = "neutral") -> Observation:
        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kf_id < 0:
            raise ValueError(f"Keyframe '{keyframe}' not found in model keyframes.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation(with_ee=True, with_jacobian=True)

    def step(self, u: np.ndarray) -> Observation:
        """
        Step the simulation with either position targets ("pos") or direct torques ("torque").
        """
        u = np.asarray(u, dtype=np.float64).copy()

        if self.command_type == "pos":
            # Your current actuators are "general" with gain/bias => behaves like a position servo.
            # Expect u as either (7,) or (nu,)
            if u.shape == (7,):
                self.data.ctrl[:] = 0.0
                self.data.ctrl[:7] = u
            elif u.shape == (self.model.nu,):
                self.data.ctrl[:] = u
            else:
                raise ValueError(f"For command_type='pos', u must be shape (7,) or (nu,), got {u.shape}")

        else:  # "torque"
            # Apply joint torques directly to the 7 DoFs.
            if u.shape != (7,):
                raise ValueError(f"For command_type='torque', u must be shape (7,), got {u.shape}")

            # Clear previous applied forces then apply new ones
            self.data.qfrc_applied[:] = 0.0
            for k, dof in enumerate(self.dof_adr):
                self.data.qfrc_applied[dof] = u[k]

            # Optional: also clear ctrl so actuators don't fight you
            self.data.ctrl[:] = 0.0

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation(with_ee=True, with_jacobian=True)

    def get_observation(self, with_ee: bool = True, with_jacobian: bool = False) -> Observation:
        q = self.data.qpos[self.qpos_adr].copy()
        dq = self.data.qvel[self.dof_adr].copy()

        # ---- torque proxy ----
        # data.qfrc_actuator: generalized forces from actuators at DoF level (best "measured torque proxy")
        tau_meas = self.data.qfrc_actuator[self.dof_adr].copy()

        # ---- EE contact force ----
        f_world, f_normal = self._ee_contact_force_world()

        ee_pos = None
        ee_quat = None
        J_pos = None
        J_rot = None

        if with_ee:
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()

            # MuJoCo gives site orientation as 3x3; convert to quaternion (w,x,y,z)
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

    # ---------------------------
    # Internals
    # ---------------------------

    def _ee_contact_force_world(self) -> Tuple[np.ndarray, float]:
        """
        Accumulate contact forces for any contact involving ee_collision geom.
        Returns:
          f_world: (3,) total world force applied at EE collision geom
          f_normal: scalar normal component (sum of normal forces)
        Notes:
          mj_contactForce gives forces in CONTACT frame.
          MuJoCo contact frame axis convention: normal is along +x in contact frame.
          contact.frame stores the contact frame axes in world coordinates.
        """
        f_world_total = np.zeros(3, dtype=np.float64)
        f_normal_total = 0.0

        # Iterate contacts and extract those involving ee geom
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 != self.ee_geom_id and c.geom2 != self.ee_geom_id:
                continue

            # local force (contact frame): [normal, friction1, friction2, ...torques...]
            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, cf)

            f_c = cf[:3]  # (fn, ft1, ft2) in contact frame
            fn = float(cf[0])

            # contact.frame: 3x3 matrix whose rows are the contact frame axes in world coordinates:
            # row0 = normal axis (world), row1 = tangent1 (world), row2 = tangent2 (world)
            R = np.array(c.frame, dtype=np.float64).reshape(3, 3)
            f_w = R.T @ f_c  # map contact coords -> world

            f_world_total += f_w
            f_normal_total += fn

        return f_world_total, f_normal_total

    @staticmethod
    def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion (w,x,y,z).
        Robust enough for logging/costs.
        """
        # Standard matrix-to-quat conversion
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            # find dominant diagonal
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
        # normalize
        return q / (np.linalg.norm(q) + 1e-12)
