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
    tau_meas: np.ndarray          # (7,) measured-torque proxy (defaults to tau_total)
    tau_meas_filt: np.ndarray     # (7,) low-pass filtered measured-torque proxy
    tau_meas_act: np.ndarray      # (7,) actuation-side torque proxy (tau_cmd + tau_act)
    tau_meas_act_filt: np.ndarray # (7,) low-pass filtered actuation-side torque proxy
    tau_cmd: np.ndarray           # (7,) qfrc_applied command torque
    tau_act: np.ndarray           # (7,) MuJoCo actuator torque contribution
    tau_constraint: np.ndarray    # (7,) MuJoCo constraint/contact torque contribution
    tau_total: np.ndarray         # (7,) tau_cmd + tau_act + tau_constraint
    tau_bias: np.ndarray          # (7,) mujoco bias torque (gravity+coriolis)
    f_contact_world: np.ndarray   # (3,) total contact force ON ee geom, world
    f_contact_normal: float       # scalar contact normal along table normal (>=0)
    f_contact_normal_world_z: float  # scalar contact normal along world +z (>=0)
    f_contact_tangent: float      # scalar tangential contact force magnitude
    contact_count_ee: int         # number of active contacts involving ee geom
    contact_count_table: int      # number of active ee<->table_contact contacts
    table_normal_world: np.ndarray  # (3,) table surface normal in world frame

    ee_pos: Optional[np.ndarray] = None    # (3,)
    ee_quat: Optional[np.ndarray] = None   # (4,) (w,x,y,z)
    J_pos: Optional[np.ndarray] = None     # (3,7)
    J_rot: Optional[np.ndarray] = None     # (3,7)
    ee_vel: Optional[np.ndarray] = None    # (3,) world linear vel of ee_site


class FrankaMujocoSim:
    """
    Minimal MuJoCo wrapper for Franka Panda tasks.

    torque mode:
      - applies u (7,) directly to qfrc_applied for the 7 arm DoFs
      - exposes explicit torque channels in Observation:
          tau_cmd (qfrc_applied), tau_act (qfrc_actuator),
          tau_constraint (qfrc_constraint), tau_total,
          tau_meas_filt (total-proxy LPF), tau_meas_act_filt (actuation-proxy LPF)
      - returns tau_bias so controllers can do gravity compensation
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
        tau_meas_lpf_alpha: float = 0.2,
    ):
        self.scene_xml = str(scene_xml)
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        if command_type not in ("pos", "torque"):
            raise ValueError("command_type must be 'pos' or 'torque'")
        self.command_type = command_type
        self.n_substeps = int(n_substeps)
        self.tau_meas_lpf_alpha = float(np.clip(tau_meas_lpf_alpha, 0.0, 1.0))

        # 7 arm joints
        if arm_joint_names is None:
            arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_names = arm_joint_names

        self.jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        if any(j < 0 for j in self.jnt_ids):
            missing = [n for n, j in zip(self.joint_names, self.jnt_ids) if j < 0]
            raise ValueError(f"Missing joint(s) in XML: {missing}")

        self.qpos_adr = [int(self.model.jnt_qposadr[j]) for j in self.jnt_ids]
        self.dof_adr = [int(self.model.jnt_dofadr[j]) for j in self.jnt_ids]

        # actuators (only used in pos mode)
        if arm_actuator_names is None:
            arm_actuator_names = [f"actuator{i}" for i in range(1, 8)]
        self.actuator_names = arm_actuator_names

        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.actuator_names]
        if any(a < 0 for a in self.act_ids):
            missing = [n for n, a in zip(self.actuator_names, self.act_ids) if a < 0]
            raise ValueError(f"Missing actuator(s) in XML: {missing}")

        # EE site + collision geom
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise ValueError(f"Missing site '{ee_site_name}'")

        self.ee_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ee_collision_geom_name)
        if self.ee_geom_id < 0:
            raise ValueError(f"Missing geom '{ee_collision_geom_name}'")
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.table_contact_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_contact")
        if self.table_contact_geom_id < 0:
            self.table_contact_geom_id = self.table_geom_id

        self._tau_meas_filt = np.zeros(7, dtype=np.float64)
        self._tau_meas_act_filt = np.zeros(7, dtype=np.float64)

        # In torque mode we want qfrc_applied to be the only actuation source.
        # Panda's default actuators are position servos with nonzero bias terms;
        # leaving them enabled injects large hidden torques even when ctrl is zero.
        if self.command_type == "torque":
            self.data.ctrl[:] = 0.0
            self.model.actuator_gainprm[self.act_ids, :] = 0.0
            self.model.actuator_biasprm[self.act_ids, :] = 0.0

        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep * self.n_substeps)

    def reset(self, keyframe: str = "neutral") -> Observation:
        kf_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if kf_id < 0:
            raise ValueError(f"Keyframe '{keyframe}' not found.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kf_id)

        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        tau_cmd = self.data.qfrc_applied[self.dof_adr].copy()
        tau_act = self.data.qfrc_actuator[self.dof_adr].copy()
        tau_constraint = self.data.qfrc_constraint[self.dof_adr].copy()
        tau_meas_act = tau_cmd + tau_act
        self._tau_meas_filt = tau_cmd + tau_act + tau_constraint
        self._tau_meas_act_filt = tau_meas_act.copy()
        return self.get_observation(with_ee=True, with_jacobian=True)

    def step(self, u: np.ndarray) -> Observation:
        u = np.asarray(u, dtype=np.float64).reshape(-1).copy()

        if self.command_type == "pos":
            if u.shape == (7,):
                self.data.ctrl[:] = 0.0
                self.data.ctrl[self.act_ids] = u
            elif u.shape == (self.model.nu,):
                self.data.ctrl[:] = u
            else:
                raise ValueError(f"pos mode expects (7,) or (nu,), got {u.shape}")
            self.data.qfrc_applied[:] = 0.0

        else:  # torque mode
            if u.shape != (7,):
                raise ValueError(f"torque mode expects (7,), got {u.shape}")

            self.data.ctrl[:] = 0.0
            self.data.qfrc_applied[:] = 0.0
            for k, dof in enumerate(self.dof_adr):
                self.data.qfrc_applied[dof] = u[k]

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self.get_observation(with_ee=True, with_jacobian=True)

    def bias_torque(self) -> np.ndarray:
        """Return MuJoCo bias torques (gravity + Coriolis) for arm joints."""
        return self.data.qfrc_bias[self.dof_adr].copy()

    def get_observation(self, with_ee: bool = True, with_jacobian: bool = False) -> Observation:
        q = self.data.qpos[self.qpos_adr].copy()
        dq = self.data.qvel[self.dof_adr].copy()

        tau_bias = self.data.qfrc_bias[self.dof_adr].copy()
        tau_cmd = self.data.qfrc_applied[self.dof_adr].copy()
        tau_act = self.data.qfrc_actuator[self.dof_adr].copy()
        tau_constraint = self.data.qfrc_constraint[self.dof_adr].copy()
        tau_meas_act = tau_cmd + tau_act
        tau_total = tau_cmd + tau_act + tau_constraint
        # Measured effort proxy used by higher-level controllers; keep explicit and deterministic.
        tau_meas = tau_total.copy()
        a = self.tau_meas_lpf_alpha
        self._tau_meas_filt = (1.0 - a) * self._tau_meas_filt + a * tau_meas
        tau_meas_filt = self._tau_meas_filt.copy()
        self._tau_meas_act_filt = (1.0 - a) * self._tau_meas_act_filt + a * tau_meas_act
        tau_meas_act_filt = self._tau_meas_act_filt.copy()

        f_world, f_normal, f_normal_z, f_tan, n_contacts_ee, n_contacts_table = self._ee_contact_force_world()
        table_normal = self._table_normal_world()

        ee_pos = ee_quat = ee_vel = None
        J_pos = J_rot = None

        if with_ee:
            ee_pos = self.data.site_xpos[self.ee_site_id].copy()
            xmat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
            ee_quat = self._mat_to_quat_wxyz(xmat)

            # EE world linear velocity from jacobian * qvel
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            ee_vel = (jacp @ self.data.qvel).copy()

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
            tau_meas_filt=tau_meas_filt,
            tau_meas_act=tau_meas_act,
            tau_meas_act_filt=tau_meas_act_filt,
            tau_cmd=tau_cmd,
            tau_act=tau_act,
            tau_constraint=tau_constraint,
            tau_total=tau_total,
            tau_bias=tau_bias,
            f_contact_world=f_world,
            f_contact_normal=float(f_normal),
            f_contact_normal_world_z=float(f_normal_z),
            f_contact_tangent=float(f_tan),
            contact_count_ee=int(n_contacts_ee),
            contact_count_table=int(n_contacts_table),
            table_normal_world=table_normal,
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            ee_vel=ee_vel,
            J_pos=J_pos,
            J_rot=J_rot,
        )

    def _table_normal_world(self) -> np.ndarray:
        table_geom_id = int(self.table_contact_geom_id)
        if table_geom_id < 0:
            table_geom_id = int(self.table_geom_id)
        if table_geom_id < 0:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        R_table = self.data.geom_xmat[table_geom_id].reshape(3, 3)
        n_table = np.asarray(R_table[:, 2], dtype=np.float64)
        n_table /= (np.linalg.norm(n_table) + 1.0e-12)
        return n_table

    def _ee_contact_force_world(self) -> Tuple[np.ndarray, float, float, float, int, int]:
        """
        Returns force ON ee_collision geom in world coordinates.
        Correct MuJoCo conventions:
          mj_contactForce -> cf[0]=normal, cf[1]=friction1, cf[2]=friction2 (in contact frame)
          contact.frame: frame[0:3]=normal, frame[3:6]=tangent1, frame[6:9]=tangent2
        """
        f_world_total = np.zeros(3, dtype=np.float64)
        f_world_table = np.zeros(3, dtype=np.float64)
        fn_table_total = 0.0
        ft_table_total = 0.0
        n_contacts_ee = 0
        n_contacts_table = 0

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 != self.ee_geom_id) and (c.geom2 != self.ee_geom_id):
                continue
            n_contacts_ee += 1

            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, cf)

            fr = np.asarray(c.frame, dtype=np.float64)
            n_world = fr[0:3]
            t1_world = fr[3:6]
            t2_world = fr[6:9]

            fn = float(cf[0])   # <-- FIXED
            ft1 = float(cf[1])
            ft2 = float(cf[2])

            f_w_on_geom2 = n_world * fn + t1_world * ft1 + t2_world * ft2

            # convert to force ON ee geom
            if c.geom2 == self.ee_geom_id:
                f_w_on_ee = f_w_on_geom2
            else:
                f_w_on_ee = -f_w_on_geom2

            f_world_total += f_w_on_ee
            other_geom = int(c.geom1) if int(c.geom2) == self.ee_geom_id else int(c.geom2)
            if int(self.table_contact_geom_id) >= 0 and other_geom == int(self.table_contact_geom_id):
                n_contacts_table += 1
                # Contact-normal signal for control/metrics: scalar compression magnitude.
                fn_table_total += abs(float(cf[0]))
                ft_table_total += float(np.linalg.norm(cf[1:3]))
                f_world_table += f_w_on_ee

        if n_contacts_table > 0:
            fn_world_z = max(float(f_world_table[2]), 0.0)
            return (
                f_world_table,
                float(fn_table_total),
                float(fn_world_z),
                float(ft_table_total),
                int(n_contacts_ee),
                int(n_contacts_table),
            )

        fn_world_z = max(float(f_world_total[2]), 0.0)
        return (
            f_world_total,
            0.0,
            float(fn_world_z),
            0.0,
            int(n_contacts_ee),
            0,
        )

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
