from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.mpc.crocoddyl_classical import ClassicalCrocoddylMPC, ClassicalMPCConfig
from src.sim.franka_sim import FrankaMujocoSim
from src.tasks.trajectories import make_approach_then_circle

SCENE = Path("assets/scenes/panda_table_scene.xml")


def _table_geometry_world(sim: FrankaMujocoSim):
    table_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if table_geom_id < 0:
        raise RuntimeError("table_top geom not found in scene")

    # model.geom_pos is local to parent body; use data.geom_xpos for world coordinates.
    table_center = sim.data.geom_xpos[table_geom_id].copy()
    table_size = sim.model.geom_size[table_geom_id].copy()  # half-sizes
    z_table_top = float(table_center[2] + table_size[2])

    return table_geom_id, table_center, table_size, z_table_top


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(np.minimum(np.maximum(value, lo), hi))


def main(no_viewer: bool = False, total_time: float = 40.0):
    print("=" * 80)
    print("Classical Panda MPC (Paper-Faithful)")
    print("=" * 80)

    sim = FrankaMujocoSim(SCENE, command_type="torque", n_substeps=5)
    obs = sim.reset("neutral")
    print(f"Simulation initialized (dt={sim.dt:.4f}s)")
    print(f"Initial EE position: {obs.ee_pos}")

    _, table_center, table_half, z_table_top = _table_geometry_world(sim)
    r_tool = float(sim.model.geom_size[sim.ee_geom_id][0])
    z_contact = z_table_top + r_tool - 0.0010  # slight penetration target
    z_pre = z_contact + 0.06

    radius = 0.04
    omega = 0.6
    margin = radius + 0.03

    x_min = float(table_center[0] - table_half[0] + margin)
    x_max = float(table_center[0] + table_half[0] - margin)
    y_min = float(table_center[1] - table_half[1] + margin)
    y_max = float(table_center[1] + table_half[1] - margin)

    center_x = _clamp(obs.ee_pos[0], x_min, x_max)
    center_y = _clamp(obs.ee_pos[1], y_min, y_max)
    center = np.array([center_x, center_y, z_contact], dtype=float)

    print(f"Table top (world): center={table_center}, half-size={table_half}, z_top={z_table_top:.4f}m")
    print(f"Contact target: z_contact={z_contact:.4f}m (tool radius={r_tool:.4f}m)")
    print(f"Circle center (clamped on table): {center}, radius={radius:.3f}m")

    traj = make_approach_then_circle(
        center=center,
        radius=radius,
        omega=omega,
        z_pre=z_pre,
        z_contact=z_contact,
        t_approach=1.2,
        ee_start=obs.ee_pos.copy(),
        t_pre=1.2,
    )
    print("Trajectory generated (approach + circle)")

    cfg = ClassicalMPCConfig(
        horizon=20,
        dt=sim.dt,
        z_contact=z_contact,
        z_press=0.0070,
        w_ee_pos=2.5e2,
        w_ee_ori=1.0e1,
        w_posture=5.0e-1,
        w_v=2.5e-1,
        w_tau=1.0e-3,
        w_tau_smooth=5.0e-2,
        w_tangent_pos=2.0e2,
        w_tangent_vel=1.0e2,
        w_plane_z=6.0e3,
        w_vz=6.0e2,
        fn_contact_on=2.0,
        fn_contact_off=0.5,
        z_contact_band=0.01,
        max_iters=20,
        debug_every=10,
    )
    print("MPC config created")

    mpc = ClassicalCrocoddylMPC(sim=sim, traj_fn=traj, config=cfg)
    print("MPC initialized")
    print()

    t = 0.0
    steps = int(total_time / sim.dt)

    log_data = {
        "t": [],
        "ee_pos_x": [],
        "ee_pos_y": [],
        "ee_pos_z": [],
        "ee_vel_x": [],
        "ee_vel_y": [],
        "ee_vel_z": [],
        "f_contact_normal": [],
    }

    print(f"Running simulation for {total_time:.1f}s ({steps} steps)...")
    print()

    def _step_once(k: int, t_now: float, obs_now):
        tau = mpc.compute_control(obs_now, t_now)
        obs_next = sim.step(tau)
        t_next = t_now + sim.dt

        log_data["t"].append(t_next)
        log_data["ee_pos_x"].append(float(obs_next.ee_pos[0]))
        log_data["ee_pos_y"].append(float(obs_next.ee_pos[1]))
        log_data["ee_pos_z"].append(float(obs_next.ee_pos[2]))
        log_data["ee_vel_x"].append(float(obs_next.ee_vel[0]))
        log_data["ee_vel_y"].append(float(obs_next.ee_vel[1]))
        log_data["ee_vel_z"].append(float(obs_next.ee_vel[2]))
        log_data["f_contact_normal"].append(float(obs_next.f_contact_normal))

        if k % 100 == 0:
            p_ref, _, surf = traj(t_next)
            err = np.linalg.norm(obs_next.ee_pos - p_ref)
            print(
                f"k={k:4d} t={t_next:6.3f}s | "
                f"EE=[{obs_next.ee_pos[0]:.3f}, {obs_next.ee_pos[1]:.3f}, {obs_next.ee_pos[2]:.4f}] | "
                f"|v|={np.linalg.norm(obs_next.ee_vel):.3f}m/s | "
                f"Fn={obs_next.f_contact_normal:.2f}N | "
                f"|p-p_ref|={err:.4f}m | surf={int(surf)}"
            )

        return t_next, obs_next

    if no_viewer:
        for k in range(steps):
            t, obs = _step_once(k, t, obs)
    else:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -30

            for k in range(steps):
                t, obs = _step_once(k, t, obs)
                try:
                    viewer.sync()
                except Exception:
                    pass

    print()
    print("=" * 80)
    print("Simulation complete!")
    print("=" * 80)

    ee_pos_z = np.array(log_data["ee_pos_z"])
    f_contact = np.array(log_data["f_contact_normal"])

    print("\nSummary statistics:")
    print(f"  EE Z: min={ee_pos_z.min():.4f}, max={ee_pos_z.max():.4f}, mean={ee_pos_z.mean():.4f}")
    print(f"  Contact force: min={f_contact.min():.2f}N, max={f_contact.max():.2f}N, mean={f_contact.mean():.2f}N")
    print(f"  Time in contact (Fn>0.5N): {(f_contact > 0.5).sum() / len(f_contact) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-viewer", action="store_true", help="Run without MuJoCo viewer.")
    parser.add_argument("--time", type=float, default=12.0, help="Total simulation time [s].")
    args = parser.parse_args()
    main(no_viewer=args.no_viewer, total_time=args.time)
