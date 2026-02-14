#!/usr/bin/env python3
"""
Quick debug script to check if MPC is generating nonzero controls.
"""
from pathlib import Path
import numpy as np
import mujoco

from src.sim.franka_sim import FrankaMujocoSim
from src.tasks.trajectories import make_approach_then_circle
from src.mpc.crocoddyl_classical import ClassicalCrocoddylMPC, ClassicalMPCConfig

SCENE = Path("assets/scenes/panda_table_scene.xml")

def main():
    # Initialize simulator
    sim = FrankaMujocoSim(SCENE, command_type="torque", n_substeps=5)
    obs = sim.reset("neutral")
    print(f"Initial EE position: {obs.ee_pos}")

    # Get table geometry
    table_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    z_table = float(sim.model.geom_pos[table_geom_id][2] + sim.model.geom_size[table_geom_id][2])
    r_tool = float(sim.model.geom_size[sim.ee_geom_id][0])
    z_contact = z_table + r_tool - 0.0005

    # Trajectory
    center = np.array([0.4, 0.0, z_contact], dtype=float)
    traj = make_approach_then_circle(
        center=center,
        radius=0.05,
        omega=0.5,
        z_contact=z_contact,
        t_approach=2.0,
        ee_start=obs.ee_pos.copy(),
    )

    # MPC with RADICAL weights
    cfg = ClassicalMPCConfig(
        horizon=15,
        dt=sim.dt,
        z_contact=z_contact,
        z_press=0.0010,
        w_ee_pos=1.0e2,  # Will be multiplied by 10x in free-space
        w_ee_ori=1.0e1,
        w_posture=0.01,  # VERY weak
        w_v=0.1,
        w_tau=1.0e-1,
        w_tau_smooth=1.0,
        w_tangent_pos=1.0e2,
        w_tangent_vel=2.0e2,
        w_plane_z=2.0e3,
        w_vz=1.0e3,
        kp_posture=100.0,
        kd_posture=15.0,
        tau_trust_inf=25.0,
        max_iters=20,
    )
    mpc = ClassicalCrocoddylMPC(sim=sim, traj_fn=traj, config=cfg)

    print("\n" + "=" * 80)
    print("TESTING MPC CONTROL OUTPUT")
    print("=" * 80)
    print(f"\nCost weights:")
    print(f"  w_posture: {cfg.w_posture}")
    print(f"  w_v: {cfg.w_v}")
    print(f"  w_ee_pos (free-space): {cfg.w_ee_pos * 10.0}")
    print(f"  kp_posture: {cfg.kp_posture}")
    print()

    # Run a few steps
    t = 0.0
    for step in range(50):
        tau = mpc.compute_control(obs, t)

        print(f"\nStep {step:3d}, t={t:6.3f}s")
        print(f"  tau_mpc: {tau}")
        print(f"  |tau|: {np.linalg.norm(tau):.6f}")
        print(f"  EE: {obs.ee_pos}")

        # Check if controls are ALL zeros
        if np.allclose(tau, 0.0, atol=1e-6):
            print("  ⚠️  WARNING: All controls are zero!")
        else:
            print("  ✓ Controls are nonzero")

        # Step simulation
        obs = sim.step(tau)
        t += sim.dt

        if step == 4:
            print("\n... (continuing, will print last 5 steps) ...")
            first_step_idx = 45

        if step >= 45:
            continue

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
