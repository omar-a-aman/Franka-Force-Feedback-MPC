#!/usr/bin/env python3
"""
Headless test to diagnose MPC control generation - no viewer needed.
"""
from pathlib import Path
import sys
import os

# Set headless mode before importing mujoco
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'

import numpy as np

try:
    import mujoco
    from src.sim.franka_sim import FrankaMujocoSim
    from src.tasks.trajectories import make_approach_then_circle
    from src.mpc.crocoddyl_classical import ClassicalCrocoddylMPC, ClassicalMPCConfig
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

SCENE = Path("assets/scenes/panda_table_scene.xml")

def main():
    print("=" * 80)
    print("HEADLESS MPC CONTROL DIAGNOSTIC")
    print("=" * 80)
    
    try:
        # Initialize simulator
        sim = FrankaMujocoSim(SCENE, command_type="torque", n_substeps=5)
        obs = sim.reset("neutral")
        print(f"✓ Simulator initialized")
        print(f"  Initial EE: {obs.ee_pos}")

        # Get table
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
        print(f"✓ Trajectory created")

        # MPC config
        cfg = ClassicalMPCConfig(
            horizon=15,
            dt=sim.dt,
            z_contact=z_contact,
            z_press=0.0010,
            w_ee_pos=1.0e2,
            w_ee_ori=1.0e1,
            w_posture=0.01,  # Very weak
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
        print(f"✓ MPC config created")
        print(f"  Cost weights: w_posture={cfg.w_posture}, w_v={cfg.w_v}, w_ee_pos={cfg.w_ee_pos}")

        # Initialize MPC
        mpc = ClassicalCrocoddylMPC(sim=sim, traj_fn=traj, config=cfg)
        print(f"✓ MPC initialized\n")

        # Run a few steps
        t = 0.0
        print(f"Running first 10 steps to check controls...")
        print("-" * 80)
        
        for step in range(10):
            tau = mpc.compute_control(obs, t)
            
            is_zero = np.allclose(tau, 0.0, atol=1e-6)
            zero_marker = "⚠️ ZERO!" if is_zero else "✓ NONZERO"
            
            print(f"\nStep {step}: t={t:.3f}s")
            print(f"  tau: {tau}")
            print(f"  |tau|: {np.linalg.norm(tau):.6f} {zero_marker}")
            print(f"  EE: {obs.ee_pos}")
            
            if step == 0 and is_zero:
                print("\n⚠️  CRITICAL: First control is ZERO! MPC problem likely has no costs.")
            
            # Step simulation
            obs = sim.step(tau)
            t += sim.dt

        print("\n" + "=" * 80)
        print("Test complete!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
