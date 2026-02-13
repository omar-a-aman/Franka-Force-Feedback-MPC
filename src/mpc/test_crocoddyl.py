# src/mpc/test_crocoddyl.py
from pathlib import Path
import numpy as np
import mujoco.viewer

from src.sim.franka_sim import FrankaMujocoSim
from src.tasks.trajectories import make_circle_trajectory
from src.mpc.crocoddyl_classical import ClassicalCrocoddylMPC, ClassicalMPCConfig

SCENE = Path("assets/scenes/panda_table_scene.xml")

def main():
    sim = FrankaMujocoSim(SCENE, command_type="torque", n_substeps=5)
    obs = sim.reset("neutral")

    # table top z=0.32, ee sphere radius:
    z_table_top = 0.32
    r = float(sim.model.geom_size[sim.ee_geom_id][0])

    # keep it slightly ABOVE contact for now (baseline)
    center = np.array([-0.5, 0.0, z_table_top + r + 0.02], dtype=float)
    traj = make_circle_trajectory(center=center, radius=0.1, omega=0.8)

    cfg = ClassicalMPCConfig(
        horizon=15,
        dt=0.02,
        w_ee_pos=2.0e3,
        w_posture=2.0,
        w_tau=1.0e-3,
        max_iters=10,
        verbose=False,
    )
    mpc = ClassicalCrocoddylMPC(sim=sim, traj_fn=traj, config=cfg)

    t = 0.0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance = 1.6

        steps = int(100.0 / sim.dt)
        for k in range(steps):
            tau = mpc.compute_control(obs, t)
            obs = sim.step(tau)
            t += sim.dt
            viewer.sync()

            if k % 50 == 0:
                print(f"k={k:4d} t={t:6.3f} ee={obs.ee_pos} fn={obs.f_contact_normal:6.2f}")

if __name__ == "__main__":
    main()
