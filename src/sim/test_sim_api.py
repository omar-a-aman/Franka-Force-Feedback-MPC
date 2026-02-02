import numpy as np
from pathlib import Path
from franka_sim import FrankaMujocoSim

SCENE = Path("assets/scenes/panda_table_scene.xml")

sim = FrankaMujocoSim(SCENE, command_type="pos", n_substeps=5)
obs = sim.reset("neutral")
print("q:", obs.q)
print("f_contact_world:", obs.f_contact_world, "f_n:", obs.f_contact_normal)

# hold position targets at current q (should remain steady)
for k in range(200):
    obs = sim.step(obs.q)  # position targets
    if k % 50 == 0:
        print(k, "ee z:", obs.ee_pos[2], "fn:", obs.f_contact_normal)
