import numpy as np
from src.tasks.trajectories import make_circle_trajectory

traj = make_circle_trajectory(center=np.array([-0.5, 0.0, 0.35]), radius=0.1, omega=1.0)

for t in [0.0, 0.5, 1.0, 2.0]:
    p, v = traj(t)
    print(t, "p=", p, "v=", v)
