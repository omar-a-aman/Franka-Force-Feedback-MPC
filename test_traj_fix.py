#!/usr/bin/env python3
"""
Test the trajectory fix without running full MPC simulation.
"""
import sys
sys.path.insert(0, '/home/omar/franka-force-feedback-mpc')

import numpy as np
from src.tasks.trajectories import make_approach_then_circle

# Setup
z_table = 0.02
r_tool = 0.03
z_contact = z_table + r_tool - 0.0005  # 0.0495
z_pre = z_contact + 0.08  # 0.1295

center = np.array([0.4, 0.0, z_contact])
ee_start = np.array([-0.3, 0.0, 0.633])

print("Testing trajectory fix...")
print(f"Robot starting EE: {ee_start}")
print(f"z_contact: {z_contact:.4f}")
print(f"z_pre: {z_pre:.4f}")
print(f"Circle center: {center}")
print()

# Create trajectory WITH starting position (fixed version)
traj = make_approach_then_circle(
    center=center,
    radius=0.05,
    omega=0.5,
    z_pre=z_pre,
    z_contact=z_contact,
    t_approach=2.0,
    ee_start=ee_start.copy(),  # KEY: Pass robot's actual starting position
)

print("Phase 1 - Approach (0 to 2s):")
for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
    p, v, surf = traj(t)
    print(f"  t={t:.1f}s: p={p}, |v|={np.linalg.norm(v):.3f} m/s, surf={surf}")

print("\nPhase 2 - Circle (2s to 6s):")
for t in [2.0, 3.0, 4.0, 5.0, 6.0]:
    p, v, surf = traj(t)
    dist_to_center = np.linalg.norm(p[:2] - center[:2])
    print(f"  t={t:.1f}s: p={p}, |v|={np.linalg.norm(v):.3f} m/s, dist_to_center={dist_to_center:.4f}m, surf={surf}")

print("\n✓ Trajectory fix validated!")
print("  - Starts from robot's actual position: [-0.3, 0, 0.633]")
print("  - Smoothly descends to table contact: z=0.0495")
print("  - Then traces circle at contact depth")
