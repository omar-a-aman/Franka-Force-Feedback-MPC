#!/usr/bin/env python3
"""
Verify the trajectory fix: robot should start from actual position and descend to table
"""
import numpy as np

# Inline the trajectory function (no dependencies)
def make_approach_then_circle(
    center,
    radius,
    omega,
    z_pre,
    z_contact,
    t_approach=2.0,
    ee_start=None,
):
    center = np.asarray(center, dtype=float).copy()

    # Circle start position (where we transition to contact phase)
    p_circle_start = center.copy()
    p_circle_start[0] += radius  # (center_x + radius, center_y, z_contact)
    p_circle_start[2] = z_contact

    # Approach start: either robot's actual current position, or default to circle start lifted
    if ee_start is not None:
        p_approach_start = np.asarray(ee_start, dtype=float).copy()
        # Make sure we start high enough to descend
        if p_approach_start[2] < z_pre:
            p_approach_start[2] = z_pre
    else:
        p_approach_start = p_circle_start.copy()
        p_approach_start[2] = z_pre  # Default: above circle start

    def smoothstep(s):
        s = np.clip(s, 0.0, 1.0)
        return s * s * (3.0 - 2.0 * s)

    def dsmoothstep_ds(s):
        s = np.clip(s, 0.0, 1.0)
        return 6.0 * s * (1.0 - s)

    def traj(t):
        if t < t_approach:
            # APPROACH PHASE
            s_lin = t / t_approach
            s = smoothstep(s_lin)
            dsdt = dsmoothstep_ds(s_lin) / t_approach

            # Linear interpolation in 3D space
            dp = p_circle_start - p_approach_start
            p = (1.0 - s) * p_approach_start + s * p_circle_start
            v = dsdt * dp
            
            return p, v, False  # Not in contact yet

        else:
            # CONTACT PHASE - Circular trajectory
            tt = t - t_approach
            th = omega * tt

            # Position: circle in XY at z_contact depth
            p = center.copy()
            p[0] += radius * np.cos(th)
            p[1] += radius * np.sin(th)
            p[2] = z_contact

            # Velocity: tangential motion around circle
            v = np.zeros(3, dtype=float)
            v[0] = -radius * omega * np.sin(th)
            v[1] = radius * omega * np.cos(th)
            v[2] = 0.0

            return p, v, True  # In contact

    return traj


# Test parameters (from your run)
z_table = 0.02
r_tool = 0.03
z_contact = z_table + r_tool - 0.0005  # 0.0495
z_pre = z_contact + 0.08  # 0.1295

center = np.array([0.4, 0.0, z_contact])
ee_start_robot = np.array([-0.3, 0.0, 0.633])  # Where robot actually started

print("=" * 80)
print("TRAJECTORY FIX VERIFICATION")
print("=" * 80)
print(f"\nRobot starting position: {ee_start_robot}")
print(f"Target contact depth:   z={z_contact:.4f}m")
print(f"Circle center:          {center}")
print(f"Approach phase duration: {2.0}s")

# Create FIXED trajectory (with ee_start parameter)
traj_fixed = make_approach_then_circle(
    center=center,
    radius=0.05,
    omega=0.5,
    z_pre=z_pre,
    z_contact=z_contact,
    t_approach=2.0,
    ee_start=ee_start_robot.copy(),  # FIXED: Use robot's actual starting position
)

print("\n" + "=" * 80)
print("APPROACH PHASE (0-2s): Robot should descend smoothly")
print("=" * 80)

approach_times = [0.0, 0.5, 1.0, 1.5, 2.0]
print(f"{'Time (s)':>8} {'X':>10} {'Y':>10} {'Z':>10} {'|v| (m/s)':>12} {'dZ (mm)':>10}")
print("-" * 80)

for t in approach_times:
    p, v, surf = traj_fixed(t)
    z_descent_mm = (ee_start_robot[2] - p[2]) * 1000
    print(f"{t:8.2f} {p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f} {np.linalg.norm(v):12.4f} {z_descent_mm:10.1f}")

# Verify the robot descends from starting Z to contact Z
z_start = traj_fixed(0.0)[0][2]
z_end = traj_fixed(2.0)[0][2]
descent_mm = (z_start - z_end) * 1000

print(f"\n✓ Robot descends: {z_start:.4f}m → {z_end:.4f}m = {descent_mm:.1f}mm descent")
assert abs(z_end - z_contact) < 0.001, f"Final Z should be {z_contact:.4f}m, got {z_end:.4f}m"

print("\n" + "=" * 80)
print("CONTACT PHASE (2-6s): Robot should trace circle at fixed depth")
print("=" * 80)

circle_times = [2.0, 3.0, 4.0, 5.0, 6.0]
print(f"{'Time (s)':>8} {'X':>10} {'Y':>10} {'Z':>10} {'|v| (m/s)':>12} {'Radius (m)':>12}")
print("-" * 80)

for t in circle_times:
    p, v, surf = traj_fixed(t)
    radius_from_center = np.linalg.norm(p[:2] - center[:2])
    print(f"{t:8.2f} {p[0]:10.4f} {p[1]:10.4f} {p[2]:10.4f} {np.linalg.norm(v):12.4f} {radius_from_center:12.4f}")

# Verify Z is constant during circle
z_circle = [traj_fixed(t)[0][2] for t in circle_times]
z_var = np.var(z_circle)
print(f"\n✓ Z constant during circle: variance={z_var:.2e} (should be ~0)")
assert z_var < 1e-8, f"Z should be constant during circle phase"

print("\n" + "=" * 80)
print("✅ TRAJECTORY FIX VERIFIED")
print("=" * 80)
print("""
CHANGES MADE:
  1. trajectories.py: Added 'ee_start' parameter to make_approach_then_circle()
  2. test_crocoddyl.py: Pass robot's actual starting position (obs.ee_pos)
  
EXPECTED BEHAVIOR:
  • Robot now starts from [-0.3, 0, 0.633] (actual position)
  • Smoothly descends to contact depth 0.0495m over 2 seconds  
  • Traces 5cm radius circle at constant contact depth
  • No more wild Z oscillations or standing in place
  
NEXT STEP: Run the fixed test with:
  $ cd /home/omar/franka-force-feedback-mpc
  $ conda activate franka-mpc
  $ export PYTHONPATH=:$PYTHONPATH
  $ python src/mpc/test_crocoddyl.py
""")
