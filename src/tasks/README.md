# `src/tasks/` - Task and Trajectory Definitions

This folder contains trajectory generators used by runners.

## Files

- `trajectories.py`
  - `make_approach_then_circle(...)`
  - staged reference:
    1. optional pre-approach,
    2. approach/descend,
    3. contact-phase circular tracking
- `test_traj.py`
  - legacy script for trajectory sanity checks

## Main trajectory output

`traj(t) -> (p_ref, v_ref, surface_mode)`

- `p_ref`: desired EE position (3,)
- `v_ref`: desired EE linear velocity (3,)
- `surface_mode`: whether contact-phase dynamics should be active

Run scripts consume this interface directly.
