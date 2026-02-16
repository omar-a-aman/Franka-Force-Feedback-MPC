# `src/mpc/` - Controller Implementations

This folder contains both MPC controllers used in evaluations.

## Files

- `crocoddyl_classical.py`
  - `ClassicalMPCConfig`
  - `ClassicalCrocoddylMPC`
  - OCP in state `(q, v)` with torque control
- `crocoddyl_force_feedback.py`
  - `ForceFeedbackMPCConfig`
  - `ForceFeedbackCrocoddylMPC`
  - Augmented OCP with torque-state dynamics `(q, v, tau_hat)`
- `pinocchio_ee_align.py`
  - helper to add MuJoCo-like tool frame in Pinocchio models

## Shared design points

- Crocoddyl shooting problem + FDDP/BoxFDDP
- phase-based behavior (approach/contact)
- normal contact option (`normal_1d`) for sliding tasks
- frame-aligned references from MuJoCo to Pinocchio
- receding-horizon warm start and policy reuse

## Contact model options

- `normal_1d` (default): normal constraint only, tangential motion allowed
- `point3d`: rigid point contact (can suppress tangential sliding)

Use `normal_1d` for circle-on-surface tracking unless you are explicitly testing
rigid no-slip behavior.
