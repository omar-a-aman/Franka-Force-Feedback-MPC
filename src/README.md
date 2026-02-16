# `src/` Package Guide

Core project code lives in `src/` and is split by responsibility:

- `src/mpc/`: MPC controller implementations
- `src/run/`: executable experiment runners
- `src/sim/`: MuJoCo simulator wrapper and smoke tests
- `src/tasks/`: reference trajectory generators
- `src/utils/`: logging and plotting helpers

Recommended entry points for normal use:

- `python3 src/run/run_classical.py ...`
- `python3 src/run/run_force_feedback.py ...`

These scripts construct trajectories, configure controllers, run simulations,
log metrics, and export plots.
