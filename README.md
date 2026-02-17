# Franka Force-Feedback MPC

This repository implements and evaluates two torque-space MPC controllers for
contact-rich end-effector tasks in MuJoCo with a Franka Panda model:

- `ClassicalCrocoddylMPC` (state `(q, v)`, control `tau`)
- `ForceFeedbackCrocoddylMPC` (augmented state `(q, v, tau_hat)`, control `w`)

Both controllers are configured for a staged task:

1. approach the table,
2. establish contact,
3. track a circle on the table while regulating normal force.

The codebase is organized to make controller comparison reproducible across
flat, tilted, and uncertainty scenarios.

## Repository Structure

- `src/mpc/`: controller implementations and MPC configs
- `src/run/`: runnable evaluation scripts and scenario setup
- `src/sim/`: MuJoCo simulation wrapper and smoke-test scripts
- `src/tasks/`: trajectory generators
- `src/utils/`: logging and plotting utilities
- `assets/scenes/`: Panda + table MuJoCo scenes used by evaluations
- `assets/mujoco_menagerie/`: vendored third-party robot assets
- `scripts/`: helper shell scripts
- `results/`: generated run artifacts

Each major subfolder has its own README.

## Environment Setup

### 1) Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate franka-mpc
```

### 2) Sanity-check key imports

```bash
python3 -c "import mujoco, crocoddyl, pinocchio; print('ok')"
```

## Quick Start

Run from repository root.

### Classical MPC

```bash
python3 src/run/run_classical.py --scenario flat --time 20.0 --no-viewer
```

### Force-Feedback MPC

```bash
python3 src/run/run_force_feedback.py --scenario flat --time 20.0 --no-viewer
```

### Sweep all scenarios

```bash
python3 src/run/run_classical.py --all-scenarios
python3 src/run/run_force_feedback.py --all-scenarios
```

`--all-scenarios` automatically runs without the viewer.

## Scenarios

Available in both run scripts:

- `flat`
- `tilted_5`
- `tilted_10`
- `tilted_15`
- `actuation_uncertainty`

Backward-compatible alias:

- `tilted` (legacy single tilted setup)

## Runtime Modes and Controllers

Both runners default to `--benchmark-mode`.

- `--benchmark-mode`:
  - uses benchmark-oriented controller settings,
  - uses 1 kHz MuJoCo physics step with `n_substeps=5` (`sim.dt = 0.005 s`),
  - enables shared uncertainty injection in `actuation_uncertainty`.
- `--no-benchmark-mode`:
  - uses development settings and disables benchmark uncertainty profile.

## Common CLI Options

For both `run_classical.py` and `run_force_feedback.py`:

- `--scenario {flat,tilted_5,tilted_10,tilted_15,actuation_uncertainty,tilted}`
- `--all-scenarios`
- `--no-viewer`
- `--time <seconds>`
- `--results-dir <path>`
- `--no-plots`
- `--contact-model {normal_1d,point3d}`
- `--phase-source {trajectory,force_latch}`
- `--circle-radius <m>` (default `0.10`)
- `--circle-omega <rad/s>` (default `1.5`)
- `--mpc-iters <int>` (hard override of solver iterations)
- `--low-budget` (only affects non-benchmark default budgets)
- `--use-command-filter`
- `--align-check-samples <int>` (MuJoCo/Pinocchio alignment check, `0` disables)

Additional Force-Feedback option:

- `--ff-tau-state-source {auto,tau_meas_act_filt,tau_meas_act,tau_cmd,tau_meas_filt,tau_meas,tau_total}`

## Output Artifacts

Each run creates:

- `results/<controller>_eval/logs/<timestamp>_<run_name>/data.npz`
- `results/<controller>_eval/logs/<timestamp>_<run_name>/data.csv`
- `results/<controller>_eval/logs/<timestamp>_<run_name>/meta.json`
- PNG plots (unless `--no-plots`)

Primary metrics printed in terminal and stored in metadata include:

- tangential tracking error,
- 3D tracking error,
- average absolute force error,
- max normal force,
- contact-loss percentage.

## Typical Evaluation Commands

### A) Baseline benchmark comparison

```bash
python3 src/run/run_classical.py --scenario flat --time 30 --no-viewer
python3 src/run/run_force_feedback.py --scenario flat --time 30 --no-viewer
```

### B) Tilt robustness

```bash
python3 src/run/run_classical.py --scenario tilted_10 --time 30 --no-viewer
python3 src/run/run_force_feedback.py --scenario tilted_10 --time 30 --no-viewer
```

### C) Uncertainty robustness

```bash
python3 src/run/run_classical.py --scenario actuation_uncertainty --time 30 --no-viewer
python3 src/run/run_force_feedback.py --scenario actuation_uncertainty --time 30 --no-viewer
```

## Troubleshooting

- Viewer issues on headless machines:
  - run with `--no-viewer`
- Divergent/unstable behavior:
  - reduce task aggressiveness (`--circle-omega`, `--circle-radius`)
  - increase `--mpc-iters`
  - run `--align-check-samples 16` or higher to confirm frame alignment
- Very slow runs:
  - use `--no-viewer`
  - reduce experiment time

## Notes

- `assets/mujoco_menagerie/` is a vendored third-party asset bundle.
- Project-specific scene files for this repository are in `assets/scenes/`.

## References

### Paper

- S. Kleff, E. Dantec, G. Saurel, N. Mansard, and L. Righetti, "Introducing Force Feedback in Model Predictive Control," *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2022, pp. 13379-13385.
- DOI: https://doi.org/10.1109/IROS47612.2022.9982003
- Open-access entry: https://is.mpg.de/publications/forcefeedback_righetti2022
