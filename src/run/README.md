# `src/run/` - Experiment Runners

This folder contains the top-level scripts used to evaluate controllers.

## Files

- `run_classical.py`: runs the classical MPC controller
- `run_force_feedback.py`: runs the force-feedback MPC controller
- `uncertainty_profiles.py`: shared uncertainty injection used by benchmark runs

## Common Usage

Run from repository root.

### Classical

```bash
python3 src/run/run_classical.py --scenario flat --time 20 --no-viewer
```

### Force-feedback

```bash
python3 src/run/run_force_feedback.py --scenario flat --time 20 --no-viewer
```

### Scenario sweep

```bash
python3 src/run/run_classical.py --all-scenarios
python3 src/run/run_force_feedback.py --all-scenarios
```

## Scenarios

- `flat`
- `tilted_5`
- `tilted_10`
- `tilted_15`
- `actuation_uncertainty`
- `tilted` (legacy alias)

## Important CLI options

Shared:

- `--benchmark-mode` / `--no-benchmark-mode`
- `--contact-model {normal_1d,point3d}`
- `--phase-source {trajectory,force_latch}`
- `--circle-radius`
- `--circle-omega`
- `--mpc-iters`
- `--use-command-filter`
- `--align-check-samples`

Force-feedback only:

- `--ff-tau-state-source {auto,tau_meas_act_filt,tau_meas_act,tau_cmd,tau_meas_filt,tau_meas,tau_total}`

## Outputs

By default, outputs are written to:

- Classical: `results/classical_eval/logs/...`
- FF-MPC: `results/force_feedback_eval/logs/...`

Each run directory contains:

- `data.npz`
- `data.csv`
- `meta.json`
- plot PNG files (if plotting enabled)
