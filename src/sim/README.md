# `src/sim/` - Simulation Layer

This folder contains the MuJoCo interface used by controllers and run scripts.

## Files

- `franka_sim.py`
  - main simulator wrapper (`FrankaMujocoSim`)
  - exposes consistent observation channels for control and logging
- `panda_table_demo.py`
  - quick scene-load and viewer smoke test
- `mujoco_viewer.py`
  - generic MuJoCo viewer smoke test
- `test_sim_api.py`
  - legacy/manual script for API experimentation

## `FrankaMujocoSim` highlights

- torque-mode stepping through `qfrc_applied`
- explicit torque channels:
  - `tau_cmd`, `tau_act`, `tau_constraint`, `tau_total`
  - filtered variants for measured proxies (`tau_meas_filt`, `tau_meas_act_filt`)
- end-effector kinematics:
  - `ee_pos`, `ee_quat`, `ee_vel`, `J_pos`, `J_rot`
- contact channels:
  - normal force, tangential force, table normal, contact counts

This API is the source of truth for controller observations.
