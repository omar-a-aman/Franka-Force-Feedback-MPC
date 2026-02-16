# `assets/scenes/` - Project Scene Definitions

Primary scene files used by the run scripts:

- `panda_table_scene.xml`
  - world + table setup for contact task
  - includes `panda_robot.xml`
  - defines:
    - `table_top` (visual geometry)
    - `table_contact` (dedicated collision/contact surface)
- `panda_robot.xml`
  - Panda model with tool site and collision sphere:
    - `ee_site`
    - `ee_collision`

Current runners (`run_classical.py`, `run_force_feedback.py`) use:

- `assets/scenes/panda_table_scene.xml`
