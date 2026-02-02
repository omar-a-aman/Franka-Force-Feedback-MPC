import time
from pathlib import Path

import mujoco
import numpy as np

try:
    import mujoco.viewer
except Exception as e:
    raise RuntimeError(
        "Failed to import mujoco.viewer. On Ubuntu, ensure OpenGL libs are installed.\n"
        "If using SSH, you may need X forwarding or offscreen rendering."
    ) from e


ROOT = Path(__file__).resolve().parents[2]
MENAGERIE = ROOT / "assets" / "mujoco_menagerie"
# Panda scene from menagerie (works out of the box)
PANDA_XML = MENAGERIE / "franka_emika_panda" / "mjx_scene.xml"


def main():
    assert PANDA_XML.exists(), f"Missing: {PANDA_XML}"
    model = mujoco.MjModel.from_xml_path(str(PANDA_XML))
    data = mujoco.MjData(model)

    # Small settle
    for _ in range(50):
        mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running() and (time.time() - t0) < 10.0:
            # tiny torque dither if actuators exist
            if model.nu > 0:
                data.ctrl[:] = 0.0
                # drive first 7 actuators gently
                data.ctrl[: min(7, model.nu)] = 0.3 * np.sin(2.0 * np.pi * 0.5 * (time.time() - t0))

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print("Viewer smoke test OK.")


if __name__ == "__main__":
    main()