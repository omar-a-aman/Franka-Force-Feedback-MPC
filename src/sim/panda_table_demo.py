import time
from pathlib import Path
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assets" / "scenes" / "panda_table_scene.xml"

def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data  = mujoco.MjData(model)

    # Reset to keyframe pose
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "neutral")
    mujoco.mj_resetDataKeyframe(model, data, kid)

    mujoco.mj_forward(model, data)

    for _ in range(100):
        mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running() and time.time() - t0 < 100.0:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    print("Loaded panda_table_scene OK.")

if __name__ == "__main__":
    main()
