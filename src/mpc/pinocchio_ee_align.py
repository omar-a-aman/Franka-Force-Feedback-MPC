import numpy as np
import pinocchio as pin


def quat_wxyz_to_R(q):
    q = np.asarray(q, dtype=float).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def add_mujoco_tool_frame(
    model: pin.Model,
    parent_frame_name: str = "panda_link8",
    new_frame_name: str = "ee_mj",
    tool_pos: np.ndarray = np.array([0.0, 0.0, 0.107]),
    tool_quat_wxyz: np.ndarray = np.array([0.3826834, 0.0, 0.0, 0.9238795]),
):
    """
    Adds a Pinocchio operational frame meant to match the MuJoCo ee_site / tool transform.
    """
    if model.existFrame(new_frame_name):
        return model.getFrameId(new_frame_name)

    if not model.existFrame(parent_frame_name):
        raise ValueError(f"Pinocchio model has no frame '{parent_frame_name}'")

    parent_fid = model.getFrameId(parent_frame_name)
    parent_fr = model.frames[parent_fid]

    R_tool = quat_wxyz_to_R(tool_quat_wxyz)
    X_parent_tool = pin.SE3(R_tool, np.asarray(tool_pos, dtype=float).reshape(3))

    placement = parent_fr.placement * X_parent_tool

    new_fr = pin.Frame(
        new_frame_name,
        parent_fr.parentJoint,
        parent_fr.parentFrame,
        placement,
        pin.FrameType.OP_FRAME,
    )
    model.addFrame(new_fr)
    return model.getFrameId(new_frame_name)
