import numpy as np
from pathlib import Path
import mujoco.viewer

from franka_sim import FrankaMujocoSim
# add these imports at top
from src.utils.logging import RunLogger
from src.utils.plotting import plot_sanding_run
from datetime import datetime

SCENE = Path("assets/scenes/panda_table_scene.xml")


def clip_vec(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


# --------------------------
# Quaternion helpers (wxyz)
# --------------------------
def quat_conj(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)

def rotmat_to_quat_wxyz(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    return quat_normalize(np.array([w, x, y, z], dtype=float))


def make_vertical_down_quat():
    """
    Desired EE orientation in WORLD:
      - EE z-axis points DOWN (world): [0,0,-1]
      - EE x-axis aligned with world x: [1,0,0]
    """
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    return rotmat_to_quat_wxyz(R)


def orientation_error_world(q_des, q_cur):
    """
    eR in world coordinates using quaternion error:
      q_err = q_des * conj(q_cur)
      eR ≈ 2 * v_err (small-angle)
    """
    q_des = quat_normalize(q_des)
    q_cur = quat_normalize(q_cur)
    q_err = quat_mul(q_des, quat_conj(q_cur))
    q_err = quat_normalize(q_err)
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def main():
    sim = FrankaMujocoSim(SCENE, command_type="torque", n_substeps=5)
    obs = sim.reset("neutral")

    

    # Table: body z=0.3, half-height=0.02 => top = 0.32
    z_table_top = 0.32
    x_table, y_table = -0.5, 0.0

    # EE collision geom is a sphere; touchdown should respect its radius
    r = float(sim.model.geom_size[sim.ee_geom_id][0])  # sphere radius
    penetration = 0.004  # 4mm "target penetration" to ensure contact engages

    p_touch = np.array([x_table, y_table, z_table_top + r - penetration])
    p_pre   = np.array([x_table, y_table, z_table_top + r + 0.10])

    # Desired vertical tool orientation (avoid arm/table collision)
    q_des = make_vertical_down_quat()

    # Posture reference
    q_ref = obs.q.copy()
    dq_ref = np.zeros_like(obs.dq)

    # Gains
    Kp_q = np.array([40, 40, 30, 25, 15, 10, 8], dtype=float)
    Kd_q = np.array([4,  4,  3,  2,  1.5, 1.0, 0.8], dtype=float)

    Kp_p = np.array([600.0, 600.0, 600.0])
    Kd_p = np.array([40.0,  40.0,  40.0])

    Kp_R = np.array([60.0, 60.0, 60.0])
    Kd_R = np.array([4.0,  4.0,  4.0])

    # Force control
    Fn_des = 20.0
    Kf = 0.8
    Ki = 30.0          # optional integral help (kept bounded)
    contact_on = 0.5   # don’t switch early

    # Limits
    tau_lim = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
    Fz_min, Fz_max = -80.0, 0.0  # only push down

    phase = 0
    integ_f = 0.0

    run_name = "sanding_pd_force"
    logger = RunLogger(
        run_name=run_name,
        results_dir="results",
        notes={
            "scene": str(SCENE),
            "Fn_des": Fn_des,
            "Kp_p": Kp_p,
            "Kd_p": Kd_p,
            "Kp_R": Kp_R,
            "Kd_R": Kd_R,
            "Kf": Kf,
            "Ki": Ki,
            "contact_on": contact_on,
            "p_pre": p_pre,
            "p_touch": p_touch,
            "q_des": q_des,
            "tau_lim": tau_lim,
            "Fz_minmax": [Fz_min, Fz_max],
        },
    )
    logger.set_meta(dt=sim.dt)
    t = 0.0

    print("Start ee:", obs.ee_pos)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance = 1.6

        for k in range(5000):
            obs = sim.get_observation(with_ee=True, with_jacobian=True)

            q, dq = obs.q, obs.dq
            p = obs.ee_pos
            q_cur = obs.ee_quat

            Jp = obs.J_pos
            Jr = obs.J_rot
            v = Jp @ dq
            w = Jr @ dq

            fn = obs.f_contact_normal  # (>=0) magnitude after the franka_sim fix

            # --------------------------
            # Phase switching
            # --------------------------
            if phase == 0:
                if np.linalg.norm(p - p_pre) < 0.01:
                    phase = 1
            elif phase == 1:
                if fn > contact_on:
                    phase = 2
                    integ_f = 0.0

            # --------------------------
            # Position -> Force
            # --------------------------
            if phase == 0:
                F = Kp_p * (p_pre - p) - Kd_p * v
            elif phase == 1:
                F = Kp_p * (p_touch - p) - Kd_p * v
            else:
                # keep x,y stiff; regulate normal force in z
                Fxy = Kp_p[:2] * (p_touch[:2] - p[:2]) - Kd_p[:2] * v[:2]

                # ✅ correct sign: if fn < Fn_des => e_f positive => push MORE
                e_f = (Fn_des - fn)
                integ_f = np.clip(integ_f + e_f * sim.dt, -1.0, 1.0)

                # negative z pushes down
                Fz = -(Fn_des + Kf * e_f + Ki * integ_f)
                Fz = float(np.clip(Fz, Fz_min, Fz_max))

                F = np.array([Fxy[0], Fxy[1], Fz], dtype=float)

            tau_pos = Jp.T @ F

            # --------------------------
            # Orientation -> Moment
            # --------------------------
            eR = orientation_error_world(q_des, q_cur)
            M = Kp_R * eR - Kd_R * w
            tau_rot = Jr.T @ M

            # --------------------------
            # Posture + gravity
            # --------------------------
            tau_posture = Kp_q * (q_ref - q) + Kd_q * (dq_ref - dq)
            tau_bias = sim.bias_torque()

            # Nullspace projector for 6D task
            J6 = np.vstack([Jp, Jr])      # (6,7)
            J6_pinv = np.linalg.pinv(J6)  # (7,6)
            N = np.eye(7) - J6_pinv @ J6  # (7,7)

            tau = tau_bias + tau_pos + tau_rot + N @ tau_posture
            tau = clip_vec(tau, -tau_lim, tau_lim)

            # log what you COMMAND
            logger.log(
                t=t,
                dt=sim.dt,
                phase=phase,
                q=q.copy(),
                dq=dq.copy(),
                p=p.copy(),
                quat=q_cur.copy(),
                fn=float(fn),
                eR=eR.copy(),
                F_cmd=F.copy(),
                tau_cmd=tau.copy(),
            )

            obs = sim.step(tau)
            t += sim.dt
            viewer.sync()


            if k % 50 == 0:
                print(
                    f"{k:4d} phase={phase} p={obs.ee_pos} fn={obs.f_contact_normal:.2f} "
                    f"|eR|={np.linalg.norm(eR):.3f} F={F}"
                )
    logger.save()
    plot_dir = plot_sanding_run(logger.path_npz, out_dir="results/plots")
    print(f"Saved logs to:  {logger.run_dir}")
    print(f"Saved plots to: {plot_dir}")



if __name__ == "__main__":
    main()
