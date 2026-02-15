from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.mpc.crocoddyl_classical import ClassicalCrocoddylMPC, ClassicalMPCConfig
from src.sim.franka_sim import FrankaMujocoSim
from src.tasks.trajectories import make_approach_then_circle
from src.utils.logging import RunLogger

SCENE = Path("assets/scenes/panda_table_scene.xml")
SCENARIOS = ("flat", "tilted", "actuation_uncertainty")


def _table_geometry_world(sim: FrankaMujocoSim):
    table_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if table_geom_id < 0:
        raise RuntimeError("table_top geom not found in scene")

    table_center = sim.data.geom_xpos[table_geom_id].copy()
    table_size = sim.model.geom_size[table_geom_id].copy()  # half-sizes
    z_table_top = float(table_center[2] + table_size[2])
    return table_geom_id, table_center, table_size, z_table_top


def _scenario_settings(name: str):
    if name == "flat":
        return {
            "tilt_deg": 0.0,
            "torque_scale": np.ones(7, dtype=float),
            "label": "Flat table",
        }
    if name == "tilted":
        return {
            "tilt_deg": 8.0,
            "torque_scale": np.ones(7, dtype=float),
            "label": "Tilted table (8deg)",
        }
    if name == "actuation_uncertainty":
        return {
            "tilt_deg": 0.0,
            "torque_scale": np.array([0.90, 1.08, 0.92, 1.05, 0.88, 1.10, 0.86], dtype=float),
            "label": "Actuation gain mismatch",
        }
    raise ValueError(f"Unknown scenario '{name}'")


def _apply_table_tilt(sim: FrankaMujocoSim, tilt_deg: float) -> None:
    if abs(float(tilt_deg)) < 1e-12:
        return

    table_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if table_body_id < 0:
        raise RuntimeError("table body not found in scene")

    angle = np.deg2rad(float(tilt_deg))
    # Tilt around world y-axis.
    quat = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=float)
    sim.model.body_quat[table_body_id] = quat
    mujoco.mj_forward(sim.model, sim.data)


def _save_paper_plots(npz_path: Path, out_dir: Path, fn_des: float) -> None:
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    t = data["t"]
    err_tan = data["err_tan"]
    fn_meas = data["fn_meas"]
    fn_pred = data["fn_pred"]

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(t, err_tan, label="Tangential error")
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title("Tangential tracking error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "tangential_error.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(t, fn_meas, label="Fn_meas")
    plt.plot(t, np.full_like(t, float(fn_des)), "--", label="Fn_des")
    plt.xlabel("time [s]")
    plt.ylabel("normal force [N]")
    plt.title("Measured normal force")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fn_meas_vs_des.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(t, fn_pred, label="Fn_pred")
    plt.plot(t, np.full_like(t, float(fn_des)), "--", label="Fn_des")
    plt.xlabel("time [s]")
    plt.ylabel("normal force [N]")
    plt.title("Predicted normal force (OCP)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fn_pred_vs_des.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(t, fn_meas, label="Fn_meas")
    plt.plot(t, fn_pred, label="Fn_pred")
    plt.plot(t, np.full_like(t, float(fn_des)), "--", label="Fn_des")
    plt.xlabel("time [s]")
    plt.ylabel("normal force [N]")
    plt.title("Measured vs predicted normal force")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "fn_meas_vs_pred.png", dpi=180)
    plt.close()


def _run_single(
    scenario: str,
    total_time: float,
    no_viewer: bool,
    results_dir: Path,
    save_plots: bool,
) -> dict:
    settings = _scenario_settings(scenario)

    print("=" * 80)
    print(f"Classical Panda MPC (Paper-Faithful) - Scenario: {settings['label']}")
    print("=" * 80)

    sim = FrankaMujocoSim(SCENE, command_type="torque", n_substeps=5)
    obs = sim.reset("neutral")
    # Build the controller references from the nominal flat geometry first.
    # For disturbed scenarios, the world is tilted afterwards (unknown to the controller model).
    obs = sim.get_observation(with_ee=True, with_jacobian=True)

    print(f"Simulation initialized (dt={sim.dt:.4f}s)")
    print(f"Initial EE position: {obs.ee_pos}")
    print(f"Scenario: {settings['label']}")

    _, table_center, table_half, z_table_top = _table_geometry_world(sim)
    r_tool = float(sim.model.geom_size[sim.ee_geom_id][0])
    z_contact = z_table_top + r_tool + 2.0e-4
    z_pre = z_contact + 0.08

    # Requested benchmark: large-radius circle centered at the table center.
    radius = 0.10
    omega = 0.14
    center = np.array([table_center[0], table_center[1], z_contact], dtype=float)

    print(f"Table top (world): center={table_center}, half-size={table_half}, z_top={z_table_top:.4f}m")
    print(f"Contact target: z_contact={z_contact:.4f}m (tool radius={r_tool:.4f}m)")
    print(f"Circle center (table center): {center}, radius={radius:.3f}m")

    traj = make_approach_then_circle(
        center=center,
        radius=radius,
        omega=omega,
        z_pre=z_pre,
        z_contact=z_contact,
        t_approach=2.0,
        ee_start=obs.ee_pos.copy(),
        t_pre=2.2,
    )
    t_contact_phase = 4.2  # must match t_pre + t_approach above
    print("Trajectory generated (approach + circle)")

    cfg = ClassicalMPCConfig(
        horizon=30,
        dt=sim.dt,
        z_contact=z_contact,
        z_press=0.0095,
        w_ee_pos=8.0e2,
        w_ee_ori=8.0e1,
        ori_weights=np.array([2.5, 2.5, 0.1], dtype=float),
        w_posture=2.0e-1,
        w_v=2.0e-1,
        w_tau=1.0e-3,
        w_tau_smooth=5.0e-2,
        w_tangent_pos=3.1e3,
        w_tangent_vel=1.0e3,
        w_plane_z=1.3e3,
        w_vz=8.0e2,
        w_friction_cone=0.0,
        w_unilateral=2.0e1,
        mu=1.0,
        contact_model="normal_1d",
        contact_gains=np.array([82.0, 45.0], dtype=float),
        fn_des=28.5,
        w_fn=2.85e1,
        w_wdamp=6.0e1,
        w_wdamp_weights=np.array([1.6, 1.6, 0.2], dtype=float),
        fn_contact_on=1.0,
        fn_contact_off=0.05,
        z_contact_band=0.012,
        max_iters=55,
        mpc_update_steps=1,
        use_feedback_policy=True,
        feedback_gain_scale=0.12,
        max_tau_raw_inf=1.5e2,
        contact_release_steps=180,
        debug_every=100,
    )
    print("MPC config created")

    mpc = ClassicalCrocoddylMPC(sim=sim, traj_fn=traj, config=cfg)
    print("MPC initialized")
    print()

    if abs(float(settings["tilt_deg"])) > 1e-12:
        _apply_table_tilt(sim, settings["tilt_deg"])
        obs = sim.get_observation(with_ee=True, with_jacobian=True)
        print(
            f"Applied hidden table tilt: {settings['tilt_deg']:.1f} deg "
            "(controller references remain from nominal flat table)."
        )

    logger = RunLogger(
        run_name=f"classical_{scenario}",
        results_dir=results_dir,
        notes={"scenario": scenario, "scene": str(SCENE)},
    )

    t = 0.0
    steps = int(total_time / sim.dt)
    contact_threshold = 0.5

    print(f"Running simulation for {total_time:.1f}s ({steps} steps)...")
    print()

    summary = {
        "t": [],
        "err_tan": [],
        "err_3d": [],
        "fn_meas": [],
        "fn_pred": [],
        "contact": [],
    }

    torque_scale = settings["torque_scale"]

    def _step_once(k: int, t_now: float, obs_now):
        tau_cmd = mpc.compute_control(obs_now, t_now)
        tau_applied = tau_cmd * torque_scale
        obs_next = sim.step(tau_applied)
        t_next = t_now + sim.dt

        p_ref, v_ref, surf_ref = traj(t_next)
        err = np.asarray(obs_next.ee_pos, dtype=float) - np.asarray(p_ref, dtype=float)
        err_tan = float(np.linalg.norm(err[:2]))
        err_3d = float(np.linalg.norm(err))
        fn_meas = float(obs_next.f_contact_normal)
        in_contact = bool(fn_meas > contact_threshold)

        info = dict(mpc.last_info)
        fn_pred = float(info.get("fn_pred", np.nan))

        summary["t"].append(t_next)
        summary["err_tan"].append(err_tan)
        summary["err_3d"].append(err_3d)
        summary["fn_meas"].append(fn_meas)
        summary["fn_pred"].append(fn_pred)
        summary["contact"].append(1.0 if in_contact else 0.0)

        logger.log(
            t=t_next,
            ee_pos=np.asarray(obs_next.ee_pos, dtype=float).copy(),
            ee_ref=np.asarray(p_ref, dtype=float).copy(),
            ee_vel=np.asarray(obs_next.ee_vel, dtype=float).copy(),
            ee_vel_ref=np.asarray(v_ref, dtype=float).copy(),
            err_tan=err_tan,
            err_3d=err_3d,
            fn_meas=fn_meas,
            fn_pred=fn_pred,
            fn_des=float(cfg.fn_des),
            tau_cmd=np.asarray(tau_cmd, dtype=float).copy(),
            tau_meas=np.asarray(obs_next.tau_meas, dtype=float).copy(),
            tau_applied=np.asarray(tau_applied, dtype=float).copy(),
            contact=int(in_contact),
            surface_ref=int(surf_ref),
            solver_iters=int(info.get("iters", -1)),
            solver_cost=float(info.get("cost", np.nan)),
            solver_success=int(bool(info.get("ok", False))),
            solver_unstable=int(bool(info.get("unstable", False))),
            solver_solved_now=int(bool(info.get("solved_now", False))),
            solver_policy_idx=int(info.get("policy_idx", -1)),
            tau_raw_inf=float(info.get("tau_raw_inf", np.nan)),
            tau_cmd_inf=float(info.get("tau_cmd_inf", np.nan)),
        )

        if k % 100 == 0:
            print(
                f"k={k:4d} t={t_next:6.3f}s | "
                f"EE=[{obs_next.ee_pos[0]:.3f}, {obs_next.ee_pos[1]:.3f}, {obs_next.ee_pos[2]:.4f}] | "
                f"|p-p_ref|={err_3d:.4f}m | err_tan={err_tan:.4f}m | "
                f"Fn_meas={fn_meas:.2f}N Fn_pred={fn_pred:.2f}N | "
                f"contact={int(in_contact)}"
            )

        return t_next, obs_next

    if no_viewer:
        for k in range(steps):
            t, obs = _step_once(k, t, obs)
    else:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            viewer.cam.distance = 1.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -30
            for k in range(steps):
                t, obs = _step_once(k, t, obs)
                try:
                    viewer.sync()
                except Exception:
                    pass

    t_arr = np.asarray(summary["t"], dtype=float)
    err_tan_arr = np.asarray(summary["err_tan"], dtype=float)
    err_3d_arr = np.asarray(summary["err_3d"], dtype=float)
    fn_meas_arr = np.asarray(summary["fn_meas"], dtype=float)
    fn_pred_arr = np.asarray(summary["fn_pred"], dtype=float)
    contact_arr = np.asarray(summary["contact"], dtype=float)
    contact_phase_mask = t_arr >= float(t_contact_phase)
    contact_arr_phase = contact_arr[contact_phase_mask]
    fn_phase = fn_meas_arr[contact_phase_mask]
    err_tan_phase = err_tan_arr[contact_phase_mask]

    rms_tan = float(np.sqrt(np.mean(err_tan_arr ** 2))) if err_tan_arr.size > 0 else np.nan
    rms_3d = float(np.sqrt(np.mean(err_3d_arr ** 2))) if err_3d_arr.size > 0 else np.nan
    rms_tan_phase = float(np.sqrt(np.mean(err_tan_phase ** 2))) if err_tan_phase.size > 0 else np.nan
    max_fn = float(np.max(fn_meas_arr)) if fn_meas_arr.size > 0 else np.nan
    contact_loss_pct = float((1.0 - np.mean(contact_arr)) * 100.0) if contact_arr.size > 0 else np.nan
    contact_loss_phase_pct = (
        float((1.0 - np.mean(contact_arr_phase)) * 100.0) if contact_arr_phase.size > 0 else np.nan
    )
    fn_mean_phase = float(np.mean(fn_phase)) if fn_phase.size > 0 else np.nan

    logger.set_meta(
        total_time=float(total_time),
        dt=float(sim.dt),
        scenario_label=settings["label"],
        scenario_tilt_deg=float(settings["tilt_deg"]),
        torque_scale=np.asarray(torque_scale, dtype=float),
        fn_des=float(cfg.fn_des),
        rms_tangential_error=rms_tan,
        rms_tangential_error_contact_phase=rms_tan_phase,
        rms_3d_error=rms_3d,
        max_fn=max_fn,
        contact_loss_pct=contact_loss_pct,
        contact_loss_contact_phase_pct=contact_loss_phase_pct,
        fn_mean_contact_phase=fn_mean_phase,
        contact_phase_start_s=float(t_contact_phase),
        cfg_summary={
            "horizon": int(cfg.horizon),
            "dt": float(cfg.dt),
            "z_contact": float(cfg.z_contact),
            "z_press": float(cfg.z_press),
            "w_fn": float(cfg.w_fn),
            "fn_des": float(cfg.fn_des),
            "contact_model": str(cfg.contact_model),
            "w_friction_cone": float(cfg.w_friction_cone),
            "w_unilateral": float(cfg.w_unilateral),
        },
    )
    logger.save()

    if save_plots:
        _save_paper_plots(logger.path_npz, logger.run_dir, cfg.fn_des)

    print()
    print("=" * 80)
    print("Simulation complete!")
    print("=" * 80)
    print("\nSummary statistics:")
    print(f"  RMS tangential error: {rms_tan:.4f} m")
    print(f"  RMS tangential error (contact phase): {rms_tan_phase:.4f} m")
    print(f"  RMS 3D tracking error: {rms_3d:.4f} m")
    print(f"  Fn_meas: min={fn_meas_arr.min():.2f}N, max={fn_meas_arr.max():.2f}N, mean={fn_meas_arr.mean():.2f}N")
    print(f"  Fn_meas (contact phase mean): {fn_mean_phase:.2f}N")
    if np.isfinite(fn_pred_arr).any():
        print(
            f"  Fn_pred: min={np.nanmin(fn_pred_arr):.2f}N, "
            f"max={np.nanmax(fn_pred_arr):.2f}N, mean={np.nanmean(fn_pred_arr):.2f}N"
        )
    print(f"  Contact loss: {contact_loss_pct:.1f}%")
    print(f"  Contact loss (contact phase): {contact_loss_phase_pct:.1f}%")
    print(f"  Logs saved to: {logger.run_dir}")

    return {
        "scenario": scenario,
        "rms_tan": rms_tan,
        "rms_3d": rms_3d,
        "max_fn": max_fn,
        "contact_loss_pct": contact_loss_pct,
        "contact_loss_phase_pct": contact_loss_phase_pct,
        "run_dir": str(logger.run_dir),
    }


def main(
    scenario: str,
    all_scenarios: bool,
    no_viewer: bool,
    total_time: float,
    results_dir: Path,
    no_plots: bool,
):
    if all_scenarios:
        if not no_viewer:
            print("[info] Forcing --no-viewer while sweeping all scenarios.")
        metrics = []
        for sc in SCENARIOS:
            metrics.append(
                _run_single(
                    scenario=sc,
                    total_time=total_time,
                    no_viewer=True,
                    results_dir=results_dir,
                    save_plots=(not no_plots),
                )
            )

        print()
        print("=" * 80)
        print("Scenario Sweep Summary")
        print("=" * 80)
        for m in metrics:
            print(
                f"{m['scenario']:>22s} | "
                f"RMS_tan={m['rms_tan']:.4f} m | "
                f"max_fn={m['max_fn']:.2f} N | "
                f"contact_loss={m['contact_loss_pct']:.1f}% | "
                f"contact_loss_phase={m['contact_loss_phase_pct']:.1f}%"
            )
        return

    _run_single(
        scenario=scenario,
        total_time=total_time,
        no_viewer=no_viewer,
        results_dir=results_dir,
        save_plots=(not no_plots),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="flat", help="Evaluation scenario.")
    parser.add_argument("--all-scenarios", action="store_true", help="Run flat/tilted/actuation_uncertainty.")
    parser.add_argument("--no-viewer", action="store_true", help="Run without MuJoCo viewer.")
    parser.add_argument("--time", type=float, default=12.0, help="Total simulation time [s].")
    parser.add_argument("--results-dir", type=Path, default=Path("results/classical_eval"), help="Output folder.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    main(
        scenario=args.scenario,
        all_scenarios=args.all_scenarios,
        no_viewer=args.no_viewer,
        total_time=args.time,
        results_dir=args.results_dir,
        no_plots=args.no_plots,
    )
