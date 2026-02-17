from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

from src.mpc.crocoddyl_force_feedback import ForceFeedbackCrocoddylMPC, ForceFeedbackMPCConfig
from src.run.uncertainty_profiles import ScenarioUncertaintyInjector, config_for_scenario
from src.sim.franka_sim import FrankaMujocoSim
from src.tasks.trajectories import make_approach_then_circle
from src.utils.logging import RunLogger
from src.utils.evaluation_plots import save_evaluation_plots

SCENE = Path("assets/scenes/panda_table_scene.xml")
SCENARIOS = ("flat", "tilted_5", "tilted_10", "tilted_15", "actuation_uncertainty")


def _scenario_seed(name: str) -> int:
    seeds = {
        "flat": 11,
        "tilted_5": 12,
        "tilted_10": 13,
        "tilted_15": 14,
        "actuation_uncertainty": 15,
        "tilted": 16,
    }
    return int(seeds.get(name, 99))


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
    if name == "tilted_5":
        return {
            "tilt_deg": 5.0,
            "torque_scale": np.ones(7, dtype=float),
            "label": "Tilted table (5deg)",
        }
    if name == "tilted_10":
        return {
            "tilt_deg": 10.0,
            "torque_scale": np.ones(7, dtype=float),
            "label": "Tilted table (10deg)",
        }
    if name == "tilted_15":
        return {
            "tilt_deg": 15.0,
            "torque_scale": np.ones(7, dtype=float),
            "label": "Tilted table (15deg)",
        }
    if name == "actuation_uncertainty":
        return {
            "tilt_deg": 0.0,
            "torque_scale": np.array([0.90, 1.08, 0.92, 1.05, 0.88, 1.10, 0.86], dtype=float),
            "label": "Actuation gain mismatch",
        }
    # backward-compat alias
    if name == "tilted":
        return {
            "tilt_deg": 8.0,
            "torque_scale": np.ones(7, dtype=float),
            "label": "Tilted table (8deg)",
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


def _save_run_plots(npz_path: Path, out_dir: Path, fn_des: float) -> None:
    save_evaluation_plots(npz_path=npz_path, out_dir=out_dir, fn_des=fn_des)


def _check_pin_mj_alignment(sim: FrankaMujocoSim, mpc: ForceFeedbackCrocoddylMPC, samples: int = 16, seed: int = 0) -> dict:
    """
    Quick consistency test for MuJoCo site pose vs. Pinocchio EE frame mapping.
    Returns aggregate position/rotation errors in MuJoCo world coordinates.
    """
    samples = int(max(samples, 0))
    if samples == 0:
        return {"samples": 0, "max_pos_m": np.nan, "rms_pos_m": np.nan, "max_rot_deg": np.nan, "rms_rot_deg": np.nan}

    qpos0 = sim.data.qpos.copy()
    qvel0 = sim.data.qvel.copy()
    qacc0 = sim.data.qacc.copy()

    rng = np.random.default_rng(int(seed))
    q_ref = np.asarray(sim.data.qpos[sim.qpos_adr], dtype=float).copy()
    jnt_range = np.asarray(sim.model.jnt_range[sim.jnt_ids], dtype=float)

    pos_errs = []
    rot_errs = []

    try:
        for _ in range(samples):
            q = q_ref.copy()
            for j in range(7):
                lo, hi = float(jnt_range[j, 0]), float(jnt_range[j, 1])
                if np.isfinite(lo) and np.isfinite(hi) and (hi > lo):
                    mid = 0.5 * (lo + hi)
                    half = 0.4 * (hi - lo)  # stay away from hard limits
                    q[j] = rng.uniform(mid - half, mid + half)
                else:
                    q[j] = q_ref[j] + rng.normal(scale=0.2)

            sim.data.qvel[:] = 0.0
            for j, adr in enumerate(sim.qpos_adr):
                sim.data.qpos[adr] = q[j]
            mujoco.mj_forward(sim.model, sim.data)

            p_mj = sim.data.site_xpos[sim.ee_site_id].copy()
            R_mj = sim.data.site_xmat[sim.ee_site_id].reshape(3, 3).copy()

            pin.forwardKinematics(mpc.model, mpc.data, q, np.zeros(mpc.model.nv))
            pin.updateFramePlacements(mpc.model, mpc.data)
            oMf = mpc.data.oMf[mpc.ee_fid]
            p_off = np.asarray(getattr(mpc, "p_site_minus_frame_pin", np.zeros(3, dtype=float)), dtype=float)
            p_mj_pred = mpc.R_mj_from_pin @ (oMf.translation + p_off)
            R_mj_pred = mpc.R_mj_from_pin @ oMf.rotation @ mpc.R_site_from_pin_ee

            pos_errs.append(float(np.linalg.norm(p_mj - p_mj_pred)))
            R_err = R_mj_pred.T @ R_mj
            c = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
            rot_errs.append(float(np.arccos(c)))
    finally:
        sim.data.qpos[:] = qpos0
        sim.data.qvel[:] = qvel0
        sim.data.qacc[:] = qacc0
        mujoco.mj_forward(sim.model, sim.data)

    pos_arr = np.asarray(pos_errs, dtype=float)
    rot_arr = np.asarray(rot_errs, dtype=float)
    return {
        "samples": int(samples),
        "max_pos_m": float(np.max(pos_arr)),
        "rms_pos_m": float(np.sqrt(np.mean(pos_arr ** 2))),
        "max_rot_deg": float(np.rad2deg(np.max(rot_arr))),
        "rms_rot_deg": float(np.rad2deg(np.sqrt(np.mean(rot_arr ** 2)))),
    }


def _run_single(
    scenario: str,
    total_time: float,
    no_viewer: bool,
    results_dir: Path,
    save_plots: bool,
    contact_model: str,
    low_budget: bool,
    mpc_iters: Optional[int],
    use_command_filter: bool,
    align_check_samples: int,
    ff_tau_state_source: str,
    phase_source: str,
    circle_radius: float,
    circle_omega: float,
    benchmark_mode: bool,
) -> dict:
    settings = _scenario_settings(scenario)

    print("=" * 80)
    print(f"Force-Feedback Panda MPC (Benchmark) - Scenario: {settings['label']}")
    print("=" * 80)

    sim = FrankaMujocoSim(SCENE, command_type="torque", n_substeps=5)
    if benchmark_mode:
        # Benchmark protocol uses a 1 kHz physics loop.
        sim.model.opt.timestep = 0.001
        mujoco.mj_forward(sim.model, sim.data)
    obs = sim.reset("neutral")
    # Build the controller references from the nominal flat geometry first.
    # For disturbed scenarios, the world is tilted afterwards (unknown to the controller model).
    obs = sim.get_observation(with_ee=True, with_jacobian=True)

    print(f"Simulation initialized (dt={sim.dt:.4f}s)")
    print(f"Initial EE position: {obs.ee_pos}")
    print(f"Scenario: {settings['label']}")
    print(
        f"Contact model: {contact_model} | command_filter={int(use_command_filter)} | "
        f"phase_source={phase_source} | benchmark_mode={int(bool(benchmark_mode))}"
    )
    print(f"FF tau_state source: {ff_tau_state_source}")

    _, table_center, table_half, z_table_top = _table_geometry_world(sim)
    r_tool = float(sim.model.geom_size[sim.ee_geom_id][0])
    z_contact_offset = -8.0e-3 if benchmark_mode else 2.0e-4
    z_contact = z_table_top + r_tool + z_contact_offset
    z_pre = z_contact + (0.05 if benchmark_mode else 0.08)

    radius = float(circle_radius)
    omega = float(circle_omega)
    center = np.array([table_center[0], table_center[1], z_contact], dtype=float)

    print(f"Table top (world): center={table_center}, half-size={table_half}, z_top={z_table_top:.4f}m")
    print(f"Contact target: z_contact={z_contact:.4f}m (tool radius={r_tool:.4f}m)")
    print(f"Circle center (table center): {center}, radius={radius:.3f}m")

    t_approach = 0.55 if benchmark_mode else 1.4
    t_pre = 0.25 if benchmark_mode else 1.4
    traj_base = make_approach_then_circle(
        center=center,
        radius=radius,
        omega=omega,
        z_pre=z_pre,
        z_contact=z_contact,
        t_approach=t_approach,
        ee_start=obs.ee_pos.copy(),
        t_pre=t_pre,
    )
    t_contact_phase = float(t_pre + t_approach)
    t_contact_stabilize = 0.2 if benchmark_mode else 0.0

    def traj(t_query: float):
        p_ref, v_ref, surf_ref = traj_base(t_query)
        if surf_ref and (float(t_query) < (t_contact_phase + t_contact_stabilize)):
            p_hold, _, _ = traj_base(t_contact_phase)
            return np.asarray(p_hold, dtype=float), np.zeros(3, dtype=float), True
        return p_ref, v_ref, surf_ref

    print("Trajectory generated (approach + circle)")

    if mpc_iters is not None:
        max_iters = int(mpc_iters)
    elif benchmark_mode:
        max_iters = 10
    else:
        max_iters = 3 if low_budget else 10
    print(
        f"MPC budget: max_iters={max_iters} "
        f"({'benchmark default' if (benchmark_mode and mpc_iters is None) else ('low-budget' if low_budget and mpc_iters is None else 'custom/dev')})"
    )

    if benchmark_mode:
        cfg = ForceFeedbackMPCConfig(
            horizon=40,
            dt=sim.dt,
            dt_ocp=0.01,
            z_contact=z_contact,
            z_press=0.0065,
            w_ee_pos=1.2e3,
            w_ee_ori=4.5e1,
            ori_weights=np.array([2.2, 2.2, 0.3], dtype=float),
            w_posture=1.0e-1,
            w_v=5.0e-2,
            posture_ref_mode="q_nom",
            w_tau=8.0e-4,
            w_w=6.0e-4,
            w_w_soft_limits=2.0,
            w_y=8.0e-4,
            y_q_weights=np.array([0.15, 0.15, 0.15, 0.15, 0.08, 0.08, 0.08], dtype=float),
            y_v_weights=np.array([0.05, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03], dtype=float),
            y_tau_weights=np.array([0.12, 0.12, 0.12, 0.12, 0.08, 0.08, 0.08], dtype=float),
            use_inner_state_reg=True,
            use_inner_tau_reg=True,
            torque_ref_mode="gravity_x0",
            w_tau_soft_limits=1.5,
            w_q_soft_limits=8.0,
            q_soft_limit_margin=0.05,
            w_tau_smooth=0.0,
            w_tangent_pos=3.6e3,
            w_tangent_vel=1.2e3,
            w_plane_z=9.0e2,
            w_vz=3.0e2,
            w_friction_cone=0.0,
            w_unilateral=3.0e1,
            mu=1.0,
            contact_gains=np.array([145.0, 85.0], dtype=float),
            fn_des=22.0,
            w_fn=3.0e1,
            w_wdamp=7.0e1,
            w_wdamp_weights=np.array([1.8, 1.8, 0.3], dtype=float),
            fn_contact_on=1.0,
            fn_contact_off=0.1,
            z_contact_band=0.012,
            max_iters=max_iters,
            mpc_update_steps=1,
            use_feedback_policy=True,
            feedback_gain_scale=0.55,
            max_solver_cost=1.0e8,
            max_tau_raw_inf=3.0e2,
            contact_release_steps=80,
            contact_model=contact_model,
            phase_source=phase_source,
            apply_command_filter=False,
            strict_force_residual_dim=True,
            ff_tau_state_source=ff_tau_state_source,
            ff_cutoff_hz=25.0,
            ff_inverse_actuation_model=True,
            ff_tau_feedback_gain=1.0,
            debug_every=500,
        )
    else:
        cfg = ForceFeedbackMPCConfig(
            horizon=50,
            dt=sim.dt,
            dt_ocp=0.01,
            z_contact=z_contact,
            z_press=0.0080,
            w_ee_pos=1.8e3,
            w_ee_ori=5.5e1,
            ori_weights=np.array([2.4, 2.4, 0.2], dtype=float),
            w_posture=6.0e-2,
            w_v=2.0e-2,
            posture_ref_mode="q_nom",
            w_tau=2.0e-3,
            w_w=3.0e-4,
            w_w_soft_limits=3.0,
            w_y=5.0e-4,
            y_q_weights=np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=float),
            y_v_weights=np.array([0.08, 0.08, 0.08, 0.08, 0.05, 0.05, 0.05], dtype=float),
            y_tau_weights=np.array([0.10, 0.10, 0.10, 0.10, 0.06, 0.06, 0.06], dtype=float),
            use_inner_state_reg=True,
            use_inner_tau_reg=True,
            torque_ref_mode="gravity_x0",
            w_tau_soft_limits=1.5,
            w_q_soft_limits=10.0,
            q_soft_limit_margin=0.10,
            w_tau_smooth=5.0e-2,
            w_tangent_pos=4.8e3,
            w_tangent_vel=1.9e3,
            w_plane_z=5.0e2,
            w_vz=2.0e2,
            w_friction_cone=0.0,
            w_unilateral=3.0e1,
            mu=1.0,
            contact_gains=np.array([150.0, 90.0], dtype=float),
            fn_des=26.0,
            w_fn=4.0e1,
            w_wdamp=8.0e1,
            w_wdamp_weights=np.array([2.0, 2.0, 0.3], dtype=float),
            fn_contact_on=1.0,
            fn_contact_off=0.05,
            z_contact_band=0.012,
            max_iters=max_iters,
            mpc_update_steps=1,
            use_feedback_policy=True,
            feedback_gain_scale=0.60,
            max_tau_raw_inf=2.2e2,
            contact_release_steps=80,
            contact_model=contact_model,
            phase_source=phase_source,
            apply_command_filter=use_command_filter,
            strict_force_residual_dim=True,
            ff_tau_state_source=ff_tau_state_source,
            ff_cutoff_hz=90.0,
            ff_inverse_actuation_model=True,
            debug_every=500,
        )
    print("MPC config created")

    mpc = ForceFeedbackCrocoddylMPC(sim=sim, traj_fn=traj, config=cfg)
    print("MPC initialized")
    align_stats = _check_pin_mj_alignment(sim, mpc, samples=align_check_samples, seed=0)
    if align_stats["samples"] > 0:
        print(
            "EE alignment check: "
            f"rms_pos={align_stats['rms_pos_m']*1e3:.2f}mm "
            f"max_pos={align_stats['max_pos_m']*1e3:.2f}mm | "
            f"rms_rot={align_stats['rms_rot_deg']:.3f}deg "
            f"max_rot={align_stats['max_rot_deg']:.3f}deg"
        )
    print()

    if abs(float(settings["tilt_deg"])) > 1e-12:
        _apply_table_tilt(sim, settings["tilt_deg"])
        obs = sim.get_observation(with_ee=True, with_jacobian=True)
        print(
            f"Applied hidden table tilt: {settings['tilt_deg']:.1f} deg "
            "(controller references remain from nominal flat table)."
        )

    uncertainty = None
    uncertainty_meta = None
    if benchmark_mode:
        unc_cfg = config_for_scenario(scenario, seed=_scenario_seed(scenario))
        if unc_cfg is not None:
            uncertainty = ScenarioUncertaintyInjector(
                dt=float(sim.dt),
                nu=7,
                config=unc_cfg,
                tau_lpf_alpha=float(sim.tau_meas_lpf_alpha),
            )
            uncertainty_meta = uncertainty.meta()
            print(
                "Uncertainty profile enabled: "
                f"a=[{unc_cfg.a_min:.3f},{unc_cfg.a_max:.3f}] "
                f"b=[{unc_cfg.b_min:.3f},{unc_cfg.b_max:.3f}] "
                f"obs_delay={unc_cfg.delta_obs_cycles} "
                f"cmd_delay={unc_cfg.delta_cmd_s*1e3:.2f}ms"
            )

    logger = RunLogger(
        run_name=f"force_feedback_{scenario}",
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
        "fn_pred_raw": [],
        "contact": [],
    }

    torque_scale = settings["torque_scale"]

    def _step_once(k: int, t_now: float, obs_now):
        ctrl_obs = uncertainty.observation_for_controller(obs_now) if uncertainty is not None else obs_now
        tau_cmd = mpc.compute_control(ctrl_obs, t_now)
        if uncertainty is not None:
            tau_applied = uncertainty.command_for_plant(tau_cmd)
        else:
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
        fn_pred_raw = float(info.get("fn_pred_raw", np.nan))

        summary["t"].append(t_next)
        summary["err_tan"].append(err_tan)
        summary["err_3d"].append(err_3d)
        summary["fn_meas"].append(fn_meas)
        summary["fn_pred"].append(fn_pred)
        summary["fn_pred_raw"].append(fn_pred_raw)
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
            fn_pred_raw=fn_pred_raw,
            fn_des=float(cfg.fn_des),
            tau_cmd=np.asarray(tau_cmd, dtype=float).copy(),
            tau_meas=np.asarray(obs_next.tau_meas, dtype=float).copy(),
            tau_meas_filt=np.asarray(obs_next.tau_meas_filt, dtype=float).copy(),
            tau_meas_act=np.asarray(obs_next.tau_meas_act, dtype=float).copy(),
            tau_meas_act_filt=np.asarray(obs_next.tau_meas_act_filt, dtype=float).copy(),
            tau_cmd_sim=np.asarray(obs_next.tau_cmd, dtype=float).copy(),
            tau_act=np.asarray(obs_next.tau_act, dtype=float).copy(),
            tau_constraint=np.asarray(obs_next.tau_constraint, dtype=float).copy(),
            tau_total=np.asarray(obs_next.tau_total, dtype=float).copy(),
            tau_applied=np.asarray(tau_applied, dtype=float).copy(),
            contact=int(in_contact),
            surface_ref=int(surf_ref),
            solver_iters=int(info.get("iters", -1)),
            solver_cost=float(info.get("cost", np.nan)),
            solver_success=int(bool(info.get("ok", False))),
            solver_unstable=int(bool(info.get("unstable", False))),
            solver_solved_now=int(bool(info.get("solved_now", False))),
            solver_policy_idx=int(info.get("policy_idx", -1)),
            tau_des_inf=float(info.get("tau_des_inf", np.nan)),
            tau_meas_state_inf=float(info.get("tau_meas_state_inf", np.nan)),
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
    avg_abs_pos_err = float(np.mean(np.abs(err_tan_arr))) if err_tan_arr.size > 0 else np.nan
    avg_abs_force_err = float(np.mean(np.abs(fn_meas_arr - float(cfg.fn_des)))) if fn_meas_arr.size > 0 else np.nan
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
        tau_meas_definition="tau_total = tau_cmd + tau_act + tau_constraint",
        tau_meas_act_definition="tau_meas_act = tau_cmd + tau_act (constraint torque excluded)",
        fn_pred_definition="Predicted normal-force variable in the OCP contact model (may not equal physical table-normal force under tilt mismatch).",
        contact_definition="in_contact = (fn_meas > 0.5 N)",
        tau_meas_lpf_alpha=float(sim.tau_meas_lpf_alpha),
        benchmark_mode=bool(benchmark_mode),
        uncertainty_profile=uncertainty_meta,
        torque_scale=np.asarray(torque_scale, dtype=float),
        fn_des=float(cfg.fn_des),
        avg_abs_position_err=avg_abs_pos_err,
        avg_abs_force_err=avg_abs_force_err,
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
            "dt_ocp": float(cfg.dt_ocp if cfg.dt_ocp is not None else cfg.dt),
            "z_contact": float(cfg.z_contact),
            "z_press": float(cfg.z_press),
            "w_fn": float(cfg.w_fn),
            "w_w": float(cfg.w_w),
            "w_w_soft_limits": float(cfg.w_w_soft_limits),
            "posture_ref_mode": str(cfg.posture_ref_mode),
            "fn_des": float(cfg.fn_des),
            "circle_radius": float(radius),
            "circle_omega": float(omega),
            "contact_model": str(cfg.contact_model),
            "w_friction_cone": float(cfg.w_friction_cone),
            "w_unilateral": float(cfg.w_unilateral),
            "w_q_soft_limits": float(cfg.w_q_soft_limits),
            "q_soft_limit_margin": float(cfg.q_soft_limit_margin),
            "max_iters": int(cfg.max_iters),
            "phase_source": str(cfg.phase_source),
            "apply_command_filter": bool(cfg.apply_command_filter),
            "ff_cutoff_hz": float(cfg.ff_cutoff_hz),
            "ff_alpha_override": None if cfg.ff_alpha_override is None else float(cfg.ff_alpha_override),
            "ff_use_tau_meas_filt": bool(cfg.ff_use_tau_meas_filt),
            "ff_tau_state_source": str(cfg.ff_tau_state_source),
            "ff_inverse_actuation_model": bool(cfg.ff_inverse_actuation_model),
            "ff_tau_feedback_gain": float(cfg.ff_tau_feedback_gain),
        },
        frame_alignment=align_stats,
    )
    logger.save()

    if save_plots:
        _save_run_plots(logger.path_npz, logger.run_dir, cfg.fn_des)

    print()
    print("=" * 80)
    print("Simulation complete!")
    print("=" * 80)
    print("\nSummary statistics:")
    print(f"  RMS tangential error: {rms_tan:.4f} m")
    print(f"  RMS tangential error (contact phase): {rms_tan_phase:.4f} m")
    print(f"  Avg abs. tangential error: {avg_abs_pos_err:.4f} m")
    print(f"  RMS 3D tracking error: {rms_3d:.4f} m")
    print(f"  Avg abs. force error: {avg_abs_force_err:.2f} N")
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
        "avg_abs_pos_err": avg_abs_pos_err,
        "avg_abs_force_err": avg_abs_force_err,
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
    contact_model: str,
    low_budget: bool,
    mpc_iters: Optional[int],
    use_command_filter: bool,
    align_check_samples: int,
    ff_tau_state_source: str,
    phase_source: str,
    circle_radius: float,
    circle_omega: float,
    benchmark_mode: bool,
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
                    contact_model=contact_model,
                    low_budget=low_budget,
                    mpc_iters=mpc_iters,
                    use_command_filter=use_command_filter,
                    align_check_samples=align_check_samples,
                    ff_tau_state_source=ff_tau_state_source,
                    phase_source=phase_source,
                    circle_radius=circle_radius,
                    circle_omega=circle_omega,
                    benchmark_mode=benchmark_mode,
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
        contact_model=contact_model,
        low_budget=low_budget,
        mpc_iters=mpc_iters,
        use_command_filter=use_command_filter,
        align_check_samples=align_check_samples,
        ff_tau_state_source=ff_tau_state_source,
        phase_source=phase_source,
        circle_radius=circle_radius,
        circle_omega=circle_omega,
        benchmark_mode=benchmark_mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS + ("tilted",),
        default="flat",
        help="Evaluation scenario.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run flat/tilted_5/tilted_10/tilted_15/actuation_uncertainty.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Run without MuJoCo viewer.")
    parser.add_argument("--time", type=float, default=12.0, help="Total simulation time [s].")
    parser.add_argument("--results-dir", type=Path, default=Path("results/force_feedback_eval"), help="Output folder.")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--contact-model", choices=("normal_1d", "point3d"), default="normal_1d")
    parser.add_argument(
        "--phase-source",
        choices=("trajectory", "force_latch"),
        default="trajectory",
        help="Contact phase selection mode.",
    )
    parser.add_argument(
        "--ff-tau-state-source",
        choices=("auto", "tau_meas_act_filt", "tau_meas_act", "tau_cmd", "tau_meas_filt", "tau_meas", "tau_total"),
        default="tau_meas_act_filt",
        help="Torque-state source used by force-feedback augmentation.",
    )
    parser.add_argument("--low-budget", action="store_true", help="Use low DDP iteration budget.")
    parser.add_argument("--mpc-iters", type=int, default=None, help="Override DDP max iterations per MPC solve.")
    parser.add_argument("--circle-radius", type=float, default=0.10, help="Circle radius [m].")
    parser.add_argument("--circle-omega", type=float, default=1.5, help="Circle angular velocity [rad/s].")
    parser.add_argument(
        "--use-command-filter",
        action="store_true",
        help="Enable command smoothing/trust/rate filtering in _safe_tau.",
    )
    parser.add_argument(
        "--align-check-samples",
        type=int,
        default=16,
        help="Number of random q samples for MuJoCo↔Pinocchio EE alignment check (0 disables).",
    )
    parser.add_argument(
        "--benchmark-mode",
        dest="benchmark_mode",
        action="store_true",
        help="Enable benchmark protocol (default).",
    )
    parser.add_argument(
        "--no-benchmark-mode",
        dest="benchmark_mode",
        action="store_false",
        help="Disable benchmark protocol and use development settings.",
    )
    parser.set_defaults(benchmark_mode=True)
    args = parser.parse_args()

    main(
        scenario=args.scenario,
        all_scenarios=args.all_scenarios,
        no_viewer=args.no_viewer,
        total_time=args.time,
        results_dir=args.results_dir,
        no_plots=args.no_plots,
        contact_model=args.contact_model,
        low_budget=args.low_budget,
        mpc_iters=args.mpc_iters,
        use_command_filter=args.use_command_filter,
        align_check_samples=args.align_check_samples,
        ff_tau_state_source=args.ff_tau_state_source,
        phase_source=args.phase_source,
        circle_radius=args.circle_radius,
        circle_omega=args.circle_omega,
        benchmark_mode=args.benchmark_mode,
    )
