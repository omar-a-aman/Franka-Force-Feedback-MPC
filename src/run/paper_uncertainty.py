from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from src.sim.franka_sim import Observation


@dataclass
class PaperUncertaintyConfig:
    """
    Uncertainty model used for paper-mode protocol (Eq. 21 style):
      - actuation gain/bias uncertainty on delayed command torque
      - delayed/noisy state measurements
      - noisy delayed torque measurements
    """

    a_min: float = 0.95
    a_max: float = 1.05
    b_min: float = -0.1
    b_max: float = 0.1
    sigma_q: float = 5.0e-4
    sigma_dq: float = 2.0e-3
    sigma_tau: float = 5.0e-2
    delta_obs_cycles: int = 2
    delta_cmd_s: float = 1.0e-3
    seed: int = 0


def config_for_scenario(scenario: str, seed: int = 0) -> Optional[PaperUncertaintyConfig]:
    """
    Shared paper-mode uncertainty presets.
    Returns None for scenarios without injected uncertainty.
    """
    name = str(scenario).strip().lower()
    if name == "actuation_uncertainty":
        # Matches the paper-style uncertainty stress test:
        # delayed command, delayed/noisy observation, uncertain actuation map.
        return PaperUncertaintyConfig(
            a_min=0.95,
            a_max=1.05,
            b_min=-0.10,
            b_max=0.10,
            sigma_q=5.0e-4,
            sigma_dq=2.0e-3,
            sigma_tau=5.0e-2,
            delta_obs_cycles=2,
            delta_cmd_s=1.0e-3,
            seed=int(seed),
        )
    return None


def _copy_optional(arr):
    if arr is None:
        return None
    return np.asarray(arr, dtype=float).copy()


def _copy_observation(obs: Observation) -> Observation:
    return replace(
        obs,
        q=np.asarray(obs.q, dtype=float).copy(),
        dq=np.asarray(obs.dq, dtype=float).copy(),
        tau_meas=np.asarray(obs.tau_meas, dtype=float).copy(),
        tau_meas_filt=np.asarray(obs.tau_meas_filt, dtype=float).copy(),
        tau_meas_act=np.asarray(obs.tau_meas_act, dtype=float).copy(),
        tau_meas_act_filt=np.asarray(obs.tau_meas_act_filt, dtype=float).copy(),
        tau_cmd=np.asarray(obs.tau_cmd, dtype=float).copy(),
        tau_act=np.asarray(obs.tau_act, dtype=float).copy(),
        tau_constraint=np.asarray(obs.tau_constraint, dtype=float).copy(),
        tau_total=np.asarray(obs.tau_total, dtype=float).copy(),
        tau_bias=np.asarray(obs.tau_bias, dtype=float).copy(),
        f_contact_world=np.asarray(obs.f_contact_world, dtype=float).copy(),
        ee_pos=_copy_optional(obs.ee_pos),
        ee_quat=_copy_optional(obs.ee_quat),
        J_pos=_copy_optional(obs.J_pos),
        J_rot=_copy_optional(obs.J_rot),
        ee_vel=_copy_optional(obs.ee_vel),
    )


class PaperUncertaintyInjector:
    def __init__(
        self,
        dt: float,
        nu: int,
        config: PaperUncertaintyConfig,
        tau_lpf_alpha: float = 0.2,
    ):
        self.dt = float(max(dt, 1.0e-9))
        self.nu = int(nu)
        self.cfg = config
        self.rng = np.random.default_rng(int(config.seed))

        self.a = float(self.rng.uniform(float(config.a_min), float(config.a_max)))
        self.b = float(self.rng.uniform(float(config.b_min), float(config.b_max)))

        self.obs_delay_steps = int(max(config.delta_obs_cycles, 0))
        self.cmd_delay_steps = int(max(np.round(float(config.delta_cmd_s) / self.dt), 0))

        self._obs_hist: deque[Observation] = deque(maxlen=self.obs_delay_steps + 1)
        self._cmd_hist: deque[np.ndarray] = deque(maxlen=self.cmd_delay_steps + 1)
        for _ in range(self.cmd_delay_steps + 1):
            self._cmd_hist.append(np.zeros(self.nu, dtype=float))

        self._tau_hat_filt = np.zeros(self.nu, dtype=float)
        self._tau_lpf_alpha = float(np.clip(tau_lpf_alpha, 0.0, 1.0))

    def meta(self) -> dict:
        return {
            "a": float(self.a),
            "b": float(self.b),
            "sigma_q": float(self.cfg.sigma_q),
            "sigma_dq": float(self.cfg.sigma_dq),
            "sigma_tau": float(self.cfg.sigma_tau),
            "delta_obs_cycles": int(self.obs_delay_steps),
            "delta_cmd_steps": int(self.cmd_delay_steps),
            "delta_cmd_s": float(self.cfg.delta_cmd_s),
            "seed": int(self.cfg.seed),
        }

    def _delayed_command(self) -> np.ndarray:
        return np.asarray(self._cmd_hist[0], dtype=float).reshape(self.nu)

    def _sample_tau_hat(self) -> np.ndarray:
        noise = self.rng.normal(0.0, float(self.cfg.sigma_tau), size=self.nu)
        return self.a * self._delayed_command() + self.b + noise

    def observation_for_controller(self, obs: Observation) -> Observation:
        obs_copy = _copy_observation(obs)
        if len(self._obs_hist) == 0:
            for _ in range(self.obs_delay_steps + 1):
                self._obs_hist.append(_copy_observation(obs_copy))
        else:
            self._obs_hist.append(obs_copy)

        delayed = _copy_observation(self._obs_hist[0])

        delayed.q = delayed.q + self.rng.normal(0.0, float(self.cfg.sigma_q), size=self.nu)
        delayed.dq = delayed.dq + self.rng.normal(0.0, float(self.cfg.sigma_dq), size=self.nu)

        tau_hat = self._sample_tau_hat()
        self._tau_hat_filt = (1.0 - self._tau_lpf_alpha) * self._tau_hat_filt + self._tau_lpf_alpha * tau_hat

        delayed.tau_meas = tau_hat.copy()
        delayed.tau_meas_filt = self._tau_hat_filt.copy()
        delayed.tau_meas_act = tau_hat.copy()
        delayed.tau_meas_act_filt = self._tau_hat_filt.copy()
        return delayed

    def command_for_plant(self, tau_cmd_nominal: np.ndarray) -> np.ndarray:
        tau_cmd_nominal = np.asarray(tau_cmd_nominal, dtype=float).reshape(self.nu)
        self._cmd_hist.append(tau_cmd_nominal.copy())
        return self._sample_tau_hat()
