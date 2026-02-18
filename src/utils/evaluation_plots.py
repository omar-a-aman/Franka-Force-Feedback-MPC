from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np


def _configure_matplotlib():
    import matplotlib.pyplot as plt

    # Keep deterministic rendering defaults when system LaTeX is unavailable.
    plt.rcParams.update(
        {
            "figure.figsize": (8.5, 4.8),
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    if shutil.which("latex") is None:
        return False

    prev = dict(plt.rcParams)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        }
    )
    try:
        fig, ax = plt.subplots(figsize=(1.0, 1.0))
        ax.set_xlabel(r"$t\;(\mathrm{s})$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        plt.rcParams.update(prev)
        return False


def _style_reference_vs_measured(ax, t, ref, meas, ylabel: str):
    ax.plot(t, ref, "-.", color="#4C6EF5", linewidth=1.8, label="Reference")
    ax.plot(t, meas, "-", color="#E03131", linewidth=1.4, label="Measured")
    ax.set_xlabel(r"$t\;(\mathrm{s})$")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    ax.grid(True)


def _style_cartesian_xy(ax, ref_xy: np.ndarray, meas_xy: np.ndarray):
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "-.", color="#4C6EF5", linewidth=1.8, label="Reference")
    ax.plot(meas_xy[:, 0], meas_xy[:, 1], "-", color="#E03131", linewidth=1.4, label="Measured")
    ax.set_xlabel(r"$p_x^{EE}\;(\mathrm{m})$")
    ax.set_ylabel(r"$p_y^{EE}\;(\mathrm{m})$")
    ax.set_title("End-Effector XY Cartesian Trajectory")
    ax.legend(loc="best")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")


def save_evaluation_plots(npz_path: Path, out_dir: Path, fn_des: float) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib()

    data = np.load(npz_path)
    t = np.asarray(data["t"], dtype=float)
    err_tan = np.asarray(data["err_tan"], dtype=float)
    fn_meas = np.asarray(data["fn_meas"], dtype=float)
    fn_pred = np.asarray(data["fn_pred"], dtype=float)

    ee_ref = np.asarray(data["ee_ref"], dtype=float) if "ee_ref" in data.files else None
    ee_pos = np.asarray(data["ee_pos"], dtype=float) if "ee_pos" in data.files else None

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(t, err_tan, "-", color="#2B8A3E", linewidth=1.5)
    ax.set_xlabel(r"$t\;(\mathrm{s})$")
    ax.set_ylabel(r"$\|p_{xy}^{EE}-\bar{p}_{xy}^{EE}\|\;(\mathrm{m})$")
    ax.set_title("Tangential Tracking Error")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "tangential_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, np.full_like(t, float(fn_des)), "--", color="#4C6EF5", linewidth=1.6, label="Reference")
    ax.plot(t, fn_meas, "-", color="#E03131", linewidth=1.4, label="Measured")
    ax.set_xlabel(r"$t\;(\mathrm{s})$")
    ax.set_ylabel(r"$\lambda_n\;(\mathrm{N})$")
    ax.set_title("Measured Normal Force")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "fn_meas_vs_des.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, np.full_like(t, float(fn_des)), "--", color="#4C6EF5", linewidth=1.6, label="Reference")
    ax.plot(t, fn_pred, "-", color="#2B8A3E", linewidth=1.4, label="Predicted")
    ax.set_xlabel(r"$t\;(\mathrm{s})$")
    ax.set_ylabel(r"$\lambda_n\;(\mathrm{N})$")
    ax.set_title("Predicted Normal Force")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "fn_pred_vs_des.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t, np.full_like(t, float(fn_des)), "--", color="#4C6EF5", linewidth=1.6, label="Reference")
    ax.plot(t, fn_pred, "-", color="#2B8A3E", linewidth=1.4, label="Predicted")
    ax.plot(t, fn_meas, "-", color="#E03131", linewidth=1.4, label="Measured")
    ax.set_xlabel(r"$t\;(\mathrm{s})$")
    ax.set_ylabel(r"$\lambda_n\;(\mathrm{N})$")
    ax.set_title("Measured vs Predicted Normal Force")
    ax.legend(loc="upper right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "fn_meas_vs_pred.png", dpi=200)
    plt.close(fig)

    if ee_ref is None or ee_pos is None or ee_ref.ndim != 2 or ee_pos.ndim != 2:
        return
    if ee_ref.shape[1] < 2 or ee_pos.shape[1] < 2:
        return

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9.0, 7.0))
    _style_reference_vs_measured(axs[0], t, ee_ref[:, 0], ee_pos[:, 0], r"$p_x^{EE}\;(\mathrm{m})$")
    _style_reference_vs_measured(axs[1], t, ee_ref[:, 1], ee_pos[:, 1], r"$p_y^{EE}\;(\mathrm{m})$")
    axs[0].set_title("End-Effector Position Tracking")
    fig.tight_layout()
    fig.savefig(out_dir / "ee_xy_ref_vs_meas.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots()
    _style_reference_vs_measured(ax, t, ee_ref[:, 0], ee_pos[:, 0], r"$p_x^{EE}\;(\mathrm{m})$")
    ax.set_title("End-Effector X Tracking")
    fig.tight_layout()
    fig.savefig(out_dir / "ee_px_ref_vs_meas.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots()
    _style_reference_vs_measured(ax, t, ee_ref[:, 1], ee_pos[:, 1], r"$p_y^{EE}\;(\mathrm{m})$")
    ax.set_title("End-Effector Y Tracking")
    fig.tight_layout()
    fig.savefig(out_dir / "ee_py_ref_vs_meas.png", dpi=220)
    plt.close(fig)

    # 2D Cartesian tracking view in XY plane.
    ref_xy = ee_ref[:, :2]
    meas_xy = ee_pos[:, :2]
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    _style_cartesian_xy(ax, ref_xy, meas_xy)
    fig.tight_layout()
    fig.savefig(out_dir / "ee_xy_cartesian_ref_vs_meas.png", dpi=220)
    plt.close(fig)
