# src/utils/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_sanding_run(
    npz_path: Path | str,
    out_dir: Path | str = "results/plots",
    run_tag: Optional[str] = None,
) -> Path:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    t = data["t"] if "t" in data.files else np.arange(len(data["fn"])) * float(data["dt"][0])
    fn = data["fn"]
    phase = data["phase"].astype(int) if "phase" in data.files else None

    out_dir = Path(out_dir)
    tag = run_tag or npz_path.parent.name
    run_dir = ensure_dir(out_dir / tag)

    # --- 1) Normal force ---
    plt.figure()
    plt.plot(t, fn)
    plt.xlabel("time [s]")
    plt.ylabel("fn [N]")
    plt.title("Contact normal force (on EE)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(run_dir / "fn.png", dpi=160)
    plt.close()

    # --- 2) EE position ---
    if "p" in data.files:
        p = data["p"]
        plt.figure()
        plt.plot(t, p[:, 0], label="x")
        plt.plot(t, p[:, 1], label="y")
        plt.plot(t, p[:, 2], label="z")
        plt.xlabel("time [s]")
        plt.ylabel("position [m]")
        plt.title("End-effector position")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / "ee_pos.png", dpi=160)
        plt.close()

    # --- 3) Orientation error norm ---
    if "eR" in data.files:
        eR = data["eR"]
        eR_norm = np.linalg.norm(eR, axis=1)
        plt.figure()
        plt.plot(t, eR_norm)
        plt.xlabel("time [s]")
        plt.ylabel("|eR|")
        plt.title("Orientation error norm")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / "eR_norm.png", dpi=160)
        plt.close()

    # --- 4) Commanded F (optional) ---
    if "F_cmd" in data.files:
        F = data["F_cmd"]
        plt.figure()
        plt.plot(t, F[:, 0], label="Fx")
        plt.plot(t, F[:, 1], label="Fy")
        plt.plot(t, F[:, 2], label="Fz")
        plt.xlabel("time [s]")
        plt.ylabel("commanded wrench force [N]")
        plt.title("Commanded task-space force")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / "F_cmd.png", dpi=160)
        plt.close()

    # --- 5) Phase (optional) ---
    if phase is not None:
        plt.figure()
        plt.plot(t, phase)
        plt.xlabel("time [s]")
        plt.ylabel("phase")
        plt.title("Phase index")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / "phase.png", dpi=160)
        plt.close()

    return run_dir
