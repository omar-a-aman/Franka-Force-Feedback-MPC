# src/utils/logging.py
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np


def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion to JSON-serializable."""
    if x is None:
        return None
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    # fallback
    return str(x)


class RunLogger:
    """
    Simple run logger:
      - log(step_data) each iteration
      - save() at end -> npz + csv + meta.json
    """

    def __init__(
        self,
        run_name: str,
        results_dir: Path | str = "results",
        notes: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ):
        self.results_dir = Path(results_dir)
        self.logs_dir = self.results_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.logs_dir / f"{stamp}_{run_name}"
        if self.run_dir.exists() and not overwrite:
            raise FileExistsError(f"Run dir exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._rows: list[dict[str, Any]] = []

        self.meta: Dict[str, Any] = {
            "run_name": run_name,
            "timestamp": stamp,
            "notes": _to_jsonable(notes or {}),
        }

    @property
    def path_npz(self) -> Path:
        return self.run_dir / "data.npz"

    @property
    def path_csv(self) -> Path:
        return self.run_dir / "data.csv"

    @property
    def path_meta(self) -> Path:
        return self.run_dir / "meta.json"

    def log(self, **kwargs: Any) -> None:
        """
        Log one timestep. You can pass numpy arrays; they'll be saved in NPZ.
        For CSV we flatten scalars + vector fields.
        """
        self._rows.append(kwargs)

    def set_meta(self, **kwargs: Any) -> None:
        self.meta.update(_to_jsonable(kwargs))

    def save(self) -> None:
        if not self._rows:
            return

        # -------- NPZ: store arrays as stacked where possible --------
        keys = sorted(self._rows[0].keys())
        out_npz: Dict[str, Any] = {}

        for k in keys:
            vals = [r.get(k, None) for r in self._rows]

            # try stack
            if isinstance(vals[0], np.ndarray):
                try:
                    out_npz[k] = np.stack(vals, axis=0)
                    continue
                except Exception:
                    pass

            # try float array
            try:
                out_npz[k] = np.array(vals, dtype=float)
            except Exception:
                out_npz[k] = np.array([_to_jsonable(v) for v in vals], dtype=object)

        np.savez_compressed(self.path_npz, **out_npz)

        # -------- CSV: flatten scalars + small vectors --------
        # We’ll export scalars and vectors (len<=10) as separate columns.
        header: list[str] = []
        rows_csv: list[list[Any]] = []

        # build header once
        sample = self._rows[0]
        for k in keys:
            v = sample.get(k, None)
            if np.isscalar(v) or v is None:
                header.append(k)
            elif isinstance(v, np.ndarray) and v.ndim == 1 and v.size <= 10:
                header.extend([f"{k}[{i}]" for i in range(v.size)])
            else:
                # too big/structured for CSV
                header.append(k)

        for r in self._rows:
            row_out: list[Any] = []
            for k in keys:
                v = r.get(k, None)
                if np.isscalar(v) or v is None:
                    row_out.append(v)
                elif isinstance(v, np.ndarray) and v.ndim == 1 and v.size <= 10:
                    row_out.extend(v.tolist())
                else:
                    row_out.append(_to_jsonable(v))
            rows_csv.append(row_out)

        import csv
        with open(self.path_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_csv)

        # -------- meta --------
        with open(self.path_meta, "w") as f:
            json.dump(self.meta, f, indent=2)
