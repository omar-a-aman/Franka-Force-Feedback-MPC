# `src/utils/` - Logging and Plotting Utilities

This folder provides reusable utilities for run outputs.

## Files

- `logging.py`
  - `RunLogger` writes:
    - `data.npz` (dense arrays),
    - `data.csv` (flattened scalar/vector columns),
    - `meta.json` (run metadata and config summary)
- `evaluation_plots.py`
  - standardized per-run plots:
    - tangential error,
    - measured vs desired force,
    - predicted vs desired force,
    - measured vs predicted force,
    - EE x/y reference-vs-measured tracking
- `plotting.py`
  - legacy plotting helper for older scripts

`evaluation_plots.py` is the plotting utility used by current run scripts.
