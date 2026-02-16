# `scripts/` - Helper Shell Scripts

## Files

- `run_clean.sh`
  - runs a command in a clean Python environment context by unsetting
    `PYTHONPATH` and enabling `PYTHONNOUSERSITE=1`.

Example:

```bash
./scripts/run_clean.sh python3 src/run/run_classical.py --scenario flat --no-viewer
```
