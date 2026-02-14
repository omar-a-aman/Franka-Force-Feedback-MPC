# Franka Force-Feedback MPC - Classical MPC Fixed

## Status: ✅ COMPLETE

All critical bugs in the classical MPC controller have been identified and fixed. The controller now:
- ✅ Smoothly approaches the table
- ✅ Maintains stable contact with proper force
- ✅ Follows circular trajectory without oscillation
- ✅ Handles torque limits and safety constraints

## Quick Start

### Run the fixed controller:
```bash
python src/mpc/test_crocoddyl.py
```

### Verify all fixes:
```bash
python verify_fixes.py
```

### See what changed:
```bash
python show_changes.py
```

## Documentation

- **[README_FIXES.md](README_FIXES.md)** - Quick reference of what was fixed
- **[FIXES_COMPLETE.md](FIXES_COMPLETE.md)** - Complete analysis of all 9 issues
- **[FIXES_DETAILED.md](FIXES_DETAILED.md)** - Technical deep-dive into each bug
- **[CHECKLIST_VISUAL.txt](CHECKLIST_VISUAL.txt)** - Visual before/after comparison

## The 9 Critical Bugs Fixed

| # | Issue | Fix |
|---|-------|-----|
| 1 | Torque smoothing non-functional | Proper control regularization |
| 2 | Z-press logic inverted | Correct penetration depth interpretation |
| 3 | Posture control too weak | Increased kp from 50 to 100 |
| 4 | Velocity damping too strong | Reduced w_v from 60 to 1.0 |
| 5 | Torque smoothing too strong | Reduced w_tau_smooth from 6 to 1.0 |
| 6 | Z velocity not damped | Separate XY/Z tracking with proper masks |
| 7 | Joint-unaware velocity damping | Per-joint weights [1,1,1,1,0.3,0.3,0.3] |
| 8 | Unreachable circle center | Moved from behind (bad) to forward (reachable) |
| 9 | Unbalanced position tracking | Z weight = 0.5x during approach (gravity-assisted) |

## Files Modified

- ✅ `src/mpc/crocoddyl_classical.py` - Completely rewritten with correct cost formulation
- ✅ `src/mpc/test_crocoddyl.py` - Improved test harness and logging
- ✅ `src/tasks/trajectories.py` - Better documentation

## Next Steps

This is the **baseline** classical MPC (no force feedback). Next:

1. Verify stability with disturbances
2. Implement force-feedback MPC (augmented state with filtered torque)
3. Compare controllers on same task
4. Measure improvement from force feedback

## Project Timeline

Phase 1: ✅ **Classical MPC (COMPLETE)**
Phase 2: ⏳ **Force-Feedback MPC (NEXT)**  
Phase 3: ⏳ **Comparison & Analysis**
