"""
Verification script to check that all fixes are in place
Run: python verify_fixes.py
"""

import sys
from pathlib import Path

def check_file_content(filepath, search_strings, description):
    """Check if file contains expected content"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        found = all(s in content for s in search_strings)
        status = "✓" if found else "✗"
        print(f"{status} {description}")
        return found
    except FileNotFoundError:
        print(f"✗ {description} (file not found)")
        return False

def main():
    print("=" * 80)
    print("VERIFICATION: Classical MPC Fixes")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # Check 1: crocoddyl_classical.py has new config
    all_ok &= check_file_content(
        "src/mpc/crocoddyl_classical.py",
        ["kp_posture: float = 100.0", "w_v: float = 1.0", "v_damp_weights"],
        "✓ Config: Strong posture (kp=100), light velocity damping (w_v=1)"
    )
    
    # Check 2: crocoddyl_classical.py has proper cost weighting
    all_ok &= check_file_content(
        "src/mpc/crocoddyl_classical.py",
        ["w_plane_z: float = 2.0e3", "w_vz: float = 1.0e3", "w_tau_smooth: float = 1.0"],
        "✓ Costs: Strong Z control (2000), strong Z damping (1000), weak smoothing (1.0)"
    )
    
    # Check 3: crocoddyl_classical.py has per-joint damping
    all_ok &= check_file_content(
        "src/mpc/crocoddyl_classical.py",
        ["v_damp_weights: np.ndarray", "[1.0, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3]"],
        "✓ Per-joint damping: strong arm [1.0], light wrist [0.3]"
    )
    
    # Check 4: crocoddyl_classical.py has separate XY/Z tracking
    all_ok &= check_file_content(
        "src/mpc/crocoddyl_classical.py",
        ["p_ref_xy = p_ref.copy()", "p_ref_xy[2] = 0", "act_z = crocoddyl.ActivationModelWeightedQuad(np.array([0.0, 0.0, 1.0]"],
        "✓ Tracking: Separate XY and Z costs with proper activation masks"
    )
    
    # Check 5: test_crocoddyl.py has better setup
    all_ok &= check_file_content(
        "src/mpc/test_crocoddyl.py",
        ["center_x = 0.4", "center_y = 0.0", "total_time = 12.0"],
        "✓ Test: Circle at reachable position (x=0.4), 12s runtime"
    )
    
    # Check 6: test_crocoddyl.py has logging
    all_ok &= check_file_content(
        "src/mpc/test_crocoddyl.py",
        ["log_data", "ee_pos_x", "f_contact_normal"],
        "✓ Logging: Trajectory data collection for analysis"
    )
    
    # Check 7: trajectories.py is well documented
    all_ok &= check_file_content(
        "src/tasks/trajectories.py",
        ["Phase 1 (0 to t_approach)", "Phase 2 (t_approach to inf)"],
        "✓ Documentation: Clear trajectory phases"
    )
    
    # Check 8: Documentation files exist
    for doc in ["README_FIXES.md", "FIXES_DETAILED.md"]:
        exists = Path(doc).exists()
        status = "✓" if exists else "✗"
        print(f"{status} {doc} exists")
        all_ok &= exists
    
    print()
    print("=" * 80)
    if all_ok:
        print("✅ ALL FIXES VERIFIED - Ready to run!")
        print()
        print("To test the fixed controller:")
        print("  python src/mpc/test_crocoddyl.py")
        print()
        return 0
    else:
        print("❌ Some fixes missing - check implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
