#!/bin/bash
# Run the fixed classical MPC test

set -e

cd /home/omar/franka-force-feedback-mpc

echo "════════════════════════════════════════════════════════════════════"
echo "Classical Panda MPC - Fixed Version"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Activating conda environment..."

# Activate conda
source /home/omar/miniforge3/etc/profile.d/conda.sh
conda activate franka-mpc

# Set Python path
export PYTHONPATH=/home/omar/franka-force-feedback-mpc:$PYTHONPATH

echo "✓ Environment ready"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "TRAJECTORY FIX:"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  ✓ trajectories.py: Added ee_start parameter"
echo "    Robot now starts from ACTUAL position instead of hardcoded value"
echo ""
echo "  ✓ test_crocoddyl.py: Passes obs.ee_pos.copy() to trajectory"  
echo "    Robot starts at [-0.3, 0, 0.633] (not stuck at wrong reference)"
echo ""
echo "EXPECTED BEHAVIOR:"
echo "  • 0-2s:   Smooth descent from 0.633m to 0.0495m (table contact)"
echo "  • 2-12s:  Smooth circular motion maintaining contact force ~10-20N"
echo "  • No jittering, no standing still, no Z oscillation"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Running test... (watch the MuJoCo viewer)"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python src/mpc/test_crocoddyl.py

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "Test complete!"
echo "════════════════════════════════════════════════════════════════════"
