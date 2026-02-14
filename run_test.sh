#!/bin/bash
cd /home/omar/franka-force-feedback-mpc
source /home/omar/miniforge3/etc/profile.d/conda.sh
conda activate franka-mpc
export PYTHONPATH=/home/omar/franka-force-feedback-mpc:$PYTHONPATH
exec python src/mpc/test_crocoddyl.py "$@"
