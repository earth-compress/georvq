#!/bin/bash
# Run both ERA5 store builds outside Slurm (network-bound, ~2 cores of
# actual CPU). Resumable: rerun after any interruption and each split
# continues from its last committed window.
cd /data/ERA5-samples
PY=~/miniconda3/envs/ring-space/bin/python
setsid nohup $PY build_store_era5.py --split train --n-windows 200 \
    --cubes-per-window 16 --workers 6 --verify \
    >> build_train.log 2>&1 &
echo "train pid $!"
setsid nohup $PY build_store_era5.py --split val --n-windows 20 \
    --cubes-per-window 16 --workers 6 --verify \
    >> build_val.log 2>&1 &
echo "val pid $!"
