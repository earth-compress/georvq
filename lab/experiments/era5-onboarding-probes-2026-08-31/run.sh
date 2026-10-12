#!/bin/bash
PY=/home/ali/miniconda3/envs/ring-space/bin/python
cd /home/ali/projects/georvq/.claude/worktrees/era5-model
OUT=/home/ali/.claude/jobs/0e80fb90/tmp/probes
for v in t z u v q; do
  $PY analysis/data_probe.py --store /data/ERA5-samples/val --source era5:$v \
      --chips 40 --entropy-bands 37 > $OUT/data_probe_$v.txt 2>&1
  $PY analysis/pyramid_sizing.py --store /data/ERA5-samples/val --source era5:$v \
      --chips 40 --window 64 --grids 1,4,16 > $OUT/pyramid_$v.txt 2>&1
  $PY analysis/pyramid_sizing.py --store /data/ERA5-samples/val --source era5:$v+std \
      --chips 40 --window 64 --grids 1,4,16 > $OUT/pyramid_${v}_std.txt 2>&1
  echo "$v done"
done
echo ALL_DONE
