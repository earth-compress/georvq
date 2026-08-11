#!/bin/bash
# Ceiling sweep v2: remaining limits on GPU 1 (512 runs to completion first;
# 1024 runs separately on GPU 0). Waits for GPU 1 to free, then for enough
# indexed chips, per limit.
set -u
WT=/home/ali/projects/georvq/.claude/worktrees/tessera-source
PY=/home/ali/miniconda3/envs/ring-space/bin
STORE=/data/tessera-samples/chips_512

while pgrep -f "run_tag=tessera_ceil_512" > /dev/null; do sleep 120; done
echo "=== GPU1 free (512 run finished) $(date) ==="

for LIM in 2048 4096 8192 16384; do
  NEED=$(( LIM * 100 / 96 + 64 ))
  while :; do
    N=$($PY/python -c "import pandas as pd;print(len(pd.read_parquet('$STORE/index.parquet')))" 2>/dev/null || echo 0)
    [ "$N" -ge "$NEED" ] && break
    [ -f "$STORE/manifest.json" ] && break
    echo "limit $LIM: waiting for $NEED indexed chips (have $N)"
    sleep 300
  done
  echo "=== limit $LIM start (indexed: $N) $(date) ==="
  PYTHONPATH=$WT $PY/georvq-train-stage1 -c "$WT/configs/tessera_stage1_probe.yaml" --gpu 1 \
      data.limit_chips=$LIM log.run_tag=tessera_ceil_$LIM \
      log.out_dir=/home/ali/projects/ring-space/checkpoints
  echo "=== limit $LIM exit $? $(date) ==="
done
echo "SWEEP COMPLETE"
