#!/bin/bash
# Ceiling sweep v3: final rungs on GPU 1 after the 2048 run finishes.
# pool.size=24000 (150 GiB) so it coexists with the 4096 run on GPU 0.
set -u
WT=/home/ali/projects/georvq/.claude/worktrees/tessera-source
PY=/home/ali/miniconda3/envs/ring-space/bin
STORE=/data/tessera-samples/chips_512

while pgrep -f "run_tag=tessera_ceil_2048" > /dev/null; do sleep 120; done
echo "=== GPU1 free (2048 run finished) $(date) ==="

for LIM in 8192 16384; do
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
      data.limit_chips=$LIM log.run_tag=tessera_ceil_$LIM data.pool.size=24000 \
      log.out_dir=/home/ali/projects/ring-space/checkpoints
  echo "=== limit $LIM exit $? $(date) ==="
done
echo "SWEEP3 COMPLETE"
