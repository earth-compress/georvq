#!/bin/bash
# Extend the tessera chip store after the initial build finishes, sized so
# /mnt/raid keeps at least RESERVE_GIB free. Resumable: uses --append.
set -u
PY=/home/ali/miniconda3/envs/ring-space/bin/python
STORE=/data/tessera-samples/chips_512
RESERVE_GIB=2560          # keep >= 2.5 TiB free on /mnt/raid
CHIP_MIB=33               # 32 MiB chip + 0.5 MiB scales sidecar
MAX_EXTRA=32768           # sanity cap

while pgrep -f "build_store_tessera.py --n-chips 16384" > /dev/null; do sleep 600; done
[ -f "$STORE/manifest.json" ] || { echo "initial build did not finalize; aborting"; exit 1; }

AVAIL_GIB=$(df -BG --output=avail /mnt/raid | tail -1 | tr -dc 0-9)
EXTRA=$(( (AVAIL_GIB - RESERVE_GIB) * 1024 / CHIP_MIB ))
[ "$EXTRA" -gt "$MAX_EXTRA" ] && EXTRA=$MAX_EXTRA
if [ "$EXTRA" -le 0 ]; then echo "no headroom above reserve; not extending"; exit 0; fi

echo "=== extending store by $EXTRA chips (avail ${AVAIL_GIB}G, reserve ${RESERVE_GIB}G) $(date) ==="
cd /data/tessera-samples
exec $PY build_store_tessera.py --n-chips "$EXTRA" --workers 12 --append \
    --out-dir "$STORE"
