#!/bin/bash
# Supervise the train-split build: restart it on any exit until it
# completes, and kill it if the log goes quiet for 30 min (a hung remote
# read survives every in-process guard; the builder resumes from its
# last committed window, so kill -9 is safe).
cd /data/ERA5-samples
PY=~/miniconda3/envs/ring-space/bin/python
while :; do
  $PY build_store_era5.py --split train --n-windows 200 \
      --cubes-per-window 16 --workers 24 --verify >> build_train.log 2>&1 &
  pid=$!
  echo "$(date -u +%FT%TZ) builder started pid $pid"
  while kill -0 $pid 2>/dev/null; do
    sleep 300
    age=$(( $(date +%s) - $(stat -c %Y build_train.log) ))
    if [ $age -gt 1800 ]; then
      echo "$(date -u +%FT%TZ) log stale ${age}s, killing $pid"
      kill -9 $pid
      sleep 5
      break
    fi
  done
  wait $pid; rc=$?
  if [ $rc -eq 0 ]; then
    echo "$(date -u +%FT%TZ) builder finished, rc=0"
    break
  fi
  echo "$(date -u +%FT%TZ) builder exited rc=$rc, restarting in 60s"
  sleep 60
done
