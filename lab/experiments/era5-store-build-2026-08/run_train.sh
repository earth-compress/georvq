#!/bin/bash
# Launch the supervised train-split build (see watchdog_train.sh).
cd /data/ERA5-samples
setsid nohup bash watchdog_train.sh >> build_train_watchdog.log 2>&1 &
echo "watchdog pid $!"
