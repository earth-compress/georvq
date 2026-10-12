#!/bin/bash
# Bulk ERA5 cube store build, train split. Resumable: rerun on timeout
# and it continues from the last committed window.

#SBATCH --job-name=era5-store-train
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

~/miniconda3/envs/ring-space/bin/python \
    /data/ERA5-samples/build_store_era5.py \
    --split train --n-windows 200 --cubes-per-window 16 --verify

echo "Finished $(date)"
