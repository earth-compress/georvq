#!/bin/bash
# ERA5 cube store build, val split (held-out years 2019 and 2024).

#SBATCH --job-name=era5-store-val
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

~/miniconda3/envs/ring-space/bin/python \
    /data/ERA5-samples/build_store_era5.py \
    --split val --n-windows 20 --cubes-per-window 16 --verify

echo "Finished $(date)"
