#!/bin/bash
#SBATCH --job-name=dask-scheduler
#SBATCH --partition=all
#SBATCH --account=incus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00

pixi run --frozen python feature-detection-test02-numthresh.py