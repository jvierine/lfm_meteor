#!/usr/bin/env bash
#SBATCH --job-name=nov17-calibrated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=nov17-calibrated-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
bash campaign_20261117/run.slurm
bash campaign_20261117/physical.slurm
bash campaign_20261117/figures.slurm
