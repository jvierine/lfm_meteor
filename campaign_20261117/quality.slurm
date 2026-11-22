#!/usr/bin/env bash
#SBATCH --job-name=nov17-quality
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=nov17-quality-%j.log
set -euo pipefail
cd /home/juha/src/lfm_meteor
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u campaign_20261117/quality.py
bash campaign_20261117/figures.slurm
