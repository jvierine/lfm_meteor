#!/usr/bin/env bash
#SBATCH --job-name=syisr-meteors
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=syisr-%j.log
set -euo pipefail
# Load the same compiler, HDF5 and MPI modules used by build.sh.
# module load gcc hdf5 openmpi
export OMP_NUM_THREADS=1
mpirun -np "${SLURM_NTASKS}" ./target/release/syisr-native process \
    --config config.toml --output "output-${SLURM_JOB_ID}"
