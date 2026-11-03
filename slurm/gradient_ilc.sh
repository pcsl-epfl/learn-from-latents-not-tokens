#!/bin/bash
#SBATCH --job-name=gradient-ilc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=1-165%10
#SBATCH --output=data/logs/gradient_%A_%a.out

set -euo pipefail
# Submit from the repository root after: mkdir -p data/logs
# Use an absolute interpreter path in an existing environment. No shell startup files.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PYTHON="${PYTHON:-python3}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TQDM_DISABLE=1
cd "$REPO_ROOT"
printf 'Repository revision: '
git rev-parse HEAD
"$PYTHON" -u simulation/run_gradient.py "$@"
