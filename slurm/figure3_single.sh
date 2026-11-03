#!/bin/bash
#SBATCH --job-name=repro-fig3-m8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=mig24gb
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=data/logs/figure3_single_%j.out
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PYTHON="${PYTHON:-python3}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TQDM_DISABLE=1
cd "$REPO_ROOT"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/data/figure3_single_${SLURM_JOB_ID:-local}}"
printf 'Repository revision: '
git rev-parse HEAD
"$PYTHON" -u simulation/run_gradient.py --sweep simulation/figure3_single.csv \
  --task 1 --max-training-seconds 1200 --checkpoint-min-step 10000 --output-root "$RUN_ROOT"
"$PYTHON" -u simulation/aggregate_gradient.py --results "$RUN_ROOT/results" \
  --expected-sweep simulation/figure3_single.csv --output "$RUN_ROOT/summary/df_ptc_m_sweep_best.parquet"
