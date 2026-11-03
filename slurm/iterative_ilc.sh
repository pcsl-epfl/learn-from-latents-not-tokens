#!/bin/bash
#SBATCH --job-name=iterative-ilc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=1-18%3
#SBATCH --output=data/logs/iterative_%A_%a.out

set -euo pipefail
# CPU-only; bigmem is a Jed partition. Adjust partition/resources for other clusters.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
PYTHON="${PYTHON:-python3}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
cd "$REPO_ROOT"
printf 'Repository revision: '
git rev-parse HEAD
TASK="${SLURM_ARRAY_TASK_ID:-1}"
if (( TASK < 1 || TASK > 18 )); then echo 'Task must be in 1..18' >&2; exit 2; fi
MS=(6 8 12 16 22 32)
M="${MS[$(((TASK - 1) / 3))]}"
SEED="$(((TASK - 1) % 3))"
OUT=data/iterative_ilc/results
if [[ "${SMOKE:-0}" == 1 ]]; then OUT=data/iterative_ilc/smoke; fi
"$PYTHON" -u simulation/iterative_ilc/run.py --v 8 --m "$M" --s 3 --L 5 --rules_seed "$SEED" --output-dir "$OUT" "$@"
